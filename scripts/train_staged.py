"""
Staged training script for MultiModalCreature.

This script supports training in phases:
1. Phase 1: Text-only (internal/external voice encoder + text decoders + transformer)
2. Phase 2: Add audio (unfreeze audio encoder/decoder)
3. Phase 3: Add vision (unfreeze visual encoder)
4. Phase 4: Add body (unfreeze proprioception, touch, animation)

This approach allows:
- Faster initial training (fewer active parameters)
- Modular addition of capabilities
- Weight preservation between phases

Usage:
    # Phase 1: Text only
    python scripts/train_staged.py --config configs/training_fast.yaml \\
        --model-config configs/model_100m.yaml \\
        --data "path/to/tokens.bin" \\
        --phase text

    # Phase 2: Add audio (loads checkpoint from phase 1)
    python scripts/train_staged.py --config configs/training_fast.yaml \\
        --model-config configs/model_100m.yaml \\
        --data "path/to/multimodal_data" \\
        --phase audio \\
        --checkpoint checkpoints/staged/text_final.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import sys
import os
import time
import math
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multimodal_transformer import MultiModalCreature
from src.data.text_dataset import TextDataset
from src.config import Config


# Define which modules belong to each training phase
PHASE_MODULES = {
    "text": [
        "internal_voice_encoder",
        "external_voice_encoder",
        "internal_text_decoder",
        "external_text_decoder",
        "transformer",
        "fusion_module",
    ],
    "audio": [
        "audio_encoder",
        "audio_decoder",
    ],
    "vision": [
        "visual_encoder",
    ],
    "body": [
        "proprioception_encoder",
        "touch_encoder",
        "animation_decoder",
    ],
}

# Cumulative phases (each phase includes previous)
CUMULATIVE_PHASES = {
    "text": ["text"],
    "audio": ["text", "audio"],
    "vision": ["text", "audio", "vision"],
    "full": ["text", "audio", "vision", "body"],
}


def get_trainable_modules(phase: str) -> list:
    """Get list of module names that should be trainable for a given phase."""
    phases_to_include = CUMULATIVE_PHASES.get(phase, ["text"])
    modules = []
    for p in phases_to_include:
        modules.extend(PHASE_MODULES.get(p, []))
    return modules


def freeze_model_except(model: torch.nn.Module, trainable_module_names: list):
    """
    Freeze all parameters except those in specified modules.

    Args:
        model: The MultiModalCreature model
        trainable_module_names: List of module attribute names to keep trainable
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze specified modules
    unfrozen_params = 0
    for name in trainable_module_names:
        if hasattr(model, name):
            module = getattr(model, name)
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - unfrozen_params

    print(f"Parameter breakdown:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {unfrozen_params:,} ({100*unfrozen_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    return unfrozen_params


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Staged multi-modal training")
    parser.add_argument("--config", type=str, default="configs/training_fast.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_100m.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--phase", type=str, default="text",
                        choices=["text", "audio", "vision", "full"],
                        help="Training phase (determines which modules are trainable)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    args = parser.parse_args()

    # ========== PERFORMANCE OPTIMIZATIONS ==========
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"STAGED TRAINING - Phase: {args.phase.upper()}")
    print("=" * 70)

    # Load config
    config_obj = Config.from_files(
        training_config_path=args.config,
        model_config_path=args.model_config
    )

    config = {}
    if config_obj.model:
        config["model"] = config_obj.model.get("model", config_obj.model)
    if config_obj.training:
        config["training"] = config_obj.training.get("training", config_obj.training)

    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    # Settings
    seq_length = args.seq_length
    batch_size = args.batch_size or training_cfg.get("batch_size", 8)
    max_steps = args.max_steps or training_cfg.get("max_steps", 100000)

    opt_cfg = training_cfg.get("optimizer", {})
    max_lr = float(opt_cfg.get("lr", 6e-4))
    min_lr = float(training_cfg.get("scheduler", {}).get("min_lr", 6e-5))
    warmup_steps = training_cfg.get("scheduler", {}).get("warmup_steps", 1000)
    weight_decay = float(opt_cfg.get("weight_decay", 0.1))
    grad_accum = training_cfg.get("gradient_accumulation_steps", 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ========== DATA ==========
    print(f"\nLoading data from {args.data}...")

    # For text phase, use TextDataset
    if args.phase == "text":
        dataset = TextDataset(args.data, seq_length=seq_length)
    else:
        # For other phases, would use MultiModalDataset
        # For now, still support text data
        dataset = TextDataset(args.data, seq_length=seq_length)

    # Use multiprocessing on Linux/WSL, not on Windows
    num_workers = 4 if os.name != 'nt' else 0
    if num_workers == 0:
        print("Note: Using num_workers=0. For better performance, use WSL2 or Linux.")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # ========== MODEL ==========
    print("\nInitializing MultiModalCreature...")

    # Wrap config properly for MultiModalCreature
    full_config = {"model": model_cfg} if "model" not in config else config
    model = MultiModalCreature(full_config).to(device)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded weights from step {ckpt.get('step', 'unknown')}")

    # Freeze parameters based on phase
    print(f"\nConfiguring for phase: {args.phase}")
    trainable_modules = get_trainable_modules(args.phase)
    print(f"Trainable modules: {trainable_modules}")
    trainable_params = freeze_model_except(model, trainable_modules)

    # Compile if requested
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    # ========== OPTIMIZER ==========
    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True if device == "cuda" else False,
    )

    scaler = GradScaler()
    use_amp = device == "cuda"

    # ========== TRAINING ==========
    print(f"\n{'=' * 70}")
    print(f"Training config:")
    print(f"  Phase: {args.phase}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"{'=' * 70}\n")

    model.train()
    optimizer.zero_grad()

    save_dir = Path(f"checkpoints/staged/{args.phase}")
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    tokens_processed = 0
    start_time = time.time()
    log_interval = 10
    save_interval = 5000

    data_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, desc="Training")

    # Get vocab size from model
    vocab_size = model.internal_text_decoder.vocab_size

    while step < max_steps:
        step_start = time.time()

        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input"].to(device)
            targets = batch["target"].to(device)

            with autocast(dtype=torch.float16, enabled=use_amp):
                # Forward pass - only use internal_voice for text-only phase
                outputs = model(
                    internal_voice_tokens=input_ids,
                    return_hidden_states=True,
                )

                # Get hidden states and compute loss manually
                hidden_states = outputs["hidden_states"]

                # Use internal text decoder to get logits
                logits = model.internal_text_decoder.projection(hidden_states)

                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            tokens_processed += input_ids.numel()

        # Update
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            1.0
        )

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        step_time = time.time() - step_start

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            if device == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1e9
            else:
                mem_used = 0

            pbar.set_description(
                f"Loss: {loss.item() * grad_accum:.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"LR: {lr:.2e} | "
                f"Mem: {mem_used:.1f}GB"
            )

        pbar.update(1)

        if step % save_interval == 0:
            ckpt_path = save_dir / f"{args.phase}_step_{step}.pt"
            torch.save({
                'step': step,
                'phase': args.phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"\nSaved checkpoint to {ckpt_path}")

    pbar.close()

    # Final save
    final_path = save_dir / f"{args.phase}_final.pt"
    torch.save({
        'step': step,
        'phase': args.phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Phase '{args.phase}' complete!")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Average throughput: {tokens_processed / elapsed:,.0f} tokens/sec")
    print(f"Final model saved to {final_path}")
    print(f"\nTo continue with next phase, run:")
    print(f"  python scripts/train_staged.py --phase <next_phase> --checkpoint {final_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
