"""
Fast text-only training script optimized for speed.

This uses a lightweight GPT-2 style model without multi-modal overhead,
matching the training approach from:
https://www.gilesthomas.com/2025/12/llm-from-scratch-28-training-a-base-model-from-scratch

Key optimizations:
- Pure text transformer (no unused encoders/decoders)
- TF32 tensor cores enabled
- FP16 mixed precision (better on RTX 3090 than BF16)
- Multi-worker data loading
- No gradient checkpointing (not needed for <500M models on 24GB)
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.text_only_transformer import TextOnlyTransformer
from src.data.text_dataset import TextDataset
from src.config import Config


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Fast text-only training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_100m.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to tokenized .bin file")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-length", type=int, default=None, help="Override sequence length")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # ========== CRITICAL PERFORMANCE OPTIMIZATIONS ==========

    # 1. Enable TF32 tensor cores (22% speedup on RTX 3090)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. Enable cuDNN benchmarking for convolutions
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("FAST TEXT-ONLY TRAINING")
    print("=" * 70)
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Load config
    config_obj = Config.from_files(
        training_config_path=args.config,
        model_config_path=args.model_config
    )

    config = {}
    if config_obj.model:
        config.update(config_obj.model)
    if config_obj.training:
        config.update(config_obj.training)

    # Override settings for speed
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    seq_length = args.seq_length or model_cfg.get("encoders", {}).get("internal_voice", {}).get("max_seq_length", 512)
    batch_size = args.batch_size or training_cfg.get("batch_size", 16)
    max_steps = args.max_steps or training_cfg.get("max_steps", 100000)

    # Learning rate settings
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
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========== DATA LOADING ==========
    print(f"\nLoading data from {args.data}...")
    dataset = TextDataset(args.data, seq_length=seq_length)

    # 3. Multi-worker data loading (critical for throughput)
    num_workers = 4 if os.name != 'nt' else 0  # Windows doesn't handle multiprocessing well
    if os.name == 'nt':
        print("Warning: Windows detected, using num_workers=0. Consider using Linux for better performance.")

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
    print("\nInitializing TextOnlyTransformer...")
    model = TextOnlyTransformer(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 4. Optionally compile model (can help after warmup)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ========== OPTIMIZER ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True if device == "cuda" else False,  # Fused AdamW is faster
    )

    # 5. FP16 mixed precision (better than BF16 on RTX 3090)
    scaler = GradScaler()
    use_amp = device == "cuda"

    # ========== RESUME FROM CHECKPOINT ==========
    start_step = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step}")

    # ========== TRAINING LOOP ==========
    print(f"\n{'=' * 70}")
    print(f"Training config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Tokens per step: {batch_size * seq_length * grad_accum:,}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Learning rate: {max_lr} -> {min_lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Mixed precision: FP16")
    print(f"  Num workers: {num_workers}")
    print(f"{'=' * 70}\n")

    model.train()
    optimizer.zero_grad()

    save_dir = Path("checkpoints/text_fast")
    save_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    tokens_processed = 0
    start_time = time.time()
    log_interval = 10
    save_interval = 5000

    data_iter = iter(train_loader)

    pbar = tqdm(total=max_steps - start_step, initial=0, desc="Training")

    while step < max_steps:
        step_start = time.time()

        # Accumulate gradients
        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input"].to(device)
            targets = batch["target"].to(device)

            # Forward with mixed precision
            with autocast(dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids)
                logits = outputs["logits"]

                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    targets.view(-1),
                )
                loss = loss / grad_accum

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            tokens_processed += input_ids.numel()

        # Update weights
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        step_time = time.time() - step_start

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            samples_per_sec = batch_size * grad_accum / step_time

            # Memory stats
            if device == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                mem_used = mem_reserved = 0

            pbar.set_description(
                f"Loss: {loss.item() * grad_accum:.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"{samples_per_sec:.1f} samp/s | "
                f"LR: {lr:.2e} | "
                f"Mem: {mem_used:.1f}GB"
            )

        pbar.update(1)

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = save_dir / f"model_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"\nSaved checkpoint to {ckpt_path}")

    pbar.close()

    # Final save
    final_path = save_dir / "model_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Average throughput: {tokens_processed / elapsed:,.0f} tokens/sec")
    print(f"Final model saved to {final_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
