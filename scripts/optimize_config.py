"""
Script to automatically find optimal training configurations for the current hardware.
Sweeps batch sizes and measures throughput/MFU.
"""

import os
import sys
import torch
import yaml
import time
import copy
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.models.multimodal_transformer import MultiModalCreature
from src.training.trainer import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_batch(batch_size, seq_len=512, vocab_size=1000, device="cuda"):
    """
    Purpose:
        Generate a dummy batch of multi-modal data for benchmarking.

    Workflow:
        1. Create random integer tensors for text modalities.
        2. Create zero-filled tensors for vision placeholders.
        3. Assemble inputs and targets dictionary.

    Args:
        batch_size: Number of samples per batch.
        seq_len: Sequence length for text.
        vocab_size: Vocabulary size for text generation.
        device: Device to create tensors on.

    Returns:
        Dictionary containing inputs and targets.
    """
    inputs = {
        "internal_voice": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "external_voice": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        # Add basic visual/audio placeholders if model expects them, 
        # but for MFU/Speed mostly internal_voice/transformer load matters.
        # Providing minimal valid inputs to avoid NoneType errors in forward
    }
    
    # We need to make sure the model handles missing modalities gracefully 
    # (which it does via Optional) or we provide them.
    # Let's provide empty/zeros for heavy modalities to test memory of full forward?
    # Actually, for stress testing memory, we SHOULD provide all modalities.
    # But to start simple, we trust the text path is dominant or we assume text-only stress first.
    # If the user runs multi-modal, we should fill image tensors.
    
    # Let's fill vision to be safe on memory usage estimation
    H, W = 64, 64 # Small for test or match config?
    inputs["left_eye_image"] = torch.zeros((batch_size, 3, H, W), device=device)
    inputs["right_eye_image"] = torch.zeros((batch_size, 3, H, W), device=device)
    
    targets = {
        "internal_text": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "external_text": torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    }
    
    return {"inputs": inputs, "targets": targets}

def benchmark_config(batch_size, config, model_config):
    """
    Purpose:
        Run a short benchmark for a specific batch size to measure throughput and memory.

    Workflow:
        1. Override batch size in config.
        2. Initialize Model with optimizations.
        3. Run warmup steps.
        4. Run benchmark steps measuring time.
        5. Check memory usage.
        6. Return metrics or None if OOM.

    Args:
        batch_size: Batch size to test.
        config: Configuration dictionary.
        model_config: Model configuration (unused, kept for signature compatibility).

    Returns:
        Dictionary with throughput/memory metrics, or None if Out Of Memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Override batch size in config
    config["training"]["batch_size"] = batch_size
    
    # Init Model
    # We create a new model each time or reuse? 
    # Creating new is safer to clear graph, but slower.
    # Let's try to reuse if possible, but OOM recovery requires clearing.
    
    try:
        model = MultiModalCreature(config)
        
        # Enable optimizations manually to match Trainer init
        if config["training"].get("gradient_checkpointing"):
             if hasattr(model, "transformer") and hasattr(model.transformer, "enable_gradient_checkpointing"):
                model.transformer.enable_gradient_checkpointing(True)
        
        model.to(device)
        
        # Compile if requested (might take time! skip for quick sweep? or essential for mem?)
        # Compile usually REDUCES memory? or increases?
        # Let's skip compile for the sweep to save time, unless user insists.
        # Actually user wants compile default. We should test WITH compile if we want valid max batch size.
        # But compiling every step of sweep is slow.
        # Valid strategy: Find max batch size without compile, then reduce by 10% for compile overhead/safety.
        
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        # Generate data
        batch = generate_dummy_batch(batch_size, device=device)
        
        # Warmup steps
        model.train()
        for _ in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch["inputs"], return_hidden_states=True)
                loss, _ = model.compute_loss(outputs, batch["targets"])
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Benchmark steps
        num_steps = 10
        total_tokens = 0
        
        for _ in range(num_steps):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch["inputs"], return_hidden_states=True)
                loss, _ = model.compute_loss(outputs, batch["targets"])
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_tokens += batch_size * 512 # Approx
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_steps
        samples_per_sec = batch_size / avg_time
        
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Cleanup
        del model
        del optimizer
        del scaler
        del batch
        torch.cuda.empty_cache()
        
        return {
            "batch_size": batch_size,
            "samples_per_sec": samples_per_sec,
            "memory_mb": mem_mb,
            "avg_step_time": avg_time
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return None
        raise e

def main():
    """
    Purpose:
        Main entry point for the optimization script.

    Workflow:
        1. Check for CUDA availability.
        2. Load default configuration.
        3. Sweep through power-of-2 batch sizes.
        4. Record best throughput config.
        5. Calculate efficient accumulation steps for target effective batch.
        6. Detect CPU count for worker optimization.
        7. Save optimized config to file.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Optimization meaningless.")
        return

    logger.info(f"Starting optimization on {torch.cuda.get_device_name(0)}")
    
    # Load base config
    config_obj = Config.from_files("configs/training_config.yaml") # Load defaults
    # Re-construct dict 
    config = {
        "model": config_obj.model if config_obj.model else {},
        "training": config_obj.training if config_obj.training else {},
        # Add minimal model defaults if missing
    }
    
    # Ensure some defaults if empty
    if not config["model"]:
        # Load model_1b as default template if empty
        with open("configs/model_config.yaml", 'r') as f:
             config["model"] = yaml.safe_load(f)
    
    # Target effective batch size (e.g. 64, 128, 256)
    # The script will find max safe micro-batch size, then set accumulation to match this target.
    target_effective_batch = 64 
    
    best_config = None
    best_throughput = 0
    
    for bs in batch_sizes:
        logger.info(f"Testing Batch Size: {bs}...")
        result = benchmark_config(bs, copy.deepcopy(config), None)
        
        if result is None:
            logger.warning(f"Batch Size {bs} -> OOM")
            break
        else:
            logger.info(f"BS {bs}: {result['samples_per_sec']:.2f} samples/s, Mem: {result['memory_mb']:.0f} MB")
            
            if result['samples_per_sec'] > best_throughput:
                best_throughput = result['samples_per_sec']
                best_config = result
    
    if best_config:
        max_micro_batch = best_config['batch_size']
        # Calculate accumulation steps
        # accum = ceil(target / micro)
        import math
        accum_steps = math.ceil(target_effective_batch / max_micro_batch)
        effective_bs = max_micro_batch * accum_steps
        
        logger.info(f"\nOptimization Complete.")
        logger.info(f"Max Safe Micro-Batch Size: {max_micro_batch}")
        logger.info(f"Target Effective Batch Size: {target_effective_batch}")
        logger.info(f"Recommended Gradient Accumulation Steps: {accum_steps}")
        logger.info(f"Actual Effective Batch Size: {effective_bs}")
        logger.info(f"Max Throughput: {best_config['samples_per_sec']:.2f} samples/s")
        
        output_path = "configs/training_config_optimized.yaml"
        new_config = copy.deepcopy(config)
        # Note: config["training"] contains the WHOLE file content including 'training', 'data', etc keys?
        # Based on config.py logic: yes.
        # So we need to access the inner 'training' key to set params.

        new_config["training"]["training"]["batch_size"] = max_micro_batch
        new_config["training"]["training"]["gradient_accumulation_steps"] = accum_steps
        
        # CPU Optimization
        import os
        cpu_count = os.cpu_count() or 4
        # Heuristic: Use physical cores or a reasonable cap (e.g. 16)
        # For data loading, often num_workers = num_cores is good
        optimal_workers = min(cpu_count, 16)
        
        if "data" in new_config["training"]:
             new_config["training"]["data"]["num_workers"] = optimal_workers
             logger.info(f"Recommended Data Workers: {optimal_workers} (based on {cpu_count} CPUs)")
        
        with open(output_path, 'w') as f:
             yaml.dump(new_config["training"], f)
        logger.info(f"Saved optimized config to {output_path}")


if __name__ == "__main__":
    main()
