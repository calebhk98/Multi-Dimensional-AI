# Model Scaling Estimates

Comprehensive estimates for training Multi-Dimensional AI models at various scales with exact assumptions documented.

## Exact Assumptions

These estimates are based on the following specific parameters:

| Parameter                  | Value       | Notes                     |
| -------------------------- | ----------- | ------------------------- |
| **Sequence Length**        | 2048 tokens | Per modality stream       |
| **Frame Rate**             | 30 FPS      | For vision/proprioception |
| **Audio Sample Rate**      | 48 kHz      | Mono audio                |
| **Batch Size**             | 32          | Per GPU effective batch   |
| **Precision**              | BF16        | Mixed precision training  |
| **Gradient Checkpointing** | Enabled     | For models >500M params   |
| **Optimizer**              | AdamW       | With weight decay 0.01    |
| **Learning Rate**          | 3e-4        | With cosine schedule      |

## Baseline Configuration

| Parameter  | Value |
| ---------- | ----- |
| Hidden dim | 512   |
| Layers     | 6     |
| Heads      | 8     |
| Context    | 2048  |

## Scaling Table

| Scale      | Params | Training Samples | Training Steps | Tokens Seen | GPU Hours (RTX 3090) | GPU Hours (2x RTX 3090) | GPU Hours (A100 80GB) |
| ---------- | ------ | ---------------- | -------------- | ----------- | -------------------- | ----------------------- | --------------------- |
| **Tiny**   | 10M    | 1M               | 31,250         | ~64M        | ~15                  | ~8                      | ~10                   |
| **Small**  | 100M   | 10M              | 312,500        | ~640M       | ~200                 | ~100                    | ~100                  |
| **Medium** | 500M   | 50M              | 1,562,500      | ~3.2B       | ~1200                | ~600                    | ~500                  |
| **Large**  | 1B     | 100M             | 3,125,000      | ~6.4B       | ~2500                | ~1250                   | ~1000                 |
| **XL**     | 3B     | 300M             | 9,375,000      | ~19.2B      | ~8000                | ~4000                   | ~3000                 |

## 10M Parameter Model (Tiny) - Detailed Estimates

### Hardware-Specific Training Times

#### RTX 3090 (24GB, 350W TDP)

- **Conservative Estimate**: 15-20 GPU hours
- **Throughput**: ~800-1000 samples/sec
- **Effective Tokens/sec**: ~1,600,000-2,000,000
- **Wall Clock**: ~16-20 hours (continuous training)
- **Limitations**: None - model fits comfortably in memory with headroom

#### 2x RTX 3090 (Data Parallel)

- **Conservative Estimate**: 8-10 GPU hours
- **Throughput**: ~1400-1800 samples/sec (w/ DDP overhead)
- **Effective Tokens/sec**: ~2,800,000-3,600,000
- **Wall Clock**: ~8-10 hours (continuous training)
- **Notes**: Near-perfect scaling achieved; DDP overhead minimal for small models

#### A100 80GB (400W TDP)

- **Conservative Estimate**: 10-12 GPU hours
- **Throughput**: ~1000-1200 samples/sec
- **Effective Tokens/sec**: ~2,000,000-2,400,000
- **Wall Clock**: ~10-12 hours (continuous training)
- **Advantages**: Higher memory bandwidth, can use larger batch sizes (64-128)

### Detailed Breakdown (10M params, Tiny Configuration)

Based on recommended training steps for Tiny model:

| Metric                           | Value          | Calculation                                   |
| -------------------------------- | -------------- | --------------------------------------------- |
| **Total Steps**                  | 31,250         | For 1M samples ÷ batch 32                     |
| **Effective Batch Size**         | 32             | Per GPU or accumulated                        |
| **Tokens per Step**              | 4              | 1 token emitted per output stream (4 streams) |
| **Total Output Tokens**          | ~4,000,000     | 31,250 steps × 32 batch × 4 tokens            |
| **Total Samples**                | ~1,000,000     | 31,250 steps × 32 batch                       |
| **Input Tokens per Sample**      | ~8,192         | Across 6 input modalities                     |
| **Total Input Tokens Processed** | ~8,192,000,000 | 1M samples × 8,192 tokens                     |

### Memory Requirements (BF16)

| Component           | Size   | Notes                                    |
| ------------------- | ------ | ---------------------------------------- |
| **Model Weights**   | 40 MB  | 10M params × 4 bytes (FP32 master copy)  |
| **Optimizer State** | 120 MB | AdamW: 2× params for momentum + variance |
| **Activations**     | 1.5 GB | Batch 32, sequence 2048                  |
| **Gradients**       | 40 MB  | Same size as weights                     |
| **Peak Total**      | ~2 GB  | Plenty of headroom on any modern GPU     |

### Training Characteristics

- **Convergence**: Fast - typically sees good results after 10-15K steps
- **Stability**: Very stable - can use higher learning rates safely
- **Overfitting Risk**: Moderate - ensure sufficient data diversity
- **Checkpoint Size**: ~40 MB per checkpoint (compress to ~10 MB)
- **Evaluation Time**: ~30 seconds on validation set (10K samples)

### Cost Analysis

| Hardware      | Hourly Rate | Total Cost | Cost per 1M Samples |
| ------------- | ----------- | ---------- | ------------------- |
| RTX 3090      | $1.10/hr    | ~$17-22    | ~$17-22             |
| 2x RTX 3090   | $2.20/hr    | ~$18-22    | ~$18-22             |
| A100 80GB     | $2.50/hr    | ~$25-30    | ~$25-30             |
| **Spot A100** | $0.80/hr    | ~$8-10     | ~$8-10              |

> [!TIP]
> The Tiny model is ideal for pipeline validation and rapid experimentation. Train overnight on a single consumer GPU to verify data quality and model architecture before scaling up.

## 100M Parameter Model (Small) - Detailed Estimates

### Hardware-Specific Training Times

#### RTX 3090 (24GB, 350W TDP)

- **Conservative Estimate**: 200-250 GPU hours
- **Throughput**: ~150-200 samples/sec
- **Effective Tokens/sec**: ~300,000-400,000
- **Wall Clock**: ~8-10 days (continuous training)
- **Limitations**: Manageable - batch size 32 works well, some memory headroom

#### 2x RTX 3090 (Data Parallel)

- **Conservative Estimate**: 100-125 GPU hours
- **Throughput**: ~280-350 samples/sec (w/ DDP overhead)
- **Effective Tokens/sec**: ~560,000-700,000
- **Wall Clock**: ~4-5 days (continuous training)
- **Notes**: Good scaling efficiency (~85-90%); ideal setup for this model size

#### A100 80GB (400W TDP)

- **Conservative Estimate**: 100-120 GPU hours
- **Throughput**: ~300-350 samples/sec
- **Effective Tokens/sec**: ~600,000-700,000
- **Wall Clock**: ~4-5 days (continuous training)
- **Advantages**: Can use batch size 64, better memory bandwidth, consistent performance

### Detailed Breakdown (100M params, Small Configuration)

Based on recommended training steps for Small model:

| Metric                           | Value           | Calculation                                   |
| -------------------------------- | --------------- | --------------------------------------------- |
| **Total Steps**                  | 312,500         | For 10M samples ÷ batch 32                    |
| **Effective Batch Size**         | 32              | Per GPU or accumulated                        |
| **Tokens per Step**              | 4               | 1 token emitted per output stream (4 streams) |
| **Total Output Tokens**          | ~40,000,000     | 312,500 steps × 32 batch × 4 tokens           |
| **Total Samples**                | ~10,000,000     | 312,500 steps × 32 batch                      |
| **Input Tokens per Sample**      | ~8,192          | Across 6 input modalities                     |
| **Total Input Tokens Processed** | ~81,920,000,000 | 10M samples × 8,192 tokens                    |

### Memory Requirements (BF16)

| Component           | Size   | Notes                                    |
| ------------------- | ------ | ---------------------------------------- |
| **Model Weights**   | 400 MB | 100M params × 4 bytes (FP32 master copy) |
| **Optimizer State** | 1.2 GB | AdamW: 2× params for momentum + variance |
| **Activations**     | 6 GB   | Batch 32, sequence 2048                  |
| **Gradients**       | 400 MB | Same size as weights                     |
| **Peak Total**      | ~8 GB  | Comfortable fit on RTX 3090 (24GB)       |

### Training Characteristics

- **Convergence**: Steady - evaluation plateau typically around 150-200K steps
- **Stability**: Stable - standard learning rate schedules work well
- **Overfitting Risk**: Low with 10M+ samples - model has good capacity
- **Checkpoint Size**: ~400 MB per checkpoint (compress to ~100 MB)
- **Evaluation Time**: ~3-5 minutes on validation set (100K samples)
- **Gradient Checkpointing**: Optional - helps if using larger batches

### Cost Analysis

| Hardware      | Hourly Rate | Total Cost | Cost per 1M Samples | Days to Complete |
| ------------- | ----------- | ---------- | ------------------- | ---------------- |
| RTX 3090      | $1.10/hr    | ~$220-275  | ~$22-27             | 8-10 days        |
| 2x RTX 3090   | $2.20/hr    | ~$220-275  | ~$22-27             | 4-5 days         |
| A100 80GB     | $2.50/hr    | ~$250-300  | ~$25-30             | 4-5 days         |
| **Spot A100** | $0.80/hr    | ~$80-96    | ~$8-10              | 4-5 days         |

### Use Cases & Performance Expectations

- **Capability Level**: Solid baseline for multi-modal understanding
- **Quality**: Good cross-modal translations for simple patterns
- **Production Readiness**: Suitable for prototypes and MVPs
- **Recommended For**: Initial production deployments, A/B testing baselines
- **Limitations**: May struggle with complex multi-step reasoning

> [!NOTE]
> The 100M Small model represents the minimum scale for production deployment. It provides a good balance of training cost (~$220) vs. capability, making it ideal for validating your data quality before committing to larger model training.

## 500M Parameter Model (Medium) - Detailed Estimates

### Hardware-Specific Training Times

#### RTX 3090 (24GB, 350W TDP)

- **Conservative Estimate**: 1200-1500 GPU hours
- **Throughput**: ~40-50 samples/sec
- **Effective Tokens/sec**: ~80,000-100,000
- **Wall Clock**: ~50-63 days (continuous training)
- **Limitations**: **CRITICAL** - Requires gradient checkpointing; batch size may need reduction to 16-24

#### 2x RTX 3090 (Data Parallel)

- **Conservative Estimate**: 600-750 GPU hours
- **Throughput**: ~70-90 samples/sec (w/ DDP overhead)
- **Effective Tokens/sec**: ~140,000-180,000
- **Wall Clock**: ~25-32 days (continuous training)
- **Notes**: Good scaling efficiency (~80-85%); gradient checkpointing required; recommended minimum setup

#### A100 80GB (400W TDP)

- **Conservative Estimate**: 500-600 GPU hours
- **Throughput**: ~90-110 samples/sec
- **Effective Tokens/sec**: ~180,000-220,000
- **Wall Clock**: ~21-25 days (continuous training)
- **Advantages**: Full batch size 32-64 without checkpointing; superior memory bandwidth; more reliable

### Detailed Breakdown (500M params, Medium Configuration)

Based on recommended training steps for Medium model:

| Metric                           | Value            | Calculation                                   |
| -------------------------------- | ---------------- | --------------------------------------------- |
| **Total Steps**                  | 1,562,500        | For 50M samples ÷ batch 32                    |
| **Effective Batch Size**         | 32               | May require gradient accumulation on 3090     |
| **Tokens per Step**              | 4                | 1 token emitted per output stream (4 streams) |
| **Total Output Tokens**          | ~200,000,000     | 1,562,500 steps × 32 batch × 4 tokens         |
| **Total Samples**                | ~50,000,000      | 1,562,500 steps × 32 batch                    |
| **Input Tokens per Sample**      | ~8,192           | Across 6 input modalities                     |
| **Total Input Tokens Processed** | ~409,600,000,000 | 50M samples × 8,192 tokens                    |

### Memory Requirements (BF16)

| Component           | Size   | Notes                                               |
| ------------------- | ------ | --------------------------------------------------- |
| **Model Weights**   | 2 GB   | 500M params × 4 bytes (FP32 master copy)            |
| **Optimizer State** | 6 GB   | AdamW: 2× params for momentum + variance            |
| **Activations**     | 15 GB  | Batch 32, sequence 2048 (w/o checkpointing)         |
| **Gradients**       | 2 GB   | Same size as weights                                |
| **Peak Total**      | ~25 GB | **Exceeds 3090 - gradient checkpointing mandatory** |

#### With Gradient Checkpointing

| Component        | Size Modified  | Reduction  |
| ---------------- | -------------- | ---------- |
| **Activations**  | ~4-6 GB        | 60-75%     |
| **Peak Total**   | ~14-16 GB      | Fits 3090  |
| **Speed Impact** | ~20-30% slower | Acceptable |

### Training Characteristics

- **Convergence**: Gradual - meaningful improvements continue through full training run
- **Stability**: Requires careful tuning - learning rate warmup essential
- **Overfitting Risk**: Very low with 50M+ samples - model benefits from all data
- **Checkpoint Size**: ~2 GB per checkpoint (compress to ~500 MB)
- **Evaluation Time**: ~15-20 minutes on validation set (500K samples)
- **Gradient Checkpointing**: **Required** on RTX 3090, optional on A100

### Cost Analysis

| Hardware      | Hourly Rate | Total Cost    | Cost per 1M Samples | Days to Complete |
| ------------- | ----------- | ------------- | ------------------- | ---------------- |
| RTX 3090      | $1.10/hr    | ~$1,320-1,650 | ~$26-33             | 50-63 days       |
| 2x RTX 3090   | $2.20/hr    | ~$1,320-1,650 | ~$26-33             | 25-32 days       |
| A100 80GB     | $2.50/hr    | ~$1,250-1,500 | ~$25-30             | 21-25 days       |
| **Spot A100** | $0.80/hr    | ~$400-480     | ~$8-10              | 21-25 days       |

### Use Cases & Performance Expectations

- **Capability Level**: Professional-grade multi-modal understanding
- **Quality**: Strong cross-modal translations with nuanced patterns
- **Production Readiness**: Suitable for production services with quality requirements
- **Recommended For**: Production deployments, customer-facing applications
- **Limitations**: May need fine-tuning for highly specialized domains
- **Sweet Spot**: Best balance of cost vs. capability for most production use cases

### Critical Considerations

> [!WARNING]
> The 500M Medium model pushes RTX 3090 to its limits. Training will be **slow** (~50-63 days on single GPU) and **requires gradient checkpointing**. Consider 2x RTX 3090 setup or A100 for practical training timelines.

> [!IMPORTANT]
> At this scale, **checkpoint management becomes critical**. With 2 GB per checkpoint and ~31K checkpoints (saving every 5%), you'll need 60+ GB storage. Implement aggressive checkpoint pruning (keep every 10th after first 50%).

> [!TIP]
> **Spot A100 is the winner** for this scale - at ~$400-480 total cost and 21-25 day timeline, it's far more practical than months on consumer hardware. The cost savings vs. time saved easily justify the slightly higher hourly rate.

## 1B Parameter Model - Detailed Estimates

### Hardware-Specific Training Times

#### RTX 3090 (24GB, 350W TDP)

- **Conservative Estimate**: 2500-3000 GPU hours
- **Throughput**: ~40-50 samples/sec
- **Effective Tokens/sec**: ~80,000-100,000
- **Wall Clock**: ~105-125 days (continuous training)
- **Limitations**: Memory constraints may require batch size 16-24

#### 2x RTX 3090 (Data Parallel)

- **Conservative Estimate**: 1250-1500 GPU hours
- **Throughput**: ~80-100 samples/sec (w/ DDP overhead)
- **Effective Tokens/sec**: ~160,000-200,000
- **Wall Clock**: ~50-65 days (continuous training)
- **Notes**: Near-linear scaling achieved with proper DDP setup

#### A100 80GB (400W TDP)

- **Conservative Estimate**: 1000-1200 GPU hours
- **Throughput**: ~100-120 samples/sec
- **Effective Tokens/sec**: ~200,000-240,000
- **Wall Clock**: ~40-50 days (continuous training)
- **Advantages**: Larger batch sizes (32-64), better memory bandwidth

### Detailed Breakdown (1B params, configs/training_config.yaml)

Based on `max_steps: 1,000,000` in configs:

| Metric                           | Value            | Calculation                                   |
| -------------------------------- | ---------------- | --------------------------------------------- |
| **Total Steps**                  | 1,000,000        | From config                                   |
| **Effective Batch Size**         | 64               | 32/GPU × 2 GPUs or gradient accumulation      |
| **Tokens per Step**              | 4                | 1 token emitted per output stream (4 streams) |
| **Total Output Tokens**          | ~256,000,000     | 1M steps × 64 batch × 4 tokens                |
| **Total Samples**                | ~64,000,000      | 1M steps × 64 batch                           |
| **Input Tokens per Sample**      | ~8,192           | Across 6 input modalities                     |
| **Total Input Tokens Processed** | ~524,000,000,000 | 64M samples × 8192 tokens                     |

## 3B Parameter Model (XL) - Detailed Estimates

### Hardware-Specific Training Times

#### RTX 3090 (24GB, 350W TDP) - Single GPU

- **Conservative Estimate**: **NOT RECOMMENDED** - Insufficient memory even with aggressive checkpointing
- **Throughput**: N/A - Model cannot fit in 24GB with optimizer states
- **Effective Tokens/sec**: N/A
- **Wall Clock**: N/A
- **Limitations**: **CRITICAL** - 3B params require ~48-60 GB peak memory; single 3090 is inadequate

#### 4x RTX 3090 (Model + Data Parallel)

- **Conservative Estimate**: 2000-2500 GPU hours (500-625 hours wall clock)
- **Throughput**: ~60-75 samples/sec (w/ pipeline + DDP overhead)
- **Effective Tokens/sec**: ~120,000-150,000
- **Wall Clock**: ~21-26 days (continuous training)
- **Notes**: Requires **ZeRO-3** or model parallelism; complex setup; 75-80% scaling efficiency
- **Complexity**: Advanced distributed training expertise required

#### 4x A100 80GB (Recommended Minimum)

- **Conservative Estimate**: 750-1000 GPU hours (188-250 hours wall clock)
- **Throughput**: ~150-180 samples/sec
- **Effective Tokens/sec**: ~300,000-360,000
- **Wall Clock**: ~8-10 days (continuous training)
- **Advantages**: Sufficient memory per GPU; can use tensor parallelism; superior interconnect (NVLink)
- **Configuration**: Tensor parallel 2-way + data parallel 2-way, or full data parallel with ZeRO-3

#### 8x A100 80GB (Optimal Setup)

- **Conservative Estimate**: 375-500 GPU hours (47-63 hours wall clock)
- **Throughput**: ~280-340 samples/sec
- **Effective Tokens/sec**: ~560,000-680,000
- **Wall Clock**: ~2-3 days (continuous training)
- **Advantages**: Near-linear scaling; faster iteration; can afford larger batch sizes
- **Notes**: Best price/performance for 3B scale when wall-clock time matters

### Detailed Breakdown (3B params, XL Configuration)

Based on recommended training steps for XL model:

| Metric                           | Value              | Calculation                                   |
| -------------------------------- | ------------------ | --------------------------------------------- |
| **Total Steps**                  | 9,375,000          | For 300M samples ÷ batch 32                   |
| **Effective Batch Size**         | 32-64              | Distributed across GPUs                       |
| **Tokens per Step**              | 4                  | 1 token emitted per output stream (4 streams) |
| **Total Output Tokens**          | ~1,200,000,000     | 9,375,000 steps × 32 batch × 4 tokens         |
| **Total Samples**                | ~300,000,000       | 9,375,000 steps × 32 batch                    |
| **Input Tokens per Sample**      | ~8,192             | Across 6 input modalities                     |
| **Total Input Tokens Processed** | ~2,457,600,000,000 | 300M samples × 8,192 tokens (~2.5 trillion)   |

### Memory Requirements (BF16)

#### Per-GPU Breakdown (without model parallelism)

| Component           | Total Size  | Per GPU (4x A100) | Notes                                    |
| ------------------- | ----------- | ----------------- | ---------------------------------------- |
| **Model Weights**   | 12 GB       | 3 GB (sharded)    | With ZeRO-3 optimizer state sharding     |
| **Optimizer State** | 36 GB       | 9 GB (sharded)    | AdamW: 2× params for momentum + variance |
| **Activations**     | 40-60 GB    | 10-15 GB          | Batch 32, sequence 2048, per GPU         |
| **Gradients**       | 12 GB       | 3 GB (sharded)    | Same size as weights                     |
| **Peak Total**      | ~100-120 GB | ~25-30 GB/GPU     | With ZeRO-3 sharding                     |

#### With Tensor Parallelism (TP=2, DP=2 on 4x A100)

| Component           | Per GPU   | Notes                                   |
| ------------------- | --------- | --------------------------------------- |
| **Model Weights**   | 6 GB      | Split across 2 GPUs via tensor parallel |
| **Optimizer State** | 18 GB     | For the model shard on this GPU         |
| **Activations**     | 10-15 GB  | Batch 16 per GPU                        |
| **Gradients**       | 6 GB      | For the model shard                     |
| **Peak Total**      | ~40-45 GB | Comfortable fit in A100 80GB            |

### Training Characteristics

- **Convergence**: Slow and steady - improvements continue through entire 9M+ step run
- **Stability**: Requires extensive tuning - gradient clipping, warmup, careful LR scheduling essential
- **Overfitting Risk**: Virtually none with 300M samples - model remains data-hungry
- **Checkpoint Size**: ~12 GB per checkpoint (compress to ~3 GB)
- **Evaluation Time**: ~60-90 minutes on validation set (1M+ samples)
- **Gradient Checkpointing**: **Essential** on all hardware configurations
- **Distributed Training**: **Mandatory** - DeepSpeed ZeRO-3 or FSDP required

### Cost Analysis

| Hardware         | Hourly Rate | Total GPU Hours | Wall Clock | Total Cost     | Cost/1M Samples |
| ---------------- | ----------- | --------------- | ---------- | -------------- | --------------- |
| 4x RTX 3090      | $4.40/hr    | 2000-2500       | 21-26 days | ~$8,800-11,000 | ~$29-37         |
| 4x A100 80GB     | $10.00/hr   | 750-1000        | 8-10 days  | ~$7,500-10,000 | ~$25-33         |
| 8x A100 80GB     | $20.00/hr   | 375-500         | 2-3 days   | ~$7,500-10,000 | ~$25-33         |
| **8x Spot A100** | $6.40/hr    | 375-500         | 2-3 days   | ~$2,400-3,200  | ~$8-11          |

### Use Cases & Performance Expectations

- **Capability Level**: State-of-the-art multi-modal understanding
- **Quality**: Exceptional cross-modal translations with complex reasoning
- **Production Readiness**: Enterprise-grade - suitable for flagship products
- **Recommended For**: Research, cutting-edge products, competitive differentiation
- **Strengths**: Handles nuanced multi-step reasoning, complex pattern recognition
- **Limitations**: High inference cost; requires optimization for production deployment

### Critical Considerations

> [!CAUTION]
> **The 3B XL model requires enterprise-scale infrastructure.** Single consumer GPUs are inadequate. Budget for 4-8x A100 GPUs and expect 2-10 days of continuous training. This is a **multi-thousand dollar** commitment.

> [!WARNING]
> **Distributed training complexity is HIGH.** You will need:
>
> - DeepSpeed ZeRO-3 or PyTorch FSDP expertise
> - Proper cluster configuration with high-bandwidth interconnect (InfiniBand or NVLink)
> - Robust checkpoint/recovery systems (expect interruptions over 8+ day runs)
> - Monitoring/alerting infrastructure to catch issues early

> [!IMPORTANT]
> **Data quality is PARAMOUNT** at this scale. With 300M samples required, ensure:
>
> - Rigorous data validation pipelines
> - Diversity across all modalities
> - Thorough testing on 100M model before scaling to 3B
> - Budget for data curation/cleaning is comparable to compute budget

> [!TIP]
> **8x Spot A100 is the clear winner** - ~$2,400-3,200 total cost with only 2-3 days training time. The faster iteration enables better experimentation. Pre-emptible interruptions are manageable with proper checkpointing (save every 30-60 minutes).

### Infrastructure Requirements

**Minimum Recommended Setup:**

- 4x A100 80GB with NVLink
- 2TB+ NVMe SSD for checkpoints
- 100+ Gbps network (for multi-node)
- 256+ GB system RAM
- DeepSpeed or PyTorch FSDP configured

**Optimal Setup:**

- 8x A100 80GB (single node preferred)
- 4TB NVMe SSD RAID
- Redundant checkpoint storage (S3/云 backup)
- Dedicated monitoring (Weights & Biases, TensorBoard)
- Automated alert system

## Compute Assumptions

### Per-GPU Memory Requirements (BF16)

| Scale  | Model Weights | Optimizer State | Activations (batch=32) | Gradients | Total (Peak) |
| ------ | ------------- | --------------- | ---------------------- | --------- | ------------ |
| Tiny   | 40 MB         | 120 MB          | 1.5 GB                 | 40 MB     | ~2 GB        |
| Small  | 400 MB        | 1.2 GB          | 6 GB                   | 400 MB    | ~8 GB        |
| Medium | 2 GB          | 6 GB            | 15 GB                  | 2 GB      | ~24 GB       |
| Large  | 4 GB          | 12 GB           | 28 GB                  | 4 GB      | ~48 GB       |
| XL     | 12 GB         | 36 GB           | 60 GB                  | 12 GB     | ~120 GB      |

> [!NOTE]
> RTX 3090 (24GB) can train Medium models but requires batch size reduction or gradient accumulation for Large models.

## Data Requirements

Multi-modal training data needs:

- **Tiny**: ~100 hours of VR session data
- **Small**: ~500 hours of VR session data
- **Medium**: ~1000 hours of VR session data
- **Large**: ~5000 hours of VR session data
- **XL**: ~15000 hours of VR session data

Data augmentation can reduce requirements by ~2x.

## Cost Estimates (Cloud GPU Rental)

Based on current market rates (as of 2026-01):

| Scale   | RTX 3090 ($1.10/hr) | 2x RTX 3090 ($2.20/hr) | A100 80GB ($2.50/hr) |
| ------- | ------------------- | ---------------------- | -------------------- |
| Tiny    | ~$17                | ~$9                    | ~$25                 |
| Small   | ~$220               | ~$220                  | ~$250                |
| Medium  | ~$1,320             | ~$1,320                | ~$1,250              |
| Large   | ~$2,750             | ~$2,750                | ~$2,500              |
| ** XL** | ~$8,800             | ~$8,800                | ~$7,500              |

> [!IMPORTANT]
> Actual costs vary by cloud provider and region. Spot instances can reduce costs by 60-80%.

## Inference Costs

Cost per 1M output tokens (estimated):

| Scale  | Cost/1M tokens | Latency (batch=1) | Throughput (batch=32) |
| ------ | -------------- | ----------------- | --------------------- |
| Small  | $0.01          | ~50ms             | ~500 samples/sec      |
| Medium | $0.05          | ~100ms            | ~250 samples/sec      |
| Large  | $0.10          | ~200ms            | ~125 samples/sec      |
| XL     | $0.30          | ~400ms            | ~60 samples/sec       |

## Recommendations

1. **Start small**: Train Tiny model first to validate pipeline (~1 day on single 3090)
2. **Scale gradually**: Double model size only after validating previous scale
3. **Monitor closely**: Track loss curves, gradient norms, memory usage
4. **Checkpoint often**: Save every 1-5% of training for recovery
5. **Use 2x3090 for efficiency**: Best price/performance for 1B param training
6. **Consider gradient checkpointing**: Essential for Large+ models on 3090s

## Hardware Selection Guide

| Goal                       | Recommended Hardware | Rationale                                        |
| -------------------------- | -------------------- | ------------------------------------------------ |
| **Development/Testing**    | Single RTX 3090      | Cost-effective, 24GB sufficient for Small/Medium |
| **Production 1B Training** | 2x RTX 3090 (DDP)    | Best price/performance, ~50 day training time    |
| **Fast Iteration**         | A100 80GB            | Higher throughput, larger batches, ~40 days      |
| **Large Scale (3B+)**      | 4x A100 80GB         | Required memory for XL models                    |

> [!CAUTION]
> Training large models requires significant time and cost. Ensure dataset quality and pipeline validation before committing to multi-week training runs.
