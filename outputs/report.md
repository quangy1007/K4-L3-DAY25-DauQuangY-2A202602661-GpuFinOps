# NimbusAI - GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) | Share of Savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Purchasing (spot/reserved) | $10,040 | 80.3% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

## Deep-Dive: Root Cause of the "GPU-Util Lie" & Unit Economics

### 1. Hardware Mechanism Behind the Lie
In telemetry analysis, **`gpu-h100-4`** exhibited **98% GPU-Util** but only **0.202 MFU** (Model FLOPs Utilization).
- **What `nvidia-smi` measures:** The standard `gpu_util_pct` metric samples the fraction of time over the past second that at least one kernel was active on the GPU SMs (Streaming Multiprocessors). It treats an SM executing a scalar memory wait instruction identically to an SM saturating FP16 Tensor Cores at 990 TFLOPs.
- **The Root Cause (Memory Stall & Decode Bottleneck):** Autoregressive LLM generation operates in two phases: prefill (compute-bound, arithmetic intensity ~455 FLOP/byte) and decode (memory-bound, arithmetic intensity ~1-2 FLOP/byte on H100). When serving small batches or uncoalesced attention kernels, the Tensor Cores spend >75% of cycles stalled waiting for weights to stream from HBM3 memory (memory bandwidth stall) or waiting on kernel launches across small tensor chunks.
- **Financial Impact:** NimbusAI pays the full **$2.50/GPU-hr** on-demand rate for H100 while obtaining less than 20% of its computational capability. Downscaling or right-sizing these workloads to memory-optimized or lower-cost instances preserves identical throughput at a fraction of the cost.

## Prioritized Action Plan (Ranked by ROI)

1. **Priority 1 (Quick Win, Immediate ROI: Days 1-7): Purchasing Optimization (`$10,040/month saved`)**
   - **Action:** Transition continuous training/inference workloads with duty cycle >= 55% to 3-Year Reserved commitments (45% discount), and convert fault-tolerant training jobs (`interruptible=1`) to Spot Instances with automated checkpointing (40% discount).
   - **Rationale:** Requires zero application code changes, zero downtime risk, and captures >80% of total company savings immediately.

2. **Priority 2 (High ROI, Fast Implementation: Weeks 2-3): Inference Optimization Levers (`$1,212/month saved`)**
   - **Action:** Enforce Prompt Caching for repeated system prefixes (90% read discount), enable Batch API for offline evaluation/nightly jobs (50% discount), and route simple queries through Cascade Routing to the small tier ($0.20/$0.40 per 1M tokens vs $3.00/$15.00).
   - **Unit Economic Metric:** Drops serving unit cost from **$6.488/1M-token** to **$1.126/1M-token** (an **82.6% reduction** in $/1M-token).

3. **Priority 3 (Medium ROI: Weeks 3-4): Right-sizing Memory-Bound Workloads (`$655/month saved`)**
   - **Action:** Migrate jobs flagged with GPU-Util lies from H100 ($2.50/hr) to A100 ($1.79/hr) or L4 ($0.80/hr) where memory bandwidth and capacity match actual operational requirements.

4. **Priority 4 (Hygiene & Governance: Ongoing): Automated Idle Reaping (`$600/month saved`)**
   - **Action:** Implement an automated daemon to terminate or suspend GPU instances idling overnight (<10% utilization for >2 consecutive hours), preventing $20/day ($600/month) in burned capital.

## Sustainability & Multi-Region Carbon Economics

- **Energy per query:** 0.24 Wh
- **Carbon per query (us-east-1):** 0.091 gCO2e
- **Cheapest + Cleanest Region:** `europe-north1`

### Regional Grid Carbon & Electricity Cost Trade-Offs
| Region | Power Cost ($/kWh) | Carbon Intensity (gCO2/kWh) | Assessment & Strategy |
|---|---|---|---|
| `europe-north1` (Norway) | $0.090 | 30 | **Optimal Green Tier:** 92% cleaner than us-east-1; best for async batch jobs & training. |
| `us-east-wa` (Washington) | $0.055 | 90 | **Lowest Power Cost:** Hydro-heavy grid, lowest power rate ($0.055/kWh), very low carbon. |
| `us-west-2` (Oregon) | $0.070 | 120 | **Balanced US Tier:** Low carbon hydro, excellent latency for US West inference. |
| `us-east-1` (N. Virginia) | $0.120 | 380 | **Default Baseline:** High carbon (380 gCO2/kWh), standard cloud pricing. |
| `europe-central2` (Poland) | $0.180 | 660 | **Caution (Dirtiest):** Coal-heavy grid (660 gCO2/kWh), highest power price ($0.18/kWh). |

> **Key Sustainability Insight:** Migrating interruptible batch and training jobs from `us-east-1` to `europe-north1` cuts operational carbon footprint by **92.1%** while maintaining cost advantages. Latency trade-offs are negligible for asynchronous workloads.

## Summary of Advanced Extensions ("Your Turn")

- **Extension 1 (Tier Policy with Spot Risk & Duration):** Evaluates per-GPU interruption rates and job duration before reserving instances.
- **Extension 2 (Right-Sizing by MBU & $/GB-VRAM):** Matches memory-bound inference workloads to A100/L4 based on required memory bandwidth rather than peak FLOPs.
- **Extension 3 (Cache Economic Viability):** Calculates break-even read counts ($N_{break-even} \approx 1.39$ reads) to guarantee positive ROI on prefix caching.
- **Extension 4 (Reasoning Workload Budgeting):** Separates reasoning tokens (~80x energy multiplier), showing reasoning accounts for disproportionate energy and cost, and introduces task complexity routing rules.
- **Extension 5 (Carbon-Aware Scheduling):** Quantifies carbon savings across global cloud regions and establishes green workload migration.

_Figures are June-2026 as-of snapshots; re-baseline before acting._