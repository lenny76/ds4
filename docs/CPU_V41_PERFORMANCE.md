# DeepSeek V4.1 CPU performance history

This table records measured milestones for the experimental CPU graph. Results
come from the dual-socket Xeon reference machine, usually with 40 workers after
the thread-count sweep. They are useful as an engineering history, but rows with
different prompt lengths or context sizes are not direct A/B comparisons.

## Generation and short-context milestones

| Revision | Implementation | Workload | Before | After | Result |
|---|---|---:|---:|---:|---:|
| initial CPU graph | Scalar/reference-heavy V4.1 CPU path | short prompt, 48 workers | - | 1.22 t/s generation | baseline |
| `70d95e8` | AVX-512 F32 x Q8_0 attention projections | short prompt | 1.22 | 1.88 t/s | +54% |
| `6d65474` | AVX-512 F16 row dot product | short prompt | 1.88 | 2.02 t/s | +7% step, +66% overall |
| `a3762a0` | Vectorized BF16 rounding | short prompt | 2.02 | about 2.08 t/s | about +3% |
| `9f43afd` | Skip output head for intermediate prompt tokens | prefill | previous revision | about +2% | small prefill gain |
| `2c4ae8e` | Parallel attention heads | 751-token context | 1.73 | 2.31 t/s generation | +34% |
| `0a3ee8f` | Stream attention rows once per head block | generation | previous revision | about +2% | small gain |
| `cce93b3` | Keep worker-pool threads spinning briefly | pool dispatch | 52 us | 7.7 us | 6.8x lower latency |
| `4e13974` | Parallel MoE router | short prompt | 2.61 | 3.84 t/s generation | +47% |
| `894f2fe` | 256-bit VNNI expert kernels | short prompt | about 3.86 | about 3.92 t/s | about +1.5% |
| `284d9ae` | Parallel collection of selected KV rows | 751-token context | 3.49 | 3.63 t/s generation | about +4% |

The short-prompt path therefore progressed from **1.22 t/s** to approximately
**3.86-3.91 t/s**, about **3.2x** overall. Exact values fluctuate with page
cache state, CPU frequency, thread count, and prompt length.

## Long-context and prefill milestones

| Revision | Implementation | Context | Prefill | Generation | Comparison |
|---|---|---:|---:|---:|---|
| `284d9ae` | Parallel KV row collection | 2048 | 3.23 t/s | 3.11 t/s | long-context baseline |
| `284d9ae` | Parallel KV row collection | 4096 | 3.13 t/s | 2.97 t/s | long-context baseline |
| `de0aa80` | Exact heap-based KV top-k | 4096 | 3.39 t/s | 3.20 t/s | +8.3% prefill, +7.7% generation |
| `bd55e48` | Layer-major, 16-token projection batching | 512 | 6.13 t/s | - | +66% versus 3.69 token-major |
| `bd55e48` | Layer-major, 16-token projection batching | 4096 | 5.46 t/s | 3.22 t/s | +61% prefill versus `de0aa80` |
| `beb4a00` | Batched Linux Engram reads | 512 | 6.18 t/s | - | +0.8%, near measurement noise |
| `e44b325` | Batched attention output projections | 512 | **6.67 t/s** | 3.63 t/s | +7.9% prefill versus `beb4a00` |
| `e44b325` | Batched attention output projections | 2048 | **6.06 t/s** | **3.59 t/s** | current measured frontier |

## Rejected experiments

| Experiment | Reference | Candidate | Decision |
|---|---:|---:|---|
| Exact routed-MoE batching | 6.67 t/s prefill | 4.56 t/s | Reverted: -31.6% |
| F16 hyper-connection batching | 6.47/6.65 t/s control | 6.62/6.62 t/s | Reverted: average effect below 1% |
| Interleave memory across NUMA nodes | 0.95 t/s generation | 0.48 t/s | Rejected: about -49% |
| Relocate Q8 weights onto anonymous huge pages | existing mapping | about -13% | Reverted |

Unless stated otherwise, throughput is measured after warming the relevant
weight pages. The CPU graph remains experimental, and parity with the Metal
backend is still pending. Tests within the CPU implementation verify exact
prompt and continuation logits for changes from `a3762a0` onward; `70d95e8`
and `6d65474` change floating-point accumulation order.
