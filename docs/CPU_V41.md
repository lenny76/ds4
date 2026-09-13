# DeepSeek V4.1 CPU port: development status

This branch does **not yet run V4.1 inference on CPU**. The engine's Metal-only
admission check is intentionally still present until the CPU graph and session
state are implemented and validated. `make cpu` alone does not enable V4.1.

## Implemented

`ds4_v41_cpu.c` contains CPU primitives for the V4.1 numerical boundaries:
BF16 rounding, FP8/E8M0 and FP4/E8M0/E4M3 activation round trips, tail RoPE,
two-token compression pooling, and Engram residual gating. These primitives
are currently exercised by tests, not connected to the inference engine.

Build and test without a model or GPU:

```sh
make test-deepseek41-cpu
make test-engram test-deepseek41-gguf
```

The CPU tests include all BF16 encodings at rounding boundaries, an independent
enumeration oracle for FP8/FP4, midpoint ties and signed zero, high-position
RoPE checks, stable pooling and Engram gate direction. They do not establish
end-to-end logits parity with Metal or the official model.

## Next integration work

1. Implement the V4.1 CPU graph with forty sliding windows, four owners of
   compressed KV/index state, reused index selections and Engram history.
2. Preserve BF16 boundaries in projections, shared/routed experts and
   hyper-connections; the existing V4 CPU layer is not interchangeable.
3. Connect graph allocation, reset, prompt synchronization and decode to CPU
   sessions. Reject unsupported snapshot/speculative/vision operations until
   their V4.1 state semantics are implemented.
4. Validate layer intermediates and logits, then prefill/decode continuity on
   a real V4.1 checkpoint before removing the Metal-only admission check.
5. Measure thread counts and NUMA placement before adding ISA optimizations.

## Current development server

The Debian 13 LXC container currently exposes 24 logical CPUs on a two-socket
Xeon Gold 6262 host. It reports approximately 250 GiB RAM and 530 GiB free
filesystem space after expansion. Query these again before model loading;
the host also runs llama.cpp/llama-swap and existing models.

Keep development under `/home/ds4dev/ds4` and candidate model files under
`/home/ds4dev/models`. Do not modify existing services or their configurations.
The upstream Q2 is approximately 341 GiB including Engram tables; Engram is
read on demand. On shared hardware, disk capacity alone does not establish
that a full inference run has sufficient available memory or I/O capacity.
