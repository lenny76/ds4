# DeepSeek V4.1 CPU port: development status

This branch contains an experimental scalar-token V4.1 CPU graph. Real Q2
checkpoint prefill and decode have been validated on the development server;
cross-backend logits parity is still pending. The graph remains behind
`DS4_CPU_V41_EXPERIMENTAL=1`, so `make cpu` alone does not opt into V4.1.

## Implemented

`ds4_v41_cpu.c` contains CPU primitives for the V4.1 numerical boundaries:
BF16 rounding, FP8/E8M0 and FP4/E8M0/E4M3 activation round trips, tail RoPE,
two-token compression pooling, and Engram residual gating. These primitives
are exercised independently by tests. `ds4_v41_cpu_graph.inc` connects them to
the 40-layer decode transition, CPU session allocation, prompt synchronization
and token evaluation. The graph owns forty 128-row sliding windows, four shared
compressed/index caches and the two disk-backed Engram readers.

Build and test without a model or GPU:

```sh
make test-deepseek41-cpu
make test-engram test-deepseek41-gguf
```

The CPU tests include all BF16 encodings at rounding boundaries, an independent
enumeration oracle for FP8/FP4, midpoint ties and signed zero, high-position
RoPE checks, stable pooling and Engram gate direction. A real Q2 run produced
finite, deterministic logits across multiple tokens and completed a public CLI
prompt. This does not yet establish exact logits parity with Metal.

## Next integration work

1. Compare layer intermediates and logits against Metal on the same checkpoint
   before removing the experimental admission gate.
2. Add snapshot serialization for the V4.1 cache state. Snapshot APIs currently
   reject the experimental CPU graph explicitly.
3. Optimize prompt prefill after decode correctness is established; the current
   implementation deliberately uses the same scalar-token transition.
4. Measure thread counts and NUMA placement before adding ISA optimizations.
