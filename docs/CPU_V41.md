# DeepSeek V4.1 CPU port: development status

This branch contains an experimental scalar-token V4.1 CPU graph. Real Q2
checkpoint prefill and decode have been validated on the development server;
cross-backend logits parity is still pending. The graph remains behind
`DS4_CPU_V41_EXPERIMENTAL=1`, so `make cpu` alone does not opt into V4.1.

On x86-64, `make cpu` uses `-march=native`. CPUs exposing AVX-512 VNNI and
AVX-512 VL therefore use exact integer VNNI kernels for the IQ2_XXS and Q2_K
expert projections; other x86 CPUs retain the scalar implementation.

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

Build the CPU CLI and run a Q2 checkpoint:

```sh
make -j2 cpu
DS4_CPU_V41_EXPERIMENTAL=1 ./ds4 --cpu -t 48 \
  -m /path/to/DeepSeek-V4.1-Flash-Q2.gguf \
  --ctx 4096 --nothink --temp 0 --tokens 32 \
  --prompt "Write a short greeting."
```

The worker pool accepts up to 64 threads. More threads do not necessarily
improve throughput: benchmark the same deterministic prompt at several thread
counts. On the current reference host, increasing the effective pool from 32
to 48 threads improved both prefill and decode by roughly 13-14%. Forcing
`numactl --interleave=all` was substantially slower, so NUMA policy must be
measured rather than assumed.

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
4. Optimize attention projections, now the largest stable warm-cache CPU cost
   after the routed-expert VNNI kernels.

## Known limitations

- Scalar-token prefill deliberately favors correctness over throughput.
- Exact logits parity with the Metal quality path has not been measured.
- Session snapshot serialization is rejected for the CPU V4.1 graph.
- Vision, tensor parallelism and SSD streaming are not admitted by this path.
- The implementation must be enabled explicitly with
  `DS4_CPU_V41_EXPERIMENTAL=1`.

## Profiling

Set `DS4_CPU_V41_PROFILE=1` to print per-token timings for Engram, attention,
shared and routed experts, and the output head. The profiler is intended for
development and remains off by default.

On a dual-socket Cascade Lake system with 48 physical cores, warm-cache A/B
measurements reduced the routed-expert stage from about 334 ms to 232 ms and
the complete token from about 832 ms to 733 ms. Cold expert reads from SATA can
still dominate an individual token, so compare kernels only after warming the
same expert pages.
