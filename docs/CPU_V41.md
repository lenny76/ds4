# DeepSeek V4.1 CPU port: development status

This branch contains an experimental V4.1 CPU graph with layer-major prompt batching and a scalar-token decode transition. Real Q2
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
3. Extend exact batching to the routed experts; the existing generic grouped
   implementation changes V4.1 reduction order and therefore cannot be reused.
4. Compare batch sizes and attention growth at 4096 tokens.

## Known limitations

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

At long context the CPU graph selects up to 512 compressed KV rows. It keeps a
512-entry min-heap instead of inserting every candidate into a full sorted
array, while preserving the earlier-row tie rule and final position order.
Tie-heavy randomized tests compare the selected set against the reference
algorithm. At 2048 tokens this improved decode from 3.24 to 3.32 t/s on the
reference host.

## CPU prompt batching

Prompt synchronization processes up to 16 tokens layer by layer. Q8_0 attention
projections and the shared expert keep each weight row hot across the token
batch while preserving F32 activations, BF16 boundaries, causal cache
publication, and per-token KV selections. A full-model test compares both the
final prompt logits and a continuation step against the scalar-token graph.

The two attention output projections use the same row-major batch traversal.
The attention core first records every token's 64 heads, then scans each
Q8_0 output row across the whole chunk before applying the second projection.
This preserves the scalar dot-product order and is bit-exact in the full-model
prompt and continuation comparison.

On the reference dual-socket Xeon host, 512-token prefill improved from 3.69 to
6.13 t/s (+66%), and batch-8 reached 5.20 t/s at 2048 tokens versus about 3.50
t/s for scalar-token prefill. Set `DS4_CPU_V41_DISABLE_BATCH_PREFILL=1` for an
A/B rollback, or `DS4_CPU_V41_BATCH_PREFILL=N` to test a size from 2 through 16.
The rollback variable takes precedence. Decode still uses `ds41c_step`.

On top of the Engram batch reader, batching the attention output projections
raised warm 512-token prefill from 6.18 to 6.67 t/s (+7.9%) on the reference
host. A 2048-token run reached 6.06 t/s prefill and 3.59 t/s decode.
