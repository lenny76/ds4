#ifndef DS4_V41_CPU_H
#define DS4_V41_CPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* V4.1 activation/cache round trips, held in F32 storage. These are not GGUF
 * weight formats. Keep their BF16 boundaries distinct from the V4 FP16 path. */
typedef enum {
    DS41_CPU_BF16,
    DS41_CPU_FP8_E8M0,
    DS41_CPU_FP4_E8M0,
    DS41_CPU_FP4_E4M3
} ds41_cpu_format;

float ds41_cpu_bf16(float x);
bool ds41_cpu_quantize(float *x, size_t capacity, uint32_t width,
                       uint32_t rows, ds41_cpu_format format);
bool ds41_cpu_rope(float *x, size_t capacity, uint32_t width, uint32_t heads,
                   uint32_t rows, uint32_t start, uint32_t stride,
                   bool compressed, bool inverse);
/* One complete pair, independent of the caller's token/cache bookkeeping.
 * out may alias either KV input; scores and KV are F32 projections. */
void ds41_cpu_pool2(float *out, const float *a, const float *b,
                    const float *score_a, const float *score_b, uint32_t width);
/* One text token: residual [4,width], projected KV [5,width], norm weights
 * [4,width]. Callers skip this operation for masked/image tokens. */
void ds41_cpu_engram_add(float *residual, const float *kv,
                         const float *q_weight, const float *k_weight,
                         uint32_t width, float eps);

#endif
