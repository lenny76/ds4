/* CPU counterparts of the numerical boundaries in metal/dsv41.metal.
 * Compile without fast-math: ties, signed zero and BF16 rounding are part of
 * the model, and reassociation can move values across quantization bins. */
#include "ds4_v41_cpu.h"

#include <math.h>
#include <string.h>

float ds41_cpu_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    bits &= 0xffff0000u;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static float fp8_value(unsigned code) {
    return code < 8 ? ldexpf((float)code, -9) :
        ldexpf(1.0f + (code & 7u) * 0.125f, (int)(code >> 3u) - 7);
}

static float fp8_nearest(float x) {
    const float a = fminf(fabsf(x), 448.0f);
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        const unsigned mid = (lo + hi + 1u) / 2u;
        if (fp8_value(mid) <= a) lo = mid;
        else hi = mid - 1u;
    }
    if (lo < 126) {
        const float lower = a - fp8_value(lo), upper = fp8_value(lo + 1);
        if (upper - a < lower || (upper - a == lower && (lo & 1u))) lo++;
    }
    return copysignf(fp8_value(lo), x);
}

static float fp4_nearest(float x) {
    static const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    /* Midpoints select the even encoding, not the even numerical value. */
    const float a = fabsf(x);
    for (unsigned i = 0; i < 7; i++) {
        const float midpoint = (values[i] + values[i + 1]) * .5f;
        if (a < midpoint || (a == midpoint && !(i & 1u)))
            return copysignf(values[i], x);
    }
    return copysignf(6.0f, x);
}

static float pow2_ceil(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits = (bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u);
    memcpy(&x, &bits, sizeof(x));
    return x;
}

bool ds41_cpu_quantize(float *x, size_t capacity, uint32_t width,
                       uint32_t rows, ds41_cpu_format format) {
    if (!x || !width || !rows || format < DS41_CPU_BF16 || format > DS41_CPU_FP4_E4M3 ||
        (uint64_t)width * rows > capacity) return false;
    const unsigned block = format == DS41_CPU_FP4_E4M3 ? 16 : 32;
    if (format != DS41_CPU_BF16 && width % block) return false;
    const size_t n = (size_t)width * rows;
    if (format == DS41_CPU_BF16) {
        for (size_t i = 0; i < n; i++) x[i] = ds41_cpu_bf16(x[i]);
        return true;
    }
    /* Fail before modifying any block if the input cannot be quantized. */
    for (size_t i = 0; i < n; i++)
        if (!isfinite(ds41_cpu_bf16(x[i]))) return false;
    for (size_t off = 0; off < n; off += block) {
        float values[32], amax = 0;
        for (unsigned i = 0; i < block; i++) {
            values[i] = ds41_cpu_bf16(x[off + i]);
            amax = fmaxf(amax, fabsf(values[i]));
        }
        const float scale = format == DS41_CPU_FP8_E8M0 ?
            pow2_ceil(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f)) :
            format == DS41_CPU_FP4_E8M0 ?
            pow2_ceil(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f)) :
            fp8_nearest(fmaxf(amax, 0.01171875f) / 6.0f);
        for (unsigned i = 0; i < block; i++) {
            const float v = values[i] / scale;
            x[off + i] = ds41_cpu_bf16((format == DS41_CPU_FP8_E8M0 ?
                fp8_nearest(v) : fp4_nearest(v)) * scale);
        }
    }
    return true;
}

bool ds41_cpu_rope(float *x, size_t capacity, uint32_t width, uint32_t heads,
                   uint32_t rows, uint32_t start, uint32_t stride,
                   bool compressed, bool inverse) {
    if (!x || width < 64 || !heads || !rows || !stride || rows > 1048576 ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > SIZE_MAX / width ||
        (size_t)heads * rows * width > capacity) return false;
    const double pi = 3.14159265358979323846;
    const float base = compressed ? 160000.0f : 10000.0f;
    const float low = (float)floor(64 * log(65536 / (32 * 2 * pi)) / (2 * log(base)));
    const float high = (float)ceil(64 * log(65536 / (2 * pi)) / (2 * log(base)));
    float freq[32];
    for (unsigned i = 0; i < 32; i++) {
        float f = 1.0f / powf(base, (float)i / 32);
        if (compressed) {
            const float ramp = fminf(1, fmaxf(0, (i - low) / (high - low)));
            const float smooth = 1 - ramp;
            f = (f / 16) * (1 - smooth) + f * smooth;
        }
        freq[i] = f;
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (unsigned i = 0; i < 32; i++) {
            const float theta = (float)(start + row * stride) * freq[i];
            const float c = cosf(theta), s = (inverse ? -1 : 1) * sinf(theta);
            for (uint32_t head = 0; head < heads; head++) {
                float *pair = x + ((size_t)row * heads + head) * width + width - 64 + 2 * i;
                const float re = pair[0], im = pair[1];
                pair[0] = ds41_cpu_bf16(re * c - im * s);
                pair[1] = ds41_cpu_bf16(re * s + im * c);
            }
        }
    }
    return true;
}

void ds41_cpu_pool2(float *out, const float *a, const float *b,
                    const float *score_a, const float *score_b, uint32_t width) {
    for (uint32_t i = 0; i < width; i++) {
        const float peak = fmaxf(score_a[i], score_b[i]);
        const float ea = expf(score_a[i] - peak), eb = expf(score_b[i] - peak);
        out[i] = ds41_cpu_bf16((a[i] * ea + b[i] * eb) / (ea + eb));
    }
}

void ds41_cpu_engram_add(float *residual, const float *kv,
                         const float *q_weight, const float *k_weight,
                         uint32_t width, float eps) {
    for (unsigned head = 0; head < 4; head++) {
        const size_t off = (size_t)head * width;
        float h2 = 0, k2 = 0, dot = 0;
        for (uint32_t i = 0; i < width; i++) {
            const float h = residual[off + i], k = ds41_cpu_bf16(kv[off + i]);
            h2 += h * h;
            k2 += k * k;
            dot += h * (q_weight[off + i] * k_weight[off + i]) * k;
        }
        dot = dot * (1 / sqrtf(h2 / width + eps)) *
            (1 / sqrtf(k2 / width + eps)) * (1 / sqrtf((float)width));
        const float gate = 1 / (1 + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot)));
        for (uint32_t i = 0; i < width; i++)
            residual[off + i] = ds41_cpu_bf16(residual[off + i] +
                gate * ds41_cpu_bf16(kv[(size_t)4 * width + i]));
    }
}
