#include "../ds4_v41_cpu.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}

static float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, 4);
    return x;
}

/* Independent scalar oracle: enumerate the representable positive values.
 * The production FP8 binary search and FP4 midpoint selection must agree at
 * every midpoint, including ties with odd/even encoding indices. */
static float nearest(float x, bool fp4) {
    const float four[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best = 0, distance = INFINITY;
    unsigned code = 0;
    for (unsigned i = 0; i < (fp4 ? 8u : 127u); i++) {
        const float value = fp4 ? four[i] : i < 8 ? i / 512.0f :
            ldexpf((float)(8 + (i & 7)), (int)(i >> 3) - 10);
        const float error = fabsf(fabsf(x) - value);
        if (error < distance || (error == distance && !(i & 1) && (code & 1))) {
            best = value; distance = error; code = i;
        }
    }
    return copysignf(best, x);
}

static void check_bf16(void) {
    /* All BF16 encodings and both sides of the rounding midpoint. */
    const unsigned tails[] = {0, 0x7fff, 0x8000, 0x8001, 0xffff};
    for (uint32_t high = 0; high < 65536; high++) {
        for (unsigned j = 0; j < 5; j++) {
            const uint32_t u = high * 65536 + tails[j];
            uint32_t expected = high;
            if ((high & 0x7f80) != 0x7f80 &&
                (tails[j] > 0x8000 || (tails[j] == 0x8000 && (high & 1)))) expected++;
            assert(bits(ds41_cpu_bf16(from_bits(u))) == expected * 65536);
        }
    }
    puts("BF16: all encodings and rounding boundaries PASS");
}

static void check_quantization(void) {
    float source[3 * 512], actual[3 * 512];
    uint32_t seed = 4141;
    for (unsigned mode = 0; mode < 4; mode++) {
        const unsigned block = mode == 3 ? 16 : 32;
        for (unsigned round = 0; round < 8; round++) {
            for (unsigned i = 0; i < 3 * 512; i++) {
                seed = seed * 1664525u + 1013904223u;
                source[i] = ldexpf(((int)(seed % 65537) - 32768) / 8192.0f,
                                   (int)round * 4 - 16);
                if (i < 512) source[i] = copysignf(0, i & 1 ? -1 : 1);
            }
            memcpy(actual, source, sizeof(source));
            assert(ds41_cpu_quantize(actual, 3 * 512, 512, 3, (ds41_cpu_format)mode));
            for (unsigned off = 0; off < 3 * 512; off += block) {
                float peak = 0, scale = 1;
                for (unsigned i = 0; i < block; i++)
                    peak = fmaxf(peak, fabsf(ds41_cpu_bf16(source[off + i])));
                if (mode == 1) scale = exp2f(ceilf(log2f(fmaxf(peak, 1e-4f) * (1.0f / 448))));
                if (mode == 2) scale = exp2f(ceilf(log2f(fmaxf(peak, 0x1.8p-124f) * (1.0f / 6))));
                if (mode == 3) scale = nearest(fmaxf(peak, 6.0f / 512) / 6, false);
                for (unsigned i = 0; i < block; i++) {
                    float expected = ds41_cpu_bf16(source[off + i]);
                    if (mode) expected = ds41_cpu_bf16(nearest(expected / scale, mode != 1) * scale);
                    assert(bits(actual[off + i]) == bits(expected));
                }
            }
        }
    }
    /* Scale=1 blocks exercise FP4 midpoints and signed zero exactly. */
    const float ties[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5};
    const float expected[] = {0, 1, 1, 2, 2, 4, 4};
    for (unsigned mode = 2; mode <= 3; mode++) {
        float row[32] = {0};
        row[15] = row[31] = 6;
        for (unsigned i = 0; i < 7; i++) { row[i] = ties[i]; row[i + 16] = -ties[i]; }
        assert(ds41_cpu_quantize(row, 32, 32, 1, (ds41_cpu_format)mode));
        for (unsigned i = 0; i < 7; i++) {
            assert(bits(row[i]) == bits(expected[i]));
            assert(bits(row[i + 16]) == bits(-expected[i]));
        }
    }
    float tiny[32] = {0};
    tiny[0] = FLT_MIN;
    assert(ds41_cpu_quantize(tiny, 32, 32, 1, DS41_CPU_FP4_E8M0));
    assert(tiny[0] == FLT_MIN);
    for (unsigned code = 0; code < 126; code++) {
        const float lower = code < 8 ? code / 512.0f :
            ldexpf((float)(8 + (code & 7)), (int)(code >> 3) - 10);
        const unsigned next = code + 1;
        const float upper = next < 8 ? next / 512.0f :
            ldexpf((float)(8 + (next & 7)), (int)(next >> 3) - 10);
        float row[32] = {0};
        row[0] = (lower + upper) * .5f;
        row[1] = -row[0]; row[31] = 448;
        assert(ds41_cpu_quantize(row, 32, 32, 1, DS41_CPU_FP8_E8M0));
        const float chosen = code & 1 ? upper : lower;
        assert(bits(row[0]) == bits(chosen) && bits(row[1]) == bits(-chosen));
    }
    float bad[32] = {1};
    bad[31] = INFINITY;
    assert(!ds41_cpu_quantize(bad, 32, 32, 1, DS41_CPU_FP8_E8M0));
    assert(bad[0] == 1 && isinf(bad[31]));
    assert(!ds41_cpu_quantize(bad, 32, 24, 1, DS41_CPU_FP8_E8M0));
    assert(!ds41_cpu_quantize(bad, 32, 32, 2, DS41_CPU_BF16));
    assert(!ds41_cpu_quantize(bad, 32, UINT32_MAX, UINT32_MAX, DS41_CPU_BF16));
    assert(!ds41_cpu_quantize(bad, 32, 32, 1, (ds41_cpu_format)4));
    puts("FP8/FP4: independent oracle, ties, zeros, tiny values and validation PASS");
}

static void check_rope(void) {
    float x[2 * 3 * 128], batch[2 * 3 * 128];
    for (unsigned kind = 0; kind < 2; kind++) {
        for (unsigned inverse = 0; inverse < 2; inverse++) {
            for (unsigned i = 0; i < 2 * 3 * 128; i++) x[i] = ds41_cpu_bf16((int)(i % 31) * .07f);
            memcpy(batch, x, sizeof(x));
            assert(ds41_cpu_rope(batch, 768, 128, 3, 2, 127, 128, kind, inverse));
            for (unsigned row = 0; row < 2; row++) {
                assert(ds41_cpu_rope(x + row * 384, 384, 128, 3, 1, 127 + row * 128, 1, kind, inverse));
                for (unsigned h = 0; h < 3; h++)
                    for (unsigned i = 0; i < 64; i++)
                        assert(x[row * 384 + h * 128 + i] == ds41_cpu_bf16((int)((row * 384 + h * 128 + i) % 31) * .07f));
            }
            assert(!memcmp(x, batch, sizeof(x)));
        }
    }
    memset(x, 0, sizeof(x));
    x[64] = 1;
    assert(ds41_cpu_rope(x, 768, 128, 1, 1, 1, 1, false, false));
    assert(x[64] == ds41_cpu_bf16(cosf(1)) && x[65] == ds41_cpu_bf16(sinf(1)));
    /* High positions catch precision loss and accidental use of V4's RoPE
     * schedule. Check the interpolated tail against a double precision oracle;
     * BF16/libm and F32 frequency rounding require an absolute tolerance. */
    const unsigned positions[] = {0, 1, 127, 4095, 65535, 262143, 1048575};
    for (unsigned kind = 0; kind < 2; kind++) {
        for (unsigned p = 0; p < sizeof(positions) / sizeof(*positions); p++) {
            memset(x, 0, sizeof(x)); x[126] = 1;
            assert(ds41_cpu_rope(x, 768, 128, 1, 1, positions[p], 1, kind, false));
            const double frequency = pow(kind ? 160000.0 : 10000.0, -31.0 / 32) / (kind ? 16 : 1);
            const double theta = positions[p] * frequency;
            assert(fabs(x[126] - cos(theta)) < .004);
            assert(fabs(x[127] - sin(theta)) < .004);
        }
    }
    assert(!ds41_cpu_rope(x, 768, 128, 1, 2, 1048575, 1, false, false));
    assert(!ds41_cpu_rope(x, 63, 64, 1, 1, 0, 1, false, false));
    assert(!ds41_cpu_rope(x, 768, 63, 1, 1, 0, 1, false, false));
    assert(!ds41_cpu_rope(x, 768, 128, 1, 1, 0, 0, false, false));
    puts("RoPE: stride, inverse, non-RoPE preservation and bounds PASS");
}

static void check_pool_engram(void) {
    float a[] = {2, 3, 4}, b[] = {6, 9, 12};
    const float sa[] = {10000, 10000, -10000}, sb[] = {10000, -10000, 10000};
    ds41_cpu_pool2(a, a, b, sa, sb, 3);
    assert(a[0] == 4 && a[1] == 3 && a[2] == 12);
    enum { W = 32 };
    float h[4 * W], kv[5 * W], qw[4 * W], kw[4 * W];
    for (unsigned i = 0; i < 4 * W; i++) { h[i] = 0; kv[i] = 1; qw[i] = kw[i] = 1; }
    for (unsigned i = 4 * W; i < 5 * W; i++) kv[i] = 2;
    ds41_cpu_engram_add(h, kv, qw, kw, W, 1e-6f);
    for (unsigned i = 0; i < 4 * W; i++) assert(h[i] == 1);
    for (unsigned i = 0; i < 4 * W; i++) { h[i] = 1; kv[i] = i < W ? -1 : 1; }
    ds41_cpu_engram_add(h, kv, qw, kw, W, 1e-6f);
    /* Opposite keys close the gate; aligned keys open it. */
    assert(h[0] > 1 && h[0] < 1.5f && h[W] > 2.5f && h[W] < 3);
    puts("Pooling and Engram: stable softmax, aliasing and gate direction PASS");
}

int main(void) {
    check_bf16(); check_quantization(); check_rope(); check_pool_engram();
    return 0;
}
