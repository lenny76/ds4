/* Included after the CPU transformer helpers, only in DS4_NO_GPU builds.
 * Scalar-token V4.1 graph. Prefill uses the same transition as decode so cache
 * ownership, Engram history and BF16 boundaries have one implementation. */
#include "ds4_v41_cpu.h"

typedef struct {
    uint32_t ctx, pos;
    bool valid;
    ds4_engram_layout engram;
    ds4_engram_history history;
    uint32_t *token_map;
    ds4_engram_table tables[2];
    float *window[40], *compressed[4], *index[4];
    float previous_kv[4][512], previous_score[4][512];
    float *residual, *after, *flat, *x, *norm, *block, *shared;
    float *q, *qr, *low, *heads, *index_q, *scores, *rows;
    float *gate, *up, *mid, *down, *engram_kv;
    float kv[512], latent[512], pool[512], pool_score[512], index_k[128];
    float index_weights[32], pre[4], attn_split[24], ffn_split[24];
    float disk_rows[2][24 * 256];
    int selected[512], blocks[2048];
    float *block_scores;
    bool *block_allowed;
} ds41_cpu_graph;

static void ds41c_free(ds41_cpu_graph *g) {
    for (unsigned i = 0; i < 2; i++) ds4_engram_table_close(&g->tables[i]);
    for (unsigned i = 0; i < 40; i++) free(g->window[i]);
    for (unsigned i = 0; i < 4; i++) { free(g->compressed[i]); free(g->index[i]); }
#define FREE(p) free(g->p)
    FREE(token_map); FREE(residual); FREE(after); FREE(flat); FREE(x); FREE(norm);
    FREE(block); FREE(shared); FREE(q); FREE(qr); FREE(low); FREE(heads); FREE(index_q);
    FREE(scores); FREE(rows); FREE(gate); FREE(up); FREE(mid); FREE(down);
    FREE(engram_kv); FREE(block_scores); FREE(block_allowed);
#undef FREE
    memset(g, 0, sizeof(*g)); g->tables[0].fd = g->tables[1].fd = -1;
}

static void ds41c_reset(ds41_cpu_graph *g) {
    g->pos = 0; g->valid = true;
    ds4_engram_history_reset(&g->history);
    /* Old rows need not be cleared: visible counts are derived from pos and
     * each owner's publication precedes every consumer in layer order. */
}

static bool ds41c_array(const ds4_model *m, const char *name, uint32_t type,
                       uint64_t count, void *out, size_t bytes) {
    ds4_array_ref a;
    if (!model_get_array(m, name, &a) || a.type != type || a.len != count) return false;
    ds4_cursor c = cursor_at(m, a.data_pos);
    return cursor_read(&c, out, bytes);
}

static bool ds41c_alloc(ds41_cpu_graph *g, const ds4_model *m, const char *path, uint32_t ctx) {
    memset(g, 0, sizeof(*g)); g->tables[0].fd = g->tables[1].fd = -1;
    if (!ctx || ctx > 32768) return false;
    g->ctx = ctx;
#define ALLOC(p, n) do { g->p = calloc((size_t)(n), sizeof(*g->p)); if (!g->p) goto fail; } while (0)
    ALLOC(token_map, DS4_N_VOCAB);
    g->engram.token_map = g->token_map; g->engram.vocab_size = DS4_N_VOCAB;
    g->engram.compressed_vocab_size = required_u32(m, "deepseek41.engram.compressed_vocab_size");
    g->engram.pad_id = required_u32(m, "deepseek41.engram.pad_id");
    if (!ds41c_array(m, "deepseek41.engram.token_map", GGUF_VALUE_UINT32, DS4_N_VOCAB,
                     g->token_map, (size_t)DS4_N_VOCAB * 4) ||
        !ds41c_array(m, "deepseek41.engram.rows", GGUF_VALUE_UINT32, 2, g->engram.rows, sizeof(g->engram.rows)) ||
        !ds41c_array(m, "deepseek41.engram.primes", GGUF_VALUE_UINT32, 48, g->engram.primes, sizeof(g->engram.primes)) ||
        !ds41c_array(m, "deepseek41.engram.multipliers", GGUF_VALUE_UINT64, 8, g->engram.multipliers, sizeof(g->engram.multipliers)) ||
        !ds4_engram_layout_valid(&g->engram)) goto fail;
    for (unsigned i = 0; i < 2; i++) {
        const ds4_tensor *t = required_tensorf(m, "blk.%u.engram_embd.weight", i ? 14u : 1u);
        if (!ds4_engram_table_open(&g->tables[i], path, t->abs_offset, g->engram.rows[i])) goto fail;
        struct stat a, b;
        if (fstat(m->fd, &a) || fstat(g->tables[i].fd, &b) || a.st_dev != b.st_dev || a.st_ino != b.st_ino) goto fail;
    }
    for (unsigned i = 0; i < 40; i++) { ALLOC(window[i], 128 * 512); }
    for (unsigned i = 0; i < 4; i++) {
        const size_t cap = ctx / (i < 3 ? 2u : 1u) + 1;
        ALLOC(compressed[i], cap * 512); ALLOC(index[i], cap * 128);
    }
    ALLOC(residual, 4 * DS4_N_EMBD); ALLOC(after, 4 * DS4_N_EMBD); ALLOC(flat, 4 * DS4_N_EMBD);
    ALLOC(x, DS4_N_EMBD); ALLOC(norm, DS4_N_EMBD); ALLOC(block, DS4_N_EMBD); ALLOC(shared, DS4_N_EMBD);
    ALLOC(q, 64 * 512); ALLOC(heads, 64 * 512); ALLOC(qr, DS4_N_LORA_Q); ALLOC(low, 8192);
    ALLOC(index_q, 32 * 128); ALLOC(scores, ctx); ALLOC(rows, (128 + 512) * 512);
    ALLOC(gate, DS4_N_FF_EXP); ALLOC(up, DS4_N_FF_EXP); ALLOC(mid, DS4_N_FF_EXP);
    ALLOC(down, DS4_N_EMBD); ALLOC(engram_kv, 5 * DS4_N_EMBD);
    ALLOC(block_scores, (ctx + 7) / 8); ALLOC(block_allowed, (ctx + 7) / 8);
#undef ALLOC
    ds41c_reset(g); return true;
fail:
    ds41c_free(g); return false;
}

static void ds41c_round(float *x, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    /* ds41_cpu_bf16 lives in another translation unit, so without LTO the
     * scalar form is one call per element. This repeats its round-to-nearest
     * even on the top 16 bits in integer lanes, leaving NaN and infinities
     * alone; it stays the reference the primitive tests check. */
    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i bias = _mm512_set1_epi32(0x7fff);
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i keep = _mm512_set1_epi32((int)0xffff0000u);
    for (; i + 16 <= n; i += 16) {
        __m512i b = _mm512_loadu_si512((const void *)(x + i));
        const __mmask16 finite =
            _mm512_cmpneq_epi32_mask(_mm512_and_si512(b, exp_mask), exp_mask);
        const __m512i inc =
            _mm512_add_epi32(bias, _mm512_and_si512(_mm512_srli_epi32(b, 16), one));
        b = _mm512_mask_add_epi32(b, finite, b, inc);
        _mm512_storeu_si512((void *)(x + i), _mm512_and_si512(b, keep));
    }
#endif
    for (; i < n; i++) x[i] = ds41_cpu_bf16(x[i]);
}

static void ds41c_mat(float *out, const ds4_model *m, const ds4_tensor *t, const float *in, bool round) {
    if (t->type == DS4_TENSOR_Q8_0) matvec_q8_0_f32_ref(out, m, t, in);
    else matvec_any(out, m, t, in);
    if (round) ds41c_round(out, t->dim[1]);
}

static void ds41c_norm(float *out, const float *in, const ds4_model *m, const ds4_tensor *w) {
    rms_norm_weight(out, in, tensor_data(m, w), w->dim[0], DS4_RMS_EPS);
    ds41c_round(out, w->dim[0]);
}

static void ds41c_hc(ds41_cpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, bool ffn) {
    float mix[24];
    rms_norm_no_weight(g->flat, ffn ? g->after : g->residual, 4 * DS4_N_EMBD, DS4_RMS_EPS);
    ds41c_mat(mix, m, ffn ? l->hc_ffn_fn : l->hc_attn_fn, g->flat, false);
    hc_split_sinkhorn_one(ffn ? g->ffn_split : g->attn_split, mix,
        tensor_data(m, ffn ? l->hc_ffn_scale : l->hc_attn_scale),
        tensor_data(m, ffn ? l->hc_ffn_base : l->hc_attn_base), 4, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS);
}

static bool ds41c_rope(float *x, uint32_t heads, uint32_t width, uint32_t pos, uint32_t il, bool inverse) {
    return ds41_cpu_rope(x, (size_t)heads * width, width, heads, 1, pos, 1, il >= 2, inverse);
}

typedef struct {
    float *scores;
    const float *index_q, *rows, *weights;
} ds41c_index_score_ctx;

static void ds41c_index_score_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const ds41c_index_score_ctx *c = vctx;
    for (uint64_t r = r0; r < r1; r++) {
        const float *k = c->rows + r * 128;
        float sum = 0;
        for (unsigned h = 0; h < 32; h++)
            sum += fmaxf(dot_f32(c->index_q + h * 128, k, 128) / 64, 0) * c->weights[h];
        c->scores[r] = sum;
    }
}

typedef struct {
    float *rows;
    const float *window, *compressed;
    const int *selected;
    unsigned raw;
    uint32_t pos;
} ds41c_gather_ctx;

/* Row r < raw comes from the sliding window, the rest from the compressed
 * cache in the order the indexer selected; each row owns its output slot. */
static void ds41c_gather_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const ds41c_gather_ctx *c = vctx;
    for (uint64_t r = r0; r < r1; r++) {
        const float *src = r < c->raw
            ? c->window + (((uint64_t)c->pos + 1 - c->raw + r) % 128) * 512
            : c->compressed + (size_t)c->selected[r - c->raw] * 512;
        memcpy(c->rows + r * 512, src, 512 * sizeof(float));
    }
}

static void ds41c_sort_indices(int *idx, unsigned n) {
    for (unsigned i = 1; i < n; i++) {
        const int value = idx[i];
        unsigned j = i;
        while (j && idx[j - 1] > value) { idx[j] = idx[j - 1]; j--; }
        idx[j] = value;
    }
}

typedef struct { float score; int index; } ds41c_topk_item;

static bool ds41c_topk_worse(ds41c_topk_item a, ds41c_topk_item b) {
    return a.score < b.score || (a.score == b.score && a.index > b.index);
}

static void ds41c_topk_sift_down(ds41c_topk_item *heap, unsigned n, unsigned p) {
    for (;;) {
        unsigned worst = p, l = 2 * p + 1, r = l + 1;
        if (l < n && ds41c_topk_worse(heap[l], heap[worst])) worst = l;
        if (r < n && ds41c_topk_worse(heap[r], heap[worst])) worst = r;
        if (worst == p) return;
        ds41c_topk_item t = heap[p]; heap[p] = heap[worst]; heap[worst] = t;
        p = worst;
    }
}

static void ds41c_topk_indices_heap(const float *score, unsigned n,
                                    unsigned k, int *idx) {
    ds41c_topk_item heap[512];
    if (k > n) k = n;
    for (unsigned i = 0; i < k; i++) heap[i] = (ds41c_topk_item){score[i], (int)i};
    for (unsigned i = k / 2; i-- > 0;) ds41c_topk_sift_down(heap, k, i);
    for (unsigned i = k; i < n; i++) {
        const ds41c_topk_item candidate = {score[i], (int)i};
        if (ds41c_topk_worse(heap[0], candidate)) {
            heap[0] = candidate;
            ds41c_topk_sift_down(heap, k, 0);
        }
    }
    for (unsigned i = 0; i < k; i++) idx[i] = heap[i].index;
    ds41c_sort_indices(idx, k);
}

static bool ds41c_attention(ds41_cpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                              uint32_t il, bool projected, bool project_output) {
    const uint32_t pos = g->pos, ratio = ds4_layer_compress_ratio(il);
    const unsigned owner = il < 8 ? 0 : il < 14 ? 1 : il < 20 ? 2 : 3;
    const uint32_t count = ratio ? (pos + 1) / ratio : 0;
    if (!projected) {
        ds41c_mat(g->qr, m, l->attn_q_a, g->norm, true);
        ds41c_norm(g->qr, g->qr, m, l->attn_q_a_norm);
        ds41c_mat(g->q, m, l->attn_q_b, g->qr, true);
        ds41c_mat(g->kv, m, l->attn_kv, g->norm, true);
        ds41c_norm(g->kv, g->kv, m, l->attn_kv_a_norm);
    }
    if (!ds41c_rope(g->q, 64, 512, pos, il, false) || !ds41c_rope(g->kv, 1, 512, pos, il, false) ||
        !ds41_cpu_quantize(g->kv, 512, 512, 1, DS41_CPU_FP8_E8M0)) return false;
    memcpy(g->window[il] + (pos % 128) * 512, g->kv, sizeof(g->kv));
    if (ds41_kv_source(il)) {
        ds41c_mat(g->pool, m, l->attn_compressor_kv, g->norm, ratio == 1);
        if (ratio == 2) {
            ds41c_mat(g->pool_score, m, l->attn_compressor_gate, g->norm, false);
            if (pos & 1) ds41_cpu_pool2(g->latent, g->previous_kv[owner], g->pool,
                                       g->previous_score[owner], g->pool_score, 512);
            else {
                memcpy(g->previous_kv[owner], g->pool, sizeof(g->pool));
                memcpy(g->previous_score[owner], g->pool_score, sizeof(g->pool_score));
            }
        } else memcpy(g->latent, g->pool, sizeof(g->pool));
        if ((pos + 1) % ratio == 0) {
            ds41c_norm(g->latent, g->latent, m, l->attn_compressor_norm);
            ds41c_mat(g->index_k, m, l->indexer_attn_k, g->latent, true);
            ds41c_norm(g->index_k, g->index_k, m, l->indexer_k_norm);
            if (!ds41c_rope(g->index_k, 1, 128, pos + 1 - ratio, il, false) ||
                !ds41_cpu_quantize(g->index_k, 128, 128, 1, DS41_CPU_FP4_E8M0) ||
                !ds41c_rope(g->latent, 1, 512, pos + 1 - ratio, il, false) ||
                !ds41_cpu_quantize(g->latent, 512, 512, 1, DS41_CPU_FP4_E4M3)) return false;
            memcpy(g->index[owner] + (size_t)(count - 1) * 128, g->index_k, sizeof(g->index_k));
            memcpy(g->compressed[owner] + (size_t)(count - 1) * 512, g->latent, sizeof(g->latent));
        }
    }
    const unsigned top = count < 512 ? count : 512;
    if (count && ds41_index_source(il)) {
        ds41c_mat(g->index_q, m, l->indexer_attn_q_b, g->qr, true);
        if (!ds41c_rope(g->index_q, 32, 128, pos, il, false) ||
            !ds41_cpu_quantize(g->index_q, 32 * 128, 128, 32, DS41_CPU_FP4_E8M0)) return false;
        ds41c_mat(g->index_weights, m, l->indexer_proj, g->norm, true);
        /* Row count tracks context length, so this outgrows every projection
         * in the layer at long context. Each row is independent. */
        const ds41c_index_score_ctx score_ctx = {
            .scores = g->scores, .index_q = g->index_q,
            .rows = g->index[owner], .weights = g->index_weights,
        };
        uint64_t score_min_rows = DS4_PARALLEL_MIN_MACS / (32 * 128);
        if (score_min_rows == 0) score_min_rows = 1;
        ds4_parallel_for_min_rows(count, ds41c_index_score_worker,
                                  (void *)&score_ctx, score_min_rows);
        const unsigned blocks = (count + 7) / 8;
        if (il == 20) {
            for (unsigned b = 0; b < blocks; b++) {
                g->block_scores[b] = -INFINITY; g->block_allowed[b] = false;
                for (unsigned r = b * 8; r < count && r < (b + 1) * 8; r++)
                    g->block_scores[b] = fmaxf(g->block_scores[b], g->scores[r]);
            }
            g->block_scores[blocks - 1] = INFINITY;
            const unsigned keep = blocks < 2048 ? blocks : 2048;
            topk_desc(g->block_scores, blocks, keep, g->blocks);
            for (unsigned b = 0; b < keep; b++) g->block_allowed[g->blocks[b]] = true;
        } else if (il > 20) {
            for (unsigned r = 0; r < count; r++) if (!g->block_allowed[r / 8]) g->scores[r] = -INFINITY;
        }
        ds41c_topk_indices_heap(g->scores, count, top, g->selected);
    }
    const unsigned raw = pos + 1 < 128 ? pos + 1 : 128;
    const ds41c_gather_ctx gather_ctx = {
        .rows = g->rows, .window = g->window[il], .compressed = g->compressed[owner],
        .selected = g->selected, .raw = raw, .pos = pos,
    };
    /* Selected rows are scattered through the compressed cache, so each copy
     * is dominated by a cache miss of about a microsecond; below this many
     * rows a pool round trip costs more than it saves. */
    ds4_parallel_for_min_rows(raw + top, ds41c_gather_worker, (void *)&gather_ctx, 64);
    layer_attention_rows_one(g->heads, m, l, g->q, g->rows, raw + top);
    ds41c_round(g->heads, 64 * 512);
    if (!ds41c_rope(g->heads, 64, 512, pos, il, true)) return false;
    if (!project_output) return true;
    /* Each output group has its own 4096 -> 1024 matrix. They are contiguous,
     * so one grouped dispatch replaces eight pool round trips. */
    matvec_q8_0_f32_ref_grouped(g->low, m, l->attn_output_a, g->heads, 8, 4096, 1024);
    ds41c_round(g->low, 8 * 1024);
    ds41c_mat(g->block, m, l->attn_output_b, g->low, true);
    return true;
}

static void ds41c_moe(ds41_cpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                      unsigned il, int token, double *shared_sec, double *routed_sec) {
    const bool profile = shared_sec && routed_sec;
    double started = profile ? now_sec() : 0.0;
    ds41c_mat(g->gate, m, l->ffn_gate_shexp, g->norm, true);
    ds41c_mat(g->up, m, l->ffn_up_shexp, g->norm, true);
    swiglu(g->mid, g->gate, g->up, DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP);
    ds41c_round(g->mid, DS4_N_FF_EXP);
    ds41c_mat(g->shared, m, l->ffn_down_shexp, g->mid, true);
    if (profile) { *shared_sec += now_sec() - started; started = now_sec(); }
    /* Upstream's routed kernel keeps an F32 reduction. Only the shared
     * expert and the final combined block have explicit BF16 boundaries. */
    layer_routed_moe_one(g->block, m, l, g->norm, il, token, DS4_SWIGLU_CLAMP_EXP, false);
    for (unsigned i = 0; i < DS4_N_EMBD; i++) g->block[i] = ds41_cpu_bf16(g->block[i] + g->shared[i]);
    if (profile) *routed_sec += now_sec() - started;
}

static void ds41c_mat_batch(float *out, const ds4_model *m, const ds4_tensor *w,
                              const float *in, uint32_t count, bool round) {
    if (w->type == DS4_TENSOR_Q8_0) matmul_q8_0_f32_ref(out, m, w, in, count);
    else for (uint32_t t = 0; t < count; t++)
        matvec_any(out + (uint64_t)t * w->dim[1], m, w,
                   in + (uint64_t)t * w->dim[0]);
    if (round) ds41c_round(out, (size_t)count * w->dim[1]);
}

/* Layer-major prompt chunk. Dense Q8 projections scan each weight row once
 * for several F32 activations. Cache publication stays token ordered inside
 * every layer, preserving causal visibility and the decode graph's numeric
 * boundaries. */
static bool ds41c_prefill_chunk(ds41_cpu_graph *g, const ds4_model *m,
                                const ds4_weights *w, const int *token,
                                uint32_t count, float *logits) {
    if (!count || count > 16 || !g->valid || g->pos + count > g->ctx) return false;
    const uint64_t hc = 4u * DS4_N_EMBD, e = DS4_N_EMBD;
#define BALLOC(name, n) float *name = calloc((size_t)(n), sizeof(float));
    BALLOC(residual, (uint64_t)count * hc) BALLOC(after, (uint64_t)count * hc)
    BALLOC(norm, (uint64_t)count * e) BALLOC(block, (uint64_t)count * e)
    BALLOC(shared, (uint64_t)count * e) BALLOC(qr, (uint64_t)count * DS4_N_LORA_Q)
    BALLOC(q, (uint64_t)count * 64u * 512u) BALLOC(kv, (uint64_t)count * 512u)
    BALLOC(heads, (uint64_t)count * 64u * 512u) BALLOC(low, (uint64_t)count * 8u * 1024u)
    BALLOC(gate, (uint64_t)count * DS4_N_FF_EXP) BALLOC(up, (uint64_t)count * DS4_N_FF_EXP)
    BALLOC(mid, (uint64_t)count * DS4_N_FF_EXP) BALLOC(pre, (uint64_t)count * 4u)
    BALLOC(attn_split, (uint64_t)count * 24u) BALLOC(ffn_split, (uint64_t)count * 24u)
    BALLOC(disk, (uint64_t)count * 2u * 24u * 256u)
#undef BALLOC
    int *selected = calloc((size_t)count * 512u, sizeof(int));
    bool *allowed = calloc((size_t)count * ((g->ctx + 7u) / 8u), sizeof(bool));
    bool ok = residual && after && norm && block && shared && qr && q && kv && heads && low && gate && up &&
              mid && pre && attn_split && ffn_split && disk && selected && allowed;
    const uint32_t initial = g->pos, allowed_stride = (g->ctx + 7u) / 8u;
    ds4_engram_history history = g->history;
    uint32_t ids[16][2][24];
    g->valid = false;
    for (uint32_t t = 0; ok && t < count; t++) {
        ok = token[t] >= 0 && (uint32_t)token[t] < DS4_N_VOCAB &&
             ds4_engram_hash(&g->engram, &history, token + t, NULL, 1, &ids[t][0][0]);
        if (ok) {
            embed_token_any(m, w, token[t], g->x);
            hc_from_plain_embedding(residual + (uint64_t)t * hc, g->x, DS4_N_EMBD, 4);
            pre[(uint64_t)t * 4u] = 1.0f;
        }
    }
    for (unsigned table = 0; ok && table < 2; table++)
        ok = ds4_engram_read_batch(&g->tables[table], &ids[0][table][0], count, 48u,
            disk + (uint64_t)table * count * 24u * 256u);
    for (unsigned il = 0; ok && il < 40; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        for (uint32_t t = 0; ok && t < count; t++) {
            memcpy(g->residual, residual + (uint64_t)t * hc, hc * sizeof(float));
            memcpy(g->pre, pre + (uint64_t)t * 4u, 4u * sizeof(float));
            if (ds41_engram_layer(il)) {
                const unsigned table = il == 1 ? 0 : 1;
                memcpy(g->disk_rows[table],
                       disk + ((uint64_t)table * count + t) * 24u * 256u,
                       sizeof(g->disk_rows[table]));
                ds41c_mat(g->engram_kv, m, l->engram_kv, g->disk_rows[table], true);
                ds41_cpu_engram_add(g->residual, g->engram_kv,
                    tensor_data(m, l->engram_q_norm), tensor_data(m, l->engram_k_norm),
                    DS4_N_EMBD, DS4_RMS_EPS);
                memcpy(residual + (uint64_t)t * hc, g->residual, hc * sizeof(float));
            }
        }
        for (uint32_t t = 0; ok && t < count; t++) {
            memcpy(g->residual, residual + (uint64_t)t * hc, hc * sizeof(float));
            memcpy(g->pre, pre + (uint64_t)t * 4u, 4u * sizeof(float));
            ds41c_hc(g, m, l, false);
            hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
            ds41c_round(g->x, DS4_N_EMBD);
            ds41c_norm(norm + (uint64_t)t * e, g->x, m, l->attn_norm);
            memcpy(attn_split + (uint64_t)t * 24u, g->attn_split, 24u * sizeof(float));
        }
        ds41c_mat_batch(qr, m, l->attn_q_a, norm, count, true);
        for (uint32_t t = 0; t < count; t++)
            ds41c_norm(qr + (uint64_t)t * DS4_N_LORA_Q,
                       qr + (uint64_t)t * DS4_N_LORA_Q, m, l->attn_q_a_norm);
        ds41c_mat_batch(q, m, l->attn_q_b, qr, count, true);
        ds41c_mat_batch(kv, m, l->attn_kv, norm, count, true);
        for (uint32_t t = 0; t < count; t++)
            ds41c_norm(kv + (uint64_t)t * 512u, kv + (uint64_t)t * 512u,
                       m, l->attn_kv_a_norm);
        for (uint32_t t = 0; ok && t < count; t++) {
            g->pos = initial + t;
            memcpy(g->norm, norm + (uint64_t)t * e, e * sizeof(float));
            memcpy(g->qr, qr + (uint64_t)t * DS4_N_LORA_Q, DS4_N_LORA_Q * sizeof(float));
            memcpy(g->q, q + (uint64_t)t * 64u * 512u, 64u * 512u * sizeof(float));
            memcpy(g->kv, kv + (uint64_t)t * 512u, 512u * sizeof(float));
            memcpy(g->selected, selected + (uint64_t)t * 512u, 512u * sizeof(int));
            memcpy(g->block_allowed, allowed + (uint64_t)t * allowed_stride,
                   allowed_stride * sizeof(bool));
            ok = ds41c_attention(g, m, l, il, true, false);
            if (ok) {
                memcpy(heads + (uint64_t)t * 64u * 512u, g->heads, 64u * 512u * sizeof(float));
                memcpy(selected + (uint64_t)t * 512u, g->selected, 512u * sizeof(int));
                memcpy(allowed + (uint64_t)t * allowed_stride, g->block_allowed,
                       allowed_stride * sizeof(bool));
            }
        }
        if (ok) {
            matmul_q8_0_f32_ref_grouped(low, m, l->attn_output_a, heads,
                                         count, 8, 4096, 1024);
            ds41c_round(low, (size_t)count * 8u * 1024u);
            ds41c_mat_batch(block, m, l->attn_output_b, low, count, true);
        }
        for (uint32_t t = 0; ok && t < count; t++) {
            memcpy(g->residual, residual + (uint64_t)t * hc, hc * sizeof(float));
            memcpy(g->block, block + (uint64_t)t * e, e * sizeof(float));
            memcpy(g->attn_split, attn_split + (uint64_t)t * 24u, 24u * sizeof(float));
            hc_post_one(g->after, g->block, g->residual, g->attn_split + 4,
                        g->attn_split + 8, DS4_N_EMBD, 4);
            ds41c_round(g->after, hc);
            memcpy(after + (uint64_t)t * hc, g->after, hc * sizeof(float));
        }
        for (uint32_t t = 0; ok && t < count; t++) {
            memcpy(g->after, after + (uint64_t)t * hc, hc * sizeof(float));
            memcpy(g->attn_split, attn_split + (uint64_t)t * 24u, 24u * sizeof(float));
            ds41c_hc(g, m, l, true);
            hc_weighted_sum_one(g->x, g->after, g->attn_split, DS4_N_EMBD, 4);
            ds41c_round(g->x, DS4_N_EMBD);
            ds41c_norm(norm + (uint64_t)t * e, g->x, m, l->ffn_norm);
            memcpy(ffn_split + (uint64_t)t * 24u, g->ffn_split, 24u * sizeof(float));
        }
        ds41c_mat_batch(gate, m, l->ffn_gate_shexp, norm, count, true);
        ds41c_mat_batch(up, m, l->ffn_up_shexp, norm, count, true);
        for (uint32_t t = 0; t < count; t++) {
            swiglu(mid + (uint64_t)t * DS4_N_FF_EXP,
                   gate + (uint64_t)t * DS4_N_FF_EXP,
                   up + (uint64_t)t * DS4_N_FF_EXP, DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP);
            ds41c_round(mid + (uint64_t)t * DS4_N_FF_EXP, DS4_N_FF_EXP);
        }
        ds41c_mat_batch(shared, m, l->ffn_down_shexp, mid, count, true);
        for (uint32_t t = 0; ok && t < count; t++) {
            layer_routed_moe_one(g->block, m, l, norm + (uint64_t)t * e,
                                 il, token[t], DS4_SWIGLU_CLAMP_EXP, false);
            for (unsigned d = 0; d < DS4_N_EMBD; d++)
                g->block[d] = ds41_cpu_bf16(g->block[d] + shared[(uint64_t)t * e + d]);
            hc_post_one(residual + (uint64_t)t * hc, g->block,
                        after + (uint64_t)t * hc, ffn_split + (uint64_t)t * 24u + 4,
                        ffn_split + (uint64_t)t * 24u + 8, DS4_N_EMBD, 4);
            ds41c_round(residual + (uint64_t)t * hc, hc);
            memcpy(pre + (uint64_t)t * 4u, ffn_split + (uint64_t)t * 24u, 4u * sizeof(float));
        }
    }
    if (ok) {
        const uint32_t last = count - 1;
        memcpy(g->residual, residual + (uint64_t)last * hc, hc * sizeof(float));
        memcpy(g->pre, pre + (uint64_t)last * 4u, 4u * sizeof(float));
        if (logits) {
            hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
            ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, w->output_norm);
            ds41c_mat(logits, m, w->output, g->norm, false);
            for (unsigned i = 0; i < DS4_N_VOCAB; i++) if (!isfinite(logits[i])) ok = false;
        }
    }
    if (ok) { g->history = history; g->pos = initial + count; g->valid = true; }
    else { g->pos = initial; g->valid = false; }
    free(allowed); free(selected); free(disk); free(ffn_split); free(attn_split); free(pre);
    free(mid); free(up); free(gate); free(low); free(heads); free(kv); free(q); free(qr); free(shared); free(block);
    free(norm); free(after); free(residual);
    return ok;
}

static bool ds41c_step(ds41_cpu_graph *g, const ds4_model *m, const ds4_weights *w, int token, float *logits) {
    if (!g->valid || g->pos >= g->ctx || token < 0 || (uint32_t)token >= DS4_N_VOCAB) return false;
    const bool profile = getenv("DS4_CPU_V41_PROFILE") != NULL;
    double engram_sec = 0, attn_hc_sec = 0, attn_sec = 0, attn_post_sec = 0;
    double ffn_hc_sec = 0, shared_sec = 0, routed_sec = 0, ffn_post_sec = 0, head_sec = 0;
    double started = profile ? now_sec() : 0.0;
    ds4_engram_history history = g->history; uint32_t ids[2][24];
    if (!ds4_engram_hash(&g->engram, &history, &token, NULL, 1, &ids[0][0])) return false;
    for (unsigned i = 0; i < 2; i++)
        if (!ds4_engram_read(&g->tables[i], ids[i], 24, g->disk_rows[i])) return false;
    if (profile) engram_sec += now_sec() - started;
    g->valid = false;
    embed_token_any(m, w, token, g->x);
    hc_from_plain_embedding(g->residual, g->x, DS4_N_EMBD, 4);
    g->pre[0] = 1; g->pre[1] = g->pre[2] = g->pre[3] = 0;
    for (unsigned il = 0; il < 40; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (ds41_engram_layer(il)) {
            started = profile ? now_sec() : 0.0;
            ds41c_mat(g->engram_kv, m, l->engram_kv, g->disk_rows[il == 1 ? 0 : 1], true);
            ds41_cpu_engram_add(g->residual, g->engram_kv, tensor_data(m, l->engram_q_norm),
                                tensor_data(m, l->engram_k_norm), DS4_N_EMBD, DS4_RMS_EPS);
            if (profile) engram_sec += now_sec() - started;
        }
        started = profile ? now_sec() : 0.0;
        ds41c_hc(g, m, l, false);
        hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, l->attn_norm);
        if (profile) attn_hc_sec += now_sec() - started;
        started = profile ? now_sec() : 0.0;
        if (!ds41c_attention(g, m, l, il, false, true)) return false;
        if (profile) attn_sec += now_sec() - started;
        started = profile ? now_sec() : 0.0;
        hc_post_one(g->after, g->block, g->residual, g->attn_split + 4, g->attn_split + 8, DS4_N_EMBD, 4);
        ds41c_round(g->after, 4 * DS4_N_EMBD);
        if (profile) attn_post_sec += now_sec() - started;
        started = profile ? now_sec() : 0.0;
        ds41c_hc(g, m, l, true);
        hc_weighted_sum_one(g->x, g->after, g->attn_split, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, l->ffn_norm);
        if (profile) ffn_hc_sec += now_sec() - started;
        ds41c_moe(g, m, l, il, token,
                  profile ? &shared_sec : NULL, profile ? &routed_sec : NULL);
        started = profile ? now_sec() : 0.0;
        hc_post_one(g->residual, g->block, g->after, g->ffn_split + 4, g->ffn_split + 8, DS4_N_EMBD, 4);
        ds41c_round(g->residual, 4 * DS4_N_EMBD); memcpy(g->pre, g->ffn_split, sizeof(g->pre));
        if (profile) ffn_post_sec += now_sec() - started;
    }
    if (logits) {
        started = profile ? now_sec() : 0.0;
        hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, w->output_norm);
        ds41c_mat(logits, m, w->output, g->norm, false);
        for (unsigned i = 0; i < DS4_N_VOCAB; i++) if (!isfinite(logits[i])) return false;
        if (profile) head_sec += now_sec() - started;
    }
    if (profile) fprintf(stderr,
        "ds4: V4.1 CPU profile pos=%u ms engram=%.3f attn_hc=%.3f attn=%.3f attn_post=%.3f "
        "ffn_hc=%.3f shared=%.3f routed=%.3f ffn_post=%.3f head=%.3f\n",
        g->pos, engram_sec * 1e3, attn_hc_sec * 1e3, attn_sec * 1e3,
        attn_post_sec * 1e3, ffn_hc_sec * 1e3, shared_sec * 1e3,
        routed_sec * 1e3, ffn_post_sec * 1e3, head_sec * 1e3);
    g->history = history; g->pos++; g->valid = true; return true;
}
