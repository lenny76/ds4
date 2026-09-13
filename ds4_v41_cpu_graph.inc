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
    for (size_t i = 0; i < n; i++) x[i] = ds41_cpu_bf16(x[i]);
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

static void ds41c_sort_indices(int *idx, unsigned n) {
    for (unsigned i = 1; i < n; i++) {
        const int value = idx[i];
        unsigned j = i;
        while (j && idx[j - 1] > value) { idx[j] = idx[j - 1]; j--; }
        idx[j] = value;
    }
}

static bool ds41c_attention(ds41_cpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il) {
    const uint32_t pos = g->pos, ratio = ds4_layer_compress_ratio(il);
    const unsigned owner = il < 8 ? 0 : il < 14 ? 1 : il < 20 ? 2 : 3;
    const uint32_t count = ratio ? (pos + 1) / ratio : 0;
    ds41c_mat(g->qr, m, l->attn_q_a, g->norm, true);
    ds41c_norm(g->qr, g->qr, m, l->attn_q_a_norm);
    ds41c_mat(g->q, m, l->attn_q_b, g->qr, true);
    ds41c_mat(g->kv, m, l->attn_kv, g->norm, true);
    ds41c_norm(g->kv, g->kv, m, l->attn_kv_a_norm);
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
        for (unsigned r = 0; r < count; r++) {
            float sum = 0;
            for (unsigned h = 0; h < 32; h++)
                sum += fmaxf(dot_f32(g->index_q + h * 128, g->index[owner] + (size_t)r * 128, 128) / 64, 0) * g->index_weights[h];
            g->scores[r] = sum;
        }
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
        topk_desc(g->scores, count, top, g->selected);
        /* The reference graph restores position order after selecting by
         * score. It is mathematically equivalent, but fixes accumulation
         * order for reproducible CPU validation. */
        ds41c_sort_indices(g->selected, top);
    }
    const unsigned raw = pos + 1 < 128 ? pos + 1 : 128;
    for (unsigned r = 0; r < raw; r++)
        memcpy(g->rows + r * 512, g->window[il] + ((pos + 1 - raw + r) % 128) * 512, 512 * sizeof(float));
    for (unsigned r = 0; r < top; r++)
        memcpy(g->rows + (raw + r) * 512, g->compressed[owner] + (size_t)g->selected[r] * 512, 512 * sizeof(float));
    layer_attention_rows_one(g->heads, m, l, g->q, g->rows, raw + top);
    ds41c_round(g->heads, 64 * 512);
    if (!ds41c_rope(g->heads, 64, 512, pos, il, true)) return false;
    /* Each output group has its own 4096 -> 1024 matrix. */
    for (unsigned group = 0; group < 8; group++) {
        ds4_tensor view = *l->attn_output_a;
        uint64_t bytes;
        if (!tensor_nbytes(view.type, 4096, &bytes)) return false;
        view.abs_offset += (uint64_t)group * 1024 * bytes;
        view.ndim = 2; view.dim[0] = 4096; view.dim[1] = 1024; view.elements = 4096 * 1024;
        view.bytes = 1024 * bytes;
        ds41c_mat(g->low + group * 1024, m, &view, g->heads + group * 4096, true);
    }
    ds41c_mat(g->block, m, l->attn_output_b, g->low, true);
    return true;
}

static void ds41c_moe(ds41_cpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, unsigned il, int token) {
    ds41c_mat(g->gate, m, l->ffn_gate_shexp, g->norm, true);
    ds41c_mat(g->up, m, l->ffn_up_shexp, g->norm, true);
    swiglu(g->mid, g->gate, g->up, DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP);
    ds41c_round(g->mid, DS4_N_FF_EXP);
    ds41c_mat(g->shared, m, l->ffn_down_shexp, g->mid, true);
    /* Upstream's routed kernel keeps an F32 reduction. Only the shared
     * expert and the final combined block have explicit BF16 boundaries. */
    layer_routed_moe_one(g->block, m, l, g->norm, il, token, DS4_SWIGLU_CLAMP_EXP, false);
    for (unsigned i = 0; i < DS4_N_EMBD; i++) g->block[i] = ds41_cpu_bf16(g->block[i] + g->shared[i]);
}

static bool ds41c_step(ds41_cpu_graph *g, const ds4_model *m, const ds4_weights *w, int token, float *logits) {
    if (!g->valid || g->pos >= g->ctx || token < 0 || (uint32_t)token >= DS4_N_VOCAB) return false;
    ds4_engram_history history = g->history; uint32_t ids[2][24];
    if (!ds4_engram_hash(&g->engram, &history, &token, NULL, 1, &ids[0][0])) return false;
    for (unsigned i = 0; i < 2; i++)
        if (!ds4_engram_read(&g->tables[i], ids[i], 24, g->disk_rows[i])) return false;
    g->valid = false;
    embed_token_any(m, w, token, g->x);
    hc_from_plain_embedding(g->residual, g->x, DS4_N_EMBD, 4);
    g->pre[0] = 1; g->pre[1] = g->pre[2] = g->pre[3] = 0;
    for (unsigned il = 0; il < 40; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (ds41_engram_layer(il)) {
            ds41c_mat(g->engram_kv, m, l->engram_kv, g->disk_rows[il == 1 ? 0 : 1], true);
            ds41_cpu_engram_add(g->residual, g->engram_kv, tensor_data(m, l->engram_q_norm),
                                tensor_data(m, l->engram_k_norm), DS4_N_EMBD, DS4_RMS_EPS);
        }
        ds41c_hc(g, m, l, false);
        hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, l->attn_norm);
        if (!ds41c_attention(g, m, l, il)) return false;
        hc_post_one(g->after, g->block, g->residual, g->attn_split + 4, g->attn_split + 8, DS4_N_EMBD, 4);
        ds41c_round(g->after, 4 * DS4_N_EMBD);
        ds41c_hc(g, m, l, true);
        hc_weighted_sum_one(g->x, g->after, g->attn_split, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, l->ffn_norm);
        ds41c_moe(g, m, l, il, token);
        hc_post_one(g->residual, g->block, g->after, g->ffn_split + 4, g->ffn_split + 8, DS4_N_EMBD, 4);
        ds41c_round(g->residual, 4 * DS4_N_EMBD); memcpy(g->pre, g->ffn_split, sizeof(g->pre));
    }
    if (logits) {
        hc_weighted_sum_one(g->x, g->residual, g->pre, DS4_N_EMBD, 4);
        ds41c_round(g->x, DS4_N_EMBD); ds41c_norm(g->norm, g->x, m, w->output_norm);
        ds41c_mat(logits, m, w->output, g->norm, false);
        for (unsigned i = 0; i < DS4_N_VOCAB; i++) if (!isfinite(logits[i])) return false;
    }
    g->history = history; g->pos++; g->valid = true; return true;
}
