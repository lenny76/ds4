/* Explicit diagnostic executable: the production admission check stays intact
 * until layer/logit parity has been measured on the pinned checkpoint. */
#include "../ds4.c"

static int test_topk_heap(void) {
    uint32_t state = 1;
    float score[4096];
    int ref[512], got[512];
    for (unsigned trial = 0; trial < 2000; trial++) {
        state = state * 1664525u + 1013904223u;
        const unsigned n = 513u + state % (4096u - 512u);
        for (unsigned i = 0; i < n; i++) {
            state = state * 1664525u + 1013904223u;
            score[i] = (float)((int)(state % 257u) - 128);
        }
        topk_desc(score, (int)n, 512, ref);
        ds41c_sort_indices(ref, 512);
        ds41c_topk_indices_heap(score, n, 512, got);
        for (unsigned i = 0; i < 512; i++) {
            if (ref[i] != got[i]) {
                fprintf(stderr, "topk mismatch trial=%u n=%u slot=%u ref=%d got=%d\n",
                        trial, n, i, ref[i], got[i]);
                return 1;
            }
        }
    }
    puts("V4.1 heap top-k: 2000 tie-heavy cases exact");
    return 0;
}

static int test_q8_f32_batch(void) {
    enum { rows = 37, in_dim = 96, blocks = 3, n_tok = 7 };
    uint8_t data[rows * blocks * 34];
    float x[n_tok * in_dim], ref[n_tok * rows], got[n_tok * rows];
    uint32_t state = 7;
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned b = 0; b < blocks; b++) {
            state = state * 1664525u + 1013904223u;
            const uint16_t scale = f32_to_f16(0.01f + (float)(state % 200u) / 100.0f);
            memcpy(data + (r * blocks + b) * 34, &scale, 2);
            for (unsigned i = 0; i < 32; i++) {
                state = state * 1664525u + 1013904223u;
                data[(r * blocks + b) * 34 + 2 + i] = (uint8_t)(int8_t)(state >> 24);
            }
        }
    }
    for (unsigned i = 0; i < n_tok * in_dim; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = (float)((int)(state % 2001u) - 1000) / 997.0f;
    }
    for (unsigned t = 0; t < n_tok; t++)
        for (unsigned r = 0; r < rows; r++)
            ref[t * rows + r] = dot_q8_0_row_f32_ref(
                data + r * blocks * 34, x + t * in_dim, in_dim, blocks);
    matmul_q8_0_f32_ref_ctx ctx = {
        .out = got, .data = data, .x = x, .n_tok = n_tok,
        .in_dim = in_dim, .out_dim = rows, .blocks = blocks,
    };
    matmul_q8_0_f32_ref_worker(&ctx, 0, rows);
    if (memcmp(ref, got, sizeof(ref))) {
        fputs("V4.1 F32 Q8 batch differs from scalar-token traversal\n", stderr);
        return 1;
    }
    puts("V4.1 F32 Q8 batch: bit-exact across 7 rows");
    return 0;
}

static int compare_prefill(int argc, char **argv) {
    if (argc < 5 || argc - 3 > DS4_V41_PREFILL_CHUNK_MAX) return 2;
    setenv("DS4_CPU_V41_EXPERIMENTAL", "1", 1);
    ds4_engine_options opt = {.model_path = argv[2], .backend = DS4_BACKEND_CPU,
        .context_size = 4096, .n_threads = 0, .power_percent = 100};
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt)) return 1;
    ds41_cpu_graph *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    float *la = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *lb = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    int tokens[DS4_V41_PREFILL_CHUNK_MAX], n = argc - 3, rc = 0;
    for (int i = 0; i < n; i++) tokens[i] = atoi(argv[i + 3]);
    if (!a || !b || !la || !lb || !ds41c_alloc(a, &e->model, argv[2], 4096) ||
        !ds41c_alloc(b, &e->model, argv[2], 4096)) rc = 1;
    for (int i = 0; !rc && i < n; i++)
        if (!ds41c_step(a, &e->model, &e->weights, tokens[i], i + 1 == n ? la : NULL)) rc = 1;
    if (!rc && !ds41c_prefill_chunk(b, &e->model, &e->weights, tokens, (uint32_t)n, lb)) rc = 1;
    if (!rc && memcmp(la, lb, (size_t)DS4_N_VOCAB * sizeof(float))) {
        fputs("V4.1 batch prefill final logits differ\n", stderr); rc = 1;
    }
    if (!rc && (!ds41c_step(a, &e->model, &e->weights, tokens[0], la) ||
                !ds41c_step(b, &e->model, &e->weights, tokens[0], lb) ||
                memcmp(la, lb, (size_t)DS4_N_VOCAB * sizeof(float)))) {
        fputs("V4.1 batch prefill continuation logits differ\n", stderr); rc = 1;
    }
    if (!rc) printf("V4.1 batch prefill: %d-token logits and continuation bit-exact\n", n);
    if (a) ds41c_free(a);
    if (b) ds41c_free(b);
    free(a); free(b); free(la); free(lb); ds4_engine_close(e); return rc;
}

/* Save a session's V4.1 CPU state after SPLIT tokens through the real
 * ds41c_save_payload()/ds41c_load_payload() functions and a real FILE*
 * (matching how ds4-agent's kvstore uses them), restore it into a THIRD,
 * never-touched graph, then continue both the untouched reference and the
 * restored graph for the remaining tokens. Bit-exact final and continuation
 * logits confirm the saved state (window/compressed/index/previous_kv/
 * previous_score/history) is complete and correctly reconstructed, not just
 * plausible-looking. */
static int compare_snapshot(int argc, char **argv) {
    if (argc < 6) return 2;
    setenv("DS4_CPU_V41_EXPERIMENTAL", "1", 1);
    ds4_engine_options opt = {.model_path = argv[2], .backend = DS4_BACKEND_CPU,
        .context_size = 4096, .n_threads = 0, .power_percent = 100};
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt)) return 1;

    char *end = NULL;
    long split = strtol(argv[3], &end, 10);
    const int n = argc - 4;
    if (end == argv[3] || *end || split < 1 || split >= n) {
        ds4_engine_close(e);
        return 2;
    }

    int *tokens = malloc((size_t)n * sizeof(*tokens));
    for (int i = 0; i < n; i++) tokens[i] = atoi(argv[i + 4]);

    ds41_cpu_graph *ref = calloc(1, sizeof(*ref));
    ds41_cpu_graph *snap = calloc(1, sizeof(*snap));
    ds41_cpu_graph *restored = calloc(1, sizeof(*restored));
    float *la = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *lb = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *lc = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    int rc = 0;
    if (!ref || !snap || !restored || !la || !lb || !lc ||
        !ds41c_alloc(ref, &e->model, argv[2], 4096) ||
        !ds41c_alloc(snap, &e->model, argv[2], 4096) ||
        !ds41c_alloc(restored, &e->model, argv[2], 4096)) rc = 1;

    for (int i = 0; !rc && i < (int)split; i++) {
        if (!ds41c_step(ref, &e->model, &e->weights, tokens[i], i + 1 == split ? la : NULL) ||
            !ds41c_step(snap, &e->model, &e->weights, tokens[i], i + 1 == split ? lb : NULL)) rc = 1;
    }
    if (!rc && memcmp(la, lb, (size_t)DS4_N_VOCAB * sizeof(float))) {
        fputs("V4.1 snapshot: phase-1 graphs differ before any save/load\n", stderr);
        rc = 1;
    }

    char err[256] = {0};
    FILE *fp = !rc ? tmpfile() : NULL;
    if (!rc && !fp) { fputs("tmpfile() failed\n", stderr); rc = 1; }

    ds4_session save_s;
    if (!rc) {
        memset(&save_s, 0, sizeof(save_s));
        save_s.v41_cpu = snap;
        save_s.logits = lb;
        save_s.checkpoint_valid = true;
        for (int i = 0; i < (int)split; i++) token_vec_push(&save_s.checkpoint, tokens[i]);
        if (ds41c_save_payload(&save_s, fp, err, sizeof(err))) {
            fprintf(stderr, "V4.1 snapshot: save failed: %s\n", err);
            rc = 1;
        }
    }

    if (!rc) {
        rewind(fp);
        uint64_t remaining = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
                             ds41c_payload_body_bytes(snap, (uint32_t)split);
        uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
        for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
            if (payload_read_u32(fp, &h[i], &remaining, err, sizeof(err))) rc = 1;
        ds4_session load_s;
        if (!rc) {
            memset(&load_s, 0, sizeof(load_s));
            load_s.v41_cpu = restored;
            load_s.logits = lc;
            if (ds41c_load_payload(&load_s, fp, h, remaining, err, sizeof(err))) {
                fprintf(stderr, "V4.1 snapshot: load failed: %s\n", err);
                rc = 1;
            } else if (memcmp(lb, lc, (size_t)DS4_N_VOCAB * sizeof(float))) {
                fputs("V4.1 snapshot: restored logits differ from the saved graph's own\n", stderr);
                rc = 1;
            } else if (restored->pos != (uint32_t)split ||
                       load_s.checkpoint.len != split ||
                       memcmp(load_s.checkpoint.v, tokens, (size_t)split * sizeof(int))) {
                fputs("V4.1 snapshot: restored pos/checkpoint do not match what was saved\n", stderr);
                rc = 1;
            }
        }
    }
    if (fp) fclose(fp);

    for (int i = (int)split; !rc && i < n; i++) {
        const bool last = i + 1 == n;
        if (!ds41c_step(ref, &e->model, &e->weights, tokens[i], last ? la : NULL) ||
            !ds41c_step(restored, &e->model, &e->weights, tokens[i], last ? lc : NULL)) rc = 1;
    }
    if (!rc && memcmp(la, lc, (size_t)DS4_N_VOCAB * sizeof(float))) {
        fputs("V4.1 snapshot: continuation after restore differs from the untouched reference\n", stderr);
        rc = 1;
    }
    if (!rc) printf("V4.1 snapshot: save/restore at token %ld of %d bit-exact "
                    "(final and %d-token continuation)\n", split, n, n - (int)split);

    if (ref) ds41c_free(ref);
    if (snap) ds41c_free(snap);
    if (restored) ds41c_free(restored);
    free(ref); free(snap); free(restored); free(la); free(lb); free(lc); free(tokens);
    ds4_engine_close(e);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test-topk")) return test_topk_heap();
    if (argc == 2 && !strcmp(argv[1], "--self-test-q8-batch")) return test_q8_f32_batch();
    if (argc >= 2 && !strcmp(argv[1], "--compare-prefill")) return compare_prefill(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "--compare-snapshot")) return compare_snapshot(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "Usage: %s MODEL.gguf TOKEN_ID [TOKEN_ID ...]\n", argv[0]);
        return 2;
    }
    /* The same explicit gate used by the CLI prevents accidental production
     * use before checkpoint parity has been established. */
    setenv("DS4_CPU_V41_EXPERIMENTAL", "1", 1);
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CPU,
        .context_size = 4096, .n_threads = 0, .power_percent = 100};
    ds4_engine *e = NULL;
    if (argc - 2 >= opt.context_size || ds4_engine_open(&e, &opt)) return 1;
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41) {
        fprintf(stderr, "V4.1 checkpoint required\n"); ds4_engine_close(e); return 1;
    }
    ds41_cpu_graph *g = calloc(1, sizeof(*g));
    float *logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (!g || !logits) { free(g); free(logits); ds4_engine_close(e); return 1; }
    if (!ds41c_alloc(g, &e->model, argv[1], opt.context_size)) {
        free(g); free(logits); ds4_engine_close(e); return 1;
    }
    int rc = 0;
    for (int i = 2; i < argc; i++) {
        char *end; errno = 0;
        long token = strtol(argv[i], &end, 10);
        if (errno || end == argv[i] || *end || token < 0 || token >= DS4_N_VOCAB) { rc = 2; break; }
        const double started = now_sec();
        if (!ds41c_step(g, &e->model, &e->weights, (int)token, logits)) { rc = 1; break; }
        int best[5]; topk_desc(logits, DS4_N_VOCAB, 5, best);
        fprintf(stdout, "position=%u token=%ld elapsed=%.3fs top", g->pos - 1, token, now_sec() - started);
        for (unsigned j = 0; j < 5; j++) fprintf(stdout, " %d:%.9g", best[j], logits[best[j]]);
        fputc('\n', stdout); fflush(stdout);
    }
    ds41c_free(g); free(g); free(logits); ds4_engine_close(e);
    return rc;
}
