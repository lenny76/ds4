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

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test-topk")) return test_topk_heap();
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
