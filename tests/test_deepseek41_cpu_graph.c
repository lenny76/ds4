/* Explicit diagnostic executable: the production admission check stays intact
 * until layer/logit parity has been measured on the pinned checkpoint. */
#include "../ds4.c"

int main(int argc, char **argv) {
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
