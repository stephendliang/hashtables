/*
 * bench_kv64_layout.c — 2D grid search: block stride × PF distance
 *
 * Sweeps N=1,2,4 × PF=4,8,12,16,20,24,28,32,40,48,64 across insert,
 * get_hit, get_miss, delete, and churn workloads.
 *
 * Build & run:
 *   cc -O3 -march=native -std=gnu11 -o bench_kv64 bench_kv64_layout.c
 *   taskset -c 0 ./bench_kv64 > results.tsv
 */

#define SIMD_MAP_NAME          kv64_n1
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 1
#include "simd_map64.h"

#define SIMD_MAP_NAME          kv64_n2
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 2
#include "simd_map64.h"

#define SIMD_MAP_NAME          kv64_n4
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 4
#include "simd_map64.h"

#include <stdio.h>
#include <time.h>

#define NUM       2000000
#define VW        1
#define CHURN     200000
#define ROUNDS    10

static const int pfs[] = {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64};
#define NPF ((int)(sizeof(pfs) / sizeof(pfs[0])))

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double elapsed_ns(struct timespec t0, struct timespec t1, int ops) {
    return ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ops;
}

static uint64_t *g_keys, *g_vals, *g_miss, *g_exkeys, *g_exvals;

static void gen_data(void) {
    g_keys   = malloc((size_t)NUM * sizeof(uint64_t));
    g_vals   = malloc((size_t)NUM * VW * sizeof(uint64_t));
    g_miss   = malloc((size_t)NUM * sizeof(uint64_t));
    int extra_n = CHURN * ROUNDS;
    g_exkeys = malloc((size_t)extra_n * sizeof(uint64_t));
    g_exvals = malloc((size_t)extra_n * VW * sizeof(uint64_t));

    uint64_t s1 = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < NUM; i++) {
        uint64_t k;
        do { k = splitmix64(&s1); } while (k == 0);
        g_keys[i] = k;
    }
    uint64_t s2 = 0x1111222233334444ULL;
    for (int i = 0; i < NUM * VW; i++) g_vals[i] = splitmix64(&s2);
    uint64_t s3 = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < NUM; i++) {
        uint64_t k;
        do { k = splitmix64(&s3); } while (k == 0);
        g_miss[i] = k;
    }
    uint64_t s4 = 0xAAAABBBBCCCCDDDDULL;
    for (int i = 0; i < extra_n; i++) {
        uint64_t k;
        do { k = splitmix64(&s4); } while (k == 0);
        g_exkeys[i] = k;
    }
    uint64_t s5 = 0x5555666677778888ULL;
    for (int i = 0; i < extra_n * VW; i++) g_exvals[i] = splitmix64(&s5);
}

#define BENCH(NAME, STRIDE_LABEL) do {                                       \
    for (int pi = 0; pi < NPF; pi++) {                                       \
        int pf = pfs[pi];                                                    \
        struct timespec t0, t1;                                              \
                                                                             \
        /* Insert (init_cap + insert_unique) */                              \
        struct NAME m;                                                       \
        NAME##_init_cap(&m, NUM);                                            \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < NUM; i++) {                                      \
            if (i + pf < NUM)                                                \
                NAME##_prefetch_insert(&m, g_keys[i + pf]);                  \
            NAME##_insert_unique(&m, g_keys[i], &g_vals[i * VW]);           \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%d\tinsert\t%.1f\n",                                     \
               STRIDE_LABEL, pf, elapsed_ns(t0, t1, NUM));                  \
                                                                             \
        /* Get hit */                                                        \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        int found = 0;                                                       \
        for (int i = 0; i < NUM; i++) {                                      \
            if (i + pf < NUM)                                                \
                NAME##_prefetch(&m, g_keys[i + pf]);                         \
            found += (NAME##_get(&m, g_keys[i]) != NULL);                    \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%d\tget_hit\t%.1f\n",                                    \
               STRIDE_LABEL, pf, elapsed_ns(t0, t1, NUM));                  \
        (void)found;                                                         \
                                                                             \
        /* Get miss */                                                       \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < NUM; i++) {                                      \
            if (i + pf < NUM)                                                \
                NAME##_prefetch(&m, g_miss[i + pf]);                         \
            NAME##_get(&m, g_miss[i]);                                       \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%d\tget_miss\t%.1f\n",                                   \
               STRIDE_LABEL, pf, elapsed_ns(t0, t1, NUM));                  \
                                                                             \
        /* Delete all */                                                     \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < NUM; i++) {                                      \
            if (i + pf < NUM)                                                \
                NAME##_prefetch_insert(&m, g_keys[i + pf]);                  \
            NAME##_delete(&m, g_keys[i]);                                    \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%d\tdelete\t%.1f\n",                                     \
               STRIDE_LABEL, pf, elapsed_ns(t0, t1, NUM));                  \
                                                                             \
        NAME##_destroy(&m);                                                  \
                                                                             \
        /* Churn */                                                          \
        NAME##_init_cap(&m, NUM);                                            \
        for (int i = 0; i < NUM; i++)                                        \
            NAME##_insert_unique(&m, g_keys[i], &g_vals[i * VW]);           \
                                                                             \
        int extra_n = CHURN * ROUNDS;                                        \
        double churn_del = 0, churn_ins = 0;                                 \
        int del_cursor = NUM - 1;                                            \
                                                                             \
        for (int r = 0; r < ROUNDS; r++) {                                   \
            clock_gettime(CLOCK_MONOTONIC, &t0);                             \
            for (int i = 0; i < CHURN; i++) {                                \
                int idx = del_cursor - (r * CHURN + i);                      \
                if (idx >= 0) {                                              \
                    int pidx = del_cursor - (r * CHURN + i + pf);            \
                    if (pidx >= 0 && i + pf < CHURN)                         \
                        NAME##_prefetch_insert(&m, g_keys[pidx]);            \
                    NAME##_delete(&m, g_keys[idx]);                          \
                }                                                            \
            }                                                                \
            clock_gettime(CLOCK_MONOTONIC, &t1);                             \
            churn_del += (t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); \
                                                                             \
            clock_gettime(CLOCK_MONOTONIC, &t0);                             \
            for (int i = 0; i < CHURN; i++) {                                \
                int eidx = r * CHURN + i;                                    \
                if (eidx + pf < extra_n && i + pf < CHURN)                   \
                    NAME##_prefetch_insert(&m, g_exkeys[eidx + pf]);         \
                NAME##_insert(&m, g_exkeys[eidx], &g_exvals[eidx * VW]);    \
            }                                                                \
            clock_gettime(CLOCK_MONOTONIC, &t1);                             \
            churn_ins += (t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); \
        }                                                                    \
        int churn_ops = CHURN * ROUNDS;                                      \
        printf("%s\t%d\tchurn_del\t%.1f\n",                                  \
               STRIDE_LABEL, pf, churn_del / churn_ops);                     \
        printf("%s\t%d\tchurn_ins\t%.1f\n",                                  \
               STRIDE_LABEL, pf, churn_ins / churn_ops);                     \
                                                                             \
        /* Post-churn get hit */                                             \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        int post_hits = 0;                                                   \
        for (int i = 0; i < churn_ops; i++) {                                \
            if (i + pf < churn_ops)                                          \
                NAME##_prefetch(&m, g_exkeys[i + pf]);                       \
            post_hits += (NAME##_get(&m, g_exkeys[i]) != NULL);              \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%d\tpost_churn_hit\t%.1f\n",                             \
               STRIDE_LABEL, pf, elapsed_ns(t0, t1, churn_ops));            \
        (void)post_hits;                                                     \
                                                                             \
        NAME##_destroy(&m);                                                  \
        fflush(stdout);                                                      \
    }                                                                        \
} while (0)

int main(void) {
    gen_data();

    printf("# stride\tpf\toperation\tns_per_op\n");

    BENCH(kv64_n1, "N=1");
    BENCH(kv64_n2, "N=2");
    BENCH(kv64_n4, "N=4");

    free(g_keys); free(g_vals); free(g_miss);
    free(g_exkeys); free(g_exvals);
    return 0;
}
