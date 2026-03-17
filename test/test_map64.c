/*
 * test_kv64.c — Correctness tests for simd_map64.h
 *
 * Tests all block strides (N=1, 2, 4, 8) with VW=1.
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o test_kv64 test_kv64.c
 *   cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_kv64_scalar test_kv64.c
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

#define SIMD_MAP_NAME          kv64_n8
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 8
#include "simd_map64.h"

/* VW=2 test */
#define SIMD_MAP_NAME          kv64_v2
#define SIMD_MAP64_VAL_WORDS    2
#define SIMD_MAP64_BLOCK_STRIDE 1
#include "simd_map64.h"

#include <stdio.h>

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

#define TEST(NAME, VW, LABEL) do {                                           \
    uint64_t *keys = malloc((size_t)N * sizeof(uint64_t));                   \
    uint64_t *vals = malloc((size_t)N * VW * sizeof(uint64_t));              \
    uint64_t s = 0xdeadbeefcafe1234ULL;                                      \
    for (int i = 0; i < N; i++) {                                            \
        uint64_t k;                                                          \
        do { k = splitmix64(&s); } while (k == 0);                           \
        keys[i] = k;                                                         \
        for (int v = 0; v < VW; v++)                                         \
            vals[i * VW + v] = splitmix64(&s);                               \
    }                                                                        \
                                                                             \
    struct NAME m;                                                           \
    NAME##_init(&m);                                                         \
    int err = 0;                                                             \
                                                                             \
    /* insert all */                                                         \
    for (int i = 0; i < N; i++) {                                            \
        int r = NAME##_insert(&m, keys[i], &vals[i * VW]);                   \
        if (r != 1) { err++; break; }                                        \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* dup insert */                                                         \
    for (int i = 0; i < 1000; i++) {                                         \
        if (NAME##_insert(&m, keys[i], &vals[i * VW]) != 0) { err++; break;}\
    }                                                                        \
                                                                             \
    /* get hit + value check */                                              \
    for (int i = 0; i < N; i++) {                                            \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) { err++; break; }                                            \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) { err++; break; }                  \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* contains hit */                                                       \
    for (int i = 0; i < N; i++) {                                            \
        if (!NAME##_contains(&m, keys[i])) { err++; break; }                 \
    }                                                                        \
                                                                             \
    /* get miss */                                                           \
    uint64_t s2 = 0xFEDCBA9876543210ULL;                                     \
    for (int i = 0; i < 10000; i++) {                                        \
        uint64_t mk;                                                         \
        do { mk = splitmix64(&s2); } while (mk == 0);                        \
        if (NAME##_get(&m, mk) != NULL) { /* may hit, ok */ }                \
    }                                                                        \
                                                                             \
    /* delete half */                                                        \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_delete(&m, keys[i]) != 1) { err++; break; }               \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) err++;                             \
                                                                             \
    /* verify deleted are gone, remaining have correct values */              \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_get(&m, keys[i]) != NULL) { err++; break; }               \
    }                                                                        \
    for (int i = N / 2; i < N; i++) {                                        \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) { err++; break; }                                            \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) { err++; break; }                  \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* re-insert deleted keys with new values */                             \
    for (int i = 0; i < N / 2; i++) {                                        \
        uint64_t nv[VW];                                                     \
        for (int j = 0; j < VW; j++) nv[j] = keys[i] + (uint64_t)j + 1;     \
        if (NAME##_insert(&m, keys[i], nv) != 1) { err++; break; }           \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* verify re-inserted values */                                          \
    for (int i = 0; i < N / 2; i++) {                                        \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) { err++; break; }                                            \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != keys[i] + (uint64_t)j + 1) { err++; break; }         \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* init_cap + insert_unique */                                           \
    NAME##_destroy(&m);                                                      \
    NAME##_init_cap(&m, N);                                                  \
    for (int i = 0; i < N; i++)                                              \
        NAME##_insert_unique(&m, keys[i], &vals[i * VW]);                    \
    if (m.count != (uint32_t)N) err++;                                       \
    for (int i = 0; i < 1000; i++) {                                         \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) { err++; break; }                                            \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) { err++; break; }                  \
        }                                                                    \
    }                                                                        \
                                                                             \
    NAME##_destroy(&m);                                                      \
    free(keys); free(vals);                                                  \
    printf("  %-16s %s\n", LABEL, err ? "FAIL" : "PASS");                    \
    if (err) fail = 1;                                                       \
} while (0)

int main(void) {
    int fail = 0;
    printf("KV64 correctness (N=%d):\n", N);

    TEST(kv64_n1, 1, "N=1 (inline):");
    TEST(kv64_n2, 1, "N=2:");
    TEST(kv64_n4, 1, "N=4:");
    TEST(kv64_n8, 1, "N=8:");
    TEST(kv64_v2, 2, "N=1, VW=2:");

    return fail;
}
