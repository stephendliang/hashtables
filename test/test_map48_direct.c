/*
 * test_map48_direct_kv.c — Map-mode correctness for simd_map48_packed.h / simd_map48_split.h
 *
 * Tests: split/packed × VW=1/2 × N=1/2 (+ N=4 for split VW=1)
 * Each variant: insert with values → dup update → get+verify → delete half →
 * verify deleted gone → verify remaining values → re-insert with new values → verify.
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o test_map48_direct_kv test_map48_direct_kv.c
 *   cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_map48_direct_kv_scalar test_map48_direct_kv.c
 */

/* --- Split VW=1 N=1 (inline) --- */
#define SIMD_MAP_NAME           sv1n1
#define SIMD_MAP48S_VAL_WORDS   1
#include "simd_map48_split.h"

/* --- Split VW=1 N=2 (superblock) --- */
#define SIMD_MAP_NAME           sv1n2
#define SIMD_MAP48S_VAL_WORDS   1
#define SIMD_MAP48S_BLOCK_STRIDE 2
#include "simd_map48_split.h"

/* --- Split VW=1 N=4 (superblock) --- */
#define SIMD_MAP_NAME           sv1n4
#define SIMD_MAP48S_VAL_WORDS   1
#define SIMD_MAP48S_BLOCK_STRIDE 4
#include "simd_map48_split.h"

/* --- Split VW=2 N=1 (inline) --- */
#define SIMD_MAP_NAME           sv2n1
#define SIMD_MAP48S_VAL_WORDS   2
#include "simd_map48_split.h"

/* --- Split VW=2 N=2 (superblock) --- */
#define SIMD_MAP_NAME           sv2n2
#define SIMD_MAP48S_VAL_WORDS   2
#define SIMD_MAP48S_BLOCK_STRIDE 2
#include "simd_map48_split.h"

/* --- Packed VW=1 N=1 (inline) --- */
#define SIMD_MAP_NAME           pv1n1
#define SIMD_MAP48P_VAL_WORDS   1
#include "simd_map48_packed.h"

/* --- Packed VW=1 N=2 (superblock) --- */
#define SIMD_MAP_NAME           pv1n2
#define SIMD_MAP48P_VAL_WORDS   1
#define SIMD_MAP48P_BLOCK_STRIDE 2
#include "simd_map48_packed.h"

/* --- Packed VW=2 N=1 (inline) --- */
#define SIMD_MAP_NAME           pv2n1
#define SIMD_MAP48P_VAL_WORDS   2
#include "simd_map48_packed.h"

/* --- Packed VW=2 N=2 (superblock) --- */
#define SIMD_MAP_NAME           pv2n2
#define SIMD_MAP48P_VAL_WORDS   2
#define SIMD_MAP48P_BLOCK_STRIDE 2
#include "simd_map48_packed.h"

#include <stdio.h>

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t make48(uint64_t *state) {
    uint64_t k;
    do { k = splitmix64(state) & 0x0000FFFFFFFFFFFFULL; } while (k == 0);
    return k;
}

#define TEST_MAP(NAME, VW, LABEL) do {                                       \
    uint64_t *keys = malloc((size_t)N * sizeof(uint64_t));                   \
    uint64_t *vals = malloc((size_t)N * VW * sizeof(uint64_t));              \
    uint64_t s = 0xdeadbeefcafe1234ULL;                                      \
    for (int i = 0; i < N; i++) {                                            \
        keys[i] = make48(&s);                                                \
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
        if (r != 1) {                                                        \
            printf("    insert fail at %d\n", i); err++; break;              \
        }                                                                    \
    }                                                                        \
    if (m.count != (uint32_t)N) {                                            \
        printf("    count=%u expected=%d\n", m.count, N); err++;             \
    }                                                                        \
                                                                             \
    /* dup insert (updates value) */                                         \
    for (int i = 0; i < 1000; i++) {                                         \
        if (NAME##_insert(&m, keys[i], &vals[i * VW]) != 0) {               \
            printf("    dup insert fail at %d\n", i); err++; break;          \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* get hit + value check */                                              \
    for (int i = 0; i < N; i++) {                                            \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) {                                                            \
            printf("    get miss at %d key=%lx\n", i,                        \
                   (unsigned long)keys[i]);                                   \
            err++; break;                                                    \
        }                                                                    \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) {                                  \
                printf("    val mismatch at %d word %d\n", i, j);            \
                err++; break;                                                \
            }                                                                \
        }                                                                    \
        if (err) break;                                                      \
    }                                                                        \
                                                                             \
    /* contains hit */                                                       \
    for (int i = 0; i < N; i++) {                                            \
        if (!NAME##_contains(&m, keys[i])) {                                \
            printf("    contains miss at %d\n", i); err++; break;            \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* delete half */                                                        \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_delete(&m, keys[i]) != 1) {                              \
            printf("    delete fail at %d\n", i); err++; break;              \
        }                                                                    \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) {                                  \
        printf("    after del count=%u expected=%d\n",                       \
               m.count, N - N / 2); err++;                                   \
    }                                                                        \
                                                                             \
    /* verify deleted are gone */                                            \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_get(&m, keys[i]) != NULL) {                              \
            printf("    deleted still present at %d\n", i);                  \
            err++; break;                                                    \
        }                                                                    \
    }                                                                        \
    /* verify remaining have correct values */                               \
    for (int i = N / 2; i < N; i++) {                                        \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) {                                                            \
            printf("    remaining missing at %d\n", i); err++; break;        \
        }                                                                    \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) {                                  \
                printf("    remaining val mismatch at %d word %d\n", i, j);  \
                err++; break;                                                \
            }                                                                \
        }                                                                    \
        if (err) break;                                                      \
    }                                                                        \
                                                                             \
    /* re-insert deleted keys with new values */                             \
    for (int i = 0; i < N / 2; i++) {                                        \
        uint64_t nv[VW];                                                     \
        for (int j = 0; j < VW; j++) nv[j] = keys[i] + (uint64_t)j + 1;    \
        if (NAME##_insert(&m, keys[i], nv) != 1) {                          \
            printf("    re-insert fail at %d\n", i); err++; break;           \
        }                                                                    \
    }                                                                        \
    if (m.count != (uint32_t)N) {                                            \
        printf("    after re-insert count=%u expected=%d\n",                 \
               m.count, N); err++;                                           \
    }                                                                        \
                                                                             \
    /* verify re-inserted values */                                          \
    for (int i = 0; i < N / 2; i++) {                                        \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) {                                                            \
            printf("    re-insert get miss at %d\n", i); err++; break;       \
        }                                                                    \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != keys[i] + (uint64_t)j + 1) {                        \
                printf("    re-insert val mismatch at %d word %d\n", i, j);  \
                err++; break;                                                \
            }                                                                \
        }                                                                    \
        if (err) break;                                                      \
    }                                                                        \
                                                                             \
    /* init_cap + insert_unique */                                           \
    NAME##_destroy(&m);                                                      \
    NAME##_init_cap(&m, N);                                                  \
    for (int i = 0; i < N; i++)                                              \
        NAME##_insert_unique(&m, keys[i], &vals[i * VW]);                    \
    if (m.count != (uint32_t)N) {                                            \
        printf("    init_cap count=%u expected=%d\n", m.count, N); err++;    \
    }                                                                        \
    for (int i = 0; i < 1000; i++) {                                         \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) {                                                            \
            printf("    init_cap get miss at %d\n", i); err++; break;        \
        }                                                                    \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != vals[i * VW + j]) {                                  \
                printf("    init_cap val mismatch at %d\n", i);              \
                err++; break;                                                \
            }                                                                \
        }                                                                    \
        if (err) break;                                                      \
    }                                                                        \
                                                                             \
    NAME##_destroy(&m);                                                      \
    free(keys); free(vals);                                                  \
    printf("  %-28s %s\n", LABEL, err ? "FAIL" : "PASS");                   \
    if (err) fail = 1;                                                       \
} while (0)

int main(void) {
    int fail = 0;
    printf("Map48 direct-compare KV correctness (N=%d):\n", N);

    /* Split variants */
    TEST_MAP(sv1n1, 1, "split VW=1 N=1 (inline):");
    TEST_MAP(sv1n2, 1, "split VW=1 N=2 (super):");
    TEST_MAP(sv1n4, 1, "split VW=1 N=4 (super):");
    TEST_MAP(sv2n1, 2, "split VW=2 N=1 (inline):");
    TEST_MAP(sv2n2, 2, "split VW=2 N=2 (super):");

    /* Packed variants */
    TEST_MAP(pv1n1, 1, "packed VW=1 N=1 (inline):");
    TEST_MAP(pv1n2, 1, "packed VW=1 N=2 (super):");
    TEST_MAP(pv2n1, 2, "packed VW=2 N=1 (inline):");
    TEST_MAP(pv2n2, 2, "packed VW=2 N=2 (super):");

    return fail;
}
