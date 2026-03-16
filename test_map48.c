/*
 * test_map48.c — Correctness tests for simd_map48.h
 *
 * Tests set mode and map mode (VW=1, VW=2).
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o test_map48 test_map48.c
 *   cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_map48_scalar test_map48.c
 */

/* Set mode */
#define SIMD_MAP_NAME set48
#include "simd_map48.h"

/* Map mode VW=1 */
#define SIMD_MAP_NAME map48_v1
#define SIMD_MAP48_VAL_WORDS 1
#include "simd_map48.h"

/* Map mode VW=2 */
#define SIMD_MAP_NAME map48_v2
#define SIMD_MAP48_VAL_WORDS 2
#include "simd_map48.h"

#include <stdio.h>

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Generate a 48-bit key (nonzero, upper 16 bits = 0) */
static inline uint64_t make48(uint64_t *state) {
    uint64_t k;
    do { k = splitmix64(state) & 0x0000FFFFFFFFFFFFULL; } while (k == 0);
    return k;
}

/* Set test */
#define TEST_SET(NAME, LABEL) do {                                           \
    uint64_t *keys = malloc((size_t)N * sizeof(uint64_t));                   \
    uint64_t s = 0xdeadbeefcafe1234ULL;                                      \
    for (int i = 0; i < N; i++) keys[i] = make48(&s);                       \
                                                                             \
    struct NAME m;                                                           \
    NAME##_init(&m);                                                         \
    int err = 0;                                                             \
                                                                             \
    /* insert all */                                                         \
    for (int i = 0; i < N; i++) {                                            \
        int r = NAME##_insert(&m, keys[i]);                                  \
        if (r != 1) { err++; break; }                                        \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* dup insert */                                                         \
    for (int i = 0; i < 1000; i++) {                                         \
        if (NAME##_insert(&m, keys[i]) != 0) { err++; break; }              \
    }                                                                        \
                                                                             \
    /* contains hit */                                                       \
    for (int i = 0; i < N; i++) {                                            \
        if (!NAME##_contains(&m, keys[i])) { err++; break; }                \
    }                                                                        \
                                                                             \
    /* contains miss */                                                      \
    uint64_t s2 = 0xFEDCBA9876543210ULL;                                     \
    for (int i = 0; i < 10000; i++) {                                        \
        uint64_t mk = make48(&s2);                                           \
        (void)NAME##_contains(&m, mk);                                       \
    }                                                                        \
                                                                             \
    /* delete half */                                                        \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_delete(&m, keys[i]) != 1) { err++; break; }              \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) err++;                             \
                                                                             \
    /* verify deleted are gone */                                            \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_contains(&m, keys[i])) { err++; break; }                 \
    }                                                                        \
    /* verify remaining are present */                                       \
    for (int i = N / 2; i < N; i++) {                                        \
        if (!NAME##_contains(&m, keys[i])) { err++; break; }                \
    }                                                                        \
                                                                             \
    /* re-insert deleted */                                                  \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_insert(&m, keys[i]) != 1) { err++; break; }              \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* init_cap + insert_unique */                                           \
    NAME##_destroy(&m);                                                      \
    NAME##_init_cap(&m, N);                                                  \
    for (int i = 0; i < N; i++) NAME##_insert_unique(&m, keys[i]);           \
    if (m.count != (uint32_t)N) err++;                                       \
    for (int i = 0; i < 1000; i++) {                                         \
        if (!NAME##_contains(&m, keys[i])) { err++; break; }                \
    }                                                                        \
                                                                             \
    NAME##_destroy(&m);                                                      \
    free(keys);                                                              \
    printf("  %-16s %s\n", LABEL, err ? "FAIL" : "PASS");                   \
    if (err) fail = 1;                                                       \
} while (0)

/* Map test */
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
        if (r != 1) { err++; break; }                                        \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* dup insert */                                                         \
    for (int i = 0; i < 1000; i++) {                                         \
        if (NAME##_insert(&m, keys[i], &vals[i * VW]) != 0) {err++; break;} \
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
        if (!NAME##_contains(&m, keys[i])) { err++; break; }                \
    }                                                                        \
                                                                             \
    /* delete half */                                                        \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_delete(&m, keys[i]) != 1) { err++; break; }              \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) err++;                             \
                                                                             \
    /* verify deleted are gone, remaining have correct values */              \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_get(&m, keys[i]) != NULL) { err++; break; }              \
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
        for (int j = 0; j < VW; j++) nv[j] = keys[i] + (uint64_t)j + 1;    \
        if (NAME##_insert(&m, keys[i], nv) != 1) { err++; break; }          \
    }                                                                        \
    if (m.count != (uint32_t)N) err++;                                       \
                                                                             \
    /* verify re-inserted values */                                          \
    for (int i = 0; i < N / 2; i++) {                                        \
        uint64_t *v = NAME##_get(&m, keys[i]);                               \
        if (!v) { err++; break; }                                            \
        for (int j = 0; j < VW; j++) {                                       \
            if (v[j] != keys[i] + (uint64_t)j + 1) { err++; break; }        \
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
    printf("  %-16s %s\n", LABEL, err ? "FAIL" : "PASS");                   \
    if (err) fail = 1;                                                       \
} while (0)

int main(void) {
    int fail = 0;
    printf("Map48 correctness (N=%d):\n", N);

    TEST_SET(set48,    "set48:");
    TEST_MAP(map48_v1, 1, "map48 VW=1:");
    TEST_MAP(map48_v2, 2, "map48 VW=2:");

    return fail;
}
