#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Sentinel: strategies 1, 2, 3 (N=4) — KW=2, VW=1 */
#define SIMD_MAP_NAME      kv_s1_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     1
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      kv_s2_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     2
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      kv_s3_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 4
#include "simd_kv_sentinel.h"

/* Bitstealing: strategies 1, 2, 3 (N=4) — KW=2, VW=1 */
#define SIMD_MAP_NAME      kv_s1_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     1
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      kv_s2_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     2
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      kv_s3_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 4
#include "simd_kv_bitstealing.h"

#define N 2000000
#define KW 2
#define VW 1

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

#define TEST(NAME) do {                                                      \
    struct NAME m;                                                           \
    NAME##_init(&m);                                                         \
    int ok = 1;                                                              \
                                                                             \
    /* 1. Insert N KV pairs */                                               \
    for (int i = 0; i < N; i++) {                                            \
        if (NAME##_insert(&m, &keys[i * KW], &vals[i * VW]) != 1)           \
            { ok = 0; break; }                                               \
    }                                                                        \
    if (m.count != (uint32_t)N) ok = 0;                                      \
                                                                             \
    /* 2. Duplicate rejection (value unchanged) */                           \
    for (int i = 0; i < 1000 && ok; i++) {                                   \
        uint64_t alt_val[VW];                                                \
        for (int w = 0; w < VW; w++) alt_val[w] = 0xBADBADBADBADBAD0ULL;    \
        if (NAME##_insert(&m, &keys[i * KW], alt_val) != 0) ok = 0;         \
        /* Verify original value preserved */                                \
        uint64_t *v = NAME##_get(&m, &keys[i * KW]);                        \
        if (!v) { ok = 0; break; }                                           \
        for (int w = 0; w < VW && ok; w++)                                   \
            if (v[w] != vals[i * VW + w]) ok = 0;                            \
    }                                                                        \
                                                                             \
    /* 3. Get hit: verify values */                                          \
    for (int i = 0; i < N && ok; i++) {                                      \
        uint64_t *v = NAME##_get(&m, &keys[i * KW]);                        \
        if (!v) { ok = 0; break; }                                           \
        for (int w = 0; w < VW && ok; w++)                                   \
            if (v[w] != vals[i * VW + w]) ok = 0;                            \
    }                                                                        \
                                                                             \
    /* 4. Get miss */                                                        \
    for (int i = 0; i < N && ok; i++) {                                      \
        if (NAME##_get(&m, &miss[i * KW]) != NULL) ok = 0;                  \
    }                                                                        \
                                                                             \
    /* 5. Contains hit/miss */                                               \
    for (int i = 0; i < 1000 && ok; i++) {                                   \
        if (!NAME##_contains(&m, &keys[i * KW])) ok = 0;                    \
        if (NAME##_contains(&m, &miss[i * KW])) ok = 0;                     \
    }                                                                        \
                                                                             \
    /* 6. Delete first half, verify get returns NULL */                       \
    for (int i = 0; i < N / 2 && ok; i++) {                                  \
        if (NAME##_delete(&m, &keys[i * KW]) != 1) ok = 0;                  \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) ok = 0;                            \
    for (int i = 0; i < N / 2 && ok; i++) {                                  \
        if (NAME##_get(&m, &keys[i * KW]) != NULL) ok = 0;                  \
    }                                                                        \
    /* Remaining keys still have correct values */                           \
    for (int i = N / 2; i < N && ok; i++) {                                   \
        uint64_t *v = NAME##_get(&m, &keys[i * KW]);                        \
        if (!v) { ok = 0; break; }                                           \
        for (int w = 0; w < VW && ok; w++)                                   \
            if (v[w] != vals[i * VW + w]) ok = 0;                            \
    }                                                                        \
                                                                             \
    /* 7. Re-insert deleted keys with DIFFERENT values */                    \
    for (int i = 0; i < N / 2 && ok; i++) {                                  \
        if (NAME##_insert(&m, &keys[i * KW], &vals2[i * VW]) != 1) ok = 0;  \
    }                                                                        \
    if (m.count != (uint32_t)N) ok = 0;                                      \
    /* Verify new values */                                                  \
    for (int i = 0; i < N / 2 && ok; i++) {                                  \
        uint64_t *v = NAME##_get(&m, &keys[i * KW]);                        \
        if (!v) { ok = 0; break; }                                           \
        for (int w = 0; w < VW && ok; w++)                                   \
            if (v[w] != vals2[i * VW + w]) ok = 0;                           \
    }                                                                        \
                                                                             \
    /* 8. Delete all */                                                      \
    for (int i = 0; i < N && ok; i++) {                                       \
        if (NAME##_delete(&m, &keys[i * KW]) != 1) ok = 0;                  \
    }                                                                        \
    if (m.count != 0) ok = 0;                                                \
                                                                             \
    /* 9. init_cap + insert_unique */                                        \
    NAME##_destroy(&m);                                                      \
    struct NAME m2;                                                          \
    NAME##_init_cap(&m2, N);                                                 \
    uint32_t cap_before = m2.cap;                                            \
    for (int i = 0; i < N; i++)                                               \
        NAME##_insert_unique(&m2, &keys[i * KW], &vals[i * VW]);            \
    if (m2.cap != cap_before || m2.count != (uint32_t)N) ok = 0;            \
    /* Verify values after insert_unique */                                  \
    for (int i = 0; i < 1000 && ok; i++) {                                   \
        uint64_t *v = NAME##_get(&m2, &keys[i * KW]);                       \
        if (!v) { ok = 0; break; }                                           \
        for (int w = 0; w < VW && ok; w++)                                   \
            if (v[w] != vals[i * VW + w]) ok = 0;                            \
    }                                                                        \
    NAME##_destroy(&m2);                                                     \
                                                                             \
    printf("  %-16s %s\n", #NAME ":", ok ? "PASS" : "FAIL");                \
    all_ok = all_ok && ok;                                                   \
} while (0)

int main(void) {
    uint64_t *keys = malloc(N * KW * sizeof(uint64_t));
    uint64_t *vals = malloc(N * VW * sizeof(uint64_t));
    uint64_t *vals2 = malloc(N * VW * sizeof(uint64_t));
    uint64_t *miss = malloc(N * KW * sizeof(uint64_t));

    uint64_t kseed = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < N * KW; i++) keys[i] = splitmix64(&kseed);

    uint64_t vseed = 0x1111222233334444ULL;
    for (int i = 0; i < N * VW; i++) vals[i] = splitmix64(&vseed);

    uint64_t v2seed = 0x5555666677778888ULL;
    for (int i = 0; i < N * VW; i++) vals2[i] = splitmix64(&v2seed);

    uint64_t mseed = 0xAAAABBBBCCCCDDDDULL;
    for (int i = 0; i < N * KW; i++) miss[i] = splitmix64(&mseed);

    int all_ok = 1;
    printf("KV map correctness (N=%d, KW=%d, VW=%d):\n", N, KW, VW);

    TEST(kv_s1_sent);
    TEST(kv_s2_sent);
    TEST(kv_s3_sent);
    TEST(kv_s1_bs);
    TEST(kv_s2_bs);
    TEST(kv_s3_bs);

    free(keys); free(vals); free(vals2); free(miss);
    return all_ok ? 0 : 1;
}
