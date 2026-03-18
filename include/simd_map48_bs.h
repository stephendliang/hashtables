/*
 * simd_map48_bs.h — direct-compare 48-bit set, 1CL/10 backshift delete
 *
 * 10 keys per 64B cache line. No overflow bits, no ctrl word.
 * Probe termination by empty slot. Delete via backshift.
 *
 * Group layout (64B):
 *   Offset 0-39:  uint32_t hi[10]  (stored as (key >> 16) + 1, so hi=0 means empty)
 *   Offset 40-59: uint16_t lo[10]  (key & 0xFFFF)
 *   Offset 60-63: 4 bytes padding
 *
 * Set mode only. Key: uint64_t (lower 48 bits, key=0 reserved,
 * keys where (key >> 16) == 0xFFFFFFFF excluded).
 * Backends: AVX2, scalar. Load factor: 75%.
 *
 *   #define SIMD_MAP_NAME my_set48bs
 *   #include "simd_map48_bs.h"
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_bs.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_BS_COMMON_H_
#define SIMD_MAP48_BS_COMMON_H_

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_2__)
#include <immintrin.h>
#endif
#if defined(__AVX2__)
#include <x86intrin.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "simd_compat.h"

#ifndef SMCAT_
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

#define SM48BS_SLOTS_ 10

/* Hash: single CRC32 (no overflow partition needed) */
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
#define sm48bs_hash(key) ((uint32_t)_mm_crc32_u64(0, (key)))
#else
static inline uint32_t sm48bs_hash(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)key;
}
#endif

/* Group access: hi at offset 0, lo at offset 40 */
static inline uint32_t *sm48bs_hi(void *grp) { return (uint32_t *)grp; }
static inline uint16_t *sm48bs_lo(void *grp) { return (uint16_t *)((char *)grp + 40); }
static inline const uint32_t *sm48bs_hi_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint16_t *sm48bs_lo_c(const void *grp) { return (const uint16_t *)((const char *)grp + 40); }

/* Stored hi = (key >> 16) + 1, so hi=0 means empty */
static inline uint64_t sm48bs_read_key(const void *grp, int slot) {
    uint32_t h = sm48bs_hi_c(grp)[slot];
    if (h == 0) return 0;
    return ((uint64_t)(h - 1) << 16) | sm48bs_lo_c(grp)[slot];
}

static inline void sm48bs_write_key(void *grp, int slot, uint64_t key) {
    sm48bs_hi(grp)[slot] = (uint32_t)(key >> 16) + 1;
    sm48bs_lo(grp)[slot] = (uint16_t)(key);
}

static inline void sm48bs_clear_slot(void *grp, int slot) {
    sm48bs_hi(grp)[slot] = 0;
    sm48bs_lo(grp)[slot] = 0;
}

/* --- SIMD match / empty --- */

#if defined(__AVX2__)

static inline uint16_t sm48bs_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48bs_hi_c(grp);
    const uint16_t *lop = sm48bs_lo_c(grp);

    /* Hi: vpcmpeqd on hi[0..7], scalar for hi[8..9] */
    __m256i h8 = _mm256_loadu_si256((const __m256i *)hip);
    __m256i kh = _mm256_set1_epi32((int)stored_hi);
    uint16_t m_hi = (uint16_t)(uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, kh)));
    if (hip[8] == stored_hi) m_hi |= (1u << 8);
    if (hip[9] == stored_hi) m_hi |= (1u << 9);

    /* Lo: vpcmpeqw on lo[0..7], scalar for lo[8..9] */
    __m128i l8 = _mm_loadu_si128((const __m128i *)lop);
    __m128i kl = _mm_set1_epi16((short)key_lo);
    uint32_t raw = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(l8, kl));
    uint16_t m_lo = (uint16_t)_pext_u32(raw, 0xAAAAu);
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    return m_hi & m_lo;
}

static inline uint16_t sm48bs_empty(const void *grp) {
    const uint32_t *hip = sm48bs_hi_c(grp);
    __m256i h8 = _mm256_loadu_si256((const __m256i *)hip);
    __m256i z = _mm256_setzero_si256();
    uint16_t em = (uint16_t)(uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, z)));
    if (hip[8] == 0) em |= (1u << 8);
    if (hip[9] == 0) em |= (1u << 9);
    return em;
}

#define sm48bs_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint16_t sm48bs_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48bs_hi_c(grp);
    const uint16_t *lop = sm48bs_lo_c(grp);

    uint32x4_t kh = vdupq_n_u32(stored_hi);
    uint16_t m_hi = (uint16_t)neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), kh));
    m_hi |= (uint16_t)(neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), kh)) << 4);
    if (hip[8] == stored_hi) m_hi |= (1u << 8);
    if (hip[9] == stored_hi) m_hi |= (1u << 9);

    uint16x8_t kl = vdupq_n_u16(key_lo);
    uint16_t m_lo = (uint16_t)neon_movemask_u16(vceqq_u16(vld1q_u16(lop), kl));
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    return m_hi & m_lo;
}

static inline uint16_t sm48bs_empty(const void *grp) {
    const uint32_t *hip = sm48bs_hi_c(grp);
    uint32x4_t z = vdupq_n_u32(0);
    uint16_t em = (uint16_t)neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), z));
    em |= (uint16_t)(neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), z)) << 4);
    if (hip[8] == 0) em |= (1u << 8);
    if (hip[9] == 0) em |= (1u << 9);
    return em;
}

#define sm48bs_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* scalar */

static inline uint16_t sm48bs_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48bs_hi_c(grp);
    const uint16_t *lop = sm48bs_lo_c(grp);
    uint16_t result = 0;
    for (int i = 0; i < SM48BS_SLOTS_; i++) {
        if (hip[i] == stored_hi && lop[i] == key_lo)
            result |= (uint16_t)(1u << i);
    }
    return result;
}

static inline uint16_t sm48bs_empty(const void *grp) {
    const uint32_t *hip = sm48bs_hi_c(grp);
    uint16_t em = 0;
    for (int i = 0; i < SM48BS_SLOTS_; i++) {
        if (hip[i] == 0)
            em |= (uint16_t)(1u << i);
    }
    return em;
}

#if defined(__SSE4_2__)
#define sm48bs_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48bs_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48_BS_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;     /* number of groups, always power of 2 */
    uint32_t mask;   /* ng - 1 */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + ((size_t)gi << 6);  /* gi * 64 bytes */
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = sm48bs_hash(key) & m->mask;
    sm48bs_prefetch_line(SM_(_group)(m, gi));
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48bs_hash(key) & m->mask;
    sm48bs_prefetch_line(SM_(_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
    size_t raw = (size_t)ng * 64;
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t ng) {
    size_t total = SM_(_mapsize)(ng);
    m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                           | MAP_POPULATE, -1, 0);
    if (m->data == MAP_FAILED) {
        m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS
                               | MAP_POPULATE, -1, 0);
        if (m->data != MAP_FAILED)
            madvise(m->data, total, MADV_HUGEPAGE);
    }
    m->ng    = ng;
    m->mask  = ng - 1;
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_ng   = m->ng;
    char    *old_data = m->data;

    SM_(_alloc)(m, old_ng * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
        const char *old_grp = old_data + ((size_t)g << 6);
        for (int s = 0; s < SM48BS_SLOTS_; s++) {
            uint64_t key = sm48bs_read_key(old_grp, s);
            if (!key) continue;

            uint32_t gi = sm48bs_hash(key) & mask;
            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t em = sm48bs_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    sm48bs_write_key(grp, pos, key);
                    m->count++;
                    break;
                }
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_ng));
}

/* --- Backshift delete --- */

static inline void SM_(_backshift_at)(struct SIMD_MAP_NAME *m,
                                       uint32_t gi, int slot) {
    uint32_t mask = m->mask;
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = (gi + 1) & mask;

    for (;;) {
        uint32_t pf_gi = (scan_gi + 2) & mask;
        sm48bs_prefetch_line(SM_(_group)(m, pf_gi));

        const char *scan_grp = SM_(_group)(m, scan_gi);
        uint16_t scan_empty = sm48bs_empty(scan_grp);

        if (scan_empty == 0x03FF) return;  /* all 10 empty */

        uint64_t cand_keys[SM48BS_SLOTS_];
        uint32_t cand_homes[SM48BS_SLOTS_];
        int cand_slots[SM48BS_SLOTS_];
        int n_cand = 0;

        for (uint16_t todo = (~scan_empty) & 0x03FF; todo; todo &= todo - 1) {
            int s = __builtin_ctz(todo);
            cand_keys[n_cand] = sm48bs_read_key(scan_grp, s);
            cand_slots[n_cand] = s;
            n_cand++;
        }

        for (int j = 0; j < n_cand; j++)
            cand_homes[j] = sm48bs_hash(cand_keys[j]) & mask;

        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            if (((hole_gi - cand_homes[j]) & mask) <
                ((scan_gi - cand_homes[j]) & mask)) {
                char *hole_grp = SM_(_group)(m, hole_gi);
                sm48bs_write_key(hole_grp, hole_slot, cand_keys[j]);
                sm48bs_clear_slot(SM_(_group)(m, scan_gi), cand_slots[j]);
                if (scan_empty) return;
                hole_gi = scan_gi;
                hole_slot = cand_slots[j];
                moved = 1;
                break;
            }
        }

        if (!moved && scan_empty) return;
        scan_gi = (scan_gi + 1) & mask;
    }
}

/* --- Public API --- */

static inline void SM_(_init)(struct SIMD_MAP_NAME *m) {
    memset(m, 0, sizeof(*m));
}

static inline void SM_(_init_cap)(struct SIMD_MAP_NAME *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    /* 75% load factor: ng*10*3/4 >= n → ng >= n*4/(10*3) = n*4/30 */
    uint64_t need = ((uint64_t)n * 4 + 29) / 30;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 4 >= (uint64_t)m->ng * SM48BS_SLOTS_ * 3) SM_(_grow)(m);

    uint32_t gi = sm48bs_hash(key) & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        if (sm48bs_match(grp, key)) return 0;

        uint16_t em = sm48bs_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            sm48bs_write_key(grp, pos, key);
            m->count++;
            return 1;
        }
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 4 >= (uint64_t)m->ng * SM48BS_SLOTS_ * 3) SM_(_grow)(m);

    uint32_t gi = sm48bs_hash(key) & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t em = sm48bs_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            sm48bs_write_key(grp, pos, key);
            m->count++;
            return;
        }
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    uint32_t gi = sm48bs_hash(key) & m->mask;

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48bs_match(grp, key)) return 1;
        if (sm48bs_empty(grp))      return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    uint32_t gi = sm48bs_hash(key) & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48bs_match(grp, key);
        uint16_t empty = sm48bs_empty(grp);
        if (mm) {
            int slot = __builtin_ctz(mm);
            sm48bs_clear_slot(grp, slot);
            m->count--;
            if (!empty) SM_(_backshift_at)(m, gi, slot);
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SIMD_MAP_NAME
