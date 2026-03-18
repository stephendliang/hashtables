/*
 * simd_map48_lembs.h — direct-compare 48-bit set/map, Lemire + backshift + ghost overflow
 *
 * Combines Lemire non-pow2 addressing, backshift deletion, and ghost
 * overflow bits for fast miss-path probe termination at 87.5% load factor.
 *
 * Group layout (64B key block):
 *   Offset 0-39:  uint32_t hi[10]  (stored as (key >> 16) + 1, hi=0 = empty)
 *   Offset 40-59: uint16_t lo[10]  (key & 0xFFFF)
 *   Offset 60-63: uint32_t ovf     (ghost overflow, 16 partitions in bits [15:0])
 *
 * Set mode when VAL_WORDS is omitted or 0, map mode when VAL_WORDS >= 1.
 * Map mode: values stored inline after keys (64B keys + 10*VW*8B values).
 *
 * Set mode:
 *   #define SIMD_MAP_NAME my_set48lb
 *   #include "simd_map48_lembs.h"
 *
 * Map mode:
 *   #define SIMD_MAP_NAME           my_map48lb
 *   #define SIMD_MAP48LB_VAL_WORDS  1
 *   #include "simd_map48_lembs.h"
 *
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_lembs.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_LEMBS_COMMON_H_
#define SIMD_MAP48_LEMBS_COMMON_H_

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

#define SM48LB_SLOTS_ 10

/* Hash: two-round CRC32 for group index + overflow partition */
struct sm48lb_h { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct sm48lb_h sm48lb_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48lb_h){a, b};
}
#else
static inline struct sm48lb_h sm48lb_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48lb_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Lemire range reduction */
static inline uint32_t sm48lb_group_idx(uint32_t hash, uint32_t ng) {
    return (uint32_t)(((uint64_t)hash * (uint64_t)ng) >> 32);
}

/* Group access: hi at offset 0, lo at offset 40, ovf at offset 60 */
static inline uint32_t *sm48lb_hi(void *grp) { return (uint32_t *)grp; }
static inline uint16_t *sm48lb_lo(void *grp) { return (uint16_t *)((char *)grp + 40); }
static inline uint32_t *sm48lb_ovf(void *grp) { return (uint32_t *)((char *)grp + 60); }
static inline const uint32_t *sm48lb_hi_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint16_t *sm48lb_lo_c(const void *grp) { return (const uint16_t *)((const char *)grp + 40); }
static inline const uint32_t *sm48lb_ovf_c(const void *grp) { return (const uint32_t *)((const char *)grp + 60); }

static inline uint64_t sm48lb_read_key(const void *grp, int slot) {
    uint32_t h = sm48lb_hi_c(grp)[slot];
    if (h == 0) return 0;
    return ((uint64_t)(h - 1) << 16) | sm48lb_lo_c(grp)[slot];
}

static inline void sm48lb_write_key(void *grp, int slot, uint64_t key) {
    sm48lb_hi(grp)[slot] = (uint32_t)(key >> 16) + 1;
    sm48lb_lo(grp)[slot] = (uint16_t)(key);
}

static inline void sm48lb_clear_slot(void *grp, int slot) {
    sm48lb_hi(grp)[slot] = 0;
    sm48lb_lo(grp)[slot] = 0;
}

/* --- SIMD match / empty --- */

#if defined(__AVX2__)

static inline uint16_t sm48lb_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48lb_hi_c(grp);
    const uint16_t *lop = sm48lb_lo_c(grp);

    __m256i h8 = _mm256_loadu_si256((const __m256i *)hip);
    __m256i kh = _mm256_set1_epi32((int)stored_hi);
    uint16_t m_hi = (uint16_t)(uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, kh)));
    if (hip[8] == stored_hi) m_hi |= (1u << 8);
    if (hip[9] == stored_hi) m_hi |= (1u << 9);

    __m128i l8 = _mm_loadu_si128((const __m128i *)lop);
    __m128i kl = _mm_set1_epi16((short)key_lo);
    uint32_t raw = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(l8, kl));
    uint16_t m_lo = (uint16_t)_pext_u32(raw, 0xAAAAu);
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    return m_hi & m_lo;
}

static inline uint16_t sm48lb_empty(const void *grp) {
    const uint32_t *hip = sm48lb_hi_c(grp);
    __m256i h8 = _mm256_loadu_si256((const __m256i *)hip);
    __m256i z = _mm256_setzero_si256();
    uint16_t em = (uint16_t)(uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, z)));
    if (hip[8] == 0) em |= (1u << 8);
    if (hip[9] == 0) em |= (1u << 9);
    return em;
}

#define sm48lb_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint16_t sm48lb_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48lb_hi_c(grp);
    const uint16_t *lop = sm48lb_lo_c(grp);

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

static inline uint16_t sm48lb_empty(const void *grp) {
    const uint32_t *hip = sm48lb_hi_c(grp);
    uint32x4_t z = vdupq_n_u32(0);
    uint16_t em = (uint16_t)neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), z));
    em |= (uint16_t)(neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), z)) << 4);
    if (hip[8] == 0) em |= (1u << 8);
    if (hip[9] == 0) em |= (1u << 9);
    return em;
}

#define sm48lb_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* scalar */

static inline uint16_t sm48lb_match(const void *grp, uint64_t key) {
    uint32_t stored_hi = (uint32_t)(key >> 16) + 1;
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48lb_hi_c(grp);
    const uint16_t *lop = sm48lb_lo_c(grp);
    uint16_t result = 0;
    for (int i = 0; i < SM48LB_SLOTS_; i++) {
        if (hip[i] == stored_hi && lop[i] == key_lo)
            result |= (uint16_t)(1u << i);
    }
    return result;
}

static inline uint16_t sm48lb_empty(const void *grp) {
    const uint32_t *hip = sm48lb_hi_c(grp);
    uint16_t em = 0;
    for (int i = 0; i < SM48LB_SLOTS_; i++) {
        if (hip[i] == 0)
            em |= (uint16_t)(1u << i);
    }
    return em;
}

#if defined(__SSE4_2__)
#define sm48lb_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48lb_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48_LEMBS_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#ifndef SIMD_MAP48LB_VAL_WORDS
#define SIMD_MAP48LB_VAL_WORDS 0
#endif

#define SM48LB_VW_ (SIMD_MAP48LB_VAL_WORDS)

#if SM48LB_VW_ > 0
#define SM48LB_VAL_SZ_   (SM48LB_VW_ * 8u)
#define SM48LB_VAL_GRP_  (SM48LB_SLOTS_ * SM48LB_VAL_SZ_)
#define SM48LB_ENTRY_SZ_ (64u + SM48LB_VAL_GRP_)
#else
#define SM48LB_ENTRY_SZ_ 64u
#endif

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;     /* exact number of groups (not pow2) */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + (size_t)gi * SM48LB_ENTRY_SZ_;
}

#if SM48LB_VW_ > 0
static inline uint64_t *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                      uint32_t gi, int slot) {
    return (uint64_t *)(SM_(_group)(m, gi) + 64 + (unsigned)slot * SM48LB_VAL_SZ_);
}
#endif

static inline uint32_t SM_(_gi)(const struct SIMD_MAP_NAME *m, uint32_t hash) {
    return sm48lb_group_idx(hash, m->ng);
}

static inline uint32_t SM_(_next)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    gi++;
    return gi < m->ng ? gi : 0;
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = SM_(_gi)(m, sm48lb_hash(key).lo);
    sm48lb_prefetch_line(SM_(_group)(m, gi));
#if SM48LB_VW_ > 0
    sm48lb_prefetch_line((const char *)SM_(_val_at)(m, gi, 0));
#endif
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = SM_(_gi)(m, sm48lb_hash(key).lo);
    sm48lb_prefetch_line(SM_(_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
    size_t raw = (size_t)ng * SM48LB_ENTRY_SZ_;
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
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_ng   = m->ng;
    char    *old_data = m->data;

    SM_(_alloc)(m, old_ng * 2);

    for (uint32_t g = 0; g < old_ng; g++) {
        const char *old_grp = old_data + (size_t)g * SM48LB_ENTRY_SZ_;
        for (int s = 0; s < SM48LB_SLOTS_; s++) {
            uint64_t key = sm48lb_read_key(old_grp, s);
            if (!key) continue;

            struct sm48lb_h h = sm48lb_hash(key);
            uint32_t gi = SM_(_gi)(m, h.lo);
            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t em = sm48lb_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    sm48lb_write_key(grp, pos, key);
#if SM48LB_VW_ > 0
                    memcpy(SM_(_val_at)(m, gi, pos),
                           old_grp + 64 + (unsigned)s * SM48LB_VAL_SZ_,
                           SM48LB_VAL_SZ_);
#endif
                    m->count++;
                    break;
                }
                *sm48lb_ovf(grp) |= (1u << (h.hi & 15));
                gi = SM_(_next)(m, gi);
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_ng));
}

/* --- Backshift delete --- */

static inline void SM_(_backshift_at)(struct SIMD_MAP_NAME *m,
                                       uint32_t gi, int slot) {
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = SM_(_next)(m, gi);

    for (;;) {
        uint32_t pf_gi = SM_(_next)(m, SM_(_next)(m, scan_gi));
        sm48lb_prefetch_line(SM_(_group)(m, pf_gi));

        const char *scan_grp = SM_(_group)(m, scan_gi);
        uint16_t scan_empty = sm48lb_empty(scan_grp);

        if (scan_empty == 0x03FF) return;

        uint64_t cand_keys[SM48LB_SLOTS_];
        uint32_t cand_homes[SM48LB_SLOTS_];
        int cand_slots[SM48LB_SLOTS_];
        int n_cand = 0;

        for (uint16_t todo = (~scan_empty) & 0x03FF; todo; todo &= todo - 1) {
            int s = __builtin_ctz(todo);
            cand_keys[n_cand] = sm48lb_read_key(scan_grp, s);
            cand_slots[n_cand] = s;
            n_cand++;
        }

        for (int j = 0; j < n_cand; j++)
            cand_homes[j] = SM_(_gi)(m, sm48lb_hash(cand_keys[j]).lo);

        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            uint32_t d_hole = (hole_gi - cand_homes[j] + m->ng) % m->ng;
            uint32_t d_scan = (scan_gi - cand_homes[j] + m->ng) % m->ng;
            if (d_hole < d_scan) {
                char *hole_grp = SM_(_group)(m, hole_gi);
                sm48lb_write_key(hole_grp, hole_slot, cand_keys[j]);
#if SM48LB_VW_ > 0
                memcpy(SM_(_val_at)(m, hole_gi, hole_slot),
                       SM_(_val_at)(m, scan_gi, cand_slots[j]),
                       SM48LB_VAL_SZ_);
#endif
                sm48lb_clear_slot(SM_(_group)(m, scan_gi), cand_slots[j]);
                if (scan_empty) return;
                hole_gi = scan_gi;
                hole_slot = cand_slots[j];
                moved = 1;
                break;
            }
        }

        if (!moved && scan_empty) return;
        scan_gi = SM_(_next)(m, scan_gi);
    }
}

/* --- Public API --- */

static inline void SM_(_init)(struct SIMD_MAP_NAME *m) {
    memset(m, 0, sizeof(*m));
}

static inline void SM_(_init_cap)(struct SIMD_MAP_NAME *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    uint32_t ng = ((uint64_t)n * 8 + 69) / 70;
    if (ng < 16) ng = 16;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

#if SM48LB_VW_ > 0
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key,
                                const uint64_t *val) {
#else
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48LB_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48lb_h h = sm48lb_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48lb_match(grp, key);
        if (mm) {
#if SM48LB_VW_ > 0
            int slot = __builtin_ctz(mm);
            memcpy(SM_(_val_at)(m, gi, slot), val, SM48LB_VAL_SZ_);
#endif
            return 0;
        }

        uint16_t em = sm48lb_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            sm48lb_write_key(grp, pos, key);
#if SM48LB_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48LB_VAL_SZ_);
#endif
            m->count++;
            return 1;
        }
        *sm48lb_ovf(grp) |= (1u << (h.hi & 15));
        gi = SM_(_next)(m, gi);
    }
}

#if SM48LB_VW_ > 0
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key,
                                        const uint64_t *val) {
#else
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48LB_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48lb_h h = sm48lb_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t em = sm48lb_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            sm48lb_write_key(grp, pos, key);
#if SM48LB_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48LB_VAL_SZ_);
#endif
            m->count++;
            return;
        }
        *sm48lb_ovf(grp) |= (1u << (h.hi & 15));
        gi = SM_(_next)(m, gi);
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48lb_h h = sm48lb_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48lb_match(grp, key)) return 1;
        if (!(*sm48lb_ovf_c(grp) & (1u << (h.hi & 15)))) return 0;
        gi = SM_(_next)(m, gi);
    }
}

#if SM48LB_VW_ > 0
static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return NULL;

    struct sm48lb_h h = sm48lb_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48lb_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            return SM_(_val_at)(m, gi, slot);
        }
        if (!(*sm48lb_ovf_c(grp) & (1u << (h.hi & 15)))) return NULL;
        gi = SM_(_next)(m, gi);
    }
}
#endif

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48lb_h h = sm48lb_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48lb_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            sm48lb_clear_slot(grp, slot);
            m->count--;
            if (!sm48lb_empty(grp)) SM_(_backshift_at)(m, gi, slot);
            return 1;
        }
        if (!(*sm48lb_ovf_c(grp) & (1u << (h.hi & 15)))) return 0;
        gi = SM_(_next)(m, gi);
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SM48LB_VW_
#undef SM48LB_ENTRY_SZ_
#ifdef SM48LB_VAL_SZ_
#undef SM48LB_VAL_SZ_
#undef SM48LB_VAL_GRP_
#endif
#undef SIMD_MAP_NAME
#undef SIMD_MAP48LB_VAL_WORDS
