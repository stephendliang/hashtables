/*
 * simd_map48_lemire.h — direct-compare 48-bit set, Lemire non-pow2 groups
 *
 * Same group layout as simd_map48_split.h (1CL/10, sentinel overflow).
 * Key change: group index uses Lemire's fast range reduction instead of
 * mask, allowing exact (non-pow2) group count for tighter memory.
 *
 *   gi = (uint32_t)(((uint64_t)hash * (uint64_t)ng) >> 32)
 *
 * Set mode only. Key: uint64_t (lower 48 bits, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 *   #define SIMD_MAP_NAME my_set48lem
 *   #include "simd_map48_lemire.h"
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_lemire.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_LEM_COMMON_H_
#define SIMD_MAP48_LEM_COMMON_H_

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

#define SM48LEM_SLOTS_    10
#define SM48LEM_OCC_MASK_ 0x03FFu
#define SM48LEM_OVF_SHIFT_ 10

/* Reuse split's hash, match, empty primitives */
#ifndef SIMD_MAP48S_COMMON_H_

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48lem_h { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct sm48lem_h sm48lem_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48lem_h){a, b};
}
#else
static inline struct sm48lem_h sm48lem_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48lem_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Group access helpers — same layout as split */
static inline uint32_t *sm48lem_ctrl(void *grp) { return (uint32_t *)grp; }
static inline uint32_t *sm48lem_hi(void *grp)   { return (uint32_t *)((char *)grp + 4); }
static inline uint16_t *sm48lem_lo(void *grp)   { return (uint16_t *)((char *)grp + 44); }

static inline const uint32_t *sm48lem_ctrl_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint32_t *sm48lem_hi_c(const void *grp)   { return (const uint32_t *)((const char *)grp + 4); }
static inline const uint16_t *sm48lem_lo_c(const void *grp)   { return (const uint16_t *)((const char *)grp + 44); }

static inline uint64_t sm48lem_read_key(const void *grp, int slot) {
    return ((uint64_t)sm48lem_hi_c(grp)[slot] << 16) | sm48lem_lo_c(grp)[slot];
}

static inline void sm48lem_write_key(void *grp, int slot, uint64_t key) {
    sm48lem_hi(grp)[slot] = (uint32_t)(key >> 16);
    sm48lem_lo(grp)[slot] = (uint16_t)(key);
}

/* --- SIMD match --- */

#if defined(__AVX2__)

static inline uint16_t sm48lem_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;

    __m256i h8 = _mm256_loadu_si256((const __m256i *)sm48lem_hi_c(grp));
    __m256i kh = _mm256_set1_epi32((int)key_hi);
    uint32_t m_hi8 = (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, kh)));
    uint16_t m_hi = (uint16_t)m_hi8;
    const uint32_t *hip = sm48lem_hi_c(grp);
    if (hip[8] == key_hi) m_hi |= (1u << 8);
    if (hip[9] == key_hi) m_hi |= (1u << 9);

    __m128i l8 = _mm_loadu_si128((const __m128i *)sm48lem_lo_c(grp));
    __m128i kl = _mm_set1_epi16((short)key_lo);
    uint32_t raw = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(l8, kl));
    uint16_t m_lo = (uint16_t)_pext_u32(raw, 0xAAAAu);
    const uint16_t *lop = sm48lem_lo_c(grp);
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    return m_hi & m_lo & (uint16_t)(ctrl & SM48LEM_OCC_MASK_);
}

static inline uint16_t sm48lem_empty(const void *grp) {
    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    return (~ctrl) & SM48LEM_OCC_MASK_;
}

#define sm48lem_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint16_t sm48lem_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48lem_hi_c(grp);
    const uint16_t *lop = sm48lem_lo_c(grp);

    uint32x4_t kh = vdupq_n_u32(key_hi);
    uint16_t m_hi = (uint16_t)neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), kh));
    m_hi |= (uint16_t)(neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), kh)) << 4);
    if (hip[8] == key_hi) m_hi |= (1u << 8);
    if (hip[9] == key_hi) m_hi |= (1u << 9);

    uint16x8_t kl = vdupq_n_u16(key_lo);
    uint16_t m_lo = (uint16_t)neon_movemask_u16(vceqq_u16(vld1q_u16(lop), kl));
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    return m_hi & m_lo & (uint16_t)(ctrl & SM48LEM_OCC_MASK_);
}

static inline uint16_t sm48lem_empty(const void *grp) {
    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    return (~ctrl) & SM48LEM_OCC_MASK_;
}

#define sm48lem_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* scalar */

static inline uint16_t sm48lem_match(const void *grp, uint64_t key) {
    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    uint16_t occ = (uint16_t)(ctrl & SM48LEM_OCC_MASK_);
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48lem_hi_c(grp);
    const uint16_t *lop = sm48lem_lo_c(grp);
    uint16_t result = 0;
    for (int i = 0; i < SM48LEM_SLOTS_; i++) {
        if ((occ >> i) & 1) {
            if (hip[i] == key_hi && lop[i] == key_lo)
                result |= (uint16_t)(1u << i);
        }
    }
    return result;
}

static inline uint16_t sm48lem_empty(const void *grp) {
    uint32_t ctrl = *sm48lem_ctrl_c(grp);
    return (~ctrl) & SM48LEM_OCC_MASK_;
}

#if defined(__SSE4_2__)
#define sm48lem_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48lem_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#else /* SIMD_MAP48S_COMMON_H_ already defined — reuse split primitives */

/* Alias split's primitives */
#define sm48lem_hash      sm48s_hash
#define sm48lem_ctrl      sm48s_ctrl
#define sm48lem_hi        sm48s_hi
#define sm48lem_lo        sm48s_lo
#define sm48lem_ctrl_c    sm48s_ctrl_c
#define sm48lem_hi_c      sm48s_hi_c
#define sm48lem_lo_c      sm48s_lo_c
#define sm48lem_read_key  sm48s_read_key
#define sm48lem_write_key sm48s_write_key
#define sm48lem_match     sm48s_match
#define sm48lem_empty     sm48s_empty
#define sm48lem_prefetch_line sm48s_prefetch_line

struct sm48lem_h { uint32_t lo, hi; };
static inline struct sm48lem_h sm48lem_hash_wrap(uint64_t key) {
    struct sm48s_h h = sm48s_hash(key);
    return *(struct sm48lem_h *)&h;
}
#undef sm48lem_hash
#define sm48lem_hash sm48lem_hash_wrap

#endif /* !SIMD_MAP48S_COMMON_H_ */

/* Lemire range reduction */
static inline uint32_t sm48lem_group_idx(uint32_t hash, uint32_t ng) {
    return (uint32_t)(((uint64_t)hash * (uint64_t)ng) >> 32);
}

#endif /* SIMD_MAP48_LEM_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;     /* exact number of groups (not necessarily pow2) */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + ((size_t)gi << 6);
}

static inline uint32_t SM_(_gi)(const struct SIMD_MAP_NAME *m, uint32_t hash) {
    return sm48lem_group_idx(hash, m->ng);
}

/* Linear probe wraps at ng (not mask) */
static inline uint32_t SM_(_next)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    gi++;
    return gi < m->ng ? gi : 0;
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = SM_(_gi)(m, sm48lem_hash(key).lo);
    sm48lem_prefetch_line(SM_(_group)(m, gi));
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = SM_(_gi)(m, sm48lem_hash(key).lo);
    sm48lem_prefetch_line(SM_(_group)(m, gi));
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
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_ng   = m->ng;
    char    *old_data = m->data;

    SM_(_alloc)(m, old_ng * 2);

    for (uint32_t g = 0; g < old_ng; g++) {
        const char *old_grp = old_data + ((size_t)g << 6);
        uint32_t ctrl = *(const uint32_t *)old_grp;
        uint16_t occ = (uint16_t)(ctrl & SM48LEM_OCC_MASK_);

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48lem_read_key(old_grp, s);
            struct sm48lem_h h = sm48lem_hash(key);
            uint32_t gi = SM_(_gi)(m, h.lo);

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t em = sm48lem_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    uint32_t *cp = sm48lem_ctrl(grp);
                    *cp |= (1u << pos);
                    sm48lem_write_key(grp, pos, key);
                    m->count++;
                    break;
                }
                uint32_t *cp = sm48lem_ctrl(grp);
                *cp |= (1u << (SM48LEM_OVF_SHIFT_ + (h.hi & 15)));
                gi = SM_(_next)(m, gi);
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_ng));
}

/* --- Public API --- */

static inline void SM_(_init)(struct SIMD_MAP_NAME *m) {
    memset(m, 0, sizeof(*m));
}

static inline void SM_(_init_cap)(struct SIMD_MAP_NAME *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    /* Load factor 7/8: ng*10*7/8 >= n → ng >= n*8/70 (exact, no pow2 rounding) */
    uint32_t ng = ((uint64_t)n * 8 + 69) / 70;
    if (ng < 16) ng = 16;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48LEM_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48lem_h h = sm48lem_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        if (sm48lem_match(grp, key)) return 0;

        uint16_t em = sm48lem_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48lem_ctrl(grp);
            *cp |= (1u << pos);
            sm48lem_write_key(grp, pos, key);
            m->count++;
            return 1;
        }
        uint32_t *cp = sm48lem_ctrl(grp);
        *cp |= (1u << (SM48LEM_OVF_SHIFT_ + (h.hi & 15)));
        gi = SM_(_next)(m, gi);
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48LEM_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48lem_h h = sm48lem_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t em = sm48lem_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48lem_ctrl(grp);
            *cp |= (1u << pos);
            sm48lem_write_key(grp, pos, key);
            m->count++;
            return;
        }
        uint32_t *cp = sm48lem_ctrl(grp);
        *cp |= (1u << (SM48LEM_OVF_SHIFT_ + (h.hi & 15)));
        gi = SM_(_next)(m, gi);
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48lem_h h = sm48lem_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48lem_match(grp, key)) return 1;
        uint32_t ctrl = *sm48lem_ctrl_c(grp);
        if (!((ctrl >> (SM48LEM_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = SM_(_next)(m, gi);
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48lem_h h = sm48lem_hash(key);
    uint32_t gi = SM_(_gi)(m, h.lo);

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48lem_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            uint32_t *cp = sm48lem_ctrl(grp);
            *cp &= ~(1u << slot);
            m->count--;
            return 1;
        }
        uint32_t ctrl = *sm48lem_ctrl_c(grp);
        if (!((ctrl >> (SM48LEM_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = SM_(_next)(m, gi);
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SIMD_MAP_NAME
