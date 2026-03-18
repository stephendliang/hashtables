/*
 * simd_map48_3cl.h — direct-compare 48-bit set, 3CL/31 fully vectorized
 *
 * 31 keys per 192B group (3 cache lines). Split hi32+lo16 layout with
 * all 32 SIMD lanes used — zero scalar tail.
 *
 * Group layout (192B, 64B-aligned):
 *   Line 0 (offset 0-63):    uint32_t hi[0..15]  (64B)
 *   Line 1 (offset 64-127):  uint32_t hi[16..31] (64B)
 *   Line 2 (offset 128-191): uint16_t lo[0..31]  (64B)
 *
 *   hi[0] = ctrl_occ: bits [30:0] occupancy (31 data slots)
 *   lo[0] = ctrl_ovf: 16-bit overflow partition
 *   Data slots: 1-31
 *
 * Set mode only. Key: uint64_t (lower 48 bits, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 *   #define SIMD_MAP_NAME my_set48_3cl
 *   #include "simd_map48_3cl.h"
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_3cl.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_3CL_COMMON_H_
#define SIMD_MAP48_3CL_COMMON_H_

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

#define SM48_3CL_SLOTS_    31
#define SM48_3CL_OCC_MASK_ 0x7FFFFFFEu  /* bits [30:1] */
#define SM48_3CL_GRP_SZ_   192

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48_3cl_h { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct sm48_3cl_h sm48_3cl_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48_3cl_h){a, b};
}
#else
static inline struct sm48_3cl_h sm48_3cl_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48_3cl_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Group access helpers */
static inline uint32_t *sm48_3cl_hi(void *grp) { return (uint32_t *)grp; }
static inline uint16_t *sm48_3cl_lo(void *grp) { return (uint16_t *)((char *)grp + 128); }
static inline const uint32_t *sm48_3cl_hi_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint16_t *sm48_3cl_lo_c(const void *grp) { return (const uint16_t *)((const char *)grp + 128); }

/* ctrl_occ is hi[0], ctrl_ovf is lo[0] */
static inline uint32_t sm48_3cl_occ(const void *grp) {
    return sm48_3cl_hi_c(grp)[0] & 0x7FFFFFFFu;
}

static inline uint64_t sm48_3cl_read_key(const void *grp, int slot) {
    return ((uint64_t)sm48_3cl_hi_c(grp)[slot] << 16) | sm48_3cl_lo_c(grp)[slot];
}

static inline void sm48_3cl_write_key(void *grp, int slot, uint64_t key) {
    sm48_3cl_hi(grp)[slot] = (uint32_t)(key >> 16);
    sm48_3cl_lo(grp)[slot] = (uint16_t)(key);
}

/* --- SIMD match --- */

#if defined(__AVX2__)

static inline uint32_t sm48_3cl_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48_3cl_hi_c(grp);
    const uint16_t *lop = sm48_3cl_lo_c(grp);

    __m256i kh = _mm256_set1_epi32((int)key_hi);

    /* hi[0..7] */
    uint32_t mh = (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 0)), kh)));
    /* hi[8..15] */
    mh |= (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 8)), kh))) << 8;
    /* hi[16..23] */
    mh |= (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 16)), kh))) << 16;
    /* hi[24..31] */
    mh |= (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 24)), kh))) << 24;

    __m256i kl = _mm256_set1_epi16((short)key_lo);

    /* lo[0..15] */
    uint32_t raw_lo = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi16(
            _mm256_loadu_si256((const __m256i *)(lop + 0)), kl));
    uint32_t ml = _pext_u32(raw_lo, 0xAAAAAAAAu);
    /* lo[16..31] */
    uint32_t raw_hi = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi16(
            _mm256_loadu_si256((const __m256i *)(lop + 16)), kl));
    ml |= _pext_u32(raw_hi, 0xAAAAAAAAu) << 16;

    uint32_t occ = hip[0] & 0x7FFFFFFFu;
    return mh & ml & occ;
}

static inline uint32_t sm48_3cl_empty(const void *grp) {
    uint32_t occ = sm48_3cl_hi_c(grp)[0] & 0x7FFFFFFFu;
    return (~occ) & 0x7FFFFFFEu;  /* exclude slot 0 (ctrl) */
}

#define sm48_3cl_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint32_t sm48_3cl_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48_3cl_hi_c(grp);
    const uint16_t *lop = sm48_3cl_lo_c(grp);

    /* hi[0..31]: 8 × vceqq_u32 (4 lanes each) → 32-bit mask */
    uint32x4_t kh = vdupq_n_u32(key_hi);
    uint32_t mh = 0;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), kh));
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), kh)) << 4;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 8), kh)) << 8;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 12), kh)) << 12;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 16), kh)) << 16;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 20), kh)) << 20;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 24), kh)) << 24;
    mh |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 28), kh)) << 28;

    /* lo[0..31]: 4 × vceqq_u16 (8 lanes each) → 32-bit mask */
    uint16x8_t kl = vdupq_n_u16(key_lo);
    uint32_t ml = 0;
    ml |= neon_movemask_u16(vceqq_u16(vld1q_u16(lop + 0), kl));
    ml |= neon_movemask_u16(vceqq_u16(vld1q_u16(lop + 8), kl)) << 8;
    ml |= neon_movemask_u16(vceqq_u16(vld1q_u16(lop + 16), kl)) << 16;
    ml |= neon_movemask_u16(vceqq_u16(vld1q_u16(lop + 24), kl)) << 24;

    uint32_t occ = hip[0] & 0x7FFFFFFFu;
    return mh & ml & occ;
}

static inline uint32_t sm48_3cl_empty(const void *grp) {
    uint32_t occ = sm48_3cl_hi_c(grp)[0] & 0x7FFFFFFFu;
    return (~occ) & 0x7FFFFFFEu;
}

#define sm48_3cl_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* scalar */

static inline uint32_t sm48_3cl_match(const void *grp, uint64_t key) {
    uint32_t occ = sm48_3cl_occ(grp);
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48_3cl_hi_c(grp);
    const uint16_t *lop = sm48_3cl_lo_c(grp);
    uint32_t result = 0;
    for (int i = 1; i < 32; i++) {
        if ((occ >> i) & 1) {
            if (hip[i] == key_hi && lop[i] == key_lo)
                result |= (1u << i);
        }
    }
    return result;
}

static inline uint32_t sm48_3cl_empty(const void *grp) {
    uint32_t occ = sm48_3cl_hi_c(grp)[0] & 0x7FFFFFFFu;
    return (~occ) & 0x7FFFFFFEu;
}

#if defined(__SSE4_2__)
#define sm48_3cl_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48_3cl_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48_3CL_COMMON_H_ */

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
    return m->data + (size_t)gi * SM48_3CL_GRP_SZ_;
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = sm48_3cl_hash(key).lo & m->mask;
    char *grp = SM_(_group)(m, gi);
    sm48_3cl_prefetch_line(grp);        /* line 0: hi[0..15] */
    sm48_3cl_prefetch_line(grp + 128);  /* line 2: lo[0..31] — ACP brings line 1 */
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48_3cl_hash(key).lo & m->mask;
    char *grp = SM_(_group)(m, gi);
    sm48_3cl_prefetch_line(grp);        /* line 0: ctrl_occ in hi[0] */
    sm48_3cl_prefetch_line(grp + 128);  /* line 2: ctrl_ovf in lo[0] */
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
    size_t raw = (size_t)ng * SM48_3CL_GRP_SZ_;
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
        const char *old_grp = old_data + (size_t)g * SM48_3CL_GRP_SZ_;
        uint32_t occ = sm48_3cl_hi_c(old_grp)[0] & 0x7FFFFFFFu;
        occ &= 0x7FFFFFFEu;  /* skip slot 0 */

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48_3cl_read_key(old_grp, s);
            struct sm48_3cl_h h = sm48_3cl_hash(key);
            uint32_t gi = h.lo & mask;

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint32_t em = sm48_3cl_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    uint32_t *hip = sm48_3cl_hi(grp);
                    hip[0] |= (1u << pos);
                    sm48_3cl_write_key(grp, pos, key);
                    m->count++;
                    break;
                }
                uint16_t *lop = sm48_3cl_lo(grp);
                lop[0] |= (uint16_t)(1u << (h.hi & 15));
                gi = (gi + 1) & mask;
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
    /* Load factor 7/8: ng*31*7/8 >= n → ng >= n*8/(31*7) */
    uint64_t need = ((uint64_t)n * 8 + 216) / 217;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48_3CL_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48_3cl_h h = sm48_3cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        if (sm48_3cl_match(grp, key)) return 0;

        uint32_t em = sm48_3cl_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *hip = sm48_3cl_hi(grp);
            hip[0] |= (1u << pos);
            sm48_3cl_write_key(grp, pos, key);
            m->count++;
            return 1;
        }
        uint16_t *lop = sm48_3cl_lo(grp);
        lop[0] |= (uint16_t)(1u << (h.hi & 15));
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48_3CL_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48_3cl_h h = sm48_3cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint32_t em = sm48_3cl_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *hip = sm48_3cl_hi(grp);
            hip[0] |= (1u << pos);
            sm48_3cl_write_key(grp, pos, key);
            m->count++;
            return;
        }
        uint16_t *lop = sm48_3cl_lo(grp);
        lop[0] |= (uint16_t)(1u << (h.hi & 15));
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48_3cl_h h = sm48_3cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48_3cl_match(grp, key)) return 1;
        uint16_t ovf = sm48_3cl_lo_c(grp)[0];
        if (!((ovf >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48_3cl_h h = sm48_3cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint32_t mm = sm48_3cl_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            uint32_t *hip = sm48_3cl_hi(grp);
            hip[0] &= ~(1u << slot);
            m->count--;
            return 1;
        }
        uint16_t ovf = sm48_3cl_lo_c(grp)[0];
        if (!((ovf >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SIMD_MAP_NAME
