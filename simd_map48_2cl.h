/*
 * simd_map48_2cl.h — direct-compare 48-bit set, 2CL/20 ACP
 *
 * 20 keys per 128B group (2 cache lines). ACP brings line 2 free.
 *
 * Group layout (128B):
 *   Line 1 (offset 0-63):   uint64_t ctrl (8B) + uint32_t hi[14] (56B)
 *   Line 2 (offset 64-127): uint32_t hi[14..19] (24B) + uint16_t lo[20] (40B)
 *
 *   ctrl: bits [19:0] occupancy, bits [35:20] overflow (16 partitions)
 *   hi at offset 8, lo at offset 88
 *
 * Set mode only. Key: uint64_t (lower 48 bits, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 *   #define SIMD_MAP_NAME my_set48_2cl
 *   #include "simd_map48_2cl.h"
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_2cl.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_2CL_COMMON_H_
#define SIMD_MAP48_2CL_COMMON_H_

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

#ifndef SMCAT_
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

#define SM48_2CL_SLOTS_     20
#define SM48_2CL_OCC_MASK_  0x000FFFFFu  /* bits [19:0] */
#define SM48_2CL_OVF_SHIFT_ 20
#define SM48_2CL_GRP_SZ_    128

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48_2cl_h { uint32_t lo, hi; };

#if defined(__SSE4_2__)
static inline struct sm48_2cl_h sm48_2cl_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48_2cl_h){a, b};
}
#else
static inline struct sm48_2cl_h sm48_2cl_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48_2cl_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Group access: ctrl at offset 0, hi at offset 8, lo at offset 88 */
static inline uint64_t *sm48_2cl_ctrl(void *grp)  { return (uint64_t *)grp; }
static inline uint32_t *sm48_2cl_hi(void *grp)    { return (uint32_t *)((char *)grp + 8); }
static inline uint16_t *sm48_2cl_lo(void *grp)    { return (uint16_t *)((char *)grp + 88); }

static inline const uint64_t *sm48_2cl_ctrl_c(const void *grp)  { return (const uint64_t *)grp; }
static inline const uint32_t *sm48_2cl_hi_c(const void *grp)    { return (const uint32_t *)((const char *)grp + 8); }
static inline const uint16_t *sm48_2cl_lo_c(const void *grp)    { return (const uint16_t *)((const char *)grp + 88); }

static inline uint64_t sm48_2cl_read_key(const void *grp, int slot) {
    return ((uint64_t)sm48_2cl_hi_c(grp)[slot] << 16) | sm48_2cl_lo_c(grp)[slot];
}

static inline void sm48_2cl_write_key(void *grp, int slot, uint64_t key) {
    sm48_2cl_hi(grp)[slot] = (uint32_t)(key >> 16);
    sm48_2cl_lo(grp)[slot] = (uint16_t)(key);
}

/* --- SIMD match --- */

#if defined(__AVX2__)

static inline uint32_t sm48_2cl_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48_2cl_hi_c(grp);
    const uint16_t *lop = sm48_2cl_lo_c(grp);

    __m256i kh = _mm256_set1_epi32((int)key_hi);

    /* hi[0..7] */
    uint32_t mh = (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 0)), kh)));
    /* hi[8..15] */
    mh |= (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i *)(hip + 8)), kh))) << 8;
    /* hi[16..19]: scalar */
    if (hip[16] == key_hi) mh |= (1u << 16);
    if (hip[17] == key_hi) mh |= (1u << 17);
    if (hip[18] == key_hi) mh |= (1u << 18);
    if (hip[19] == key_hi) mh |= (1u << 19);

    __m256i kl = _mm256_set1_epi16((short)key_lo);

    /* lo[0..15] */
    uint32_t raw = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi16(
            _mm256_loadu_si256((const __m256i *)(lop + 0)), kl));
    uint32_t ml = _pext_u32(raw, 0xAAAAAAAAu);
    /* lo[16..19]: scalar */
    if (lop[16] == key_lo) ml |= (1u << 16);
    if (lop[17] == key_lo) ml |= (1u << 17);
    if (lop[18] == key_lo) ml |= (1u << 18);
    if (lop[19] == key_lo) ml |= (1u << 19);

    uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
    return mh & ml & (uint32_t)(ctrl & SM48_2CL_OCC_MASK_);
}

static inline uint32_t sm48_2cl_empty(const void *grp) {
    uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
    return (~(uint32_t)ctrl) & SM48_2CL_OCC_MASK_;
}

#define sm48_2cl_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#else /* scalar */

static inline uint32_t sm48_2cl_match(const void *grp, uint64_t key) {
    uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
    uint32_t occ = (uint32_t)(ctrl & SM48_2CL_OCC_MASK_);
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48_2cl_hi_c(grp);
    const uint16_t *lop = sm48_2cl_lo_c(grp);
    uint32_t result = 0;
    for (int i = 0; i < SM48_2CL_SLOTS_; i++) {
        if ((occ >> i) & 1) {
            if (hip[i] == key_hi && lop[i] == key_lo)
                result |= (1u << i);
        }
    }
    return result;
}

static inline uint32_t sm48_2cl_empty(const void *grp) {
    uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
    return (~(uint32_t)ctrl) & SM48_2CL_OCC_MASK_;
}

#if defined(__SSE4_2__)
#define sm48_2cl_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48_2cl_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48_2CL_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;
    uint32_t mask;
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + (size_t)gi * SM48_2CL_GRP_SZ_;
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = sm48_2cl_hash(key).lo & m->mask;
    sm48_2cl_prefetch_line(SM_(_group)(m, gi));
    /* ACP brings line 2 (offset 64) for free */
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48_2cl_hash(key).lo & m->mask;
    sm48_2cl_prefetch_line(SM_(_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
    size_t raw = (size_t)ng * SM48_2CL_GRP_SZ_;
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
        const char *old_grp = old_data + (size_t)g * SM48_2CL_GRP_SZ_;
        uint64_t ctrl = *(const uint64_t *)old_grp;
        uint32_t occ = (uint32_t)(ctrl & SM48_2CL_OCC_MASK_);

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48_2cl_read_key(old_grp, s);
            struct sm48_2cl_h h = sm48_2cl_hash(key);
            uint32_t gi = h.lo & mask;

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint32_t em = sm48_2cl_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    uint64_t *cp = sm48_2cl_ctrl(grp);
                    *cp |= (1ull << pos);
                    sm48_2cl_write_key(grp, pos, key);
                    m->count++;
                    break;
                }
                uint64_t *cp = sm48_2cl_ctrl(grp);
                *cp |= (1ull << (SM48_2CL_OVF_SHIFT_ + (h.hi & 15)));
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
    /* Load factor 7/8: ng*20*7/8 >= n → ng >= n*8/(20*7) = n*8/140 */
    uint64_t need = ((uint64_t)n * 8 + 139) / 140;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48_2CL_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48_2cl_h h = sm48_2cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        if (sm48_2cl_match(grp, key)) return 0;

        uint32_t em = sm48_2cl_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint64_t *cp = sm48_2cl_ctrl(grp);
            *cp |= (1ull << pos);
            sm48_2cl_write_key(grp, pos, key);
            m->count++;
            return 1;
        }
        uint64_t *cp = sm48_2cl_ctrl(grp);
        *cp |= (1ull << (SM48_2CL_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48_2CL_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48_2cl_h h = sm48_2cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint32_t em = sm48_2cl_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint64_t *cp = sm48_2cl_ctrl(grp);
            *cp |= (1ull << pos);
            sm48_2cl_write_key(grp, pos, key);
            m->count++;
            return;
        }
        uint64_t *cp = sm48_2cl_ctrl(grp);
        *cp |= (1ull << (SM48_2CL_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48_2cl_h h = sm48_2cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48_2cl_match(grp, key)) return 1;
        uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
        if (!((ctrl >> (SM48_2CL_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48_2cl_h h = sm48_2cl_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint32_t mm = sm48_2cl_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            uint64_t *cp = sm48_2cl_ctrl(grp);
            *cp &= ~(1ull << slot);
            m->count--;
            return 1;
        }
        uint64_t ctrl = *sm48_2cl_ctrl_c(grp);
        if (!((ctrl >> (SM48_2CL_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SIMD_MAP_NAME
