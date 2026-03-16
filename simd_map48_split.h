/*
 * simd_map48_split.h — direct-compare 48-bit set, split hi32+lo16
 *
 * 10 keys per 64B cache line. No metadata filtering — SIMD compares
 * hi32 (vpcmpeqd) and lo16 (vpcmpeqw) separately, then AND.
 *
 * Group layout (64B):
 *   Offset 0-3:   uint32_t ctrl
 *                    bits [9:0]   = occupancy (10 slots)
 *                    bits [25:10] = overflow (16 partitions)
 *                    bits [31:26] = reserved
 *   Offset 4-43:  uint32_t hi[10]  (upper 32 bits: key >> 16)
 *   Offset 44-63: uint16_t lo[10]  (lower 16 bits: key & 0xFFFF)
 *   Total: 4 + 40 + 20 = 64 bytes
 *
 * Set mode only. Key: uint64_t (lower 48 bits, upper 16 must be 0, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 *   #define SIMD_MAP_NAME my_set48s
 *   #include "simd_map48_split.h"
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_split.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48S_COMMON_H_
#define SIMD_MAP48S_COMMON_H_

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

#define SM48S_SLOTS_    10
#define SM48S_OCC_MASK_ 0x03FFu
#define SM48S_OVF_SHIFT_ 10

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48s_h { uint32_t lo, hi; };

#if defined(__SSE4_2__)
static inline struct sm48s_h sm48s_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48s_h){a, b};
}
#else
static inline struct sm48s_h sm48s_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48s_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Group access helpers — cast char* to typed pointers */
static inline uint32_t *sm48s_ctrl(void *grp) { return (uint32_t *)grp; }
static inline uint32_t *sm48s_hi(void *grp)   { return (uint32_t *)((char *)grp + 4); }
static inline uint16_t *sm48s_lo(void *grp)   { return (uint16_t *)((char *)grp + 44); }

static inline const uint32_t *sm48s_ctrl_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint32_t *sm48s_hi_c(const void *grp)   { return (const uint32_t *)((const char *)grp + 4); }
static inline const uint16_t *sm48s_lo_c(const void *grp)   { return (const uint16_t *)((const char *)grp + 44); }

static inline uint64_t sm48s_read_key(const void *grp, int slot) {
    return ((uint64_t)sm48s_hi_c(grp)[slot] << 16) | sm48s_lo_c(grp)[slot];
}

static inline void sm48s_write_key(void *grp, int slot, uint64_t key) {
    sm48s_hi(grp)[slot] = (uint32_t)(key >> 16);
    sm48s_lo(grp)[slot] = (uint16_t)(key);
}

/* --- SIMD match --- */

#if defined(__AVX2__)

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;

    /* Hi: vpcmpeqd on hi[0..7], scalar for hi[8..9] */
    __m256i h8 = _mm256_loadu_si256((const __m256i *)sm48s_hi_c(grp));
    __m256i kh = _mm256_set1_epi32((int)key_hi);
    uint32_t m_hi8 = (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, kh)));
    uint16_t m_hi = (uint16_t)m_hi8;
    const uint32_t *hip = sm48s_hi_c(grp);
    if (hip[8] == key_hi) m_hi |= (1u << 8);
    if (hip[9] == key_hi) m_hi |= (1u << 9);

    /* Lo: vpcmpeqw on lo[0..7], scalar for lo[8..9] */
    __m128i l8 = _mm_loadu_si128((const __m128i *)sm48s_lo_c(grp));
    __m128i kl = _mm_set1_epi16((short)key_lo);
    uint32_t raw = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(l8, kl));
    uint16_t m_lo = (uint16_t)_pext_u32(raw, 0xAAAAu);
    const uint16_t *lop = sm48s_lo_c(grp);
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return m_hi & m_lo & (uint16_t)(ctrl & SM48S_OCC_MASK_);
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#define sm48s_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#else /* scalar */

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    uint16_t occ = (uint16_t)(ctrl & SM48S_OCC_MASK_);
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48s_hi_c(grp);
    const uint16_t *lop = sm48s_lo_c(grp);
    uint16_t result = 0;
    for (int i = 0; i < SM48S_SLOTS_; i++) {
        if ((occ >> i) & 1) {
            if (hip[i] == key_hi && lop[i] == key_lo)
                result |= (uint16_t)(1u << i);
        }
    }
    return result;
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#if defined(__SSE4_2__)
#define sm48s_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48s_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48S_COMMON_H_ */

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
    uint32_t gi = sm48s_hash(key).lo & m->mask;
    sm48s_prefetch_line(SM_(_group)(m, gi));
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48s_hash(key).lo & m->mask;
    sm48s_prefetch_line(SM_(_group)(m, gi));
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
        uint32_t ctrl = *(const uint32_t *)old_grp;
        uint16_t occ = (uint16_t)(ctrl & SM48S_OCC_MASK_);

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48s_read_key(old_grp, s);
            struct sm48s_h h = sm48s_hash(key);
            uint32_t gi = h.lo & mask;

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t em = sm48s_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    uint32_t *cp = sm48s_ctrl(grp);
                    *cp |= (1u << pos);
                    sm48s_write_key(grp, pos, key);
                    m->count++;
                    break;
                }
                uint32_t *cp = sm48s_ctrl(grp);
                *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
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
    uint64_t need = ((uint64_t)n * 8 + 69) / 70;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48S_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        if (sm48s_match(grp, key)) return 0;

        uint16_t em = sm48s_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp |= (1u << pos);
            sm48s_write_key(grp, pos, key);
            m->count++;
            return 1;
        }
        uint32_t *cp = sm48s_ctrl(grp);
        *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48S_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t em = sm48s_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp |= (1u << pos);
            sm48s_write_key(grp, pos, key);
            m->count++;
            return;
        }
        uint32_t *cp = sm48s_ctrl(grp);
        *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48s_match(grp, key)) return 1;
        uint32_t ctrl = *sm48s_ctrl_c(grp);
        if (!((ctrl >> (SM48S_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48s_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp &= ~(1u << slot);
            m->count--;
            return 1;
        }
        uint32_t ctrl = *sm48s_ctrl_c(grp);
        if (!((ctrl >> (SM48S_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SIMD_MAP_NAME
