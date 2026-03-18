/*
 * simd_compat.h — portable CRC32, prefetch, and mmap shims
 *
 * Include AFTER <sys/mman.h> and <immintrin.h> (if applicable).
 *
 * Provides:
 *   - ARM CRC32C intrinsic via __crc32cd, exposed as _mm_crc32_u64 compat
 *   - __builtin_prefetch as _mm_prefetch compat on non-x86
 *   - MAP_HUGETLB / MAP_POPULATE / MADV_HUGEPAGE fallback defines
 */
#ifndef SIMD_COMPAT_H_
#define SIMD_COMPAT_H_

#include <stdint.h>

/* --- ARM CRC32 + prefetch compat --- */
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
static inline uint64_t _simd_compat_crc32(uint64_t crc, uint64_t v) {
    return (uint64_t)__crc32cd((uint32_t)crc, v);
}
#define _mm_crc32_u64(crc, v) _simd_compat_crc32((crc), (v))
#ifndef _MM_HINT_T0
#define _MM_HINT_T0 3
#endif
#define _mm_prefetch(p, h) __builtin_prefetch((const void *)(p), 0, 3)
#endif

/* --- ARM NEON movemask helpers --- */
#if defined(__ARM_NEON)
#include <arm_neon.h>

/* 8 x u16 comparison result → 8-bit mask.
 * Comparison results are all-ones (0xFFFF) or all-zeros (0x0000) per lane,
 * so AND with positional weights directly extracts the bit — no shift or
 * multiply needed. 2 NEON ops (AND + ADDV) vs the old 3 (USHR + MUL + ADDV).
 * Benchmarked: 0.30 ns/op vs 0.33 ns (mul+addv), 0.38 (narrow+smul),
 * 0.40 (addp cascade) on Apple Silicon M-series. */
static inline uint32_t neon_movemask_u16(uint16x8_t v) {
    static const uint16_t pos[] = {1, 2, 4, 8, 16, 32, 64, 128};
    return vaddvq_u16(vandq_u16(v, vld1q_u16(pos)));
}

/* 4 x u32 comparison result → 4-bit mask */
static inline uint32_t neon_movemask_u32(uint32x4_t v) {
    static const uint32_t pos[] = {1, 2, 4, 8};
    return vaddvq_u32(vandq_u32(v, vld1q_u32(pos)));
}

/* 2 x u64 comparison result → 2-bit mask */
static inline uint32_t neon_movemask_u64(uint64x2_t v) {
    return (uint32_t)((vgetq_lane_u64(v, 0) & 1) |
                      ((vgetq_lane_u64(v, 1) & 1) << 1));
}

#endif /* __ARM_NEON */

/* --- mmap flag compat (macOS / non-Linux) --- */
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0
#endif
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif

#endif /* SIMD_COMPAT_H_ */
