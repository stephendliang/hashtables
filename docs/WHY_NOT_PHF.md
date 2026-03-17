# Why perfect hash maps don't work for SIMD hash sets

This directory previously contained `simd_phf64.h`, a "perfect" hash set that
guaranteed worst-case O(1) lookup with no probing. It was removed because it
consistently lost to the regular `simd_set64.h` in benchmarks. This file
explains why.

## The idea

A perfect hash function (PHF) maps N known keys into N (or slightly more)
slots with zero collisions. For a SIMD-friendly variant, we relax this to:
hash N keys into groups of 8 slots (one cache line) such that no group
receives more than 8 keys. Lookup is then a single SIMD compare against one
cache line — guaranteed, no probing, no loop.

## The balls-into-bins problem

Placing N keys into `ng` groups of 8 slots is equivalent to throwing N balls
into ng bins with capacity 8. The constraint is:

    max(bin_count) <= 8

With single-choice hashing (each key maps to exactly one group), the maximum
bin occupancy for N balls into ng bins is:

    max ≈ N/ng + O(sqrt((N/ng) · ln(ng)))

For the no-overflow constraint to hold with reasonable probability, you need
the average load to be far below the bin capacity. In practice:

| PHF64_SLOTS | Groups needed (ng) | Average load | Effective load factor |
|-------------|--------------------|--------------|-----------------------|
| 8  (1 CL)   | ≈ N               | ~1 key/group | **12.5%**             |
| 16 (2 CL)   | ≈ N/2             | ~2 keys/group| **12.5%**             |

Even with seed search (trying 1000 random hash seeds per group count), the
birthday-collision tail forces massive over-provisioning.

Trying to push average load to 50% (ng = N/4) makes the expected maximum
bin occupancy ~12-15, far exceeding 8. No seed will fix this — it's a
statistical impossibility, not an engineering problem.

## Why this kills performance

`simd_phf64` at 12% load vs `simd_map64` at 75% load, for N = 2M keys:

- **8x memory footprint** → TLB working set is 8x larger
- **L2/L3 thrashing** → at 2M keys × 8 bytes × 8 slots/key = 128MB, the
  table doesn't fit in L2 (typically 1-4MB) or even L3 on many CPUs
- `simd_map64` at 75% load: 2M × 8 bytes / 0.75 = 21MB — fits in L3

The guaranteed single-probe lookup saves ~0.3 probes on average versus
`simd_map64`, but this is overwhelmed by the cache/TLB penalty of touching
8x more address space.

Measured on EPYC 9845 (N=2M, pipelined prefetch):

    simd_map64:  4-9 ns/lookup (fits in L3)
    simd_phf64: 12-18 ns/lookup (TLB + L3 miss dominated)

## Known alternatives (and why they don't help here)

**Power of two choices**: Hash each key to 2 candidate groups, place in the
less loaded one. Max load drops from O(ln N / ln ln N) to O(ln ln N) ≈ 3-4,
allowing ~50% load. But lookup must now check 2 random cache lines — which
is what the regular hash map already does with probing, except the regular
map has better spatial locality since probe chains are sequential.

**Larger groups (32-64 slots)**: Wider bins tolerate higher average loads
before overflow. But each lookup reads 4-8 cache lines, which is worse than
`simd_map64`'s typical 1-probe hit.

**CHD / compressed MPHF**: Stores per-bucket displacement values; achieves
near 100% load. But adds a dependent indirection (read displacement table →
compute position → read key), so 2 serial random memory accesses. At that
point you've reinvented cuckoo hashing with extra metadata.

## Conclusion

The fundamental problem is that "no group overflows" with single-choice
hashing is a constraint on the **tail** of a Poisson distribution. Satisfying
it requires the mean to be far from the capacity, which wastes most of the
allocated memory, which destroys cache performance, which is the only thing
that matters at scale.

Perfect hash maps have legitimate uses in small, static, latency-critical
tables (compiler keyword lookup, CPU instruction decode, etc.) where the
table fits in L1. For general-purpose hash sets at 100K+ keys, a well-tuned
open-addressing map with SIMD probing will always win.
