# Map48 Architecture Comparison

5 split48 variants benchmarked at 2M keys, AVX2, i9-12900HK, PF=24.
All use hi32+lo16 key split with direct SIMD comparison.

## Variants

| # | Variant | Group size | Slots | Load factor | Overflow | Delete |
|---|---------|-----------|-------|-------------|----------|--------|
| 1 | **split** (baseline) | 1CL (64B) | 10 | 87.5% | sentinel 16-part | clear occ bit |
| 2 | 3CL/31 | 3CL (192B) | 31 | 87.5% | sentinel 16-part | clear occ bit |
| 3 | 2CL/20 | 2CL (128B) | 20 | 87.5% | sentinel 16-part | clear occ bit |
| 4 | backshift | 1CL (64B) | 10 | 75% | none (empty-slot) | backshift |
| 5 | lemire | 1CL (64B) | 10 | 87.5% | sentinel 16-part | clear occ bit |

## Results (ns/op, median of 3 runs)

| Variant | Insert | Contains | Mixed 50/25/25 | Alloc (MiB) |
|-----------|--------|----------|----------------|-------------|
| **split** | **3.9** | **3.4** | **7.4** | 16 |
| 3CL/31 | 6.5 | 5.3 | 11.4 | 24 |
| 2CL/20 | 5.9 | 5.1 | 9.0 | 16 |
| backshift | 5.0 | 4.9 | 7.8 | 32 |
| lemire | 5.1 | 5.3 | 7.8 | 14 |

## Analysis

### Split baseline wins everything

The 1CL/10 sentinel design is optimal. Every operation touches exactly 1
cache line per group probe. No alternative architecture came close.

### 3CL/31: +50-65% slower despite full vectorization

The hypothesis was that 31 slots × 87.5% LF in 3 cache lines would reduce
group count enough to offset the extra lines. It didn't. 3 cache lines per
probe is simply too expensive — the hardware prefetcher and ACP cannot fully
hide 2 additional line fetches when probing random groups. Insert is 67%
slower, contains 56%, mixed 54%.

The fully-vectorized match (4× vpcmpeqd + 2× vpcmpeqw, zero scalar tail) is
irrelevant — the bottleneck is memory, not compute. Eliminating 4 scalar
compares saves ~2 cycles against ~100+ cycles of memory latency.

### 2CL/20: +30-50% slower, ACP doesn't help enough

2 cache lines with ACP (Adjacent Cache Line Prefetch) bringing line 2 for
free was supposed to make this effectively 1CL cost. In practice, ACP only
helps when the access pattern gives the prefetcher time to act. With
randomized probing, ACP often delivers line 2 too late. Insert is 51% slower,
contains 50%, mixed 22%.

The mixed workload gap (22%) is smaller than insert/contains (~50%) because
the mixed workload's interleaved ops give more time between accesses to the
same group, allowing ACP to complete.

### Backshift: competitive mixed, but 2x memory

Backshift eliminates overflow bits entirely — no ctrl word, no sentinel.
Probe termination is by empty slot (like map64). This requires 75% load
factor instead of 87.5%, doubling allocation (32 MiB vs 16).

Insert is 28% slower (backshift overhead on grow), contains is 44% slower
(must scan until empty slot vs overflow bit check), but mixed is only 5%
slower (7.8 vs 7.4). The mixed competitiveness comes from zero-cost delete:
backshift keeps probe chains compact, so future lookups have shorter chains.

Not worth it: 2x memory for marginal mixed gains, with significant
insert/contains regression.

### Lemire: 12.5% less memory, 30-55% slower

Lemire's fast range reduction (`(uint32_t)(((uint64_t)hash * ng) >> 32)`)
allows exact (non-pow2) group count, saving 12.5% memory (14 vs 16 MiB).

But `imul + shr` is measurably more expensive than `and mask` on the critical
path. Insert is 31% slower, contains 56%, mixed 5%. The `imul` has 3-cycle
latency vs 1 for `and`, and more critically, the modular wrap (`gi++; if
(gi >= ng) gi = 0`) adds a branch that `(gi + 1) & mask` avoids.

Not worth it: 2 MiB saved for 30%+ regression on primary operations.

## Conclusion

**Split 1CL/10 with sentinel overflow is the optimal 48-bit set architecture.**

The memory-latency bottleneck makes cache line count the dominant factor.
No amount of compute optimization (full vectorization, fewer groups, non-pow2
addressing) can compensate for touching additional cache lines. The 1CL
designs (split, backshift, lemire) cluster together on mixed workloads
(7.4-7.8 ns), while multi-CL designs are significantly worse (9.0-11.4).
Within the 1CL cluster, sentinel overflow with pow2 masking is fastest.

Previous benchmark (packed vs split vs map48 vs map64) for reference:

| Variant | Insert | Contains | Mixed |
|---------|--------|----------|-------|
| split | 3.9 | 3.3 | 6.7 |
| packed | 4.0 | 4.3 | 7.4 |
| map64 | 4.7 | 3.8 | 6.6 |
| map48 (sentinel) | 6.7 | 10.1 | 13.6 |
