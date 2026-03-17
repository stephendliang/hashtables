# Map48 Architecture Comparison

6 split48 variants benchmarked at 2M keys, AVX2, i9-12900HK, PF=24.
All use hi32+lo16 key split with direct SIMD comparison.

## Variants

| # | Variant | Group size | Slots | LF | Overflow | Delete |
|---|---------|-----------|-------|----|----------|--------|
| 1 | **split** (baseline) | 1CL (64B) | 10 | 87.5% | sentinel 16-part | clear occ bit |
| 2 | 3CL/31 | 3CL (192B) | 31 | 87.5% | sentinel 16-part | clear occ bit |
| 3 | 2CL/20 | 2CL (128B) | 20 | 87.5% | sentinel 16-part | clear occ bit |
| 4 | backshift | 1CL (64B) | 10 | 75% | none (empty-slot) | backshift |
| 5 | lemire | 1CL (64B) | 10 | 87.5% | sentinel 16-part | clear occ bit |
| 6 | lem+bs | 1CL (64B) | 10 | 87.5% | ghost 16-part | backshift |

## Results (ns/op, median of 3 runs)

| Variant | Insert | Contains | Mixed 50/25/25 | Alloc (MiB) |
|-----------|--------|----------|----------------|-------------|
| **split** | **3.9** | **3.4** | 7.4 | 16 |
| 3CL/31 | 6.5 | 5.3 | 11.4 | 24 |
| 2CL/20 | 5.9 | 5.1 | 9.0 | 16 |
| backshift | 5.0 | 4.9 | 7.8 | 32 |
| lemire | 5.1 | 5.3 | 7.8 | 14 |
| lem+bs | 6.3 | 5.3 | **6.6** | 14 |

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

### Lem+bs: best mixed at 14 MiB via ghost overflow + backshift

Combines Lemire addressing with backshift deletion AND ghost overflow bits
in the 4B padding at offset 60. The ghost overflow bits (set on insert
overflow, never cleared on delete) provide fast miss-path probe termination
at 87.5% load factor — the same technique as `simd_bitstealing.h`.

This was developed iteratively:
1. First attempt: Lemire + backshift at 75% LF, empty-slot termination.
   Result: 18 MiB (75% LF erased the Lemire savings), mixed was unstable
   (7.9-12.4 ns) due to `% ng` division in backshift distance calculation.
2. Second attempt: push to 87.5% LF, use the 4B padding as 16-partition
   ghost overflow bits. Result: 14 MiB, mixed stabilized at 6.6-6.7 ns.

Insert is 6.3 ns (62% slower than split) — the `imul+shr` Lemire reduction,
overflow bit writes, and tighter packing all hurt. Contains is 5.3 ns (56%
slower). But mixed is **6.6 ns — 11% faster than split** (7.4 ns). The mixed
advantage comes from backshift keeping probe chains compact after deletes,
combined with ghost overflow for fast miss termination. At 14 MiB, it uses
12.5% less memory than split (16 MiB).

Backshift still uses empty-slot termination internally (to find the end of
the probe chain to repair), while the public contains/delete search path
uses overflow bits. Ghost bits accumulate under churn but grow resets them
(fresh table). Benchmarks show stable performance across 3 runs.

## Conclusion

**Split 1CL/10 with sentinel overflow is the optimal 48-bit set architecture
for insert/contains-heavy workloads.** For churn-heavy workloads (50/25/25),
lem+bs is 11% faster at 12.5% less memory.

The memory-latency bottleneck makes cache line count the dominant factor.
No amount of compute optimization (full vectorization, fewer groups, non-pow2
addressing) can compensate for touching additional cache lines. The 1CL
designs (split, backshift, lemire, lem+bs) cluster together on mixed workloads
(6.6-7.8 ns), while multi-CL designs are significantly worse (9.0-11.4).
Within the 1CL cluster, sentinel overflow with pow2 masking is fastest for
pure insert/contains; ghost overflow + backshift wins mixed churn.

Previous benchmark (packed vs split vs map48 vs map64) for reference:

| Variant | Insert | Contains | Mixed |
|---------|--------|----------|-------|
| split | 3.9 | 3.3 | 6.7 |
| packed | 4.0 | 4.3 | 7.4 |
| map64 | 4.7 | 3.8 | 6.6 |
| map48 (sentinel) | 6.7 | 10.1 | 13.6 |

## Map mode (VW=1): TCP connection map benchmark

Map-mode benchmark with 2M connections, VW=1 (8B value per key), two
workloads: 90/5/5 (packet processing) and 50/25/25 (high churn). PF sweep
16–64 with prefetch pipelined ops. Direct-compare variants only — sentinel
and bitstealing provide no benefit for 48-bit keys (h2 metadata filtering
is pointless when the entire key fits in a single SIMD comparison).

Entry sizes with VW=1: map64 = 128B (2 CL), split/packed = 144B (3 CL),
map48 = 256B+ (multi-CL metadata groups).

### Results (ns/op, best PF per variant)

| Variant | 90/5/5 | PF | 50/25/25 | PF | Alloc (MiB) | B/entry |
|---------|--------|----|----------|----|-------------|---------|
| map64 | **3.0** | 64 | **6.7** | 64 | 128 | 67.1 |
| split N=1 | 5.2 | 64 | 12.2 | 24 | 72 | 37.7 |
| map48 | 5.7 | 24 | 6.9 | 40 | 128 | 67.1 |
| packed N=1 | 5.8 | 64 | 12.1 | 24 | 72 | 37.7 |
| packed N=2 | 15.1 | 32 | 18.7 | 32 | 72 | 37.7 |
| split N=2 | 20.5 | 32 | 22.8 | 24 | 72 | 37.7 |

### Map64 wins both workloads

Map64 is the fastest map-mode container for 48-bit keys. 128B entries (2 CL:
64B keys + 64B values) give a clean power-of-2 stride. The HW adjacent-line
prefetcher pulls the value CL for free when the key CL is fetched. Only 1
fill buffer entry per operation despite touching 2 CLs.

The 2× memory cost (128 vs 72 MiB) is the tradeoff. For the TCP use case
at 2M connections, 128 MiB is acceptable.

### Split/packed 50/25/25 churn regression: 3 CLs per entry

Split/packed with VW=1 have 144B entries: 64B key block + 80B value block
(10 slots × 8B). This spans 3 cache lines. On 90/5/5 (read-heavy), the
third CL (value bytes 64–79, covering only slots 8–9) rarely matters —
most lookups resolve without touching it. But on 50/25/25, 50% of ops are
writes (insert + delete) that must touch the value region via
write-allocate, and the extra CL adds measurable fill buffer pressure.

Result: split/packed at 12 ns vs map64 at 7 ns — a ~70% churn regression
entirely explained by the extra cache line per write operation.

### Superblock layout (N=2): strictly worse

The hypothesis: separating keys (N×64B stride) from values (N×80B stride)
via superblock layout would restore HW-prefetcher-friendly access patterns
and fix the churn regression.

The result: 3–4× worse than inline. The hypothesis was wrong.

**Root cause: HW adjacent-line prefetch.** With inline layout (N=1), the
key CL at offset 0 and value CL at offset 64 of each 144B entry are
physically adjacent. Intel's L2 adjacent-line prefetcher (ACP) automatically
pulls the value CL when the key CL is fetched — effectively making the SW
prefetch of the value CL a no-op (already in L1). This means inline layout
consumes only 1 fill buffer entry per operation despite touching 2 CLs.

With superblock layout (N=2), key CLs are grouped together and value CLs
are 128+ bytes away — no longer adjacent. Every value access is a true L1
miss requiring its own fill buffer entry. At PF=24 with 2 non-adjacent
prefetches per op, that's 48 outstanding requests vs 12 fill buffer
entries — 4× oversubscribed. The theoretical minimum latency with 2
non-adjacent lines is ~17 ns/op (fill buffer throughput = 12 entries /
~100 ns DRAM latency = 0.12 lines/ns; 2 lines/op → 1 op per 17 ns).
Measured: 15–24 ns. Matches.

This is the same effect seen in the set-mode 2CL/20 variant above: ACP
cannot help when cache lines are not physically adjacent in the access
stream. The difference is that set-mode has no value CLs to worry about,
while map-mode's extra value CL is the critical factor.

### Map48 (sentinel): competitive on churn, expensive on reads

Map48 achieves 6.9 ns on 50/25/25 — nearly matching map64 (6.7 ns).
Its h2 metadata approach means delete is a single bit-clear with no
backshift, which helps churn workloads. But its multi-CL metadata groups
(256B+) cost more on read-heavy workloads: 5.7 ns vs map64's 3.0 ns.

### Conclusion (VW=1)

**Map64 is the fastest VW=1 map at PF=64**, but only because its 128B entry
aligns perfectly with ACP. At larger value sizes (VW≥2) this advantage
vanishes — see TCP state benchmark below.

## Map mode (VW=1,2,4): TCP connection state benchmark

Map-mode benchmark with 2M connections at VW=1 (8B: counter), VW=2 (16B:
timestamp + flags), VW=4 (32B: full connection state). PF=24, 4M ops per
workload. Compares map64, split, and lem+bs (the two Pareto-optimal 48-bit
direct-compare headers).

Entry sizes:
- map64: 128B/192B/320B (VW=1/2/4) — 75% LF, pow2 groups
- split: 144B/224B/384B (VW=1/2/4) — 87.5% LF, pow2 groups
- lembs: 144B/224B/384B (VW=1/2/4) — 87.5% LF, non-pow2 groups

### Results (ns/op, median of 3 runs, AVX2, PF=24)

**90/5/5 (read-heavy, packet processing)**

| Variant | VW=1 | VW=2 | VW=4 | MiB (1) | MiB (2) | MiB (4) |
|---------|------|------|------|---------|---------|---------|
| map64 | 9.5 | 13.9 | 20.8 | 128 | 192 | 320 |
| split | 9.2 | 9.6 | **9.1** | 72 | 112 | 192 |
| **lembs** | **8.8** | **9.3** | 9.2 | **64** | **98** | **168** |

**50/25/25 (high churn)**

| Variant | VW=1 | VW=2 | VW=4 | MiB (1) | MiB (2) | MiB (4) |
|---------|------|------|------|---------|---------|---------|
| map64 | **10.7** | 15.4 | 22.7 | 128 | 192 | 320 |
| split | 12.1 | **12.6** | **13.0** | 72 | 112 | 192 |
| lembs | 12.3 | 13.0 | 13.7 | **64** | **98** | **168** |

### Map64 collapses at VW≥2

Map64's VW=1 advantage (128B = 2 adjacent CLs, ACP pulls value CL free) does
not scale. At VW=2, the entry grows to 192B (3 CLs) — the third CL is no
longer adjacent to the first, ACP cannot help, and every value access is a
true L1 miss. At VW=4, the entry is 320B (5 CLs) and map64 is 2.3× slower
than split/lembs (20.8 vs 9.1 ns on 90/5/5).

Map64 also wastes the most memory due to 75% LF: VW=4 uses 320 MiB vs
lembs's 168 MiB (47% less).

### Split and lembs: same speed, lembs wins memory

Both use the same 1CL key block with inline values. The only difference is
group addressing: pow2 mask (split) vs Lemire (lembs). At VW=1-4, the
performance difference is within noise (<5%). But lembs consistently uses
11-14% less memory from non-pow2 group counts:
- VW=1: 64 vs 72 MiB (11%)
- VW=2: 98 vs 112 MiB (12.5%)
- VW=4: 168 vs 192 MiB (12.5%)

### VW-insensitivity of split/lembs

Remarkably, split and lembs show almost no speed degradation from VW=1 to
VW=4 on 90/5/5 (9.2→9.1 and 8.8→9.2 ns). This is because 90% of ops are
reads (get), and the value CL is prefetched alongside the key CL via
`_prefetch()`. The hardware prefetcher and ACP bring additional value CLs for
inline layouts. On 50/25/25, the 25% inserts cause write-allocate traffic to
value CLs, adding ~1 ns per VW doubling — a gentle slope compared to map64's
steep 10→23 ns climb.

### Conclusion

**Lembs is the optimal map-mode container for 48-bit keys at VW≥2.** It
matches split's speed while using 11-14% less memory. At VW=4 (32B
connection state), lembs is 2.3× faster than map64 at 47% less memory
(168 vs 320 MiB). For memory-constrained deployments at any VW, lembs is
the best choice. Map64 is only competitive at VW=1 on churn workloads
(10.7 vs 12.3 ns) where its 2-CL ACP advantage still holds.
