# KV Map vs boost::unordered_flat_map Benchmark

## Setup

- **CPU**: i9-12900HK (Alder Lake), AVX2 + BMI2
- **Key**: uint64_t[2] (128-bit), **Value**: uint64_t (64-bit)
- **Maps**: sentinel inline, bitstealing inline, boost::unordered_flat_map
- **N**: 2M keys, 8 runs per workload
- **Our maps**: init_cap + prefetch pipelining (PF=24), dual-mode prefetch
- **Boost**: reserve + emplace (no prefetch)

## Competitors

| Map | Metadata | Layout | Prefetch | Dup check |
|-----|----------|--------|----------|-----------|
| sentinel (inline) | 16-bit h2, 31 slots/group | interleaved KV in group | PF=24, dual-mode | insert_unique (insert-only), insert (mixed) |
| bitstealing (inline) | 11-bit h2 + 4-bit overflow, 32 slots/group | interleaved KV in group | PF=24, dual-mode | same as sentinel |
| boost::unordered_flat_map | 1-byte metadata | flat open addressing | none | always (emplace) |

## Results (8 runs, median ns/op)

### Insert-only (2M unique keys)

| Map | Median | p vs boost |
|-----|--------|------------|
| **sentinel** | **14.3** | p<0.001 *** |
| **bitstealing** | **14.4** | p<0.001 *** |
| boost | 24.4 | — |

Our maps beat boost by **41%** on pure inserts.

### Insert-and-read (500K init, 2M mixed ops)

| Workload | sentinel | bitstealing | boost | Winner |
|----------|----------|-------------|-------|--------|
| rw_95_5 (95% ins, 5% read) | 17.6 | **17.4** | 19.2 | SIMD (9% faster) |
| rw_75_25 | **24.5** | 25.0 | 29.2 | sentinel (16% faster) |
| rw_50_50 | 29.4 | **29.2** | 34.6 | SIMD (16% faster) |
| rw_25_75 | **30.1** | 30.8 | 34.0 | sentinel (11% faster) |
| rw_5_95 (5% ins, 95% read) | 28.0 | **27.6** | 30.8 | SIMD (10% faster) |

**Our maps win every insert/read ratio.** All pairwise comparisons are
p<0.001 or p<0.01. The margin ranges from 9% (rw_95_5) to 16% (rw_75_25,
rw_50_50).

### Churn (1M init, 2M mixed ops with read + insert + delete)

| Workload | sentinel | bitstealing | boost | Winner |
|----------|----------|-------------|-------|--------|
| churn_80_10_10 | 32.0 | **31.5** | 40.2 | SIMD (22% faster) |
| churn_50_25_25 | **30.7** | 31.0 | 42.1 | sentinel (27% faster) |
| churn_33_33_34 | 30.2 | **29.6** | 42.5 | SIMD (30% faster) |
| churn_20_40_40 | **27.4** | 27.7 | 41.9 | sentinel (35% faster) |

**Our maps dominate all churn workloads by 22-35%.** All pairwise differences
are p<0.001. The churn advantage is even larger than insert/read because
delete-heavy mixes benefit from metadata-only prefetch (delete only needs the
metadata line to clear a slot).

Sentinel and bitstealing remain statistically indistinguishable across all
churn profiles (no significant pairwise difference).

## Overall ranking

| Rank | Map | Normalized score |
|------|-----|-----------------|
| 1 | sentinel | 0.032 |
| 2 | bitstealing | 0.035 |
| 3 | boost | 1.000 |

Normalized score: per-workload latency normalized to [0,1] range (0=best,
1=worst), averaged across all 10 workloads. Lower is better.

## Prefetch analysis: why dual-mode prefetch matters

### The problem with uniform 5-line prefetch

The original `_prefetch()` issues 5 cache line prefetches per call (metadata +
4 key/value lines, offsets 0/64/128/192/256). At PF=24, this creates
24×5=120 outstanding prefetch requests. The Golden Cove L1 data cache has
~12 fill buffer entries. Result: **72% of cycles stalled on prefetch
instructions** waiting for fill buffer slots.

perf annotation (before fix):
```
25.3%  prefetcht0 0x40(%rax)    # key line 1
15.1%  prefetcht0 0x80(%rax)    # key line 2
16.9%  prefetcht0 0xc0(%rax)    # key line 3
15.1%  prefetcht0 0x100(%rax)   # key line 4
10.3%  jmp (loop back)
```

For insert_unique, the 4 key/value prefetches are wasted: the insert path
only READS the metadata line (to find an empty slot via SIMD). Key/value
writes go through write-allocate in the store buffer — they don't need to be
prefetched.

### The fix: `_prefetch_insert()`

New lightweight prefetch for insert/delete paths: metadata line only (1
prefetch instead of 5). The benchmark dispatches based on `op_type[i+pf]`:
- `OP_GET` → `_prefetch()` (5 lines: metadata + key data for h2 verify)
- `OP_INSERT` / `OP_DELETE` → `_prefetch_insert()` (1 line: metadata only)

### Hardware counter comparison (insert_only, pinned P-core)

| Metric | Before (5-line) | After (1-line) | Change |
|--------|----------------|----------------|--------|
| ns/op | 26.1 | 13.9 | **-47%** |
| cycles | 390M | 264M | -32% |
| IPC | 0.85 | 1.20 | +41% |
| L1d loads | 74.6M | 66.6M | -11% |
| L1d misses | 660K | 212K | -68% |
| L1d stores | 32.5M | 32.5M | unchanged |
| topdown BE-bound | 84.7% | — | reduced |

The L1d miss DECREASE (660K→212K) confirms that unnecessary prefetches were
evicting useful data from L1 (cache pollution). With metadata-only prefetch,
the working set fits better and the hardware prefetcher handles spatial
locality for the key/value writes.

## Methodology

- 8 independent runs, ANOVA with Bonferroni-corrected pairwise t-tests
- Same workload (identical key sequences, op sequences) for all 3 maps
- Workload generator simulates map state to ensure valid op sequences
- Our maps: init_cap(n) + insert_unique for pre-population, insert/get/delete
  with dual-mode prefetch pipeline for timed phase
- Boost: reserve(n) + emplace for pre-population, emplace/find/erase for
  timed phase
- Timing: clock_gettime(CLOCK_MONOTONIC), timed phase only (excludes setup)
