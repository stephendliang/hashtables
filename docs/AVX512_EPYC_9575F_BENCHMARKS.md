# AVX-512 Benchmark Results — AMD EPYC 9575F (Zen 5)

**CPU**: AMD EPYC 9575F 64-Core (Zen 5, Turin)
**ISA**: AVX-512 (F/BW/DQ/CD/VL/VNNI/IFMA/VBMI/VBMI2/BITALG/VPOPCNTDQ/VP2INTERSECT), AVX2, BMI2
**RAM**: 180GB DDR5
**Hugepages**: 2048 × 2MB (4GB reserved)
**Date**: 2026-03-16

---

## 1. simd_set64 — AVX-512 Backend (bench_map64_backends)

N=1M keys, 10M ops, Zipf s=1.0

| Operation | Mops/s | ns/op |
|-----------|--------|-------|
| insert | 378.0 | 2.6 |
| lookup (hit) | 850.6 | 1.2 |
| lookup (±) | 578.7 | 1.7 |
| delete | 340.7 | 2.9 |

**Mixed workloads (post-churn lookup Mops/s):**

| Profile | Mixed Mops/s | Post-churn hit | Post-churn miss |
|---------|-------------|----------------|-----------------|
| read-heavy 90/5/5 | 374.4 | 1072.8 | 661.5 |
| balanced 50/25/25 | 167.5 | 1070.9 | 552.9 |
| churn 33/33/34 | 146.9 | 1219.0 | 719.4 |
| write-heavy 10/50/40 | 108.1 | 850.6 | 50.0 |
| eviction 20/10/70 | 346.7 | 40.0 | 802.7 |

**Key takeaway**: Post-churn hit lookups reach **1.2 GHz** (< 1 ns/op) on the
EPYC. The AVX-512 backend's single-instruction group compare (`vpcmpeqq` on
zmm) fully hides in OoO memory-wait bubbles. Post-churn miss on the eviction
workload (live=2) achieves 802.7 Mops/s — nearly all probes terminate at the
first empty slot.

---

## 2. Generic Map Layout Grid (bench_kv_layout)

2M keys, KW=2, VW=1, AVX-512. Best PF per layout shown.

### Best configurations at optimal PF (PF=36–40)

| Layout | Overflow | BlockN | insert | get_hit | get_miss | delete |
|--------|----------|--------|--------|---------|----------|--------|
| inline | sentinel | - | 10.6 | 6.2 | 4.1 | 6.5 |
| separate | sentinel | - | 9.2 | 6.7 | 3.6 | 6.5 |
| hybrid | sentinel | 1 | 10.7 | 6.5 | 4.2 | 6.4 |
| inline | bitstealing | - | 9.7 | 6.3 | 3.7 | 6.6 |
| separate | bitstealing | - | 10.2 | 6.4 | 3.7 | 6.4 |
| **hybrid** | **bitstealing** | **1** | **8.5** | **6.1** | **3.8** | **6.1** |

*(All values in ns/op at PF=40 unless noted)*

### Winner: hybrid-bitstealing N=1

**Hybrid bitstealing N=1 dominates every operation** at high PF:
- Insert: **8.5 ns** (vs 9.2–10.7 for others) — 12–20% faster
- Get hit: **6.1 ns** (vs 6.2–6.7) — marginally best
- Delete: **6.1 ns** (vs 6.4–6.6) — marginally best
- Get miss: 3.8 ns (separate layouts slightly better at 3.6–3.7)

### Churn performance (PF=36–40)

| Layout | Overflow | churn_del | churn_ins | post_churn_hit |
|--------|----------|-----------|-----------|----------------|
| inline-sent | - | 6.3 | 13.7 | 10.5 |
| separate-sent | - | 6.8 | 13.7 | 11.4 |
| hybrid-sent N=1 | 1 | 6.3 | 13.2 | 10.6 |
| inline-bs | - | 6.3 | 12.1 | 10.6 |
| separate-bs | - | 6.7 | 11.7 | 11.1 |
| **hybrid-bs N=1** | 1 | **6.4** | **12.4** | **10.9** |

Bitstealing variants consistently beat sentinel on churn_ins by ~1 ns (12.1–12.4 vs 13.2–13.7).

### Hybrid block stride effect (bitstealing, PF=40)

| BlockN | insert | get_hit | get_miss | delete |
|--------|--------|---------|----------|--------|
| **1** | **8.5** | **6.1** | **3.8** | **6.1** |
| 2 | 11.1 | 7.0 | 3.7 | 7.1 |
| 4 | 11.2 | 7.3 | 3.7 | 7.2 |
| 8 | 11.2 | 7.3 | 3.8 | 7.5 |
| 16 | 12.1 | 7.4 | 3.8 | 7.2 |
| 32 | 11.5 | 7.5 | 3.8 | 7.1 |

**N=1 wins decisively** — same conclusion as the AVX2 laptop benchmarks.
N≥2 costs 2.5–3.5 ns more on insert and ~1 ns on reads. The stride=1
degenerate case (keys+values adjacent, 128B/group for VW=1) has the best
cache-line locality. This matches the i9-12900HK finding.

### PF convergence

All layouts converge by PF=36–40. Beyond PF=40, diminishing returns. The
optimal PF window on this EPYC is **36–40** for reads, **20** for
insert-only — matching the laptop pattern but shifted slightly higher,
consistent with the EPYC's deeper memory hierarchy.

---

## 3. Map64 Layout Grid (bench_kv64_layout)

2M keys, VW=1, AVX-512.

### Best results per stride at PF=64 (asymptotic)

| Stride | insert | get_hit | get_miss | delete | churn_del | churn_ins |
|--------|--------|---------|----------|--------|-----------|-----------|
| **N=1** | **4.7** | **2.3** | **3.2** | **3.3** | **7.3** | **5.6** |
| N=2 | 4.9 | 2.7 | 3.6 | 3.6 | 7.9 | 5.7 |
| N=4 | 4.7 | 2.6 | 3.5 | 3.6 | 7.2 | 5.5 |

### PF sensitivity (N=1)

| PF | insert | get_hit | get_miss | delete |
|----|--------|---------|----------|--------|
| 4 | 11.6 | 7.0 | 8.2 | 9.4 |
| 16 | 6.9 | 4.9 | 6.0 | 6.4 |
| 28 | 5.0 | 3.6 | 4.4 | 5.0 |
| 40 | 4.6 | 3.0 | 3.7 | 4.1 |
| 48 | 4.7 | 2.8 | 3.5 | 3.8 |
| 64 | 4.7 | 2.3 | 3.2 | 3.3 |

**map64 is significantly faster than the generic map** on this CPU. At PF=48+:
- Get hit: **2.3–2.8 ns** (vs 6.1 ns generic) — 2.4× faster
- Insert: **4.7 ns** (vs 8.5 ns generic) — 1.8× faster
- Delete: **3.3 ns** (vs 6.1 ns generic) — 1.8× faster

This is expected: map64 uses direct 64-bit comparison (1 AVX-512 instruction)
with no h2 metadata, touching only 1–2 cache lines per probe vs 5 for
sentinel/bitstealing. The gap is larger on this EPYC than on the laptop,
likely due to the deeper DRAM latency amplifying the cache-line count advantage.

**PF does not plateau until ~48–64** on this EPYC (vs ~24–28 on the laptop).
The EPYC's higher DRAM latency and deeper memory hierarchy benefit from
longer prefetch distances. Consider raising the default PF for map64 from 24
to 40–48 on server platforms.

---

## 4. PF Tuning + Delete Prefetch Mode (bench_kv_pf_tuning)

Sentinel inline, KW=2, VW=1, 2M keys.

### Insert-only

| PF | ns/op |
|----|-------|
| 4 | 11.69 |
| 8 | 10.07 |
| 12 | 9.29 |
| 16 | 8.41 |
| **20** | **7.98** |
| 24 | 8.23 |
| 40 | 7.90 |
| 64 | 8.06 |

Optimal insert PF: **20** (7.98 ns), with a broad flat region from 20–64.
Same optimal as the laptop.

### Churn 50/25/25 — meta vs full delete prefetch

| PF | meta (ns/op) | full (ns/op) | delta |
|----|-------------|-------------|-------|
| 4 | 29.61 | 29.88 | +0.9% |
| 12 | 26.98 | 26.54 | -1.6% |
| 24 | 26.68 | 26.27 | -1.5% |
| 32 | 27.59 | 26.32 | -4.6% |
| 48 | 27.35 | 26.43 | -3.4% |
| 64 | 27.61 | 26.30 | -4.7% |

**Surprise: full prefetch wins on the EPYC** at PF≥12, by 1.5–4.7%.
This is the **opposite** of the laptop result (where meta-only won by 6–10%).
The EPYC 9575F has more L1 fill buffer entries than the i9-12900HK, so the
5-line prefetch for deletes doesn't saturate the fill buffer. The
recommendation may need to be platform-dependent: **meta for client (12
fill entries), full for server (more fill entries)**.

---

## 5. Map48 Comparison (bench_map48)

2M keys, PF=24.

| Variant | insert | contains_hit | mixed 50/25/25 | bytes/entry |
|---------|--------|-------------|----------------|-------------|
| map48 (h2) | 5.1 | 3.8 | 6.6 | 16.8 |
| sentinel (KW=1) | 4.0 | 5.0 | 6.8 | 21.0 |
| **map64** | **3.7** | **2.7** | **5.2** | **16.8** |

**map64 wins across the board** — fastest insert (3.7 ns), fastest lookup
(2.7 ns), fastest mixed (5.2 ns), and ties map48 on memory efficiency
(16.8 B/entry). map48's h2 metadata overhead hurts more on the EPYC where
the extra instruction pressure matters less but the extra memory access
(metadata line) matters more.

Sentinel is 25% larger per entry (21.0 vs 16.8 B) and 24% slower on mixed.

---

## 6. Map48 Direct-Compare (bench_map48_direct)

2M keys, PF=24.

| Variant | insert | contains_hit | mixed 50/25/25 |
|---------|--------|-------------|----------------|
| packed (3×u16) | 3.8 | 3.9 | 6.3 |
| **split (hi32+lo16)** | **2.3** | **3.3** | **5.9** |
| map48 (h2) | 4.5 | 3.8 | 6.7 |
| **map64** | **3.7** | **2.8** | **5.2** |

**Split wins among 48-bit variants** — 2.3 ns insert is the fastest insert
of any 48-bit design, beating packed by 39% and h2-map48 by 49%.

**map64 still wins overall** on contains (2.8 vs 3.3 ns) and mixed (5.2
vs 5.9 ns). Split is competitive on insert (2.3 vs 3.7 ns — split actually
wins insert by 38%).

The split layout's `vpcmpeqd` + `vpcmpeqw` approach is more efficient than
packed's broadcast+coalesce+PEXT, especially on Zen 5 where 32-bit compares
are fast.

---

## 7. Map48 Architecture Variants (bench_map48_arch)

2M keys, PF=24. Compares all 48-bit set designs.

| Variant | Group size | Keys/grp | Load | insert | contains_hit | mixed 50/25/25 | alloc (MiB) |
|---------|-----------|----------|------|--------|-------------|----------------|-------------|
| split (hi32+lo16) | 64B (1CL) | 10 | 87.5% | 3.2 | 3.5 | 6.5 | 16 |
| **2CL/20** | **128B (2CL)** | **20** | **87.5%** | **2.3** | **3.7** | **8.2** | **16** |
| 3CL/31 | 192B (3CL) | 31 | 87.5% | 3.1 | 3.2 | 10.1 | 24 |
| backshift | 64B (1CL) | 10 | 75% | 4.5 | 4.3 | 7.0 | 32 |
| lemire | 64B (1CL) | 10 | 87.5% | 3.9 | 4.7 | 7.0 | 14 |

### EPYC vs laptop shift

On the laptop (AVX2, i9-12900HK), split dominated everything. On the EPYC:

- **2CL/20 wins insert** at 2.3 ns (28% faster than split's 3.2 ns). The
  adjacent cache line prefetch (ACP) on Zen 5 makes the 2-line group
  essentially free — the hardware fetches both lines together. This was
  a regression on the laptop.
- **3CL/31 wins contains** at 3.2 ns (9% faster than split's 3.5 ns). With
  31 slots per group and 87.5% load, fewer groups need probing. But mixed
  workload collapses to 10.1 ns — deletes across 3 cache lines are expensive.
- **Split wins mixed** at 6.5 ns — the 1-CL group keeps deletes fast and
  balanced across all operation types.
- **Backshift is worst** — 75% load factor doubles allocation (32 MiB) and
  backshift deletes are inherently slower than occupancy-bit clears.
- **Lemire saves memory** (14 MiB, 12% less than split) but is 12–34% slower
  on all operations. The non-power-of-2 modular arithmetic adds instruction
  overhead that isn't hidden by memory latency.

---

## 8. 48-bit Overall Assessment

### The full 48-bit family ranked (EPYC, set mode, PF=24)

| Rank | Variant | insert | contains | mixed | memory | verdict |
|------|---------|--------|----------|-------|--------|---------|
| 1 | **split** | 2.3–3.2 | 3.3–3.5 | 5.9–6.5 | 16 MiB | **Best all-rounder** |
| 2 | 2CL/20 | **2.3** | 3.7 | 8.2 | 16 MiB | Insert king, poor mixed |
| 3 | packed | 3.8 | 3.9 | 6.3 | 16 MiB | Decent but split is strictly better |
| 4 | 3CL/31 | 3.1 | **3.2** | 10.1 | 24 MiB | Lookup king, terrible mixed |
| 5 | map48 (h2) | 4.5–5.1 | 3.8 | 6.6–6.7 | 16.8 | Only one with map mode |
| 6 | lemire | 3.9 | 4.7 | 7.0 | **14 MiB** | Most memory-efficient |
| 7 | backshift | 4.5 | 4.3 | 7.0 | 32 MiB | 75% load wastes memory |

*(insert ns from bench_map48_direct where available, else bench_map48_arch)*

### vs map64

| | map64 (PF=24) | split (PF=24) | split advantage |
|--|--------------|---------------|-----------------|
| insert | 3.7 ns | 2.3 ns | split **38% faster** |
| contains | 2.7–2.8 ns | 3.3–3.5 ns | map64 **18–23% faster** |
| mixed | 5.2 ns | 5.9–6.5 ns | map64 **12–20% faster** |
| keys/CL | 8 | 10 | split 25% denser |
| load factor | 75% | 87.5% | split 17% higher |
| key range | full 64-bit | 48-bit only | map64 more general |
| map mode | yes (VW≥1) | **no** | map64 only |
| AVX-512 | yes (1 instr) | **no** (AVX2 only) | map64 benefits more |

**Bottom line**: Split is the fastest 48-bit set and wins insert
throughput, but map64 wins reads and mixed workloads. The gap narrows on
EPYC vs laptop. Neither packed nor split support map mode (values), so
if you need a 48-bit KV map, map48 (h2) is the only option today — and
it's the slowest 48-bit variant.

### When to use 48-bit

- **Use map64** when keys fit in 64 bits and you need map mode or read-heavy
  workloads. It's simpler, faster on lookups, and has AVX-512 support.
- **Use split** when keys are naturally 48-bit (pointers, truncated hashes),
  you only need set mode, and either (a) insert throughput is critical, or
  (b) memory pressure matters (25% more keys per cache line, 17% higher
  load factor).
- **Don't use packed** — split is strictly better on EPYC (and was already
  better on the laptop).
- **Don't use map48 (h2)** for sets — it's slower than both split and map64.
  Only use it if you need 48-bit KV map mode.

---

## 9. TCP Connection Map — Pareto Frontier (bench_tcp_pareto)

Target: IPv4 (32-bit) + src_port (16-bit) = 48-bit key → connection state
(VW=1). 2M concurrent connections, 4M ops per workload.

Memory includes 2× headroom (init_cap for 4M to avoid regrow during churn).
For target-capacity memory (no headroom), halve the MiB values.

### 90/5/5 — Packet processing (read-heavy)

| Variant | Mode | PF=16 | PF=24 | PF=32 | PF=48 | PF=64 | MiB | B/entry |
|---------|------|-------|-------|-------|-------|-------|-----|---------|
| **map64** | **map** | 6.1 | 5.4 | 4.5 | **3.5** | **3.1** | **128** | **67** |
| map48 | map | 6.9 | 5.9 | 6.2 | 6.0 | 6.2 | 128 | 67 |
| sentinel | map | 7.1 | 6.3 | 6.3 | 6.6 | 7.0 | 144 | 76 |
| bitstealing | map | 7.4 | 6.1 | 6.2 | 6.5 | 6.5 | 144 | 76 |
| split | set | 5.3 | - | - | - | 3.0 | 32 | 17 |
| packed | set | 5.9 | - | - | - | 4.0 | 32 | 17 |

**map64 dominates all map variants** — 3.1 ns at PF=64 vs 5.9+ for
everything else. It's both the smallest (128 vs 144 MiB) and fastest
map. The gap is **2×** at high PF.

The multi-CL designs (map48: 4CL, sentinel: 5CL) **cannot benefit from
high PF** — they plateau or regress beyond PF=24 because each prefetch
consumes 4–5 fill buffer entries. map64's 1-CL access keeps improving
all the way to PF=64.

Split set-mode (3.0 ns at PF=64) matches map64's speed at 4× less memory
— the clear motivation for adding VW=1 to split.

### 50/25/25 — High churn

| Variant | Mode | PF=16 | PF=24 | PF=32 | PF=48 | PF=64 | MiB | B/entry |
|---------|------|-------|-------|-------|-------|-------|-----|---------|
| **map64** | **map** | 8.7 | 7.6 | 7.2 | 7.2 | **7.0** | **128** | **67** |
| map48 | map | 8.9 | 8.1 | 7.6 | 7.3 | 7.2 | 128 | 67 |
| sentinel | map | 8.9 | 7.9 | 7.4 | 7.2 | 7.2 | 144 | 76 |
| bitstealing | map | 9.1 | 8.1 | 7.5 | 7.3 | 7.4 | 144 | 76 |
| split | set | 7.1 | - | - | - | 6.2 | 32 | 17 |
| packed | set | 7.8 | - | - | - | 6.7 | 32 | 17 |

Under high churn, **all map variants converge to 7.0–7.4 ns** at high PF.
Write operations (insert + delete = 50% of ops) equalize performance
because every variant touches similar cache-line counts on writes. map64's
read advantage shrinks from 2× to just 3%.

### Pareto frontier (map mode, best PF per variant)

```
  Memory (B/entry, with 2x headroom)
  76 |                         ● sentinel (7.2 ns)
     |                         ● bitstealing (7.3 ns)
  67 |   ● map64 (3.1 ns)      ● map48 (5.9 ns)     ← 90/5/5
     |   ● map64 (7.0 ns)      ● map48 (7.2 ns)     ← 50/25/25
     |
  17 |   ★ split-set (3.0 ns)                         ← no values (yet)
     +---+-------+-------+-------+-------+---→
         3       5       6       7       8     ns/op
```

**map64 is the only Pareto-optimal map variant** — it dominates map48
(same memory, faster), sentinel (less memory, faster), and bitstealing
(less memory, faster). There is no speed-vs-memory tradeoff among the
existing map designs; map64 wins both axes.

### The split gap: 4× memory, same speed

Split set-mode achieves map64's speed (3.0 vs 3.1 ns at PF=64) at
**4× less memory** (17 vs 67 B/entry). This is the largest inefficiency
in the current design space — map64 wastes 50 B/entry on value
infrastructure that split doesn't need (but split has no value support).

A hypothetical split-VW=1 would add ~80B per group (10 values × 8B) in
a separate mmap, totaling ~36 MiB at target capacity:
- 10 keys/CL at 87.5% load → 262K groups × 64B = 16 MiB (keys)
- 262K groups × 10 × 8B = 20 MiB (values) → 22 MiB rounded to 2MB pages
- **Total: ~38 MiB → ~20 B/entry** (vs 67 for map64)

If split-VW=1 retains split's set-mode speed, it would be **3.3×
more memory-efficient** than map64 at comparable throughput. This is
the single highest-value feature to build for the TCP use case.

### Implications for TCP connection tracking

For a typical server handling 1M–10M concurrent TCP connections:

| Connections | map64 VW=1 | split VW=1 (est.) | savings |
|-------------|-----------|-------------------|---------|
| 1M | 64 MiB | ~19 MiB | 45 MiB |
| 2M | 128 MiB | ~38 MiB | 90 MiB |
| 10M | 640 MiB | ~190 MiB | 450 MiB |

At 10M connections, the split design would save **450 MiB** — significant
for a kernel or DPDK datapath where every MB of cache and TLB coverage
matters.

---

## Summary: EPYC 9575F vs i9-12900HK

| Observation | EPYC 9575F (AVX-512) | i9-12900HK (AVX2) |
|-------------|---------------------|--------------------|
| map64 get_hit (PF=48) | **2.3 ns** | ~4–5 ns |
| map64 insert (PF=48) | **4.7 ns** | ~6–7 ns |
| Generic best (hybrid-bs N=1) | 6.1 ns hit, 8.5 ns ins | ~6.3 hit, ~9.7 ins |
| Optimal PF for map64 | 48–64 | 24–28 |
| Optimal PF for generic | 36–40 | 24–28 |
| Delete prefetch winner | **full (5-line)** | meta (1-line) |
| Best 48-bit | split > packed | (not tested on AVX2 laptop) |
| N=1 stride wins? | Yes | Yes |
| map64 vs generic gap | **2.4× on reads** | ~1.5× |

### Key differences from laptop

1. **Higher optimal PF**: The EPYC's deeper memory hierarchy (more NUMA hops,
   higher DRAM latency) benefits from PF=40–64 vs the laptop's PF=24–28.
   Default PF values should be higher for server deployments.

2. **Full delete prefetch wins on server**: The EPYC has enough fill buffer
   entries to handle 5-line prefetch on deletes without saturation. The
   meta-only recommendation is laptop-specific.

3. **map64 advantage amplified**: Direct 64-bit comparison with 1-line access
   benefits disproportionately from the EPYC's memory bandwidth. The gap
   between map64 and generic widens from ~1.5× to ~2.4× on reads.

4. **AVX-512 makes the compute path invisible**: With single-instruction
   group match (`vpcmpeqq zmm`), the SIMD hot path is completely hidden
   in memory stalls. The bottleneck is purely cache/DRAM latency.

5. **2CL/20 resurrected on Zen 5**: The 2-cache-line group design that
   regressed on the laptop is now the fastest insert (2.3 ns) on the EPYC,
   likely due to Zen 5's adjacent cache line prefetch. Platform-dependent
   architecture selection may be worthwhile.

---

## Suggestions for Further Investigation

### Performance tuning

1. **Platform-adaptive PF defaults**: The optimal PF differs significantly
   between client (24–28) and server (40–64). Consider a runtime detection
   mechanism — e.g., probe DRAM latency at init time with a calibration
   loop, or detect CPU vendor/family and select a PF table. Alternatively,
   expose a `_set_pf()` API and let the caller tune.

2. **Platform-adaptive delete prefetch mode**: Full 5-line wins on EPYC,
   meta 1-line wins on laptop. Could detect L1 fill buffer depth at build
   time (known per microarchitecture) or make it a compile-time flag. The
   fill buffer count is the deciding factor: ≥16 entries → full, ≤12 → meta.

3. **Benchmark at higher PF values for 48-bit**: All map48 benchmarks run
   at PF=24. Given that map64 doesn't plateau until PF=48–64 on this EPYC,
   the 48-bit variants may also benefit from higher PF. Re-run
   bench_map48_direct and bench_map48_arch with a PF sweep (24–64) to find
   the true optimum. Split at PF=48 could close the gap with map64.

4. **2CL/20 deserves a PF sweep on EPYC**: It won insert at PF=24 already.
   With Zen 5 ACP fetching the adjacent line for free, 2CL may have
   different PF behavior than 1CL designs. The effective prefetch cost is
   1 line, not 2.

5. **map64 at higher key counts (4M, 8M, 16M)**: All benchmarks test at
   2M keys. The EPYC has 180GB RAM and a much larger LLC. Testing at larger
   scales would reveal whether the PF=48–64 optimum holds or shifts further
   out as the working set exceeds LLC.

### Feature gaps (TCP-critical)

6. **Add map mode to split (highest priority)**: The Pareto analysis shows
   split-VW=1 would be ~3.3× more memory-efficient than map64 at comparable
   speed. Implementation plan:
   - Separate mmap for values (like sentinel layout 2): `val_data` array
     indexed by `group_index * 10 + slot`
   - `_get()` returns `uint64_t *` into val_data
   - `_insert()` takes `(key, val)`, writes key to group + value to val_data
   - Delete already works (clear occupancy bit) — just leave stale value
   - Prefetch: `_prefetch()` adds value-line prefetch for reads,
     `_prefetch_insert()` stays metadata-only
   - Estimated memory: ~20 B/entry at target capacity (vs 34 for map64)
   - This is the **single highest-value feature** for the TCP use case.

7. **AVX-512 backend for split**: Split is AVX2-only. On this EPYC,
   map64's AVX-512 advantage is significant. A `vpcmpeqd zmm` backend for
   split could match 16 hi-words in one instruction (vs 8+2 scalar today).
   The lo16 half could use `vpcmpeqw zmm` similarly. This might close the
   remaining lookup gap vs map64. Combined with suggestion 6, a split-VW=1
   with AVX-512 could dominate map64 on both speed and memory.

8. **Split with Lemire indexing**: Lemire saves 12% memory (14 vs 16 MiB)
   but is 12–34% slower due to the modular arithmetic. On EPYC where compute
   is free (hidden in memory stalls), Lemire's overhead may shrink. Worth
   testing a split+Lemire hybrid — split's layout with Lemire's non-pow2
   group count. For TCP: saves ~2 B/entry, may not be worth the complexity.

### Architecture research

9. **Adjacent Cache Line Prefetch (ACP) characterization**: 2CL/20's insert
   win on EPYC but loss on the laptop strongly suggests ACP is the cause.
   Verify with `perf stat` — look at `L1-dcache-load-misses` per op for
   1CL vs 2CL designs. If 2CL shows ~same miss count as 1CL on EPYC, ACP
   is confirmed. Intel's ACP can be disabled via MSR for A/B testing.

10. **Fill buffer saturation profiling**: The meta-vs-full delete prefetch
    reversal is attributed to fill buffer depth. Profile with
    `perf stat -e l1d_pend_miss.fb_full` (Intel) or equivalent AMD PMC to
    confirm. This would validate the "≥16 fill entries → full wins" theory.

11. **NUMA effects**: This is a 64-core EPYC — if multi-socket or if memory
    is interleaved across CCDs, NUMA placement matters. Benchmark with
    `numactl --membind=0 --cpunodebind=0` to isolate single-CCD performance
    vs the default interleaved policy. Cross-CCD latency could explain the
    higher optimal PF.

12. **Frequency scaling effects**: EPYC server CPUs may run at lower boost
    clocks than laptop parts under sustained load. Check actual frequency
    during benchmarks with `perf stat -e cycles,task-clock` to compute
    effective GHz. ns/op comparisons across platforms need frequency
    normalization to isolate microarchitectural differences.

### New designs to explore

13. **48-bit map with inline 16-bit values**: For small values (counters,
    flags, small indices), a 48-bit key + 16-bit value could fit in the
    same 64B group as split — 10 entries of (6B key + 2B value) = 80B,
    but with creative packing (keys in 60B + values in 4B of ctrl + 20B)
    it might work in 64B. Extremely memory-efficient for frequency counting.

14. **Batched / vectorized operations**: The prefetch pipeline already
    batches logically, but a true batch API (`_insert_batch(m, keys, n)`)
    could enable tighter scheduling — software pipelining with explicit
    stages rather than the current circular buffer approach.

15. **Compare map64 with Robin Hood or Swiss Table**: The current open
    addressing uses linear probing with group stepping. Robin Hood
    (variance reduction) or Swiss Table (SIMD metadata) are well-studied
    alternatives. Benchmark against abseil `flat_hash_set` and Folly
    `F14FastSet` on this EPYC to see if the prefetch pipeline approach
    beats established libraries by the same margin as on the laptop.

### TCP connection table specific

16. **Realistic churn patterns with Zipf**: The current benchmarks use
    uniform random key selection. Real TCP traffic follows Zipf
    distributions — a few connections are very hot (many packets), most
    are cold (few packets then close). Re-run bench_tcp_pareto with
    Zipf-distributed lookups (s=1.0–1.2) to see how the hot-key cache
    effect changes the Pareto frontier.

17. **Benchmark at realistic TCP scales**: Test at 100K, 500K, 1M, 5M,
    10M concurrent connections. The EPYC has a large LLC (~384 MB L3),
    so the breakpoint where working set exceeds cache is different from
    the laptop. The relative ranking may change at scale — map48's
    denser packing could win at 10M+ connections where map64's larger
    footprint causes more LLC misses.

18. **Connection timeout / bulk expiry**: TCP connections expire in
    batches (TIME_WAIT timeout). The current delete model removes one
    key at a time. A bulk-delete or scan-and-sweep API that iterates
    groups and clears expired entries would better match the TCP lifecycle.
    Profile group-scan throughput for split vs map64 (split's 10 entries
    in 1 CL is ideal for scan).

19. **VW=2 for rich connection state**: VW=1 stores a pointer (8B) or
    compact state. VW=2 (16B) could store inline connection state:
    last_seen timestamp (8B) + seq/ack numbers or flags (8B), avoiding
    a pointer chase to a separate connection struct. Re-run the Pareto
    analysis with VW=2 — the memory gap between split and map64 would
    widen further (split: ~28 B/entry, map64: ~50 B/entry).

20. **Multi-threaded partitioned design**: For DPDK/XDP, each core owns
    a partition of the connection table. Benchmark per-core throughput
    with smaller tables (N/num_cores). At 100K–200K connections per core,
    the working set fits in L2 (1.25 MB), which changes the PF calculus
    entirely — PF may be unnecessary and the multi-CL designs may catch up.
