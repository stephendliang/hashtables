# When the bottleneck is DRAM, the compiler doesn't matter

Tested GCC 15, Clang 21, and ICX 2025 on the same benchmark
(`bench_map128_throughput.c`, 2M keys, AVX2, i9-12900HK). All three
compilers produce wall-clock identical results — the spread is <1 ns/op,
within run-to-run noise.

## Sentinel (ns/op, 2-run average)

| Operation  | GCC 15 | Clang 21 | ICX 2025 |
|------------|--------|----------|----------|
| insert     |   19.7 |     19.3 |     20.2 |
| hit        |   18.1 |     18.3 |     18.1 |
| miss       |   17.8 |     16.7 |     17.4 |
| churn del  |   19.4 |     19.6 |     19.9 |
| churn ins  |   19.9 |     19.6 |     19.5 |
| delete all |   18.7 |     18.7 |     18.7 |

## Bitstealing (ns/op, 2-run average)

| Operation  | GCC 15 | Clang 21 | ICX 2025 |
|------------|--------|----------|----------|
| insert     |   19.9 |     20.0 |     19.6 |
| hit        |   18.1 |     18.5 |     18.2 |
| miss       |   18.1 |     17.1 |     17.6 |
| churn del  |   21.4 |     20.0 |     19.5 |
| churn ins  |   21.7 |     20.1 |     19.8 |
| delete all |   18.9 |     19.2 |     19.0 |

## Why

The hot path is 4-5 AVX2 intrinsics that map 1:1 to hardware instructions.
There is no room for the compiler to be clever or dumb — `vpcmpeqw`,
`vpmovmskb`, `pext`, and a branch is the entire inner loop. All three
compilers emit structurally identical assembly.

The topdown breakdown confirms this:

- **83-84% backend-bound**, 40% memory-bound
- **14% retiring** — the actual instruction work
- **0.74-0.89 IPC** — low because the core is stalling on cache misses
- **~15% L1d miss rate**, ~20% of those go to DRAM
- **Working set ~40MB** vs L2 cache 1.25MB — no amount of instruction
  optimization changes the fact that data lives in DRAM

Each lookup touches 1 cache line (metadata) for a miss, 2 cache lines
(metadata + key) for a hit. At 2M keys with 576B groups, the working set
is 300-400x larger than L2. Prefetch pipelining (PF=24) hides what it can,
but the core still spends most of its time waiting.

## Proof: the assembly is the same

Disassembling the sentinel contains-hit probe loop from GCC and ICX confirms
they emit the same instruction sequence. The core SIMD path is identical:

**GCC 15** (probe inner loop):
```asm
crc32  %r8,%rax            ; hash klo
mov    %eax,%r11d
and    %eax,%r10d          ; group index = hash & mask
shr    $0x11,%eax          ; h2 = hash >> 17
or     $0x8000,%ax         ; h2 |= OCC_BIT
crc32  %r9,%r11            ; hash khi -> overflow partition
vmovd  %eax,%xmm0
and    $0xf,%r11d          ; partition = hash_b & 15
mov    $0x1,%eax
vpbroadcastw %xmm0,%ymm0  ; broadcast h2 to all 16 lanes
shlx   %r11d,%eax,%r11d   ; overflow bit = 1 << partition
mov    %r10d,%eax
lea    (%rax,%rax,8),%rdi  ; group_addr = base + idx * 9
shl    $0x6,%rdi           ;   * 64
add    %r13,%rdi
vpcmpeqw (%rdi),%ymm0,%ymm1        ; compare lo 16 slots
vpmovmskb %ymm1,%edx               ; movemask lo
vpcmpeqw 0x20(%rdi),%ymm0,%ymm1    ; compare hi 16 slots
pext   %ebx,%edx,%edx              ; extract odd bits lo
vpmovmskb %ymm1,%eax               ; movemask hi
pext   %ebx,%eax,%eax              ; extract odd bits hi
shl    $0x10,%eax                   ; merge hi|lo
or     %edx,%eax
and    $0x7fffffff,%eax             ; mask out slot 31 (sentinel)
jne    match
jmp    check_overflow
; key verify loop
blsr   %eax,%eax                    ; clear lowest set bit
je     check_overflow
tzcnt  %eax,%esi                    ; slot index
shl    $0x4,%rdx                    ; key offset = slot * 16
lea    0x40(%rdi,%rdx,1),%rdx       ; key addr
cmp    (%rdx),%r8                   ; compare klo
jne    blsr
cmp    0x8(%rdx),%r9                ; compare khi
jne    blsr
```

**ICX 2025** (probe inner loop):
```asm
crc32  %rsi,%r8            ; hash klo
mov    %r8,%r9
crc32  %rdi,%r9            ; hash khi -> overflow partition
mov    %r8d,%r10d
shr    $0x11,%r10d         ; h2 = hash >> 17
or     $0x8000,%r10d       ; h2 |= OCC_BIT
vmovd  %r10d,%xmm0
vpbroadcastw %xmm0,%ymm0  ; broadcast h2
and    $0xf,%r9b           ; partition = hash_b & 15
mov    $0x1,%r10d
shlx   %r9d,%r10d,%r9d    ; overflow bit = 1 << partition
and    %r14d,%r8d                   ; group index & mask
lea    (%r8,%r8,8),%r11             ; group_addr = idx * 9
shl    $0x6,%r11                    ;   * 64
vpcmpeqw (%rdx,%r11,1),%ymm0,%ymm1        ; compare lo 16 slots
vpcmpeqw 0x20(%rdx,%r11,1),%ymm0,%ymm2    ; compare hi 16 slots
vpmovmskb %ymm1,%r10d                      ; movemask lo
pext   %ecx,%r10d,%r12d                    ; extract odd bits lo
vpmovmskb %ymm2,%r10d                      ; movemask hi
pext   %ecx,%r10d,%r10d                    ; extract odd bits hi
shl    $0x10,%r10d                          ; merge hi|lo
or     %r12d,%r10d
vmovdqu 0x30(%rdx,%r11,1),%xmm1           ; ** load sentinel early **
and    $0x7fffffff,%r10d                    ; mask out slot 31
je     overflow_check
lea    0x40(%rdx,%r11,1),%r11              ; key base
; key verify loop
blsr   %r10d,%r10d                  ; clear lowest set bit
je     overflow_check
tzcnt  %r10d,%r12d                  ; slot index
shl    $0x4,%r13d                   ; key offset
cmp    %rsi,(%r11,%r13,1)           ; compare klo
jne    blsr
shl    $0x4,%r12d
cmp    %rdi,0x8(%r11,%r12,1)        ; compare khi
jne    blsr
```

**What differs:**

1. **Register allocation** — completely different register names, same count.
   Neither spills to stack in the hot loop.
2. **Instruction ordering** — GCC computes the second `crc32` (overflow
   partition) after extracting h2; ICX does both `crc32`s back-to-back.
   Both are off the critical path (OoO hides it behind memory loads).
3. **Speculative sentinel load** — ICX inserts `vmovdqu 0x30(...),%xmm1`
   to load the sentinel before checking the match mask. GCC defers it to
   the miss path. No measurable effect because the sentinel shares a cache
   line with the metadata — it's always hot.
4. **Overflow check structure** — GCC uses `jmp` to a separate block; ICX
   uses `vpextrw` inline and falls through. Same logic, different CFG.

The core SIMD sequence is **instruction-identical**: `vpcmpeqw` x2 ->
`vpmovmskb` x2 -> `pext` x2 -> `shl` -> `or` -> `and` -> branch. Same
instruction count, same dependency chain, same latency. There is nothing
for hand-written assembly to improve.

## What does help: isolated CPUs and 1GB hugepages

Re-running the same benchmark with `taskset -c 4` (pinned to an isolated
P-core, CPUs 4-11 isolated via kernel cmdline) and 16x 1GB hugepages
(`HugePages_Total: 16, Hugepagesize: 1048576 kB`) drops everything by
~1-1.5 ns/op and collapses run-to-run variance from ~1-2 ns to ~0.2-0.3 ns.

**Sentinel (ns/op, isolated CPU 4, 1GB hugepages, 2-run average):**

| Operation  | GCC 15 | Clang 21 | ICX 2025 |
|------------|--------|----------|----------|
| insert     |   18.6 |     18.5 |     19.2 |
| hit        |   17.0 |     16.9 |     18.0 |
| miss       |   16.8 |     16.3 |     16.7 |
| churn del  |   18.3 |     18.6 |     18.7 |
| churn ins  |   19.0 |     18.7 |     18.5 |
| delete all |   17.6 |     17.8 |     17.9 |

**Bitstealing (ns/op, isolated CPU 4, 1GB hugepages, 2-run average):**

| Operation  | GCC 15 | Clang 21 | ICX 2025 |
|------------|--------|----------|----------|
| insert     |   18.7 |     18.3 |     18.4 |
| hit        |   17.0 |     16.8 |     17.0 |
| miss       |   16.7 |     16.0 |     16.5 |
| churn del  |   18.4 |     18.3 |     18.3 |
| churn ins  |   18.9 |     18.6 |     18.5 |
| delete all |   17.7 |     17.9 |     18.6 |

The ~1 ns improvement comes from two sources:

1. **1GB hugepages eliminate dTLB misses entirely.** The ~40MB working set
   fits inside a single 1GB page. Previously with 2MB hugepages, dTLB
   misses cost ~3.8M per 2M-key lookup (~2 misses/op). Now: zero.
2. **CPU isolation removes scheduling jitter.** No kernel threads, no
   timers, no interrupts from other tasks landing on the benchmark core.
   This is why variance collapsed — the earlier GCC bitstealing churn
   outlier at 23 ns/op was scheduling noise that disappeared with isolation.

All three compilers remain within ~0.5 ns of each other — pure noise.
The compiler still doesn't matter. But the runtime environment does,
slightly.

## The lesson

Hand-written assembly cannot outrun DRAM. When 83% of execution time is
the backend waiting on memory, optimizing the 14% that retires instructions
changes nothing. The only wins left are:

1. **Fewer cache lines touched per operation** (layout changes)
2. **Better prefetch scheduling** (already tuned — alternatives regressed)
3. **Smaller groups** (trades capacity/collision rate for cache footprint)
4. **Runtime environment** (1GB hugepages, CPU isolation — ~1 ns/op, one-time)

The first three are data structure problems. The fourth is deployment
configuration. None are compiler problems.
