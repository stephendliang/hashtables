#!/usr/bin/env python3
"""
ANOVA analysis of KV layout benchmark results.

Segregated analysis: sentinel and bitstealing are analyzed independently.
Within each overflow scheme, layouts compete against each other.

Usage:
    python3 analyze_kv_bench.py /tmp/bench_kv_run*.tsv
"""

import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats


def load_runs(files):
    frames = []
    for i, f in enumerate(sorted(files)):
        df = pd.read_csv(f, sep='\t', comment='#',
                         names=['layout', 'overflow', 'blockN', 'PF',
                                'operation', 'ns_per_op'])
        df['run'] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def variant_label(row):
    if row['layout'] == 'inline':
        return 'inline'
    elif row['layout'] == 'separate':
        return 'separate'
    else:
        return f"hybrid_n{row['blockN']}"


def overflow_label(row):
    return 'sentinel' if 'sent' in row['overflow'] else 'bitstealing'


def best_pf_per_variant(df):
    pf_ops = ['insert', 'get_hit', 'get_miss', 'delete']
    churn_ops = ['churn_del', 'churn_ins', 'post_churn_hit']

    pf_rows = df[df['operation'].isin(pf_ops)].copy()
    churn_rows = df[df['operation'].isin(churn_ops)].copy()

    best_pf = (pf_rows.groupby(['ovf_scheme', 'variant', 'operation', 'PF'])['ns_per_op']
               .median()
               .reset_index()
               .sort_values('ns_per_op')
               .drop_duplicates(subset=['ovf_scheme', 'variant', 'operation'], keep='first'))

    result_rows = []
    for _, bp in best_pf.iterrows():
        mask = ((pf_rows['ovf_scheme'] == bp['ovf_scheme']) &
                (pf_rows['variant'] == bp['variant']) &
                (pf_rows['operation'] == bp['operation']) &
                (pf_rows['PF'] == bp['best_PF' if 'best_PF' in bp.index else 'PF']))
        result_rows.append(pf_rows[mask])

    pf_best = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    return pd.concat([pf_best, churn_rows], ignore_index=True)


def anova_and_ranking(df, operation, scheme):
    sub = df[(df['operation'] == operation) & (df['ovf_scheme'] == scheme)].copy()
    if sub.empty:
        return None

    variants = sorted(sub['variant'].unique())
    groups = [sub[sub['variant'] == v]['ns_per_op'].values for v in variants]

    valid = [(v, g) for v, g in zip(variants, groups) if len(g) >= 2]
    if len(valid) < 2:
        return None

    variants_v = [v for v, _ in valid]
    groups_v = [g for _, g in valid]

    f_stat, p_value = stats.f_oneway(*groups_v)

    grand_mean = np.concatenate(groups_v).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_v)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups_v)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    summary = []
    for v, g in zip(variants_v, groups_v):
        summary.append({
            'variant': v,
            'mean': g.mean(),
            'std': g.std(ddof=1),
            'median': np.median(g),
            'min': g.min(),
            'max': g.max(),
            'n': len(g),
        })
    summary.sort(key=lambda x: x['median'])

    n_pairs = len(variants_v) * (len(variants_v) - 1) // 2
    pairwise = []
    for i in range(len(variants_v)):
        for j in range(i + 1, len(variants_v)):
            t, p = stats.ttest_ind(groups_v[i], groups_v[j])
            p_adj = min(p * n_pairs, 1.0)
            diff = groups_v[i].mean() - groups_v[j].mean()
            pairwise.append({
                'a': variants_v[i],
                'b': variants_v[j],
                'diff_ns': diff,
                'p_adj': p_adj,
                'sig': '***' if p_adj < 0.001 else '**' if p_adj < 0.01
                       else '*' if p_adj < 0.05 else 'ns',
            })

    return {
        'operation': operation,
        'scheme': scheme,
        'F': f_stat,
        'p': p_value,
        'eta_sq': eta_sq,
        'summary': summary,
        'pairwise': pairwise,
    }


def print_results(result):
    op = result['operation']
    print(f"\n  {op.upper()}")
    print(f"  {'─' * 66}")
    print(f"  ANOVA: F={result['F']:.2f}, p={result['p']:.2e}, "
          f"η²={result['eta_sq']:.3f}", end='')
    if result['p'] < 0.001:
        print(f"  *** highly significant")
    elif result['p'] < 0.05:
        print(f"  * significant")
    else:
        print(f"  n.s.")

    print(f"\n  {'Rank':<5} {'Layout':<20} {'Median':>8} {'Mean':>8} "
          f"{'Std':>7} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for rank, s in enumerate(result['summary'], 1):
        print(f"  {rank:<5} {s['variant']:<20} {s['median']:>8.1f} "
              f"{s['mean']:>8.1f} {s['std']:>7.2f} {s['min']:>8.1f} "
              f"{s['max']:>8.1f}")

    sig_pairs = [p for p in result['pairwise'] if p['sig'] != 'ns']
    if sig_pairs:
        print(f"\n  Significant pairwise (Bonferroni):")
        sig_pairs.sort(key=lambda x: x['p_adj'])
        for p in sig_pairs[:10]:
            faster = p['a'] if p['diff_ns'] < 0 else p['b']
            slower = p['b'] if p['diff_ns'] < 0 else p['a']
            print(f"    {faster:<20} < {slower:<20} "
                  f"Δ{abs(p['diff_ns']):>5.1f} ns  "
                  f"p={p['p_adj']:.2e} {p['sig']}")


def overall_ranking(all_results, scheme):
    print(f"\n  {'━' * 66}")
    print(f"  OVERALL RANKING — {scheme.upper()}")
    print(f"  {'━' * 66}")

    variants_set = set()
    op_data = {}
    for r in all_results:
        if r is None:
            continue
        op = r['operation']
        op_data[op] = {}
        for s in r['summary']:
            op_data[op][s['variant']] = s['median']
            variants_set.add(s['variant'])

    if not op_data:
        print("  No data.")
        return

    scores = {v: [] for v in variants_set}
    for op, medians in op_data.items():
        vals = list(medians.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi > lo else 1.0
        for v in variants_set:
            if v in medians:
                scores[v].append((medians[v] - lo) / rng)

    avg_scores = []
    for v in variants_set:
        if scores[v]:
            avg_scores.append((v, np.mean(scores[v]), len(scores[v])))
    avg_scores.sort(key=lambda x: x[1])

    print(f"\n  {'Rank':<5} {'Layout':<20} {'NormScore':>10}")
    print(f"  {'-'*5} {'-'*20} {'-'*10}")
    for rank, (v, sc, nops) in enumerate(avg_scores, 1):
        bar = '█' * int(sc * 30) if sc > 0.01 else ''
        print(f"  {rank:<5} {v:<20} {sc:>10.3f}  {bar}")


def best_pf_table(df, scheme):
    pf_ops = ['insert', 'get_hit', 'get_miss', 'delete']
    sub = df[(df['ovf_scheme'] == scheme) & (df['operation'].isin(pf_ops))].copy()
    if sub.empty:
        return

    best = (sub.groupby(['variant', 'operation', 'PF'])['ns_per_op']
            .median()
            .reset_index()
            .sort_values('ns_per_op')
            .drop_duplicates(subset=['variant', 'operation'], keep='first')
            .sort_values(['operation', 'ns_per_op']))

    print(f"\n  BEST PF PER LAYOUT")
    print(f"  {'─' * 50}")
    for op in pf_ops:
        s = best[best['operation'] == op]
        print(f"\n  {op}:")
        for _, row in s.iterrows():
            print(f"    {row['variant']:<20} PF={int(row['PF']):>2}  "
                  f"{row['ns_per_op']:>6.1f} ns/op")


def analyze_scheme(df_all, df_best, scheme):
    print(f"\n{'=' * 72}")
    print(f"{'=' * 72}")
    print(f"  {scheme.upper()}")
    print(f"{'=' * 72}")
    print(f"{'=' * 72}")

    ops = sorted(df_best[df_best['ovf_scheme'] == scheme]['operation'].unique())
    all_results = []
    for op in ops:
        result = anova_and_ranking(df_best, op, scheme)
        if result:
            print_results(result)
            all_results.append(result)

    overall_ranking(all_results, scheme)
    best_pf_table(df_all, scheme)


def main():
    if len(sys.argv) < 2:
        files = sorted(glob.glob('/tmp/bench_kv_run*.tsv'))
        if not files:
            print("Usage: python3 analyze_kv_bench.py /tmp/bench_kv_run*.tsv")
            sys.exit(1)
    else:
        files = sys.argv[1:]

    print(f"Loading {len(files)} benchmark runs...")
    df = load_runs(files)
    print(f"  Total rows: {len(df)}")

    df['variant'] = df.apply(variant_label, axis=1)
    df['ovf_scheme'] = df.apply(overflow_label, axis=1)

    df_best = best_pf_per_variant(df)
    print(f"  Rows after best-PF selection: {len(df_best)}")

    analyze_scheme(df, df_best, 'sentinel')
    analyze_scheme(df, df_best, 'bitstealing')


if __name__ == '__main__':
    main()
