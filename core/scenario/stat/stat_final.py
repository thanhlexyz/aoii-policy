import numpy as np
import pandas as pd
import glob
import os

import util

N_TAIL = 1000

def tail_metrics(path, n_irsa_slot):
    df   = pd.read_csv(path)
    tail = df.tail(N_TAIL)
    return {'n_step'     : len(df),
            'avg_aoii'   : tail['avg_aoii'].mean(),
            'max_aoii'   : tail['max_aoii'].max(),
            'utilization': tail['n_send'].mean() / n_irsa_slot,
            'p_success'  : tail['n_success'].sum() / max(tail['n_send'].sum(), 1),
            'reward'     : tail['reward'].mean()}

def stat_final(args):
    rows = []
    for dataset in util.DATASETS:
        for solver in util.SOLVERS:
            pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_{solver}_s[0-9]*.csv')
            for path in sorted(glob.glob(pattern)):
                seed = int(path.rsplit('_s', 1)[1].split('.')[0])
                rows.append({'dataset': dataset, 'solver': solver, 'seed': seed,
                             **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print(f'[!] no runs under {os.path.join(args.csv_dir, args.mode)}')
        return
    df   = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'summary_per_seed.csv')
    df.to_csv(path, index=None, float_format='%.6f')
    agg = (df.groupby(['dataset', 'solver'], sort=False)
             .agg(n_seed=('seed', 'count'), n_step=('n_step', 'max'),
                  avg_aoii=('avg_aoii', 'mean'), avg_aoii_std=('avg_aoii', 'std'),
                  avg_aoii_median=('avg_aoii', 'median'), avg_aoii_max=('avg_aoii', 'max'),
                  utilization=('utilization', 'mean'), utilization_std=('utilization', 'std'),
                  p_success=('p_success', 'mean'),
                  reward=('reward', 'mean'), reward_std=('reward', 'std'))
             .reset_index())
    out = os.path.join(args.csv_dir, 'summary.csv')
    agg.to_csv(out, index=None, float_format='%.6f')
    signed_rank(df, args)
    main_table(agg, df, args)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False))
    print(f'[+] wrote {out} and {path}')

TABLE_ORDER = ['reactive', 'random', 'hybrid', 'threshold', 'ucb', 'vhct', 'mpgts']
NAME  = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
LABEL = {'reactive': r'\textsc{Reactive}', 'random': r'\textsc{Throughput}', 'hybrid': r'\textsc{Hybrid}',
         'threshold': r'\textsc{Threshold}', 'ucb': r'\textsc{UCB}', 'vhct': r'\textsc{VHCT}',
         'mpgts': r'\tool (ours)'}

def fmt(m, sd, digits):
    return f'${m:.{digits}f} \\pm {sd:.{digits}f}$'

def fmt_aoii(m, sd):
    if m >= 10:  return f'${m:.0f} \\pm {sd:.0f}$'
    if m >= 1:   return f'${m:.1f} \\pm {sd:.1f}$'
    return fmt(m, sd, 3) if m >= 0.01 else fmt(m, sd, 4)

ALPHA = 0.01

def signed_rank_p(d, a, b, metric):
    from scipy import stats
    x = d[d.solver == a].set_index('seed')[metric]
    y = d[d.solver == b].set_index('seed')[metric]
    x, y = x.align(y, join='inner')
    if len(x) < 2 or (x == y).all():
        return float('nan')
    return stats.wilcoxon(x, y, alternative='two-sided').pvalue

def seeds_in_words(df):
    n = df.groupby(['dataset', 'solver']).seed.count()
    lo, hi = int(n.min()), int(n.max())
    words = {5: 'five', 10: 'ten', 15: 'fifteen', 20: 'twenty'}
    return words.get(hi, str(hi)) if lo == hi else f'{lo} to {hi}'

SUPPLEMENT_BITS = 16 + 7

def signaling(args):
    import math
    from simulator.solver.mpgts.learner import grids
    ts_bits = math.ceil(math.log2(int(np.prod([len(g) for g in grids(args)]))))
    none = (0, 'none')
    return {'reactive': none, 'threshold': none, 'random': none, 'hybrid': none,
            'ucb': (SUPPLEMENT_BITS, 'unicast'), 'vhct': (SUPPLEMENT_BITS, 'unicast'),
            'mpgts': (ts_bits, 'broadcast')}

def main_table(agg, df, args):
    datasets = [d for d in util.DATASETS if d in set(agg.dataset)]
    aoii = agg.pivot(index='solver', columns='dataset', values='avg_aoii')
    std  = agg.pivot(index='solver', columns='dataset', values='avg_aoii_std')
    solvers = [s for s in TABLE_ORDER if s in aoii.index]
    order = aoii.loc[solvers, datasets].rank().mean(axis=1).sort_values(ascending=False).index
    sig = signaling(args)
    rows = []
    for solver in order:
        cells = []
        for dataset in datasets:
            cell = fmt_aoii(aoii.loc[solver, dataset], std.loc[solver, dataset]).replace(r' \pm ', r'{\pm}')
            if aoii.loc[solvers, dataset].idxmin() == solver:
                cell = rf'$\mathbf{{{cell[1:-1]}}}$'
            cells.append(cell)
        bits, form = sig[solver]
        rows.append(f'      {LABEL[solver]} & ' + ' & '.join(cells) + f' & {bits} & {form}' + r' \\')
    if 'mpgts' in aoii.index:
        cuts = []
        for dataset in datasets:
            d = df[df.dataset == dataset]
            others = d[d.solver != 'mpgts'].groupby('solver').avg_aoii.mean()
            best = others.idxmin()
            cut  = (others[best] - d[d.solver == 'mpgts'].avg_aoii.mean()) / others[best] * 100
            star = r'$^{*}$' if signed_rank_p(d, 'mpgts', best, 'avg_aoii') < ALPHA else ''
            cuts.append(f'{cut:.0f}\\%{star}')
        rows.append(r'      \midrule')
        rows.append(r'      \textit{AoII reduction} & ' + ' & '.join(cuts) + r' & & \\')
        m = agg[agg.solver == 'mpgts']
        ack = m.utilization * args.n_irsa_slot / args.n_sensor * m.p_success
        print(f'[i] per-packet ACK at the load of the proposed policy: '
              f'{ack.min():.3f}-{ack.max():.3f} bit per sensor per slot')
    n = len(datasets)
    tex = r"""\newcommand{\maintable}{%
  \begin{table*}[t]
    \centering
    \caption{Steady-state AoII, mean $\pm$ standard deviation over """ + seeds_in_words(df) + r""" seeds; bold: lowest per trace; $*$: $p < """ + f'{ALPHA:g}' + r"""$ against the best baseline (Wilcoxon signed-rank test paired by seed).}
    \label{tab:main}
    \footnotesize
    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l""" + 'r' * n + r"""cc@{}}
      \toprule
      & \multicolumn{""" + f'{n}' + r"""}{c}{\textbf{Steady-state AoII}} & \multicolumn{2}{c}{\textbf{\shortstack{Downlink signaling\\per sensor and round}}} \\
      \cmidrule(lr){2-""" + f'{n + 1}' + r"""} \cmidrule(l){""" + f'{n + 2}-{n + 3}' + r"""}
      \textbf{Policy} & """ + ' & '.join(rf'\texttt{{{NAME[d]}}}' for d in datasets) + r""" & bits & form \\
      \midrule
""" + '\n'.join(rows) + r"""
      \bottomrule
    \end{tabular*}
  \end{table*}%
}
"""
    out = os.path.join(args.figure_dir, 'tab-main.tex')
    with open(out, 'w') as fp:
        fp.write(tex)
    print(f'[+] wrote {out}')


def signed_rank(df, args):
    rows = []
    for dataset in util.DATASETS:
        d = df[df.dataset == dataset]
        v = d[d.solver == 'mpgts']
        others = [s for s in util.SOLVERS if s != 'mpgts' and (d.solver == s).sum() > 1]
        if len(v) < 2 or not others: continue
        means = d[d.solver.isin(others)].groupby('solver')[['avg_aoii', 'reward']].mean()
        for solver in others:
            o = d[d.solver == solver]
            for metric, best in [('avg_aoii', means.avg_aoii.idxmin()), ('reward', means.reward.idxmax())]:
                pval = signed_rank_p(d, 'mpgts', solver, metric)
                rows.append({'dataset': dataset, 'vs': solver, 'metric': metric, 'n_seed': len(v),
                             'mpgts_mean': v[metric].mean(), 'other_mean': o[metric].mean(),
                             'p': pval, 'best_baseline': solver == best,
                             'significant': pval < ALPHA, 'bonferroni': pval < ALPHA / len(others)})
    out = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'wilcoxon.csv')
    out.to_csv(path, index=None, float_format='%.4g')
    print(out.to_string(index=False)); print(f'[+] wrote {path}')
