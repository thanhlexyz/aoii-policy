import pandas as pd
import glob
import os

import util

N_TAIL = 1000

def tail_metrics(path, n_irsa_slot):
    """Steady-state figures of merit, over the last N_TAIL slots of one run."""
    df   = pd.read_csv(path)
    tail = df.tail(N_TAIL)
    return {'n_step'     : len(df),
            'avg_aoii'   : tail['avg_aoii'].mean(),
            'max_aoii'   : tail['max_aoii'].max(),
            'utilization': tail['n_send'].mean() / n_irsa_slot,
            'p_success'  : tail['n_success'].sum() / max(tail['n_send'].sum(), 1),
            'reward'     : tail['reward'].mean()}

def stat_final(args):
    """One row per (dataset, solver), aggregated over every seed on disk. This
    is the table every results claim in the paper quotes."""
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
    # mean and spread over seeds
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
    welch(df, args)
    main_table(agg, args)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False))
    print(f'[+] wrote {out} and {path}')

TABLE_ORDER = ['reactive', 'random', 'hybrid', 'threshold', 'ucb', 'vhct', 'mpgts']
NAME  = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}
LABEL = {'reactive': r'\textsc{Reactive}', 'random': r'\textsc{Throughput}', 'hybrid': r'\textsc{Hybrid}',
         'threshold': r'\textsc{Threshold}', 'ucb': r'\textsc{UCB}', 'vhct': r'\textsc{VHCT}',
         'mpgts': r'\tool (ours)'}

def fmt(m, sd, digits):
    return f'${m:.{digits}f} \\pm {sd:.{digits}f}$'

def fmt_aoii(m, sd):
    if m >= 10:  return f'${m:.0f} \\pm {sd:.0f}$'
    if m >= 1:   return f'${m:.1f} \\pm {sd:.1f}$'
    return fmt(m, sd, 3) if m >= 0.01 else fmt(m, sd, 4)

def main_table(agg, args):
    """Writes ../paper/figure/tab-main.tex, the one table RQ1 rests on:
    steady-state AoII per policy (rows) and trace (columns), the policies
    sorted from worst to best by their mean AoII rank over the traces and the
    best AoII per trace in bold."""
    datasets = [d for d in util.DATASETS if d in set(agg.dataset)]
    aoii = agg.pivot(index='solver', columns='dataset', values='avg_aoii')
    std  = agg.pivot(index='solver', columns='dataset', values='avg_aoii_std')
    solvers = [s for s in TABLE_ORDER if s in aoii.index]
    order = aoii.loc[solvers, datasets].rank().mean(axis=1).sort_values(ascending=False).index
    rows = []
    for solver in order:
        cells = []
        for dataset in datasets:
            cell = fmt_aoii(aoii.loc[solver, dataset], std.loc[solver, dataset]).replace(r' \pm ', r'{\pm}')
            if aoii.loc[solvers, dataset].idxmin() == solver: cell = rf'$\mathbf{{{cell[1:-1]}}}$'
            cells.append(cell)
        rows.append(f'      {LABEL[solver]} & ' + ' & '.join(cells) + r' \\')
    tex = r"""\newcommand{\maintable}{%
  \begin{table}[t]
    \centering
    \caption{Steady-state AoII per policy, sorted from worst to best (mean $\pm$ standard deviation over five seeds on the final 1000 slots, best in bold).}
    \label{tab:main}
    \setlength{\tabcolsep}{1pt}\scriptsize
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l""" + 'r' * len(datasets) + r"""@{}}
      \toprule
      \textbf{Policy} & """ + ' & '.join(rf'\texttt{{{NAME[d]}}}' for d in datasets) + r""" \\
      \midrule
""" + '\n'.join(rows) + r"""
      \bottomrule
    \end{tabular*}
  \end{table}%
}
"""
    out = os.path.join(args.paper_figure_dir, 'tab-main.tex')
    with open(out, 'w') as fp:
        fp.write(tex)
    print(f'[+] wrote {out}')


def welch(df, args):
    """Welch t-test, the proposed policy against every other policy, on the
    per-seed steady-state aoii and reward. This is what any 'indistinguishable'
    or 'better' claim in the paper has to point at."""
    from scipy import stats
    rows = []
    for dataset in util.DATASETS:
        d = df[df.dataset == dataset]
        v = d[d.solver == 'mpgts']
        for solver in util.SOLVERS:
            if solver == 'mpgts': continue
            o = d[d.solver == solver]
            if len(v) < 2 or len(o) < 2: continue
            for metric in ['avg_aoii', 'reward']:
                t, pval = stats.ttest_ind(v[metric], o[metric], equal_var=False)
                rows.append({'dataset': dataset, 'vs': solver, 'metric': metric,
                             'mpgts_mean': v[metric].mean(), 'other_mean': o[metric].mean(),
                             't': t, 'p': pval})
    out = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'welch.csv')
    out.to_csv(path, index=None, float_format='%.4g')
    print(out.to_string(index=False)); print(f'[+] wrote {path}')
