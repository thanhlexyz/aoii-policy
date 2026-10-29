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

ALPHA = 0.01   # significance level of every test the paper reports

def seeds_in_words(df):
    n = df.groupby(['dataset', 'solver']).seed.count()
    lo, hi = int(n.min()), int(n.max())
    words = {5: 'five', 10: 'ten', 15: 'fifteen', 20: 'twenty'}
    return words.get(hi, str(hi)) if lo == hi else f'{lo} to {hi}'

def main_table(agg, df, args):
    """Writes ../paper/figure/tab-main.tex, the one table RQ1 rests on:
    steady-state AoII per policy (rows) and trace (columns), the policies
    sorted from worst to best by their mean AoII rank over the traces, the
    best AoII per trace in bold and marked when it differs from the runner-up
    at level ALPHA under Welch's t-test on the per-seed values."""
    from scipy import stats
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
            ranked = aoii.loc[solvers, dataset].sort_values()
            if ranked.index[0] == solver:
                cell = rf'$\mathbf{{{cell[1:-1]}}}$'
                d = df[df.dataset == dataset]
                best, second = d[d.solver == ranked.index[0]].avg_aoii, d[d.solver == ranked.index[1]].avg_aoii
                if len(best) > 1 and len(second) > 1 and stats.ttest_ind(best, second, equal_var=False).pvalue < ALPHA:
                    cell += r'$^{\dagger}$'
            cells.append(cell)
        rows.append(f'      {LABEL[solver]} & ' + ' & '.join(cells) + r' \\')
    # the objective: the reward gain of the proposed policy over the best
    # baseline, starred when the Welch test against that baseline is below
    # ALPHA / (number of baselines)
    if 'mpgts' in aoii.index:
        gains = []
        for dataset in datasets:
            d = df[df.dataset == dataset]
            means = d[d.solver != 'mpgts'].groupby('solver').reward.mean()
            best = means.idxmax()
            v, o = d[d.solver == 'mpgts'].reward, d[d.solver == best].reward
            gain = (v.mean() - means[best]) / abs(means[best]) * 100
            star = r'$^{*}$' if len(v) > 1 and len(o) > 1 and \
                stats.ttest_ind(v, o, equal_var=False).pvalue < ALPHA / len(means) else ''
            gains.append(f'{gain:+.0f}\\%{star}')
        rows.append(r'      \midrule')
        rows.append(r'      \textit{reward gain} & ' + ' & '.join(gains) + r' \\')
    tex = r"""\newcommand{\maintable}{%
  \begin{table}[t]
    \centering
    \caption{Steady-state AoII per policy, sorted from worst to best (mean $\pm$ standard deviation over """ + seeds_in_words(df) + r""" seeds on the final 1000 slots; best in bold, $\dagger$: differs from the runner-up at $p < """ + f'{ALPHA:g}' + r""" $, Welch's $t$-test); last row: reward gain of \tool over the best baseline, $*$: significant at the Bonferroni level.}
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
    per-seed steady-state aoii and reward, with a Bonferroni level over the
    baselines of one trace; `best_baseline` marks the baseline with the best
    mean of that metric, the comparison RQ1 quotes. This is what any
    'indistinguishable' or 'better' claim in the paper has to point at."""
    from scipy import stats
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
                t, pval = stats.ttest_ind(v[metric], o[metric], equal_var=False)
                rows.append({'dataset': dataset, 'vs': solver, 'metric': metric, 'n_seed': len(v),
                             'mpgts_mean': v[metric].mean(), 'other_mean': o[metric].mean(),
                             't': t, 'p': pval, 'best_baseline': solver == best,
                             'significant': pval < ALPHA / len(others)})
    out = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'welch.csv')
    out.to_csv(path, index=None, float_format='%.4g')
    print(out.to_string(index=False)); print(f'[+] wrote {path}')
