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

ALPHA = 0.01   # significance level of every test the paper reports

def signed_rank_p(d, a, b, metric):
    """Two-sided p-value of the Wilcoxon signed-rank test between policies
    `a` and `b` on `metric`, paired by seed: every policy replays the same
    seeds (same device population, event sequence and channel draws), so the
    per-seed difference removes the between-seed spread the unpaired tests
    pay for. Exact distribution (scipy chooses it for n <= 50 without ties);
    nan when fewer than two paired seeds or every difference is zero."""
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

def main_table(agg, df, args):
    """Writes ../paper/figure/tab-main.tex, the one table RQ1 rests on:
    steady-state AoII per policy (rows) and trace (columns), the policies
    sorted from worst to best by their mean AoII rank over the traces, the
    best AoII per trace in bold; two summary rows give the change of the
    proposed policy in AoII and in reward against the best baseline on each
    metric, starred at level ALPHA under the Wilcoxon signed-rank test paired
    by seed."""
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
                if signed_rank_p(d, ranked.index[0], ranked.index[1], 'avg_aoii') < ALPHA:
                    cell += r'$^{\dagger}$'
            cells.append(cell)
        rows.append(f'      {LABEL[solver]} & ' + ' & '.join(cells) + r' \\')
    # one summary row per metric, each against the best baseline on that
    # metric (so the two rows may name different baselines), starred when the
    # signed-rank test against that one pre-specified baseline is below ALPHA
    if 'mpgts' in aoii.index:
        cuts, gains = [], []
        for dataset in datasets:
            d = df[df.dataset == dataset]
            others = d[d.solver != 'mpgts'].groupby('solver')[['avg_aoii', 'reward']].mean()
            vm = d[d.solver == 'mpgts'][['avg_aoii', 'reward']].mean()
            b_aoii, b_rew = others.avg_aoii.idxmin(), others.reward.idxmax()
            cut  = (others.avg_aoii[b_aoii] - vm.avg_aoii) / others.avg_aoii[b_aoii] * 100
            gain = (vm.reward - others.reward[b_rew]) / abs(others.reward[b_rew]) * 100
            star = lambda b, m: r'$^{*}$' if signed_rank_p(d, 'mpgts', b, m) < ALPHA else ''
            cuts.append(f'{cut:.0f}\\%{star(b_aoii, "avg_aoii")}')
            gains.append(f'{gain:.0f}\\%{star(b_rew, "reward")}')
        rows.append(r'      \midrule')
        rows.append(r'      \textit{AoII gain} & ' + ' & '.join(cuts) + r' \\')
        rows.append(r'      \textit{reward gain} & ' + ' & '.join(gains) + r' \\')
    tex = r"""\newcommand{\maintable}{%
  \begin{table}[t]
    \centering
    \caption{Steady-state AoII per policy, sorted from worst to best (mean $\pm$ standard deviation over """ + seeds_in_words(df) + r""" seeds on the final 1000 slots; best in bold); last two rows: gain of \tool in AoII (reduction) and in reward (increase) over the baseline that is best on that metric, negative if \tool is worse, $*$: significant at $p < """ + f'{ALPHA:g}' + r"""$ (Wilcoxon signed-rank test paired by seed). The reward is not the AoII: on \texttt{SINDIT}, \textsc{Hybrid} has the higher reward at a $12\times$ higher AoII.}
    \label{tab:main}
    \setlength{\tabcolsep}{0pt}\scriptsize
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


def signed_rank(df, args):
    """Wilcoxon signed-rank test paired by seed, the proposed policy against
    every other policy, on the per-seed steady-state aoii and reward.
    `best_baseline` marks the baseline with the best mean of that metric, the
    one pre-specified comparison RQ1 quotes at level ALPHA; `bonferroni`
    is the level for the other, exploratory comparisons of one trace. With
    ten seeds the exact two-sided p cannot fall below 2 / 2^10 = 0.002, so
    the Bonferroni level 0.01 / 6 is unreachable; that column is kept for
    larger seed counts. This is what any 'indistinguishable' or 'better'
    claim in the paper has to point at."""
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
