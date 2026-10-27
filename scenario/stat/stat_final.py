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

TABLE_ORDER = ['reactive', 'random', 'hybrid', 'threshold', 'grid', 'vhct', 'repair']
NAME  = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
LABEL = {'reactive': r'\textsc{Reactive}', 'random': r'\textsc{Throughput}', 'hybrid': r'\textsc{Hybrid}',
         'threshold': r'\textsc{Threshold}', 'grid': r'\textsc{Grid-UCB}', 'vhct': r'\textsc{VHCT}',
         'repair': r'\textsc{Repair} (ours)'}

def fmt(m, sd, digits):
    return f'${m:.{digits}f} \\pm {sd:.{digits}f}$'

def fmt_aoii(m, sd):
    if m >= 100: return f'${m:.0f} \\pm {sd:.0f}$'
    if m >= 1:   return f'${m:.1f} \\pm {sd:.1f}$'
    return fmt(m, sd, 3) if m >= 0.01 else fmt(m, sd, 4)

def main_table(agg, args):
    """Writes ../paper/figure/tab-main.tex, the one table RQ1 rests on:
    steady-state AoII, offered load and reward per policy and trace, the best
    reward per trace in bold and the proposed policy's gain over the best
    baseline in the last row of each block."""
    rows = []
    for dataset in util.DATASETS:
        d = agg[agg.dataset == dataset].set_index('solver')
        if d.empty: continue
        best = d['reward'].idxmax()
        rows.append(r'      \midrule')
        rows.append(rf'      \multirow{{{len(d)}}}{{*}}{{\rotatebox{{90}}{{{NAME[dataset]}}}}}')
        for solver in TABLE_ORDER:
            if solver not in d.index: continue
            r = d.loc[solver]
            reward = fmt(r.reward, r.reward_std, 0 if abs(r.reward) >= 10 else 3)
            if solver == best: reward = rf'$\mathbf{{{reward[1:-1]}}}$'
            rows.append(f'        & {LABEL[solver]} & {fmt_aoii(r.avg_aoii, r.avg_aoii_std)} & '
                        f'{fmt(r.utilization, r.utilization_std, 3)} & {reward} \\\\')
        if 'repair' in d.index:
            base = d.drop('repair')['reward'].max()
            gain = (d.loc['repair', 'reward'] - base) / abs(base) * 100
            rows.append(rf'        & \multicolumn{{3}}{{r}}{{\textit{{gain over best baseline}}}} & '
                        rf'\textit{{{gain:+.0f}\%}} \\\\')
    tex = r"""\newcommand{\maintable}{%
  \begin{table}[t]
    \centering
    \caption{Steady-state AoII, offered load $G$ and reward across the IoT traces (mean $\pm$ standard deviation over five seeds on the final 1000 slots, best reward in bold).}
    \label{tab:main}
    \setlength{\tabcolsep}{3pt}\scriptsize
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}}
      \toprule
      & \textbf{Policy} & \textbf{AoII} & \textbf{Load $G$} & \textbf{Reward} \\
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
        v = d[d.solver == 'repair']
        for solver in util.SOLVERS:
            if solver == 'repair': continue
            o = d[d.solver == solver]
            if len(v) < 2 or len(o) < 2: continue
            for metric in ['avg_aoii', 'reward']:
                t, pval = stats.ttest_ind(v[metric], o[metric], equal_var=False)
                rows.append({'dataset': dataset, 'vs': solver, 'metric': metric,
                             'repair_mean': v[metric].mean(), 'other_mean': o[metric].mean(),
                             't': t, 'p': pval})
    out = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'welch.csv')
    out.to_csv(path, index=None, float_format='%.4g')
    print(out.to_string(index=False)); print(f'[+] wrote {path}')
