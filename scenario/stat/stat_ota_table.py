import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

CHANNEL_TAG = {'ideal': '', 'software': '_sw', 'ota': '_ota'}
NAME  = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
# the table shows the multi-seed trace; single-seed traces are stated in prose
TABLE_DATASETS = ['sindit']
LABEL = {'reactive': 'reactive', 'random': 'throughput', 'hybrid': 'hybrid',
         'threshold': 'threshold', 'ucb': 'ucb', 'vhct': 'VHCT', 'mpgts': 'mpgts'}

def collect(args, dataset):
    rows = []
    for channel, tag in CHANNEL_TAG.items():
        for solver in util.SOLVERS:
            for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode,
                                                      f'{dataset}_{solver}{tag}_s[0-9]*.csv'))):
                rows.append({'channel': channel, 'solver': solver,
                             **tail_metrics(path, args.n_irsa_slot)})
    return pd.DataFrame(rows)

def fmt_reward(v):
    return f'${v:.0f}$' if abs(v) >= 10 else f'${v:.3f}$'

def stat_ota_table(args):
    """Writes ../paper/figure/tab-ota.tex: delivery and reward (with rank)
    per policy and channel, for every dataset that has radio runs."""
    blocks, coverage = [], []
    for dataset in util.DATASETS:
        df = collect(args, dataset)
        if df.empty or 'ota' not in set(df.channel):
            continue
        n_ota = df[df.channel == 'ota'].groupby('solver').size().min()
        n_step = int(df[df.channel == 'ota'].n_step.max())
        coverage.append(f'{NAME[dataset]} over {n_ota} seed{"s" if n_ota > 1 else ""}')
        agg = df.groupby(['channel', 'solver']).agg(p=('p_success', 'mean'),
                                                     r=('reward', 'mean')).reset_index()
        agg['rank'] = agg.groupby('channel')['r'].rank(ascending=False).astype(int)
        lines = []
        for solver in util.SOLVERS:
            cells = []
            for ch in ['ideal', 'software', 'ota']:
                a = agg[(agg.channel == ch) & (agg.solver == solver)]
                cells.append(f'{float(a.p.iloc[0]):.3f}' if len(a) else '--')
            for ch in ['ideal', 'software', 'ota']:
                a = agg[(agg.channel == ch) & (agg.solver == solver)]
                cells.append(f'{fmt_reward(float(a.r.iloc[0]))} ({int(a["rank"].iloc[0])})' if len(a) else '--')
            lines.append(f'            {LABEL[solver]:10s} & ' + ' & '.join(cells) + r' \\')
        if dataset in TABLE_DATASETS:
            blocks.append((NAME[dataset], n_step, lines))
    if not blocks:
        print('[!] no radio runs'); return
    body = []
    for name, n_step, lines in blocks:
        body.append(r'            \midrule')
        if len(blocks) > 1:
            body.append(rf'            \multicolumn{{7}}{{l}}{{\textit{{{name}}}}} \\')
        body.extend(lines)
    tex = r'''\newcommand{\otacoverage}{''' + ' and '.join(coverage) + f', {n_step} slots per run' + r'''}
\newcommand{\otatable}{%
    \begin{table}[t]
        \centering
        \caption{''' + ', '.join(b[0] for b in blocks) + r''' : delivery and reward (rank) per channel. Ideal, software: ten seeds; radio: five seeds, 2000 slots.}
        \label{tab:ota}
        \setlength{\tabcolsep}{3pt}\scriptsize
        \begin{tabular}{lrrrrrr}
            \toprule
            & \multicolumn{3}{c}{Delivery} & \multicolumn{3}{c}{Reward (rank)} \\
            \cmidrule(lr){2-4}\cmidrule(lr){5-7}
            Policy & ideal & softw. & radio & ideal & softw. & radio \\
''' + '\n'.join(body) + r'''
            \bottomrule
        \end{tabular}
    \end{table}%
}
'''
    out = os.path.join(args.paper_figure_dir, 'tab-ota.tex')
    pathlib_write(out, tex)
    print(tex); print(f'[+] wrote {out}')

def pathlib_write(path, text):
    with open(path, 'w') as fp:
        fp.write(text)
