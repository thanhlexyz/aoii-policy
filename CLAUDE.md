# Code rules

## Style

- No comments and no docstrings, in Python or in the makefile. Variable, function, class and flag names carry the meaning. When a line seems to need a comment, rename something instead.
- Explanations live in `README.md` and in this file, never in the code.
- Folder and file names are singular: `test/`, `doc/`, `test_run.py`, `requirement.txt`, never `tests/` or `test_runs.py`. Stored result files keep their frozen names.
- Wording and notation follow the paper; `../paper/paper.tex` Section II is the reference. The paper says sensor, never device. Use the code names in the glossary below for anything the paper names.
- Avoid using the bare noun "supplement" (or "supplements") in documentation, descriptions, or comments; use "background rate" (or "auxiliary rate" / "supplementary transmissions"). The frozen CSV column header `supplement` remains untouched per the frozen data schema.

## Glossary

| Paper | Paper wording | Code name | Flag |
|---|---|---|---|
| $K$ | sensors | `n_sensor` | `--n_sensor` |
| $T$ | learning rounds, `horizon` / `round_length` | | |
| $N$ | slots per round | `round_length` | `--round_length` |
| $TN$ | horizon in slots | `horizon` | `--horizon` |
| $M$ | IRSA slots per IRSA frame | `n_irsa_slot` | `--n_irsa_slot` |
| $L$ | replicas per update, on average | `avg_n_replica` | `--avg_n_replica` |
| $\epsilon$ | error tolerance | `error_tolerance` | `--error_tolerance` |
| $\alpha$ | cost weight | `cost_weight` | `--cost_weight` |
| $G$ | nominal offered load | `nominal_load` | `--nominal_load` |
| $\overline{x}$ | throughput-maximizing rate | `throughput_rate` | |
| $\Delta$ | slot interval | `slot_interval` | `--slot_interval` |
| $X_k(n)$ | source value | `source_value` | |
| $\hat X_k(n)$ at the sensor | local estimate | `local_estimate` | |
| $\hat X_k(n)$ at the server | server estimate | `server_estimate` | |
| $n - S_k(n)$ | age of the current error | `error_age` | |
| $\Omega_k(n)$ | AoII | `aoii` | |
| $\theta = (q, n_r, x_{\mathrm{bg}})$ | retransmission schedule | `schedule` | |
| $q$, $n_r$, $x_{\mathrm{bg}}$ | repeat probability, repeat budget, background rate | `repeat_probability`, `repeat_budget`, `background_rate` | `--repeat_probability_grid`, `--repeat_budget_grid`, `--background_rate_grid` |
| $\Theta$ | candidate schedules | `candidate_schedules` | |
| $n_0$ | warm-up, in sensor-rounds per schedule | `warm_up` | `--warm_up` |
| $r_k(t)$ | round reward | `round_reward` | |
| $\tau$, $\rho$ | Threshold parameters | `threshold_tau`, `threshold_rho` | `--threshold_tau`, `--threshold_rho` |

Learners of the proposed policy, `--learner`: `thompson` (the paper's multiple-play Thompson sampling at the server), `ucb_server` (the ablation "UCB1, one arm per round"), `ucb_per_sensor`, `fixed`. The server learns; the base station decodes.

## Frozen data schema

The results of the paper are final. These names are how they are stored, so they never change, even where the paper uses another word:

- Run files `data/csv/test/{dataset}_{solver}{tag}_s{seed}.csv`, the `test/` folder and `--mode test`.
- Solver keys `random`, `reactive`, `hybrid`, `threshold`, `ucb`, `vhct`, `mpgts`; `random` is the paper's Throughput.
- Every tag the makefile writes, such as `_q`, `_fix_n3_bg0.01`, `_tune_t2_r1`, `_alpha2`, `_scaleK60`.
- Per-slot csv columns: `reward`, `cost`, `avg_age`, `avg_err`, `max_err`, `avg_aoii`, `max_aoii`, `n_send`, `n_success`, `n_error`, `step`, `episode`, `utilization`, `p_success`, `n_retx`, `q`, `n_repeat`, `bg`, `agree`, `p_learned`, `supplement`, `p_event`, `nu`.
- Summary csv names, columns and stored labels in `data/csv/`, such as the ablation label "UCB per device".

No test or refactor writes into `data/` or `../paper/`, and no new experiment is run.

The summary csvs in `data/csv/` and `data/csv/stat/` are committed and are the paper's results; `make test` regenerates them and must match them byte for byte. The run files in `data/csv/test/`, the traces in `data/data/` and the figures in `data/figure/` are not committed, nor are the radio summaries `channel.csv` and `ota_calib_ota.csv`.

## Radio code

The paper does not use the over-the-air experiments for now, so the radio code (`sdr/`, `pysdr/`, the radio calibration, the Fig. 6 plot and channel statistic, `requirement-pi.txt` and the radio make targets) is kept in `../resource/radio/` with its original layout and is not part of this code. Its run files stay in `data/csv/test` with the `_ota` tag. Bringing it back means moving those files into `core/` and restoring the `--channel` and `--ota_*` arguments.

The Berkeley, BTS and SINDIT normalizers need raw data that is not on this machine, so the tests do not cover them and their internal names stay as they are.

## Tests

`make test` from `core/` runs `../test/` with pytest:

- `test_run.py` reruns paper configurations into a temporary folder and compares each run file byte for byte with `data/csv/test`.
- `test_result.py` regenerates every summary csv and Table I from the runs on disk into a temporary folder and compares them with `data/csv` and `../paper/figure/tab-main.tex`.
- `test_figure.py` regenerates the paper's figures into a temporary folder and compares their text with the copies in `../paper/figure/simulation`.

Run the tests after every change. Run `make paper` only when they pass.

The parser rejects unknown and abbreviated flags, so a stale flag anywhere fails at once.
