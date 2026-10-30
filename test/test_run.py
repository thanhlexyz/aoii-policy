import concurrent.futures
import filecmp

import pytest

from reproduction import PAPER_RUN_DIR, run_scenario

PAPER_CONFIGURATIONS = [
    '--dataset=bts --solver=reactive --seed=3',
    '--dataset=berkeley --solver=random --seed=7',
    '--dataset=sindit --solver=hybrid --seed=2',
    '--dataset=synthetic --solver=threshold --seed=5',
    '--dataset=bts --solver=ucb --seed=9',
    '--dataset=berkeley --solver=vhct --seed=1',
    '--dataset=bts --solver=mpgts --seed=0',
    '--dataset=synthetic --solver=mpgts --seed=4',
    '--dataset=sindit --solver=mpgts --seed=6 --repeat_budget_grid=5 --tag=_bgonly',
    '--dataset=bts --solver=mpgts --seed=2 --repeat_probability_grid=0.02,0.05,0.1 --repeat_budget_grid=1,3,5 --tag=_q',
    '--dataset=berkeley --solver=mpgts --seed=8 --learner=ucb_server --tag=_ucb',
    '--dataset=bts --solver=mpgts --seed=1 --round_length=25 --tag=_N25',
    '--dataset=sindit --solver=mpgts --seed=3 --learner=fixed --repeat_budget_grid=3 --background_rate_grid=0.01 --tag=_fix_n3_bg0.01',
    '--dataset=bts --solver=threshold --seed=4 --threshold_tau=0.5 --threshold_rho=0.5 --tag=_tune_t0.5_r0.5',
    '--dataset=synthetic --solver=ucb --seed=2 --causal_norm=0.1 --tag=_causal',
    '--dataset=berkeley --solver=mpgts --seed=5 --freeze_after=0.4 --tag=_frozen',
    '--dataset=bts --solver=vhct --seed=6 --cost_weight=2 --tag=_alpha2',
    '--dataset=synthetic --solver=mpgts --seed=1 --n_sensor=60 --n_irsa_slot=10 --tag=_scaleK60',
    '--dataset=sindit --solver=random --seed=0 --stream_offset=10 --tag=_o10',
    '--dataset=berkeley --solver=hybrid --seed=3 --start_slot=1500 --tag=_t1500',
]


def run_file_name(configuration):
    value = dict(flag[2:].split('=', 1) for flag in configuration.split())
    return f"{value['dataset']}_{value['solver']}{value.get('tag', '')}_s{value['seed']}.csv"


@pytest.fixture(scope='module')
def rerun_dir(tmp_path_factory):
    csv_dir = tmp_path_factory.mktemp('reruns')
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(lambda configuration: run_scenario('solve', *configuration.split(), f'--csv_dir={csv_dir}'),
                      PAPER_CONFIGURATIONS))
    return csv_dir / 'test'


@pytest.mark.parametrize('configuration', PAPER_CONFIGURATIONS)
def test_rerun_reproduces_paper_run(rerun_dir, configuration):
    name = run_file_name(configuration)
    assert filecmp.cmp(rerun_dir / name, PAPER_RUN_DIR / name, shallow=False)
