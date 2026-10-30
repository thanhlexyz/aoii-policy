import filecmp

import pytest

from reproduction import PAPER_CSV_DIR, PAPER_FIGURE_DIR, PAPER_RUN_DIR, run_scenario

SUMMARY_SCENARIOS = ['stat_final', 'stat_ablation', 'stat_fixed', 'stat_robustness', 'stat_subset',
                     'stat_scaling', 'stat_threshold']
SUMMARY_FILES = ['summary.csv', 'summary_per_seed.csv', 'wilcoxon.csv', 'ablation.csv', 'fixed_schedules.csv',
                 'fixed_best.csv', 'fixed_worst.csv', 'robustness.csv', 'subsets.csv',
                 'scaling.csv', 'scaling_per_seed.csv', 'threshold_tuning.csv']
DATASETS = ['synthetic', 'berkeley', 'bts', 'sindit']


@pytest.fixture(scope='module')
def result_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp('results')
    csv_dir, figure_dir = root / 'csv', root / 'figure'
    csv_dir.mkdir()
    (csv_dir / 'test').symlink_to(PAPER_RUN_DIR)
    for scenario in SUMMARY_SCENARIOS:
        run_scenario(scenario, f'--csv_dir={csv_dir}', f'--figure_dir={figure_dir}')
    (csv_dir / 'stat').mkdir()
    for dataset in DATASETS:
        intervals = run_scenario('stat_sampling_interval_per_sensor', f'--dataset={dataset}',
                                 f'--csv_dir={csv_dir}', f'--figure_dir={figure_dir}', capture=True)
        (csv_dir / 'stat' / f'diff_{dataset}.csv').write_text(intervals)
    return root


@pytest.mark.parametrize('name', SUMMARY_FILES)
def test_summary_reproduces_paper(result_dir, name):
    assert filecmp.cmp(result_dir / 'csv' / name, PAPER_CSV_DIR / name, shallow=False)


@pytest.mark.parametrize('dataset', DATASETS)
def test_slot_interval_statistics_reproduce_paper(result_dir, dataset):
    name = f'diff_{dataset}.csv'
    assert filecmp.cmp(result_dir / 'csv' / 'stat' / name, PAPER_CSV_DIR / 'stat' / name, shallow=False)


def test_table_reproduces_paper(result_dir):
    assert filecmp.cmp(result_dir / 'figure' / 'tab-main.tex', PAPER_FIGURE_DIR / 'tab-main.tex', shallow=False)
