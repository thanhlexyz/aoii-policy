import subprocess

import pytest

from reproduction import PAPER_CSV_DIR, PAPER_FIGURE_DIR, run_scenario

PAPER_FIGURES = ['plot_ablation.pdf', 'plot_scaling_reward.pdf', 'plot_scaling_consensus.pdf',
                 'plot_scaling_load.pdf', 'plot_scaling_legend.pdf']


def pdf_text(path):
    return subprocess.run(['pdftotext', '-layout', str(path), '-'], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope='module')
def figure_dir(tmp_path_factory):
    figure_dir = tmp_path_factory.mktemp('figures')
    run_scenario('make_figure', f'--csv_dir={PAPER_CSV_DIR}', f'--figure_dir={figure_dir}')
    return figure_dir / 'simulation'


@pytest.mark.parametrize('name', PAPER_FIGURES)
def test_figure_reproduces_paper(figure_dir, name):
    assert pdf_text(figure_dir / name) == pdf_text(PAPER_FIGURE_DIR / 'simulation' / name)
