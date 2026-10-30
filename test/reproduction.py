import pathlib
import subprocess

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
CORE_DIR = CODE_DIR / 'core'
PAPER_CSV_DIR = CODE_DIR / 'data' / 'csv'
PAPER_RUN_DIR = PAPER_CSV_DIR / 'test'
PAPER_FIGURE_DIR = CODE_DIR.parent / 'paper' / 'figure'


def run_scenario(scenario, *flags, capture=False):
    command = ['python3', 'main.py', '--no_progress_bar', '--device=cpu', f'--scenario={scenario}', *flags]
    completed = subprocess.run(command, cwd=CORE_DIR, check=True, capture_output=True, text=True)
    return completed.stdout if capture else None
