# Multifidelity importance sampling: reproducible report

Primary editable source: `main.tex`. Compile from this directory so its table and figure paths resolve. `main.pdf` is the checked PDF.

## Rerun

Use a copy of this folder if you want to preserve the supplied outputs. The experiment script overwrites results and figures. Python 3.13.7 was used; package versions are pinned in requirements.txt.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py
python build_tables.py
python verify.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```


## File map

- `models.py`: deterministic steady Burgers solver.
- `sliding_block.py`: exact stopping distance, tangent and crossing tests.
- `run_experiments.py`: model references, surrogate, proposals, independent sampling runs, time-stepping check, plots and CSVs.
- `build_tables.py`: LaTeX tables from saved CSVs.
- `verify.py`: independent output-consistency, solver and weight checks.
- `results/runs.csv`: individual estimates and standard errors.
- `results/summary.csv`: comparisons across 40 runs per configuration.
- `results/metadata.json`: seeds, proposal parameters, references and environment.
- `results/grid_convergence.csv`, `time_checks.csv`: numerical-model checks.
- `results/timings.csv`: stage timings, including pilot work.

