"""Generate the three LaTeX table fragments included by main.tex.

Run python build_tables.py after run_experiments.py. The script reads
results/summary.csv, selects the 2,000-draw comparisons, and converts relative
RMSE and coverage fractions to percentages. It also reads grid_convergence.csv
for the Burgers refinement table. Running or importing the module overwrites
burgers_table.tex, sliding_block_table.tex and grid_table.tex beside the script."""
import csv
from pathlib import Path
P = Path(__file__).resolve().parent
rows = list(csv.DictReader((P/'results/summary.csv').open()))
for case in ['Burgers', 'Sliding block']:
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Method & Mean estimate & RMSE & Rel. RMSE & Coverage \\', r'\midrule']
    for r in rows:
        if r['case'] != case or r['M'] != '2000':
            continue
        lines.append(
            f"{r['method']} & {float(r['mean']):.6f} & {float(r['rmse']):.2e} & {100*float(r['relative_rmse']):.1f}\\% & {100*float(r['coverage']):.1f}\\% \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (P/(case.lower().replace(' ', '_')+'_table.tex')).write_text('\n'.join(lines))
rows = list(csv.DictReader((P/'results/grid_convergence.csv').open()))
lines = [r'\begin{tabular}{rrr}', r'\toprule',
         r'Grid points & Failure boundary $z_*$ & Failure probability \\', r'\midrule']
for r in rows:
    lines.append(
        f"{r['points']} & {float(r['boundary']):.9f} & {float(r['probability']):.7f} \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
(P/'grid_table.tex').write_text('\n'.join(lines))
