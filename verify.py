"""Check saved experiment results against independent calculations.

Run python verify.py after the experiments, with Python assertions enabled.
The checks read metadata.json, runs.csv and summary.csv in results/, recompute
means, RMSEs and interval coverage, and check Burgers residuals and profile
monotonicity at three inputs. The block checks use constant-acceleration
kinematics, analytic crossing probabilities and one surrogate misclassification.
Running or importing this script executes the checks. On success it overwrites
results/verification.txt. A failed numerical check raises AssertionError."""

from sliding_block import stopping_distance, tangent_distance, crosses_line, surrogate_crosses_line
from pathlib import Path
import csv
import json
import numpy as np
from models import solve
root = Path(__file__).resolve().parent
meta = json.loads((root/'results/metadata.json').read_text())
raw = list(csv.DictReader((root/'results/runs.csv').open()))
summary = list(csv.DictReader((root/'results/summary.csv').open()))
checks = []
for z in [.01, .0104744086284, .1]:
    x, u, y = solve(z, ret=True)
    assert np.isclose(u[0], 1+z) and np.isclose(u[-1], -1)
    h = x[1]-x[0]
    residual = .05*np.diff(u, 2)/h**2-u[1:-1]*(u[2:]-u[:-2])/(2*h)
    assert max(abs(residual)) < 1e-9
    assert np.all(np.diff(u) <= 1e-12)
checks.append(
    'Burgers residual, monotonic spatial profile and fixed boundaries checked at three inputs.')
for row in summary:
    estimates = []
    standard_errors = []
    for run in raw:
        if run['case'] == row['case'] and run['method'] == row['method'] and run['M'] == row['M']:
            estimates.append(float(run['estimate']))
            standard_errors.append(float(run['se']))
    assert len(estimates) == 40
    estimates = np.array(estimates)
    se = np.array(standard_errors)
    if row['case'] == 'Burgers':
        p = meta['burgers_reference']
    else:
        p = meta['block_reference']
    assert np.isclose(np.mean(estimates), float(row['mean']), rtol=1e-12)
    assert np.isclose(np.sqrt(np.mean((estimates-p)**2)),
                      float(row['rmse']), rtol=1e-12)
    assert np.isclose(np.mean(abs(estimates-p) <= 1.96*se),
                      float(row['coverage']))
checks.append(
    'All main summary means, RMSEs and coverage recomputed from raw runs.')
# Check mechanics independently via constant-acceleration kinematics.
speeds = np.array([0.0, 9.0, 9.82, 10.0])
time_to_stop = speeds/2.0
assert np.allclose(stopping_distance(speeds), speeds *
                   time_to_stop-time_to_stop**2)
assert not crosses_line(9.8) and crosses_line(np.nextafter(9.8, np.inf))
assert crosses_line(9.82) and not surrogate_crosses_line(9.82)
assert np.all(stopping_distance(speeds) >= tangent_distance(speeds)-1e-13)
assert abs(meta['block_reference']-.02) < 1e-12
assert abs(meta['block_surrogate_probability'] -
           ((10-(24.01+20.25)/4.5)/10)) < 1e-12
checks.append(
    'Sliding block checked against kinematics, analytic probabilities and one missed crossing.')
(root/'results/verification.txt').write_text('\n'.join(checks)+'\nPASS\n')
print('\n'.join(checks))
print('PASS')
