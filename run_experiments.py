"""Run the report's seeded MFIS experiments for Burgers flow and a sliding block.

Run python run_experiments.py from an environment with requirements.txt
installed. The script executes at module level, including when imported.
It builds model-specific references, fits Gaussian proposals from independent
pilot samples, and compares four estimators at three production sample counts.
Each comparison uses 40 runs. A separate Burgers check refits the proposal
for each run to measure the effect of changing the pilot data.

Results overwrite the CSV and JSON outputs in results/ and PNGs in figures/
next to this script. runs.csv contains individual estimates and standard
errors. summary.csv aggregates them. metadata.json records seeds, fitted
parameters and reference probabilities. Elapsed times depend on hardware.
Run build_tables.py afterwards to regenerate the LaTeX tables.
This implementation is based on the paper's equations."""

from scipy.linalg import solve_banded
import sliding_block as block
from models import solve
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import json
import time
import platform
import sys
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
FIG = ROOT/'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
SEED = 20261105
R = 40
MS = [100, 500, 2000]
N = 20000


def save_csv(name, rows):
    """Write experiment records to a CSV file in the results directory.

    name is the output filename. rows must be a nonempty sequence of mappings
    with a common set of fields. The first record determines column order.
    Overwrite the file with a header and one row per record. Return None."""
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def nominal(x, bounds):
    """Evaluate the uniform input density at x for bounds=(lower, upper).

    x may be a scalar or NumPy array. The endpoints must be ordered. Return
    1/(upper-lower) on the closed interval and zero outside, elementwise.
    The result is a density per input unit, used in the importance weight."""
    a, b = bounds
    return ((x >= a) & (x <= b))/(b-a)


def fit(x, fail, bounds):
    """Fit one Gaussian to pilot inputs that the supplied model flags as failures.

    x is a one-dimensional NumPy array of nominal draws. fail(x) must return
    a Boolean mask of the same length. bounds records the nominal interval in
    the calling interface and is unused here because x has already been drawn.
    Return a dictionary of the retained sample mean, population-form standard
    deviation (maximum-likelihood fit), and failure count. With fewer than two
    retained inputs return None, which requests nominal sampling downstream."""
    failure_mask = fail(x)
    failure_inputs = x[failure_mask]
    if len(failure_inputs) < 2:
        return None
    return {
        'mean': float(failure_inputs.mean()),
        'sd': float(failure_inputs.std()),
        'failures': len(failure_inputs),
    }



def draw(rng, m, q, bounds, alpha=0):
    """Draw m independent production inputs using a NumPy Generator rng.

    For q=None, sample uniformly between the two bounds. Otherwise q supplies
    Gaussian mean and sd. alpha in [0, 1] optionally sets the probability
    of drawing from the nominal uniform component instead. Return a length-m
    array. Gaussian draws outside the nominal interval are kept."""
    if q is None:
        return rng.uniform(bounds[0], bounds[1], m)
    x = rng.normal(q['mean'], q['sd'], m)
    if alpha:
        take = rng.random(m) < alpha
        x[take] = rng.uniform(bounds[0], bounds[1], take.sum())
    return x


def density(x, q, bounds, alpha=0):
    """Evaluate the sampling density used by draw at a scalar or array x.

    bounds defines the nominal uniform interval. With q=None return its density.
    Otherwise use q['mean'] and q['sd'] for the Gaussian component (positive sd)
    and alpha for the uniform mixture fraction. Use the same q, bounds and alpha
    as draw so each weight uses the density that generated the input."""
    if q is None:
        return nominal(x, bounds)
    return (1-alpha)*norm.pdf(x, q['mean'], q['sd'])+alpha*nominal(x, bounds)


def terms(x, q, fail, bounds, alpha=0):
    # Reject neither out-of-support draws nor their denominator count.
    # Their nominal density and contribution are zero; avoid model calls there.
    """Compute one weighted failure contribution per production draw in x.

    x is a one-dimensional NumPy array. q, bounds and alpha describe the
    proposal used by draw. fail is a vectorized indicator for the model whose
    failure probability is estimated. Within bounds multiply that indicator
    by the nominal density divided by the proposal density. Outside bounds
    return zero without calling the model. Average the full returned array,
    including zeros, to obtain the probability estimate."""
    original_density = nominal(x, bounds)
    inside = original_density > 0
    allowed_inputs = x[inside]
    failed = fail(allowed_inputs)
    proposal_density = density(allowed_inputs, q, bounds, alpha)
    contributions = np.zeros(len(x))
    contributions[inside] = failed * original_density[inside] / proposal_density
    return contributions



def burgers(x):
    """Return high-fidelity failure labels for an iterable of boundary inputs x.

    Each perturbation is passed to the 514-node steady solver. A transition
    position strictly below 0.74 counts as failure. Return a one-dimensional
    Boolean NumPy array with one entry per input, in the original order."""
    failures = []
    for z in x:
        position = solve(float(z))
        failures.append(position < 0.74)
    return np.array(failures, dtype=bool)


def summarize(case, method, m, values, ptrue):
    """Aggregate repeated estimates against a positive reference probability.

    case and method label the experiment. m is the production draw count per
    run. Each row of values contains (estimate, estimated standard error).
    ptrue is the model-specific reference probability. Return a dictionary
    with the labels, counts, mean estimate, signed bias, RMSE, RMSE/ptrue,
    mean standard error, fraction of nominal 95% intervals covering ptrue
    (using 1.96 standard errors), and number of exactly zero estimates."""
    values = np.array(values)
    estimates = values[:, 0]
    standard_errors = values[:, 1]
    errors = estimates - ptrue
    rmse = np.sqrt(np.mean(errors**2))
    covered = abs(errors) <= 1.96 * standard_errors
    return {
        'case': case, 'method': method, 'M': m, 'runs': len(estimates),
        'mean': float(estimates.mean()), 'bias': float(estimates.mean() - ptrue),
        'rmse': float(rmse), 'relative_rmse': float(rmse / ptrue),
        'mean_se': float(standard_errors.mean()), 'coverage': float(covered.mean()),
        'zero_estimates': int((estimates == 0).sum()),
    }


def distance_from_threshold(z, grid_points):
    """Return the steady transition position minus 0.74 on the given grid.

    A root gives the input where the high-fidelity failure decision changes.
    """
    return solve(z, grid_points) - 0.74


def continuum_margin(z):
    """Evaluate the analytic steady-profile approximation minus the threshold.

    The zero is used as a separate grid-refinement check.
    """
    position = 1 - 0.05 / (1 + z) * np.log((2 + z) / z)
    return position - 0.74


start = time.perf_counter()
meta = {'seed': SEED, 'runs': R, 'M': MS, 'pilot_N': N,
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'platform': platform.platform(), 'scope': 'Adapted steady-state reproduction of section 4.3; not original POD/time integration.'}
# Reference solves and grid convergence precede sampling; never substituted for HF calls.
grids = []
for n in [66, 130, 258, 514, 1026, 2050]:
    low = solve(0.01, n)
    z = 0.01
    if low < 0.74:
        z = brentq(distance_from_threshold, 0.01, 0.1, args=(n,), xtol=1e-13)
    grids.append(dict(points=n, boundary=z,
                 probability=(z-.01)/.09, y_at_lower=low))
save_csv('grid_convergence.csv', grids)
pb = grids[3]['probability']
meta['burgers_reference'] = pb
# Dense deterministic check of monotonicity supports the interval reference.
zg = np.linspace(.01, .1, 301)
yg = np.zeros(len(zg))
for i in range(len(zg)):
    yg[i] = solve(zg[i])
assert np.all(np.diff(yg) > 0)
# 81 cheap coarse-grid solves followed by interpolation; build cost recorded separately.
t = time.perf_counter()
knots = np.linspace(.01, .1, 81)
yl = np.zeros(len(knots))
for i in range(len(knots)):
    yl[i] = solve(knots[i], 258)
sur = PchipInterpolator(knots, yl, extrapolate=False)
meta['surrogate_build_seconds'] = time.perf_counter()-t


def surrogate_margin(z):
    """Return interpolated transition position minus the 0.74 failure limit."""
    return sur(z) - 0.74


def surrogate_failure(z):
    """Return one Boolean failure decision per Burgers surrogate input."""
    return sur(z) < 0.74


zlow = brentq(surrogate_margin, .01, .1, xtol=1e-13)
meta['burgers_surrogate_probability'] = (zlow-.01)/.09
# Separate exact continuum calculation (left BC mismatch below machine precision here).
zc = brentq(continuum_margin, .01, .1, xtol=1e-13)
meta['burgers_continuum_probability'] = (zc-.01)/.09
# A uniform speed makes the failure probability the fractional tail length.
vmin, vmax = block.SPEED_BOUNDS
critical_speed = np.sqrt(2 * block.DECELERATION * block.LINE_DISTANCE)
surrogate_speed = block.TANGENT_SPEED + (block.LINE_DISTANCE - block.stopping_distance(
    block.TANGENT_SPEED)) * block.DECELERATION / block.TANGENT_SPEED
block_reference = (vmax - critical_speed) / (vmax - vmin)
meta['block_reference'] = block_reference
meta['block_surrogate_probability'] = (vmax - surrogate_speed) / (vmax - vmin)
meta['block_threshold_speeds'] = [critical_speed, surrogate_speed]
allrows = []
raw = []
proposals = {}
times = []
cases = ['Burgers', 'Sliding block']
for ci in range(len(cases)):
    case = cases[ci]
    if case == 'Burgers':
        bounds = (0.01, 0.1)
        hf = burgers
        lf = surrogate_failure
        ptrue = pb
    else:
        bounds = block.SPEED_BOUNDS
        hf = block.crosses_line
        lf = block.surrogate_crosses_line
        ptrue = block_reference
    rng = np.random.default_rng(SEED+100*ci)
    t = time.perf_counter()
    qlo = fit(rng.uniform(bounds[0], bounds[1], N), lf, bounds)
    tlo = time.perf_counter()-t
    t = time.perf_counter()
    qhi = fit(rng.uniform(bounds[0], bounds[1], N), hf, bounds)
    thi = time.perf_counter()-t
    proposals[case] = {'low': qlo, 'high': qhi}
    times.extend([dict(case=case, stage='LF pilot and fit', seconds=tlo), dict(
        case=case, stage='HF pilot and fit', seconds=thi)])
    methods = ['MC', 'HF IS', 'MFIS', 'Surrogate IS']
    for mi in range(len(methods)):
        name = methods[mi]
        q = qlo
        check_failure = hf
        if name == 'MC':
            q = None
        elif name == 'HF IS':
            q = qhi
        elif name == 'Surrogate IS':
            check_failure = lf
        for m in MS:
            t = time.perf_counter()
            values = []
            for r in range(R):
                rng = np.random.default_rng(
                    [SEED, ci, mi, m, r])
                x = draw(rng, m, q, bounds)
                v = terms(x, q, check_failure, bounds)
                estimate = float(v.mean())
                se = float(v.std(ddof=1)/np.sqrt(m))
                values.append([estimate, se])
                raw.append(dict(case=case, method=name, M=m,
                           run=r, estimate=estimate, se=se))
            times.append(dict(case=case, stage=name+' production M=' +
                         str(m)+' all runs', seconds=time.perf_counter()-t))
            row = summarize(case, name, m, values, ptrue)
            allrows.append(row)
            print(case, name, m, 'RMSE', row['rmse'], flush=True)
meta['proposals'] = proposals
# Pilot variability: re-fit each run, unlike the fixed-pilot primary analysis.
repilot = []
for r in range(40):
    rng = np.random.default_rng([SEED, 900, r])
    q = fit(rng.uniform(.01, .1, N), surrogate_failure, (.01, .1))
    x = draw(rng, 500, q, (.01, .1))
    v = terms(x, q, burgers, (.01, .1))
    repilot.append([v.mean(), v.std(ddof=1)/np.sqrt(500)])
meta['refitted_pilot_check'] = summarize(
    'Burgers', 'MFIS refit', 500, repilot, pb)
# Independent time-stepping check on three inputs. Linearized convection is implicit.
checks = []
for z in [.01, .010474408628382054, .1]:
    n = 514
    dt = .1
    nu = .05
    x = np.linspace(-1, 1, n)
    h = x[1]-x[0]
    u = np.linspace(1+z, -1, n)
    for step in range(100000):
        old = u.copy()
        v = old[1:-1]
        a = -dt*nu/h**2-dt*v/(2*h)
        c = -dt*nu/h**2+dt*v/(2*h)
        ab = np.zeros((3, n-2))
        ab[1] = 1+2*dt*nu/h**2
        ab[0, 1:] = c[:-1]
        ab[2, :-1] = a[1:]
        rhs = old[1:-1].copy()
        rhs[0] -= a[0]*(1+z)
        rhs[-1] -= c[-1]*(-1)
        u[1:-1] = solve_banded((1, 1), ab, rhs)
        if np.linalg.norm(u-old) < 1e-12:
            break
    yt = np.interp(0, u[::-1], x[::-1])
    yn = solve(z)
    checks.append(dict(z=z, steps=step+1, y_time=yt, y_newton=yn,
                  difference=abs(yt-yn), converged=step < 99999))
save_csv('time_checks.csv', checks)
# Figures and raw plotting data.
plt.rcParams.update(
    {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
for z in [.01, .02, .1]:
    x, u, y = solve(z, ret=True)
    ax[0].plot(x, u, label=f'z={z}')
ax[0].axhline(0, color='gray', lw=.7)
ax[0].set(xlabel='Position x', ylabel='Steady velocity u')
ax[0].legend()
ax[1].plot(zg, yg, label='514-point high fidelity')
plot_z = np.linspace(.01, .014, 401)
ax[1].plot(plot_z, sur(plot_z), '--', label='Low-fidelity surrogate')
ax[1].axhline(.74, color='black', ls=':', label='Failure threshold')
ax[1].set(xlabel='Boundary perturbation z',
          ylabel='Transition position y', xlim=(.01, .014), ylim=(.736, .755))
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG/'burgers.png', dpi=200)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, case in zip(axs, ['Burgers', 'Sliding block']):
    for name in ['MC', 'HF IS', 'MFIS', 'Surrogate IS']:
        errors = []
        for result in allrows:
            if result['case'] == case and result['method'] == name:
                errors.append(result['rmse'])
        ax.loglog(MS, errors, '-o', label=name)
    ax.set(title=case, xlabel='Production sample count n',
           ylabel='Empirical RMSE (40 runs)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG/'errors.png', dpi=200)
plt.close(fig)
speeds = np.linspace(9.6, 10.05, 350)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
axes[0].plot(speeds, block.stopping_distance(
    speeds), label='Work–energy model')
axes[0].plot(speeds, block.tangent_distance(
    speeds), '--', label='Tangent at 9 m/s')
axes[0].axhline(block.LINE_DISTANCE, color='black',
                ls=':', label='Line at 24.01 m')
axes[0].set(xlabel='Initial speed (m/s)', ylabel='Stopping distance (m)')
proposal = proposals['Sliding block']['low']
axes[1].plot(speeds, density(speeds, proposal,
             block.SPEED_BOUNDS), label='Fitted Gaussian')
axes[1].fill_between(speeds, 0, nominal(
    speeds, block.SPEED_BOUNDS), color='black', label='Nominal density')
axes[1].set(xlabel='Initial speed (m/s)', ylabel='Probability density (s/m)')
for axis in axes:
    axis.axvspan(critical_speed, vmax, color='orange',
                 alpha=.15, label='True crossing speeds')
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG/'sliding_block.png', dpi=200)
plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
grid_sizes = []
probabilities = []
for grid in grids:
    grid_sizes.append(grid['points'])
    probabilities.append(grid['probability'])
ax.plot(grid_sizes, probabilities, 'o-', label='Steady finite differences')
ax.axhline(.00549, color='black', ls='--', label='Paper MC reference')
ax.axhspan(.00549-1.96*7.389e-5, .00549+1.96*7.389e-5, color='gray',
           alpha=.2, label='Approx. paper reference 95% interval')
ax.axhline(meta['burgers_continuum_probability'], color='red',
           ls=':', label='Continuum steady reference')
ax.set(xscale='log', xlabel='Spatial grid points', ylabel='Failure probability')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG/'grid.png', dpi=200)
plt.close(fig)
save_csv('summary.csv', allrows)
save_csv('runs.csv', raw)
save_csv('timings.csv', times)
responses = []
for i in range(len(zg)):
    responses.append({'z': zg[i], 'y_high': yg[i], 'y_low': float(sur(zg[i]))})
save_csv('burgers_response.csv', responses)
meta['total_seconds'] = time.perf_counter()-start
(OUT/'metadata.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
