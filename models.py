"""Solve the steady one-dimensional viscous Burgers equation on [-1, 1].

The uncertain perturbation changes the left boundary velocity. The model output
is the location where the steady velocity changes sign. The experiment uses a
514-node grid for the reference response and 258 nodes for the surrogate knots.
This implementation was written separately from the paper's code."""

import numpy as np
from scipy.linalg import solve_banded


def solve(z, n=514, ret=False):
    """Compute the zero-crossing position of the discrete steady velocity profile.

    Parameters
    ----------
    z : float
        Left boundary perturbation in the study interval [0.01, 0.1]. Velocity
        is fixed to 1 + z at the left end and -1 at the right end.
    n : int, default 514
        Number of equally spaced spatial nodes, including both endpoints.
    ret : bool, default False
        If true, also return the spatial nodes and converged velocity profile.

    Returns
    -------
    y or (x, u, y)
        Linearly interpolated position where velocity is zero. With ret=True,
        x and u are one-dimensional arrays of length n.

    Raises
    ------
    ValueError
        If the interior residual fails the convergence check within 30 iterations.
        The exception records z, n and the maximum absolute residual."""
        
    nu = .05
    x = np.linspace(-1, 1, n)
    h = x[1]-x[0]
    a = 1+z
    y = 1-nu/a*np.log((a+1)/(a-1))
    u = -a*np.tanh(a*(x-y)/(2*nu))
    u[0] = 1+z
    u[-1] = -1
    converged = False
    for iteration in range(30):
        # Diffusion minus nonlinear transport at all interior nodes.
        v = u[1:-1]
        diffusion = nu * (u[:-2] - 2*v + u[2:]) / h**2
        transport = v * (u[2:] - u[:-2]) / (2*h)
        r = diffusion - transport
        if np.max(abs(r)) < 1e-9:
            converged = True
            break
        # Three Jacobian diagonals in scipy.linalg.solve_banded storage.
        ab = np.zeros((3, n-2))
        ab[1] = -2*nu/h**2-(u[2:]-u[:-2])/(2*h)
        ab[0, 1:] = nu/h**2-v[:-1]/(2*h)
        ab[2, :-1] = nu/h**2+v[1:]/(2*h)
        u[1:-1] += solve_banded((1, 1), ab, -r)
    if not converged:
        raise ValueError((z, n, np.max(abs(r))))
    # Velocity decreases with x; reverse arrays for increasing interpolation data.
    y = np.interp(0, u[::-1], x[::-1])
    if ret:
        return x, u, y
    return y
