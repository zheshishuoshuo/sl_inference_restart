"""Run MCMC sampling for the lens population parameters.

This module provides a thin wrapper around :mod:`emcee` that ties together the
mock-data generation, grid tabulation and likelihood evaluation.

Usage is intentionally simple: supply the mock observed data and a grid in
``logM_halo`` on which the lensing quantities were pre-computed.  The returned
sampler object from :mod:`emcee` can then be further analysed.
"""

from __future__ import annotations

import numpy as np
import emcee
from emcee.backends import HDFBackend

from .likelihood import log_posterior


def run_mcmc(
    grids,
    logM_sps_obs,
    *,
    nwalkers: int = 50,
    nsteps: int = 3000,
    initial_guess: np.ndarray | None = None,
    backend_file: str = "chains_eta.h5",
) -> emcee.EnsembleSampler:
    """Sample the posterior using :mod:`emcee`.

    Parameters
    ----------
    grids:
        List of :class:`~make_tabulate.LensGrid` produced by
        :func:`make_tabulate.tabulate_likelihood_grids`.
    logM_sps_obs:
        Array of observed stellar masses from SPS modelling (``log10`` scale).
    nwalkers, nsteps:
        MCMC configuration.
    initial_guess:
        Initial position of the walkers in parameter space.  Must have length 5
        corresponding to ``(mu0, beta, sigmaDM, mu_alpha, sigma_alpha)``.
    backend_file:
        Path to the HDF5 file used by :class:`emcee.backends.HDFBackend` for
        storing the chain.
    """

    ndim = 5
    if initial_guess is None:
        initial_guess = np.array([12.5, 2.0, 0.3, 0.1, 0.1])

    backend = HDFBackend(backend_file)
    backend.reset(nwalkers, ndim)

    p0 = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=(grids, logM_sps_obs),
        backend=backend,
    )
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler


__all__ = ["run_mcmc"]
