"""QAOA parameter optimization: p=1 grid scan -> COBYLA polish -> INTERP
extension to deeper p, with multi-start. Optionally a sampled CVaR objective
(focus on the best alpha-quantile of the distribution rather than the mean —
often better for optimization tasks)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..config import rng_for
from ..qubo.ising import Ising
from .circuit import expectation, run_state


@dataclass
class QAOAResult:
    params: np.ndarray
    energy: float
    p: int
    n_evals: int
    wall_time: float
    seed: int
    history: list = field(default_factory=list)


def _cvar_energy(isg: Ising, params: np.ndarray, alpha: float, n_shots: int, seed: int) -> float:
    """Mean of the best alpha-quantile of sampled energies."""
    state = run_state(isg, params)
    samples = state.sampling(n_shots, seed)
    xs = np.array([[(s >> i) & 1 for i in range(isg.n)] for s in set(samples)])
    counts = np.array([samples.count(s) for s in set(samples)])  # small unique sets in practice
    energies = np.array([isg.energy(x) for x in xs])
    order = np.argsort(energies)
    take = max(1, int(np.ceil(alpha * n_shots)))
    acc, total, csum = 0.0, 0, 0
    for k in order:
        c = min(int(counts[k]), take - total)
        acc += c * energies[k]
        total += c
        if total >= take:
            break
    return acc / total


def grid_init_p1(isg: Ising, *, n_gamma: int = 24, n_beta: int = 12) -> np.ndarray:
    """Coarse scan of the p=1 landscape; returns [gamma, beta] at the minimum.

    gamma range scales inversely with the typical |coefficient| so the scan
    covers about one period of the smallest-frequency terms.
    """
    coeffs = np.concatenate([np.abs(isg.h), np.abs(np.array(list(isg.J.values()) or [1.0]))])
    scale = float(np.median(coeffs[coeffs > 0])) or 1.0
    gammas = np.linspace(0.05, np.pi / scale / 2.0, n_gamma)
    betas = np.linspace(0.05, np.pi / 2, n_beta)
    best, best_e = None, np.inf
    for g in gammas:
        for b in betas:
            e = expectation(isg, np.array([g, b]))
            if e < best_e:
                best_e, best = e, (g, b)
    return np.array(best)


def optimize_qaoa(
    isg: Ising,
    *,
    p: int = 1,
    init: np.ndarray | None = None,
    n_starts: int = 4,
    maxiter: int = 100,
    method: str = "COBYLA",
    cvar_alpha: float | None = None,
    cvar_shots: int = 2048,
    seed: int = 0,
) -> QAOAResult:
    t0 = time.perf_counter()
    history: list = []
    evals = 0

    def objective(params: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        if cvar_alpha is not None:
            e = _cvar_energy(isg, params, cvar_alpha, cvar_shots, seed=(seed * 7919 + evals))
        else:
            e = expectation(isg, params)
        history.append(e)
        return e

    starts: list[np.ndarray] = []
    if init is not None:
        starts.append(np.asarray(init, dtype=float))
    rng = rng_for(seed, 0x0A0A)
    while len(starts) < n_starts:
        starts.append(rng.uniform(0.05, 1.0, size=2 * p))

    best_x, best_e = None, np.inf
    for s in starts:
        res = minimize(objective, s, method=method, options={"maxiter": maxiter})
        if res.fun < best_e:
            best_e, best_x = float(res.fun), np.asarray(res.x)

    return QAOAResult(
        params=best_x,
        energy=best_e,
        p=p,
        n_evals=evals,
        wall_time=time.perf_counter() - t0,
        seed=seed,
        history=history,
    )


def interp_params(params: np.ndarray) -> np.ndarray:
    """INTERP heuristic (Zhou et al. 2020): linearly interpolate the p-level
    (gamma, beta) schedules to p+1 as the next initialization."""
    p = len(params) // 2
    gammas, betas = params[0::2], params[1::2]

    def stretch(v: np.ndarray) -> np.ndarray:
        old = np.linspace(0, 1, p)
        new = np.linspace(0, 1, p + 1)
        return np.interp(new, old, v)

    out = np.empty(2 * (p + 1))
    out[0::2], out[1::2] = stretch(gammas), stretch(betas)
    return out


def p_sweep_interp(isg: Ising, *, p_max: int = 3, seed: int = 0, **kw) -> list:
    """p=1 grid + polish, then INTERP up to p_max. Returns all levels' results."""
    results = []
    init = grid_init_p1(isg)
    res = optimize_qaoa(isg, p=1, init=init, seed=seed, **kw)
    results.append(res)
    for p in range(2, p_max + 1):
        init = interp_params(results[-1].params)
        res = optimize_qaoa(isg, p=p, init=init, seed=seed, **kw)
        results.append(res)
    return results


def solve_qaoa(qubo, *, p: int = 2, n_starts: int = 4, maxiter: int = 100,
               n_shots: int = 10_000, cvar_alpha: float | None = None, seed: int = 0):
    """One-call QAOA solve: optimize (p-sweep to `p`), sample, extract.

    Returns (Solution, list[QAOAResult]).
    """
    from ..qubo.ising import qubo_to_ising
    from .extract import extract_solution, sample_bitstrings

    t0 = time.perf_counter()
    isg = qubo_to_ising(qubo)
    results = p_sweep_interp(
        isg, p_max=p, seed=seed, n_starts=n_starts, maxiter=maxiter, cvar_alpha=cvar_alpha
    )
    best = min(results, key=lambda r: r.energy)
    samples = sample_bitstrings(isg, best.params, n_shots=n_shots, seed=seed)
    sol = extract_solution(samples, qubo, source="qaoa")
    sol.wall_time = time.perf_counter() - t0
    sol.meta.update(p=best.p, energy=best.energy, n_evals=sum(r.n_evals for r in results))
    return sol, results
