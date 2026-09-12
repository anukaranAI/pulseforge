"""Bayesian optimization loop over reactor parameters (roadmap item 4c), using scikit-optimize's
Gaussian-process minimizer (gp_minimize).

This is the same active-learning approach Dong22 itself used to drive its C2H4-yield search
(Fig. 3e-g: initial sampling, GP surrogate model, qEI acquisition function, iterate) -- balancing
peak temperature, pulse width, and staging against conversion/energy objectives. mc2_sim.sweep is
the fixed-grid counterpart of this: useful when you want to see the whole response surface; this
module is for when you have a specific target and the parameter space is too large to grid-sweep
exhaustively.

Deliberately reuses mc2_sim.sweep's run_fn builders (single_stage_run_fn / array_run_fn) rather
than duplicating reactor-calling logic -- the optimizer just calls the same run_fn repeatedly with
different parameter combinations, so it inherits the same thermal-profile/propagator caching for
free whenever the search revisits a nearby pulse program.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real

from mc2_sim.config import HeaterProperties
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.sweep import array_run_fn, single_stage_run_fn

# Parameters that are naturally integer-valued get an Integer skopt dimension; everything else
# gets Real. Only n_cycles_per_stage (array mode) falls in this set today.
_INTEGER_PARAMS = {"n_cycles_per_stage"}


@dataclass
class OptimizationResult:
    best_params: dict
    best_value: float
    history: pd.DataFrame  # one row per evaluation: swept params + this run_fn's full output dict
    elapsed_s: float
    n_calls: int


def run_bayesian_optimization(
    mode: str,
    axes: dict[str, tuple[float, float]],
    fixed: dict,
    objective_metric: str,
    maximize: bool,
    heater: HeaterProperties,
    params: ArrheniusParams,
    n_calls: int = 20,
    random_state: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OptimizationResult:
    """Search `axes` (param_name -> (low, high) bounds) for the combination that
    maximizes/minimizes `objective_metric` (a key in the chosen run_fn's output dict, e.g.
    "H2_yield_mol_per_mol_CH4"), holding everything else at `fixed`.

    mode: "single_stage" or "array" -- selects which run_fn builder from mc2_sim.sweep to use.
    """

    if mode not in ("single_stage", "array"):
        raise ValueError(f"mode must be 'single_stage' or 'array', got {mode!r}")

    space = []
    param_names = []
    for name, (lo, hi) in axes.items():
        if name in _INTEGER_PARAMS:
            space.append(Integer(int(lo), int(hi), name=name))
        else:
            space.append(Real(float(lo), float(hi), name=name))
        param_names.append(name)

    run_fn = (
        single_stage_run_fn(heater, params, fixed=fixed)
        if mode == "single_stage"
        else array_run_fn(heater, params, fixed=fixed)
    )

    history_rows: list[dict] = []
    call_count = 0

    def objective(x: list[float]) -> float:
        nonlocal call_count
        combo = dict(zip(param_names, x))
        outputs = run_fn(combo)
        value = outputs[objective_metric]
        history_rows.append({**combo, **outputs})
        call_count += 1
        if progress_callback is not None:
            progress_callback(call_count, n_calls)
        return -value if maximize else value

    t0 = time.time()
    # n_initial_points: gp_minimize needs some random exploration before the GP surrogate model is
    # meaningful; skopt's own default (~10 or n_calls//5) is reasonable, but for small n_calls
    # (common in an interactive UI) we want a bit less pure-random fraction.
    n_initial = max(3, n_calls // 4)
    result = gp_minimize(
        objective, space, n_calls=n_calls, n_initial_points=min(n_initial, n_calls), random_state=random_state
    )
    elapsed_s = time.time() - t0

    best_params = dict(zip(param_names, result.x))
    best_value = -result.fun if maximize else result.fun

    return OptimizationResult(
        best_params=best_params,
        best_value=best_value,
        history=pd.DataFrame(history_rows),
        elapsed_s=elapsed_s,
        n_calls=n_calls,
    )
