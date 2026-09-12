"""Parameter-sweep runner: given axes of scenario parameters, runs the reactor engine over every
combination and returns a tidy results table.

This is the prerequisite building block for the rest of the orchestration roadmap (README.md item
4): response-surface visualization (heatmaps of conversion/H2 yield/energy vs. two parameters,
mirroring Dong22's own Fig. 3f), a Bayesian-optimization search loop, and eventually agent-driven
orchestration on top of both.

Deliberately generic: works over EITHER single-stage (mc2_sim.reactor.run_pulsed_stage) or 4-stage
array (mc2_sim.array.four_stage_ppa) scenarios, by taking a `run_fn` that maps one parameter
combination (dict) to a flat dict of outputs -- callers get `single_stage_run_fn` / `array_run_fn`
below as ready-made builders for the two common cases, so this module doesn't need to know which
reactor mode is being swept.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable

import pandas as pd

from mc2_sim.array import four_stage_ppa
from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.reactor import compute_propagator, run_pulsed_stage
from mc2_sim.tea import array_mass_balance
from mc2_sim.thermal import HeaterThermalModel


def cartesian_grid(axes: dict[str, list]) -> list[dict]:
    """axes = {"T_high_K": [1800, 2000, 2200], "pulse_on_s": [0.02, 0.055]} -> one dict per
    combination, e.g. [{"T_high_K": 1800, "pulse_on_s": 0.02}, {"T_high_K": 1800, "pulse_on_s":
    0.055}, ...]."""

    keys = list(axes.keys())
    combos = itertools.product(*(axes[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def run_sweep(
    axes: dict[str, list],
    run_fn: Callable[[dict], dict],
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run run_fn(combo) for every combination in the Cartesian product of axes, collecting each
    combination's parameters plus run_fn's outputs into one row of a results table.

    run_fn should return a flat dict of scalar outputs (e.g. {"conversion_pct": ..., "H2_yield":
    ...}) -- see single_stage_run_fn / array_run_fn below for ready-made builders.
    progress_callback(i, total), if given, is called after each run (wire it to a UI progress bar).
    """

    combos = cartesian_grid(axes)
    rows = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        outputs = run_fn(combo)
        rows.append({**combo, **outputs})
        if progress_callback is not None:
            progress_callback(i + 1, len(combos))
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = time.time() - t0
    df.attrs["n_runs"] = len(combos)
    return df


def single_stage_run_fn(
    heater: HeaterProperties, params: ArrheniusParams, fixed: dict | None = None
) -> Callable[[dict], dict]:
    """Builds a run_fn for sweeping the single-stage PHQ reactor. `fixed` supplies defaults for any
    parameter not swept; combo values override them. Recognized keys: T_high_K, pulse_on_s,
    pulse_off_s, T_env_K, tau_s (effective residence time).

    Caches both the thermal periodic-steady-state solve AND the kinetics propagator (see
    reactor.compute_propagator) per unique (T_high, on, off, T_env) -- when only one of the two
    swept axes actually changes the pulse program (e.g. sweeping T_high against tau_s), most
    combinations in the grid share both and shouldn't redo either (params is fixed for the whole
    sweep, so the propagator, not just the thermal profile, is safe to reuse across combinations
    that share a pulse program)."""

    fixed = fixed or {}
    thermal_model = HeaterThermalModel(heater)
    profile_cache: dict[tuple[float, float, float, float], object] = {}
    propagator_cache: dict[tuple[float, float, float, float], tuple] = {}

    def run_fn(combo: dict) -> dict:
        p = {**fixed, **combo}
        pulse = PulseProgram(
            T_high_K=p["T_high_K"],
            pulse_on_s=p.get("pulse_on_s", 0.02),
            pulse_off_s=p.get("pulse_off_s", 1.08),
            T_env_K=p.get("T_env_K", 300.0),
        )
        scenario = ReactorScenario(
            name="sweep",
            heater=heater,
            pulse=pulse,
            feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=p.get("tau_s", 20.0)),
        )
        key = (pulse.T_high_K, pulse.pulse_on_s, pulse.pulse_off_s, pulse.T_env_K)
        if key not in profile_cache:
            profile_cache[key] = thermal_model.periodic_steady_state(pulse)
        if key not in propagator_cache:
            propagator_cache[key] = compute_propagator(profile_cache[key], pulse, params)
        result = run_pulsed_stage(
            scenario, params, cycle_profile=profile_cache[key], propagator=propagator_cache[key]
        )
        return {
            "conversion_pct": result.conversion_pct,
            "H2_yield_mol_per_mol_CH4": result.H2_yield_mol_per_mol_CH4,
            "C2_sel_pct": result.selectivity_pct["C2"],
            "C6H6_sel_pct": result.selectivity_pct["C6H6"],
            "Coke_sel_pct": result.selectivity_pct["Coke"],
            "T_avg_K": result.T_avg_K,
        }

    return run_fn


def array_run_fn(
    heater: HeaterProperties, params: ArrheniusParams, fixed: dict | None = None
) -> Callable[[dict], dict]:
    """Builds a run_fn for sweeping the 4-stage array. Recognized keys: T_high_K, pulse_on_s,
    pulse_off_s, n_cycles_per_stage, feed_preheat_K.

    Shares one thermal-profile cache AND one propagator cache across every call this run_fn makes
    (see four_stage_ppa/run_staged_array's profile_cache/propagator_cache parameters) -- important
    whenever a sweep axis is n_cycles_per_stage, which doesn't change the pulse program at all, or
    when only one of two swept axes does. params is fixed for the whole sweep, so the propagator
    cache is safe to share the same way the thermal-profile cache is."""

    fixed = fixed or {}
    profile_cache: dict[tuple[float, float, float, float], object] = {}
    propagator_cache: dict[tuple[float, float, float, float], tuple] = {}

    def run_fn(combo: dict) -> dict:
        p = {**fixed, **combo}
        arr = four_stage_ppa(
            T_high_K=p["T_high_K"],
            pulse_on_s=p.get("pulse_on_s", 0.055),
            pulse_off_s=p.get("pulse_off_s", 1.045),
            n_cycles_per_stage=int(p.get("n_cycles_per_stage", 5)),
            heater=heater,
            params=params,
            feed_preheat_K=p.get("feed_preheat_K", 773.0),
            profile_cache=profile_cache,
            propagator_cache=propagator_cache,
        )
        overall = arr.overall
        mb = array_mass_balance(
            conversion_pct=overall.conversion_pct,
            h2_yield_mol_per_mol_ch4=overall.H2_yield_mol_per_mol_CH4,
            coke_selectivity_pct=overall.selectivity_pct["Coke"],
            c2_plus_c6h6_selectivity_pct=overall.selectivity_pct["C2"] + overall.selectivity_pct["C6H6"],
        )
        return {
            "conversion_pct": overall.conversion_pct,
            "H2_yield_mol_per_mol_CH4": overall.H2_yield_mol_per_mol_CH4,
            "Coke_sel_pct": overall.selectivity_pct["Coke"],
            "kg_ch4_fed_per_kg_h2_one_pass": mb.kg_ch4_fed_per_kg_h2_one_pass,
        }

    return run_fn
