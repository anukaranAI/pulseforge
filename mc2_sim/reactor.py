"""Couples thermal.py's heater temperature profile into kinetics.py's carbon-network ODE and
integrates over a gas parcel's residence time, following the same "astronomic time" batch-reactor
view Dong22 uses for its microkinetic comparison (a fluid element flowing past the heater
experiences a temperature-vs-time history; PFR-in-space == batch-in-time for that element).

Two run modes mirror the two branches of Fig. 3a:
  - run_pulsed_stage:      PHQ -- periodic pulsed heater profile from thermal.py
  - run_continuous_stage:  conventional continuous-furnace heating at a fixed temperature
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from mc2_sim.config import PulseProgram, ReactorScenario
from mc2_sim.kinetics import (
    N_SPECIES,
    ArrheniusParams,
    augmented_rhs,
    carbon_rhs,
    conversion_and_selectivity,
    initial_state,
)
from mc2_sim.thermal import CycleProfile, HeaterThermalModel


@dataclass
class ReactorResult:
    conversion_pct: float
    selectivity_pct: dict[str, float]
    H2_yield_mol_per_mol_CH4: float
    T_peak_K: float
    T_avg_K: float
    exposure_time_s: float
    raw_state: np.ndarray  # [C_CH4, C_C2H6, C_C2H4, C_C2H2, C_C6H6, C_Coke, H2] -- feed this into
    # the next stage's initial_state to chain reactors (see array.py) without losing precision to
    # the conversion_pct/selectivity_pct summary.


def _integrate_carbon_network(
    T_of_t, exposure_time_s: float, params: ArrheniusParams, max_step: float, y0: np.ndarray | None = None
) -> np.ndarray:
    if y0 is None:
        y0 = initial_state()
    if exposure_time_s <= 0:
        return y0

    # LSODA is fast and handles the vast majority of (T_of_t, params) combinations fine, but can
    # occasionally hit "Unexpected istate" on very stiff excursions (e.g. during calibration, while
    # the optimizer is trying out extreme rate constants far from the eventual fit). Radau is
    # ~5x slower but unconditionally stable for stiff systems, so it's a good fallback rather than
    # the default -- keeps the common case fast without sacrificing robustness on the rare case.
    for method in ("LSODA", "Radau"):
        sol = solve_ivp(
            carbon_rhs,
            (0.0, exposure_time_s),
            y0,
            args=(T_of_t, params),
            method=method,
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol.success:
            return np.clip(sol.y[:, -1], 0.0, None)

    raise RuntimeError(f"Kinetics ODE integration failed with both solvers: {sol.message}")


def _integrate_augmented(
    T_of_t, duration_s: float, params: ArrheniusParams, max_step: float, Y0: np.ndarray
) -> np.ndarray:
    """Propagate the (N_SPECIES+1) x N_SPECIES fundamental-solution matrix Y forward by
    duration_s. Returns the flattened result (shape (N_SPECIES+1)*N_SPECIES)."""

    if duration_s <= 0:
        return Y0
    for method in ("LSODA", "Radau"):
        sol = solve_ivp(
            augmented_rhs,
            (0.0, duration_s),
            Y0,
            args=(T_of_t, params),
            method=method,
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol.success:
            return sol.y[:, -1]
    raise RuntimeError(f"Augmented (propagator) ODE integration failed with both solvers: {sol.message}")


def _period_propagator(T_on, T_off, on_s: float, off_s: float, params: ArrheniusParams, max_step_on, max_step_off):
    """Solve for the linear state-transition operator over one full pulse period: returns (M, h2v)
    such that, for any starting carbon-pool vector C, one period later:
        C_next = M @ C
        H2_produced_this_period = h2v @ C

    This replaces solving the kinetics ODE once per pulse cycle (expensive when a scenario spans
    dozens of cycles) with ONE ODE solve per period (over a (N_SPECIES+1) x N_SPECIES matrix state
    instead of a single N_SPECIES+1 vector) -- exact because every reaction step is first order, so
    the carbon network is a linear time-varying ODE at any fixed pulse program. Applying M and h2v
    cycle-to-cycle afterward is then just cheap matrix-vector algebra.
    """

    Y0 = np.vstack([np.eye(N_SPECIES), np.zeros((1, N_SPECIES))]).ravel()
    Y1 = _integrate_augmented(T_on, on_s, params, max_step_on, Y0)
    Y2 = _integrate_augmented(T_off, off_s, params, max_step_off, Y1)
    Y2 = Y2.reshape(N_SPECIES + 1, N_SPECIES)
    return Y2[:N_SPECIES, :], Y2[N_SPECIES, :]


def compute_propagator(
    cycle_profile: CycleProfile, pulse: PulseProgram, params: ArrheniusParams
) -> tuple[np.ndarray, np.ndarray]:
    """Public entry point for _period_propagator, for callers that want to cache/reuse it
    themselves (see mc2_sim.sweep and mc2_sim.array) -- e.g. a parameter sweep that holds kinetics
    parameters fixed while only some axes change the pulse program shouldn't recompute this for
    every run that happens to share a (cycle_profile, params) pair."""

    on_t = cycle_profile.t_s[cycle_profile.t_s <= pulse.pulse_on_s]
    on_T = cycle_profile.T_K[: len(on_t)]
    off_t = cycle_profile.t_s[cycle_profile.t_s >= pulse.pulse_on_s] - pulse.pulse_on_s
    off_T = cycle_profile.T_K[-len(off_t) :]

    def T_on(t: float) -> float:
        return float(np.interp(t, on_t, on_T))

    def T_off(t: float) -> float:
        return float(np.interp(t, off_t, off_T))

    max_step_on = pulse.pulse_on_s / 8.0
    max_step_off = max(pulse.pulse_off_s / 10.0, 1e-4)
    return _period_propagator(T_on, T_off, pulse.pulse_on_s, pulse.pulse_off_s, params, max_step_on, max_step_off)


def _result_from_state(state: np.ndarray, T_peak: float, T_avg: float, exposure_time_s: float) -> ReactorResult:
    cs = conversion_and_selectivity(state)
    return ReactorResult(
        conversion_pct=cs["conversion_pct"],
        selectivity_pct={"C2": cs["C2"], "C6H6": cs["C6H6"], "Coke": cs["Coke"]},
        H2_yield_mol_per_mol_CH4=cs["H2_yield_mol_per_mol_CH4"],
        T_peak_K=T_peak,
        T_avg_K=T_avg,
        exposure_time_s=exposure_time_s,
        raw_state=state,
    )


def run_pulsed_stage(
    scenario: ReactorScenario,
    params: ArrheniusParams,
    cycle_profile: CycleProfile | None = None,
    inlet_state: np.ndarray | None = None,
    propagator: tuple[np.ndarray, np.ndarray] | None = None,
) -> ReactorResult:
    """Integrate the carbon network through repeated pulse cycles until the scenario's total
    exposure time is covered, carrying state forward cycle-to-cycle.

    A pulse period is solved once as a linear state-transition operator (matrix M plus an H2-yield
    vector; see compute_propagator/_period_propagator) rather than by re-solving an ODE for every
    cycle. Because every reaction step is first order, one period's evolution is an exact linear
    map on the carbon-pool vector -- so a scenario spanning dozens of cycles costs one ODE solve
    (of a slightly larger, matrix-valued system) plus cheap matrix-vector multiplication per cycle,
    instead of one ODE solve per cycle. This is what makes calibration against many
    (T_high, pulse_duration) points and long effective residence times tractable.

    cycle_profile can be precomputed/reused (e.g. during calibration, where many kinetics-parameter
    values are tried against the same fixed pulse program) to avoid re-solving the thermal
    periodic-steady-state root-find on every call. propagator goes one step further: if the SAME
    (cycle_profile, params) pair recurs across calls -- e.g. mc2_sim.sweep holding kinetics
    parameters fixed while sweeping an axis that doesn't affect the pulse program -- pass in a
    precomputed (M, h2v) from compute_propagator to skip its two ODE solves entirely too.

    inlet_state lets a scenario start from a partially-reacted gas composition instead of fresh
    CH4 -- this is how array.py chains multiple stages (a later stage's gas has already lost some
    CH4 and gained C2/H2/coke from the earlier stages it passed through).
    """

    if cycle_profile is None:
        cycle_profile = HeaterThermalModel(scenario.heater).periodic_steady_state(scenario.pulse)

    pulse = scenario.pulse
    period = pulse.period_s
    exposure_time_s = scenario.feed.residence_time_s or (scenario.n_cycles * period)
    n_full_cycles = int(exposure_time_s // period)
    remainder_s = exposure_time_s - n_full_cycles * period

    on_t = cycle_profile.t_s[cycle_profile.t_s <= pulse.pulse_on_s]
    on_T = cycle_profile.T_K[: len(on_t)]
    off_t = cycle_profile.t_s[cycle_profile.t_s >= pulse.pulse_on_s] - pulse.pulse_on_s
    off_T = cycle_profile.T_K[-len(off_t) :]

    def T_on(t: float) -> float:
        return float(np.interp(t, on_t, on_T))

    def T_off(t: float) -> float:
        return float(np.interp(t, off_t, off_T))

    max_step_on = pulse.pulse_on_s / 8.0
    max_step_off = max(pulse.pulse_off_s / 10.0, 1e-4)

    full_state = initial_state() if inlet_state is None else np.array(inlet_state, dtype=float)
    C = full_state[:N_SPECIES]
    H2 = full_state[N_SPECIES]

    if n_full_cycles > 0:
        if propagator is None:
            propagator = _period_propagator(
                T_on, T_off, pulse.pulse_on_s, pulse.pulse_off_s, params, max_step_on, max_step_off
            )
        M, h2v = propagator
        for _ in range(n_full_cycles):
            H2 += float(h2v @ C)
            C = M @ C
        C = np.clip(C, 0.0, None)

    state = np.concatenate([C, [H2]])
    if remainder_s > 0:
        if remainder_s <= pulse.pulse_on_s:
            state = _integrate_carbon_network(T_on, remainder_s, params, max_step_on, y0=state)
        else:
            state = _integrate_carbon_network(T_on, pulse.pulse_on_s, params, max_step_on, y0=state)
            state = _integrate_carbon_network(
                T_off, remainder_s - pulse.pulse_on_s, params, max_step_off, y0=state
            )

    T_avg = float(np.trapezoid(cycle_profile.T_K, cycle_profile.t_s) / period)
    return _result_from_state(state, cycle_profile.T_peak_K, T_avg, exposure_time_s)


def run_continuous_stage(T_K: float, exposure_time_s: float, params: ArrheniusParams) -> ReactorResult:
    def T_of_t(_t: float) -> float:
        return T_K

    max_step = max(exposure_time_s / 200.0, 1e-4)
    state = _integrate_carbon_network(T_of_t, exposure_time_s, params, max_step)
    return _result_from_state(state, T_K, T_K, exposure_time_s)
