"""4-Stage Staggered Parallel Pulse Array (PPA) -- our reactor architecture for driving methane
pyrolysis to completion. Instead of a single carbon-heater element freezing the pyrolysis cascade
at C2 products (Dong22's PHQ regime), this design chains several such heater stages so a gas
parcel encounters multiple consecutive thermal impulses in one pass, driving the SAME cascade
(CH4 -> C2H6 -> C2H4 -> C2H2 -> C6H6 -> Coke, per mc2_sim.kinetics) all the way to completion:
CH4 -> C(s) + 2 H2.

"Staggered" (each stage firing with a phase offset, e.g. 250 ms) is a power-electronics benefit
(smooths peak current draw across the array) rather than a chemistry one -- from the gas's
perspective, what matters is that it passes FOUR heater pulses in sequence, not their exact
relative timing between physical channels. So this module chains N single-stage reactors, each one
picking up where the previous left off (inlet_state), rather than modeling the phase relationship
explicitly.

"Inter-stage heat recovery" (hot exit gas preheats incoming feed to ~500 degC before Stage 1) is
modelled as a single feed_preheat_K applied to every stage's PulseProgram.T_env_K --
the heater now quenches toward the preheated baseline rather than room temperature between pulses,
which both (a) lowers per-pulse electrical power (smaller T_high - T_env gap: see thermal.py) and
(b) raises the "off"-phase baseline temperature the gas sits at between pulses, both physically
real effects of the preheat and both captured by the existing thermal/kinetics coupling without new
model machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.kinetics import ArrheniusParams, initial_state
from mc2_sim.reactor import ReactorResult, compute_propagator, run_pulsed_stage
from mc2_sim.thermal import HeaterThermalModel


@dataclass
class StageSpec:
    pulse: PulseProgram
    n_cycles: int = 1  # how many pulses of THIS stage's heater the gas is exposed to


@dataclass
class ArrayResult:
    overall: ReactorResult
    per_stage: list[ReactorResult]


def run_staged_array(
    stages: list[StageSpec],
    heater: HeaterProperties,
    params: ArrheniusParams,
    feed_preheat_K: float = 300.0,
    profile_cache: dict[tuple[float, float, float, float], object] | None = None,
    propagator_cache: dict[tuple[float, float, float, float], tuple] | None = None,
) -> ArrayResult:
    """Chain N heater stages: the gas exiting stage i becomes the inlet composition for stage i+1.
    Every stage's PulseProgram.T_env_K is overridden to feed_preheat_K (see module docstring).

    profile_cache maps (T_high, on, off, T_env) -> CycleProfile, propagator_cache maps the same key
    -> (M, h2v) (see reactor.compute_propagator). Stages sharing the same pulse program -- the
    common case, e.g. four_stage_ppa's 4 identical channels -- shouldn't each re-solve the thermal
    periodic-steady-state root-find (profile_cache) or the kinetics propagator's two ODE solves
    (propagator_cache) from scratch; neither depends on which stage or what the inlet gas
    composition is, and propagator_cache additionally assumes `params` is the same across every use
    of a given cache instance. Pass external dicts in (and reuse them across calls) to also share
    solves ACROSS separate run_staged_array/four_stage_ppa calls -- e.g. mc2_sim.sweep does this so
    a parameter sweep over many array configurations at fixed kinetics doesn't redo this work on
    every row."""

    state = initial_state()
    per_stage: list[ReactorResult] = []
    thermal_model = HeaterThermalModel(heater)
    if profile_cache is None:
        profile_cache = {}
    if propagator_cache is None:
        propagator_cache = {}

    for stage in stages:
        pulse = stage.pulse.model_copy(update={"T_env_K": feed_preheat_K})
        scenario = ReactorScenario(
            name="array-stage",
            heater=heater,
            pulse=pulse,
            feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=stage.n_cycles * pulse.period_s),
            n_cycles=stage.n_cycles,
        )
        key = (pulse.T_high_K, pulse.pulse_on_s, pulse.pulse_off_s, pulse.T_env_K)
        if key not in profile_cache:
            profile_cache[key] = thermal_model.periodic_steady_state(pulse)
        if key not in propagator_cache:
            propagator_cache[key] = compute_propagator(profile_cache[key], pulse, params)
        result = run_pulsed_stage(
            scenario,
            params,
            cycle_profile=profile_cache[key],
            inlet_state=state,
            propagator=propagator_cache[key],
        )
        per_stage.append(result)
        state = result.raw_state

    return ArrayResult(overall=per_stage[-1], per_stage=per_stage)


def four_stage_ppa(
    T_high_K: float,
    pulse_on_s: float,
    pulse_off_s: float,
    n_cycles_per_stage: int,
    heater: HeaterProperties,
    params: ArrheniusParams,
    feed_preheat_K: float = 773.0,  # 500 degC, our design's target preheat
    profile_cache: dict[tuple[float, float, float, float], object] | None = None,
    propagator_cache: dict[tuple[float, float, float, float], tuple] | None = None,
) -> ArrayResult:
    """Our specific 4-identical-stage configuration: same (T_high, pulse) program
    repeated across all 4 channels, gas passing through each in sequence. See run_staged_array's
    docstring for profile_cache/propagator_cache -- pass shared dicts across repeated calls (e.g.
    from mc2_sim.sweep, at fixed kinetics parameters) to reuse solves across them too, not just
    across this call's 4 stages."""

    pulse = PulseProgram(T_high_K=T_high_K, pulse_on_s=pulse_on_s, pulse_off_s=pulse_off_s)
    stages = [StageSpec(pulse=pulse, n_cycles=n_cycles_per_stage) for _ in range(4)]
    return run_staged_array(
        stages,
        heater,
        params,
        feed_preheat_K=feed_preheat_K,
        profile_cache=profile_cache,
        propagator_cache=propagator_cache,
    )
