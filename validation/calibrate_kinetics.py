"""Fit all 6 lumped Arrhenius pre-exponential factors AND activation energies, plus two effective
gas residence times (one per flow-rate group used in Dong22: 4 sccm for Fig. 3b-d, 24 sccm for
Fig. 3a), against the digitized experimental dataset.

A first attempt at this fit held Ea fixed at hand-picked literature-informed defaults and only
floated the 6 pre-factors; it converged (RMSE ~33 percentage points) but structurally could not
reproduce the data: it fixed C6H6-decay's Ea (250 kJ/mol) below C2H2->C6H6-formation's Ea
(320 kJ/mol), which forces any benzene formed to decay to coke before it can accumulate, at ANY
value of the pre-factors -- hence the fit's near-zero C6H6 selectivity everywhere versus the
observed 7-15%. Letting Ea float lets the optimizer find whatever step ordering the data actually
implies, rather than being locked into a hand-picked one.

tau_24sccm is reparametrized as tau_4sccm * frac (frac in (0, 1]) rather than an independent
free residence time. This structurally enforces "faster flow -> shorter or equal residence time"
-- the first attempt let it float independently and it pinned at the upper bound with
tau_24sccm > tau_4sccm, which is physically backwards (24 sccm is 6x faster flow than 4 sccm).

Only conditions that go through the SAME carbon-heater pathway our lumped network represents are
used as fit targets: "carbon_cont" and "carbon_phq" from Fig. 3a (both use the bare carbon paper,
just with/without pulsing) plus all of Fig. 3b/c/d (all PHQ on carbon paper). The Fe/silica-catalyst
and no-catalyst conditions in Fig. 3a use a different active-site chemistry our model doesn't
represent and are excluded from fitting (kept in the CSV for reference/discussion only).

Fig. 3's C10H8 (naphthalene) selectivity is lumped into our "Coke" bucket for comparison, since our
network doesn't carry a separate late-stage-PAH species -- documented simplification.

Run: python -m validation.calibrate_kinetics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.kinetics import ArrheniusParams, N_REACTIONS
from mc2_sim.reactor import run_continuous_stage, run_pulsed_stage
from mc2_sim.thermal import HeaterThermalModel

DATA_DIR = Path(__file__).parent / "data"
RESULTS_PATH = Path(__file__).parent / "fitted_kinetics_params.json"

# Thermal profiles depend only on (T_high, on, off) and the (fixed) heater properties -- not on the
# kinetics parameters being fit -- so solve each unique one once and reuse it for every residual
# evaluation. This is what makes the least_squares loop tractable (each pulsed reactor call would
# otherwise re-run thermal.py's periodic-steady-state root-find every time).
_HEATER = HeaterProperties()
_THERMAL_MODEL = HeaterThermalModel(_HEATER)
_PROFILE_CACHE: dict[tuple[float, float, float], object] = {}


def _get_cycle_profile(T_high: float, on_s: float, off_s: float):
    key = (T_high, on_s, off_s)
    if key not in _PROFILE_CACHE:
        pulse = PulseProgram(T_high_K=T_high, pulse_on_s=on_s, pulse_off_s=off_s)
        _PROFILE_CACHE[key] = _THERMAL_MODEL.periodic_steady_state(pulse)
    return _PROFILE_CACHE[key]


def _load_data() -> dict[str, pd.DataFrame]:
    return {
        "fig3a": pd.read_csv(DATA_DIR / "dong22_fig3a_baseline_comparison.csv", comment="#"),
        "fig3b": pd.read_csv(DATA_DIR / "dong22_fig3b_conversion_vs_thigh.csv", comment="#"),
        "fig3c": pd.read_csv(DATA_DIR / "dong22_fig3c_selectivity_vs_thigh.csv", comment="#"),
        "fig3d": pd.read_csv(DATA_DIR / "dong22_fig3d_selectivity_vs_pulse_duration.csv", comment="#"),
    }


def _build_params(x: np.ndarray) -> tuple[ArrheniusParams, float, float]:
    log10_A = x[:N_REACTIONS]
    Ea = x[N_REACTIONS : 2 * N_REACTIONS]
    log10_tau4 = x[2 * N_REACTIONS]
    log10_frac = x[2 * N_REACTIONS + 1]  # tau_24sccm = tau_4sccm * 10**log10_frac, frac in (0, 1]
    tau_4sccm = 10.0**log10_tau4
    tau_24sccm = tau_4sccm * 10.0**log10_frac
    params = ArrheniusParams(log10_A=log10_A, Ea_kJ_per_mol=Ea)
    return params, tau_4sccm, tau_24sccm


def _model_point_pulsed(T_high, on_s, off_s, tau, params) -> dict:
    scenario = ReactorScenario(
        name="fit",
        heater=_HEATER,
        pulse=PulseProgram(T_high_K=T_high, pulse_on_s=on_s, pulse_off_s=off_s),
        feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=tau),
    )
    cycle_profile = _get_cycle_profile(T_high, on_s, off_s)
    try:
        res = run_pulsed_stage(scenario, params, cycle_profile=cycle_profile)
    except RuntimeError:
        # Rare stiff-ODE excursion during the search (both LSODA and Radau failed). Return a
        # sentinel far outside [0, 100] so this point contributes a large residual and the
        # optimizer steers away, instead of the whole calibration crashing.
        return {"conversion_pct": 1000.0, "C2": 1000.0, "C6H6": 1000.0, "Coke": 1000.0}
    return {"conversion_pct": res.conversion_pct, **res.selectivity_pct}


def _model_point_continuous(T_K, tau, params) -> dict:
    try:
        res = run_continuous_stage(T_K, tau, params)
    except RuntimeError:
        return {"conversion_pct": 1000.0, "C2": 1000.0, "C6H6": 1000.0, "Coke": 1000.0}
    return {"conversion_pct": res.conversion_pct, **res.selectivity_pct}


# Conversion residuals are weighted up relative to selectivity-split residuals. Every prior
# attempt used unit weighting for all 45 residuals, and consistently traded away the
# conversion-vs-T_high curve SHAPE (the primary, physically load-bearing quantity for downstream
# use -- e.g. our H2-yield calculations depend on conversion, not on the C6H6/coke split) to chase
# the selectivity residuals, which outnumber conversion residuals ~2:1 (30 vs 15).
# This makes conversion errors cost proportionally more in the total SSE, without excluding any
# data point.
CONVERSION_WEIGHT = 3.0


def residuals(x: np.ndarray, data: dict[str, pd.DataFrame]) -> np.ndarray:
    params, tau4, tau24 = _build_params(x)
    res = []

    for _, row in data["fig3b"].iterrows():
        m = _model_point_pulsed(row.T_high_K, 0.02, 1.08, tau4, params)
        res.append(CONVERSION_WEIGHT * (m["conversion_pct"] - row.CH4_conversion_pct))

    for _, row in data["fig3c"].iterrows():
        m = _model_point_pulsed(row.T_high_K, 0.02, 1.08, tau4, params)
        res.append(CONVERSION_WEIGHT * (m["conversion_pct"] - row.CH4_conversion_pct))
        res.append(m["C2"] - row.C2_selectivity_pct)
        res.append(m["C6H6"] - row.C6H6_selectivity_pct)
        data_coke = row.C10H8_selectivity_pct + row.coke_selectivity_pct
        res.append(m["Coke"] - data_coke)

    for _, row in data["fig3d"].iterrows():
        m = _model_point_pulsed(1800.0, row.pulse_duration_s, 1.1 - row.pulse_duration_s, tau4, params)
        res.append(CONVERSION_WEIGHT * (m["conversion_pct"] - row.CH4_conversion_pct))
        res.append(m["C2"] - row.C2_selectivity_pct)
        res.append(m["C6H6"] - row.C6H6_selectivity_pct)
        data_coke = row.C10H8_selectivity_pct + row.coke_selectivity_pct
        res.append(m["Coke"] - data_coke)

    # NOTE: "carbon_cont" (1273 K continuous, 24 sccm) was tried as an excluded (not-fit) point in
    # one iteration of this calibration, on the theory that it's structurally irreconcilable with
    # the rest of the dataset. That made the OVERALL fit worse (RMSE 24.85 vs 17.00 on the
    # remaining points): with only carbon_phq left to constrain tau_24sccm, that one parameter
    # became underdetermined and drifted to a worse optimum, while the shared kinetics parameters
    # landed in essentially the same basin either way. So carbon_cont is kept in the fit -- it's
    # poorly reproduced (model ~0% vs. actual 14% conversion) but including it is still better than
    # the alternative. See validation/compare.py output / project notes for the honest accounting
    # of where this lumped 6-reaction scheme does and doesn't match the digitized data.
    fig3a = data["fig3a"].set_index("condition")
    row = fig3a.loc["carbon_cont"]
    m = _model_point_continuous(1273.0, tau24, params)
    res.append(CONVERSION_WEIGHT * (m["conversion_pct"] - row.CH4_conversion_pct))
    res.append(m["C2"] - row.C2_selectivity_pct)
    res.append(m["C6H6"] - row.C6H6_selectivity_pct)
    res.append(m["Coke"] - (row.C10H8_selectivity_pct + row.coke_selectivity_pct))

    row = fig3a.loc["carbon_phq"]
    m = _model_point_pulsed(2200.0, 0.055, 1.045, tau24, params)
    res.append(CONVERSION_WEIGHT * (m["conversion_pct"] - row.CH4_conversion_pct))
    res.append(m["C2"] - row.C2_selectivity_pct)
    res.append(m["C6H6"] - row.C6H6_selectivity_pct)
    res.append(m["Coke"] - (row.C10H8_selectivity_pct + row.coke_selectivity_pct))

    return np.array(res)


def calibrate(seed: int = 0) -> tuple[ArrheniusParams, float, float, "least_squares"]:
    """Multi-start least_squares: a single gradient-based run from one initial guess is prone to
    landing on a degenerate local minimum (an earlier attempt found "almost nothing reacts,
    default to ~100% C2 selectivity" -- see the loss-function note below for how that happened).

    Starts are structured, not fully random: a first fully-random-uniform-over-bounds multi-start
    attempt spent >30 minutes without converging, because sampling log10_A and Ea independently
    across their whole allowed ranges frequently produces pathologically stiff rate-constant
    combinations (e.g. one reaction ~instantaneous, another astronomically slow at the same T),
    which makes even the Radau fallback solver crawl. Instead we jitter around the literature-
    informed defaults (already a physically sane starting point) and vary only tau_4sccm's order
    of magnitude, which is the parameter that actually drove the earlier degenerate minimum.
    """

    data = _load_data()
    rng = np.random.default_rng(seed)

    # log10_A in [0,15]; Ea in [20,450] kJ/mol; tau_4sccm in [0.1,100] s.
    #
    # log10_A's floor was originally 5 and Ea's floor 100 kJ/mol (typical elementary-step
    # pre-factor/activation-energy ranges). Back-of-envelope arithmetic on the digitized data rules
    # that out for R1a specifically: conversion only grows ~10x from 1200 K to 2000 K (Fig. 3b), and
    # matching that growth with the model's actual per-cycle-depletion dynamics implies an apparent
    # Ea of order 50-65 kJ/mol for whatever channel dominates at low-to-moderate severity -- well
    # below 100. The corresponding pre-factor for a rate that small at 1200 K is also small (~10-50
    # /s, not the 1e8-1e15 /s typical of a true elementary gas-phase step). Physically this reads as
    # a transport/boundary-layer-limited apparent rate rather than a true activation barrier -- the
    # paper's Fig. 2g shows a steep spatial temperature gradient near the heater, which this 0D
    # lumped model has no direct way to represent except through a low *apparent* Ea. Widening both
    # floors lets the optimizer actually reach that regime instead of being blocked from it.
    #
    # frac = tau_24sccm / tau_4sccm was originally left free in [0.001, 1] (any ratio <= 1), on the
    # reasoning that faster flow can only mean shorter-or-equal residence. That was too permissive:
    # with no physical anchor on *how much* shorter, the optimizer drove frac to ~0.001 (tau_24sccm
    # ~25 ms), making the model predict essentially zero conversion for both 24 sccm Fig. 3a
    # conditions and abandoning them rather than reconciling them with the 4 sccm dataset -- a
    # residence-time ratio of ~880:1 for a flow-rate ratio of 6:1 (24 vs 4 sccm). If residence time
    # scales as 1/flow_rate for a fixed effective reaction-zone volume, frac should be close to
    # 4/24 = 1/6 ~= 0.167. We bound frac to [0.03, 0.5] -- centered near that 1/6 estimate but wide
    # enough to absorb the fact gas residence isn't purely volumetric (boundary-layer/diffusion
    # effects near the heater) -- which removes the degenerate escape route.
    lower = np.concatenate([np.full(N_REACTIONS, 0.0), np.full(N_REACTIONS, 20.0), [-1.0, np.log10(0.03)]])
    upper = np.concatenate([np.full(N_REACTIONS, 15.0), np.full(N_REACTIONS, 450.0), [2.0, np.log10(0.5)]])

    default_log10_A = ArrheniusParams().log10_A
    default_Ea = ArrheniusParams().Ea_kJ_per_mol
    tau4_starts = [1.0, 5.0, 22.0, 60.0]  # s -- spans "less than one cycle" to "many tens of cycles"

    # Informed seed: the best fit from the PREVIOUS (6-reaction, single-channel-R1) model, with the
    # new R1a channel initialized inactive (log10_A at its floor) so this point reproduces that
    # fit's behavior almost exactly -- i.e. a guaranteed floor for the richer model's cost, since
    # it's a strict superset (R1a=0 recovers the old topology exactly). Added after a first attempt
    # with only the generic structured starts above landed at RMSE 49.8 -- *worse* than the old
    # model's 17.0 -- which shouldn't be possible for a superset model; the generic starts just
    # weren't exploring the new, larger parameter space well enough on their own.
    informed_x0 = np.array(
        [
            5.0, 10.504,  # R1a (inactive), R1b (= old R1)
            13.46, 11.284, 10.441, 9.735, 9.389,  # R2-R6, unchanged from the old fit
            400.0, 380.6,  # Ea: R1a (irrelevant, A is at floor), R1b (= old R1)
            313.7, 301.3, 364.8, 241.6, 353.3,  # Ea: R2-R6
            np.log10(22.269), np.log10(0.693 / 22.269),  # tau4, frac -- old fit's values
        ]
    )
    informed_x0 = np.clip(informed_x0, lower, upper)

    # Transport-limited-channel seed: R1a set from the back-of-envelope arithmetic in the bounds
    # comment above (A~34 /s, Ea~65 kJ/mol -- fit to reproduce the observed ~10x conversion growth
    # from 1200 K to 2000 K on its own), R1b pushed to a higher Ea (420) so it stays dormant across
    # the fitted range and only matters at our design's more extreme severities. This directly
    # tests the "low apparent-Ea transport-limited channel" hypothesis rather than relying on the
    # optimizer to discover that regime from a generic or old-fit-derived start.
    transport_x0 = np.array(
        [
            np.log10(34.0), 12.0,  # R1a (transport-limited hypothesis), R1b (dormant until high T)
            13.46, 11.284, 10.441, 9.735, 9.389,  # R2-R6, unchanged from the old fit
            65.0, 420.0,  # Ea: R1a (transport-limited hypothesis), R1b
            313.7, 301.3, 364.8, 241.6, 353.3,  # Ea: R2-R6
            np.log10(22.269), np.log10(0.693 / 22.269),  # tau4, frac -- old fit's values
        ]
    )
    transport_x0 = np.clip(transport_x0, lower, upper)

    seeds = [("informed (old-fit floor)", informed_x0), ("transport-limited hypothesis", transport_x0)]
    for tau4_0 in tau4_starts:
        jitter_A = default_log10_A + rng.normal(0, 1.0, N_REACTIONS)
        jitter_Ea = default_Ea + rng.normal(0, 30.0, N_REACTIONS)
        jitter_A = np.clip(jitter_A, lower[:N_REACTIONS], upper[:N_REACTIONS])
        jitter_Ea = np.clip(jitter_Ea, lower[N_REACTIONS : 2 * N_REACTIONS], upper[N_REACTIONS : 2 * N_REACTIONS])
        x0 = np.concatenate([jitter_A, jitter_Ea, [np.log10(tau4_0)], [np.log10(1.0 / 6.0)]])
        seeds.append((f"tau4_0={tau4_0}", x0))

    best_result = None
    for label, x0 in seeds:
        # NOTE: an earlier attempt used loss="soft_l1" to down-weight the handful of hard-to-fit
        # high-leverage points (the 24 sccm panel-a conditions). That backfired badly: it let the
        # optimizer settle on a degenerate "almost nothing reacts" solution (near-zero conversion
        # everywhere), which trivially satisfies the many low-conversion data points and defaults
        # to ~100% C2 selectivity, while the down-weighting suppressed the gradient pressure from
        # the conditions that would have caught this (carbon_cont's selectivity was off by 57
        # points). Plain linear (L2) loss keeps every residual's full gradient, which is what
        # actually pushes the fit to get conversion right rather than rewarding a "nothing
        # happens" shortcut.
        result = least_squares(
            residuals, x0, args=(data,), bounds=(lower, upper), method="trf", xtol=1e-11, ftol=1e-11, max_nfev=1500
        )
        print(f"  start '{label}': cost={result.cost:.2f}", flush=True)
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    params, tau4, tau24 = _build_params(best_result.x)
    return params, tau4, tau24, best_result


if __name__ == "__main__":
    params, tau4, tau24, result = calibrate()
    rmse = float(np.sqrt(np.mean(result.fun**2)))
    max_abs_err = float(np.max(np.abs(result.fun)))

    print(f"Converged: {result.success}, cost={result.cost:.3f}")
    print(f"RMSE across all {len(result.fun)} fit residuals: {rmse:.2f} percentage points")
    print(f"Max abs residual: {max_abs_err:.2f} percentage points")
    print(f"tau_4sccm  = {tau4:.3f} s")
    print(f"tau_24sccm = {tau24:.3f} s")
    print("log10_A   =", np.round(params.log10_A, 3).tolist())
    print("Ea_kJ/mol =", np.round(params.Ea_kJ_per_mol, 1).tolist())

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "log10_A": params.log10_A.tolist(),
                "Ea_kJ_per_mol": params.Ea_kJ_per_mol.tolist(),
                "tau_4sccm_s": tau4,
                "tau_24sccm_s": tau24,
                "rmse_pct_points": rmse,
                "max_abs_residual_pct_points": max_abs_err,
            },
            indent=2,
        )
    )
    print(f"\nSaved fitted parameters to {RESULTS_PATH}")
