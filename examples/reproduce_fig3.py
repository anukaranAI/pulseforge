"""Reproduce Dong22 Fig. 3b-d as model-curve-over-data-points plots, using the calibrated kinetics
parameters from validation/fitted_kinetics_params.json.

Run: python -m examples.reproduce_fig3
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.reactor import run_pulsed_stage
from mc2_sim.thermal import HeaterThermalModel
from validation.compare import DATA_DIR, load_fitted_params

OUTPUT_DIR = Path(__file__).parent / "output"


def _run(T_high, on_s, off_s, tau, params, heater):
    scenario = ReactorScenario(
        name="reproduce_fig3",
        heater=heater,
        pulse=PulseProgram(T_high_K=T_high, pulse_on_s=on_s, pulse_off_s=off_s),
        feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=tau),
    )
    cycle = HeaterThermalModel(heater).periodic_steady_state(scenario.pulse)
    return run_pulsed_stage(scenario, params, cycle_profile=cycle)


def plot_fig3b(params, tau4, heater, ax):
    data = pd.read_csv(DATA_DIR / "dong22_fig3b_conversion_vs_thigh.csv", comment="#")
    T_model = np.linspace(1150, 2050, 25)
    conv_model = [_run(T, 0.02, 1.08, tau4, params, heater).conversion_pct for T in T_model]

    ax.plot(T_model, conv_model, "-", color="tab:blue", label="model")
    ax.plot(data.T_high_K, data.CH4_conversion_pct, "D", color="black", label="Dong22 Fig. 3b (digitized)")
    ax.set_xlabel("T_high (K)")
    ax.set_ylabel("CH4 conversion (%)")
    ax.set_title("Fig. 3b: conversion vs T_high\n(0.02s on, 1.08s off)")
    ax.legend()


def plot_fig3c(params, tau4, heater, ax):
    data = pd.read_csv(DATA_DIR / "dong22_fig3c_selectivity_vs_thigh.csv", comment="#")
    T_model = np.linspace(1150, 2050, 25)
    C2 = [_run(T, 0.02, 1.08, tau4, params, heater).selectivity_pct["C2"] for T in T_model]

    ax.plot(T_model, C2, "-", color="tab:red", label="model C2 selectivity")
    ax.plot(data.T_high_K, data.C2_selectivity_pct, "s", color="black", label="Dong22 Fig. 3c C2 (digitized)")
    ax.set_xlabel("T_high (K)")
    ax.set_ylabel("C2 selectivity (%)")
    ax.set_title("Fig. 3c: C2 selectivity vs T_high")
    ax.set_ylim(0, 105)
    ax.legend()


def plot_fig3d(params, tau4, heater, ax):
    data = pd.read_csv(DATA_DIR / "dong22_fig3d_selectivity_vs_pulse_duration.csv", comment="#")
    pd_model = np.linspace(0.015, 0.12, 20)
    C2 = [_run(1800.0, p, 1.1 - p, tau4, params, heater).selectivity_pct["C2"] for p in pd_model]

    ax.plot(pd_model, C2, "-", color="tab:green", label="model C2 selectivity")
    ax.plot(
        data.pulse_duration_s, data.C2_selectivity_pct, "^", color="black", label="Dong22 Fig. 3d C2 (digitized)"
    )
    ax.set_xlabel("Pulse duration (s)")
    ax.set_ylabel("C2 selectivity (%)")
    ax.set_title("Fig. 3d: C2 selectivity vs pulse duration\n(T_high = 1800 K)")
    ax.set_ylim(0, 105)
    ax.legend()


if __name__ == "__main__":
    params, tau4, tau24 = load_fitted_params()
    heater = HeaterProperties()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_fig3b(params, tau4, heater, axes[0])
    plot_fig3c(params, tau4, heater, axes[1])
    plot_fig3d(params, tau4, heater, axes[2])
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "reproduce_fig3.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
