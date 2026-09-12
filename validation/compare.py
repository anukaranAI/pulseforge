"""Run the (calibrated) reactor model at every digitized experimental condition and report/plot
model-vs-data agreement. This is the validation "gate": if this doesn't line up reasonably well,
the engine isn't trustworthy enough to extrapolate to our staged-array design's regime.

Run: python -m validation.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.reactor import run_continuous_stage, run_pulsed_stage
from mc2_sim.thermal import HeaterThermalModel

DATA_DIR = Path(__file__).parent / "data"
FITTED_PARAMS_PATH = Path(__file__).parent / "fitted_kinetics_params.json"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_fitted_params() -> tuple[ArrheniusParams, float, float]:
    d = json.loads(FITTED_PARAMS_PATH.read_text())
    params = ArrheniusParams(log10_A=np.array(d["log10_A"]), Ea_kJ_per_mol=np.array(d["Ea_kJ_per_mol"]))
    return params, d["tau_4sccm_s"], d["tau_24sccm_s"]


def _pulsed(T_high, on_s, off_s, tau, params, heater) -> dict:
    scenario = ReactorScenario(
        name="validate",
        heater=heater,
        pulse=PulseProgram(T_high_K=T_high, pulse_on_s=on_s, pulse_off_s=off_s),
        feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=tau),
    )
    cycle = HeaterThermalModel(heater).periodic_steady_state(scenario.pulse)
    res = run_pulsed_stage(scenario, params, cycle_profile=cycle)
    return {"conversion_pct": res.conversion_pct, **res.selectivity_pct}


def _continuous(T_K, tau, params) -> dict:
    res = run_continuous_stage(T_K, tau, params)
    return {"conversion_pct": res.conversion_pct, **res.selectivity_pct}


def build_parity_table() -> pd.DataFrame:
    params, tau4, tau24 = load_fitted_params()
    heater = HeaterProperties()
    rows = []

    fig3b = pd.read_csv(DATA_DIR / "dong22_fig3b_conversion_vs_thigh.csv", comment="#")
    for _, r in fig3b.iterrows():
        m = _pulsed(r.T_high_K, 0.02, 1.08, tau4, params, heater)
        rows.append(
            {"source": "Fig3b", "condition": f"T_high={r.T_high_K:.0f}K", "quantity": "conversion_pct",
             "data": r.CH4_conversion_pct, "model": m["conversion_pct"]}
        )

    fig3c = pd.read_csv(DATA_DIR / "dong22_fig3c_selectivity_vs_thigh.csv", comment="#")
    for _, r in fig3c.iterrows():
        m = _pulsed(r.T_high_K, 0.02, 1.08, tau4, params, heater)
        cond = f"T_high={r.T_high_K:.0f}K"
        rows.append({"source": "Fig3c", "condition": cond, "quantity": "conversion_pct",
                      "data": r.CH4_conversion_pct, "model": m["conversion_pct"]})
        rows.append({"source": "Fig3c", "condition": cond, "quantity": "C2_sel_pct",
                      "data": r.C2_selectivity_pct, "model": m["C2"]})
        rows.append({"source": "Fig3c", "condition": cond, "quantity": "C6H6_sel_pct",
                      "data": r.C6H6_selectivity_pct, "model": m["C6H6"]})
        rows.append({"source": "Fig3c", "condition": cond, "quantity": "coke+C10H8_sel_pct",
                      "data": r.C10H8_selectivity_pct + r.coke_selectivity_pct, "model": m["Coke"]})

    fig3d = pd.read_csv(DATA_DIR / "dong22_fig3d_selectivity_vs_pulse_duration.csv", comment="#")
    for _, r in fig3d.iterrows():
        m = _pulsed(1800.0, r.pulse_duration_s, 1.1 - r.pulse_duration_s, tau4, params, heater)
        cond = f"pulse={r.pulse_duration_s}s"
        rows.append({"source": "Fig3d", "condition": cond, "quantity": "conversion_pct",
                      "data": r.CH4_conversion_pct, "model": m["conversion_pct"]})
        rows.append({"source": "Fig3d", "condition": cond, "quantity": "C2_sel_pct",
                      "data": r.C2_selectivity_pct, "model": m["C2"]})
        rows.append({"source": "Fig3d", "condition": cond, "quantity": "C6H6_sel_pct",
                      "data": r.C6H6_selectivity_pct, "model": m["C6H6"]})
        rows.append({"source": "Fig3d", "condition": cond, "quantity": "coke+C10H8_sel_pct",
                      "data": r.C10H8_selectivity_pct + r.coke_selectivity_pct, "model": m["Coke"]})

    fig3a = pd.read_csv(DATA_DIR / "dong22_fig3a_baseline_comparison.csv", comment="#").set_index("condition")
    r = fig3a.loc["carbon_cont"]
    m = _continuous(1273.0, tau24, params)
    for q, d, v in [("conversion_pct", r.CH4_conversion_pct, m["conversion_pct"]),
                    ("C2_sel_pct", r.C2_selectivity_pct, m["C2"]),
                    ("C6H6_sel_pct", r.C6H6_selectivity_pct, m["C6H6"]),
                    ("coke+C10H8_sel_pct", r.C10H8_selectivity_pct + r.coke_selectivity_pct, m["Coke"])]:
        rows.append({"source": "Fig3a", "condition": "carbon_cont", "quantity": q, "data": d, "model": v})

    r = fig3a.loc["carbon_phq"]
    m = _pulsed(2200.0, 0.055, 1.045, tau24, params, heater)
    for q, d, v in [("conversion_pct", r.CH4_conversion_pct, m["conversion_pct"]),
                    ("C2_sel_pct", r.C2_selectivity_pct, m["C2"]),
                    ("C6H6_sel_pct", r.C6H6_selectivity_pct, m["C6H6"]),
                    ("coke+C10H8_sel_pct", r.C10H8_selectivity_pct + r.coke_selectivity_pct, m["Coke"])]:
        rows.append({"source": "Fig3a", "condition": "carbon_phq", "quantity": q, "data": d, "model": v})

    df = pd.DataFrame(rows)
    df["abs_error_pct_points"] = (df["model"] - df["data"]).abs()
    return df


def plot_parity(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Fig3a": "tab:red", "Fig3b": "tab:blue", "Fig3c": "tab:green", "Fig3d": "tab:orange"}
    for source, group in df.groupby("source"):
        ax.scatter(group["data"], group["model"], label=source, color=colors.get(source), alpha=0.75)
    lims = [0, 100]
    ax.plot(lims, lims, "k--", linewidth=1, label="parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Experimental value (%)")
    ax.set_ylabel("Model prediction (%)")
    ax.set_title("mc2_sim vs digitized Dong22 Fig. 3 data\n(conversion % and carbon-basis selectivity %)")
    ax.legend()
    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved parity plot to {out_path}")


if __name__ == "__main__":
    df = build_parity_table()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(df.round(2).to_string(index=False))
    print()
    print(f"RMSE: {np.sqrt((df['abs_error_pct_points']**2).mean()):.2f} percentage points")
    print(f"Mean abs error: {df['abs_error_pct_points'].mean():.2f} percentage points")
    print(f"Max abs error: {df['abs_error_pct_points'].max():.2f} percentage points")

    plot_parity(df, OUTPUT_DIR / "parity_plot.png")
