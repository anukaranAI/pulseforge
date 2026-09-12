"""Streamlit front end for mc2_sim -- branded as "PulseForge" (by anukaranAI).

Six tabs:
  1. Single-Stage PHQ  -- replicate Dong22's reactor: set a pulse program, see the heater
     temperature profile and the resulting conversion/selectivity/H2 yield.
  2. Staged Array Reactor -- run our 4-stage staged array design, see per-stage progression and
     the two-tier TEA comparison against its design targets.
  3. Parameter Sweep -- run a grid of scenarios at once and view the result as a response-surface
     heatmap (mirroring Dong22's own Fig. 3f), rather than one scenario at a time. The
     orchestration/optimization/agent roadmap items all build on top of mc2_sim.sweep, which this
     tab is the first consumer of.
  4. Bayesian Optimization -- directed search (mc2_sim.optimize, scikit-optimize's gp_minimize)
     for parameters hitting a target objective, rather than a fixed grid -- the same active-
     learning approach Dong22 itself used (Fig. 3e-g) for its own process optimization.
  5. SimOps Copilot -- a Gemini-backed chat assistant (mc2_sim.copilot) that answers questions
     about results run so far and proposes concrete follow-up simulations, which the user reviews
     and runs one at a time -- never autonomous.
  6. Validation -- the honest calibration status: parity plot, Fig. 3 reproduction, RMSE, and the
     full documented history of what was tried and why (inline, not just a pointer to files), so
     nobody mistakes this for a precision-validated tool.

Run: python -m streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc2_sim.array import four_stage_ppa
from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.copilot import ask_copilot, execute_job
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.optimize import run_bayesian_optimization
from mc2_sim.reactor import run_continuous_stage, run_pulsed_stage
from mc2_sim.sweep import array_run_fn, run_sweep, single_stage_run_fn
from mc2_sim.tea import array_mass_balance, stoichiometric_ideal, thermodynamic_energy_estimate
from mc2_sim.thermal import HeaterThermalModel

ROOT = Path(__file__).parent
FITTED_PARAMS_PATH = ROOT / "validation" / "fitted_kinetics_params.json"
LOGO_PATH = ROOT / "AnukaranAilogo_new5.png"

st.set_page_config(page_title="PulseForge — by anukaranAI", page_icon="⚡", layout="wide")


@st.cache_data
def load_fitted_params() -> dict:
    return json.loads(FITTED_PARAMS_PATH.read_text())


def build_params(d: dict) -> ArrheniusParams:
    return ArrheniusParams(log10_A=np.array(d["log10_A"]), Ea_kJ_per_mol=np.array(d["Ea_kJ_per_mol"]))


fitted = load_fitted_params()
DEFAULT_PARAMS = build_params(fitted)
DEFAULT_HEATER = HeaterProperties()

header_logo, header_title = st.columns([1, 5])
with header_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH))
with header_title:
    st.title("PulseForge")
    st.caption(
        "Pulsed Joule-Heating (PHQ) thermochemical simulation engine, by anukaranAI. Validated "
        "against Dong et al., Nature 605, 470-476 (2022), and extended to our own staged-array "
        "design for methane-to-hydrogen conversion."
    )

if "run_log" not in st.session_state:
    st.session_state.run_log = []  # every completed simulation this session, for the copilot's context
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pending_jobs" not in st.session_state:
    st.session_state.pending_jobs = []  # SuggestedJob list awaiting user approval


def log_run(entry: dict) -> None:
    st.session_state.run_log.append(entry)


tab_single, tab_array, tab_sweep, tab_optimize, tab_copilot, tab_validation = st.tabs(
    [
        "Single-Stage PHQ",
        "Staged Array Reactor",
        "Parameter Sweep",
        "Bayesian Optimization",
        "SimOps Copilot",
        "Validation Status",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Single-stage PHQ reactor
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Single-stage pulsed heater reactor")
    st.write(
        "Replicates the Dong22 reactor: one carbon-paper heater, pulsed between a peak and "
        "ambient/preheat temperature. Set a pulse program below and run it."
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        mode = st.radio("Heating mode", ["Pulsed (PHQ)", "Continuous (furnace)"], horizontal=True)
        T_high = st.slider("T_high (K)", 1000, 2400, 1800, step=50)
        if mode == "Pulsed (PHQ)":
            pulse_on = st.number_input("Pulse ON duration (s)", 0.005, 0.5, 0.02, step=0.005, format="%.3f")
            pulse_off = st.number_input("Pulse OFF duration (s)", 0.1, 3.0, 1.08, step=0.01, format="%.2f")
            T_env = st.number_input("Ambient / feed temperature (K)", 250.0, 1000.0, 300.0, step=25.0)
        else:
            exposure_s = st.number_input("Exposure / residence time (s)", 0.1, 200.0, 20.0, step=0.5)

        tau = st.number_input(
            "Effective residence time, tau (s)",
            0.1,
            200.0,
            float(fitted["tau_4sccm_s"]),
            step=0.5,
            help="How long a gas parcel dwells near the heater, accumulating exposure to repeated "
            "pulses. Defaults to the calibrated 4 sccm value from validation/.",
            disabled=(mode == "Continuous (furnace)"),
        )
        run_clicked = st.button("Run reactor", type="primary")

    with col_out:
        if run_clicked:
            if mode == "Pulsed (PHQ)":
                pulse = PulseProgram(T_high_K=T_high, pulse_on_s=pulse_on, pulse_off_s=pulse_off, T_env_K=T_env)
                scenario = ReactorScenario(
                    name="ui-single-stage",
                    heater=DEFAULT_HEATER,
                    pulse=pulse,
                    feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=tau),
                )
                cycle = HeaterThermalModel(DEFAULT_HEATER).periodic_steady_state(pulse)
                result = run_pulsed_stage(scenario, DEFAULT_PARAMS, cycle_profile=cycle)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(cycle.t_s, cycle.T_K, color="tab:red")
                ax.set_xlabel("Time within one pulse period (s)")
                ax.set_ylabel("Heater temperature (K)")
                ax.set_title(f"Periodic steady-state pulse profile (T_avg = {result.T_avg_K:.0f} K)")
                st.pyplot(fig)

                log_run(
                    {
                        "mode": "single_stage",
                        "T_high_K": T_high,
                        "pulse_on_s": pulse_on,
                        "pulse_off_s": pulse_off,
                        "tau_s": tau,
                        "conversion_pct": round(result.conversion_pct, 2),
                        "H2_yield_mol_per_mol_CH4": round(result.H2_yield_mol_per_mol_CH4, 3),
                        "C2_sel_pct": round(result.selectivity_pct["C2"], 2),
                        "C6H6_sel_pct": round(result.selectivity_pct["C6H6"], 2),
                        "Coke_sel_pct": round(result.selectivity_pct["Coke"], 2),
                    }
                )
            else:
                result = run_continuous_stage(T_high, exposure_s, DEFAULT_PARAMS)
                log_run(
                    {
                        "mode": "continuous",
                        "T_high_K": T_high,
                        "exposure_s": exposure_s,
                        "conversion_pct": round(result.conversion_pct, 2),
                        "H2_yield_mol_per_mol_CH4": round(result.H2_yield_mol_per_mol_CH4, 3),
                        "C2_sel_pct": round(result.selectivity_pct["C2"], 2),
                        "C6H6_sel_pct": round(result.selectivity_pct["C6H6"], 2),
                        "Coke_sel_pct": round(result.selectivity_pct["Coke"], 2),
                    }
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("CH4 conversion", f"{result.conversion_pct:.2f} %")
            m2.metric("H2 yield", f"{result.H2_yield_mol_per_mol_CH4:.3f} mol/mol CH4")
            m3.metric("Peak temperature", f"{result.T_peak_K:.0f} K")

            st.write("**Product selectivity (% of converted carbon)**")
            sel_df = pd.DataFrame(
                {"Product": list(result.selectivity_pct.keys()), "Selectivity (%)": list(result.selectivity_pct.values())}
            )
            fig2, ax2 = plt.subplots(figsize=(5, 2.5))
            ax2.barh(sel_df["Product"], sel_df["Selectivity (%)"], color=["tab:red", "tab:green", "tab:gray"])
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Selectivity (%)")
            st.pyplot(fig2)
        else:
            st.info("Set a pulse program on the left and click **Run reactor**.")

# ---------------------------------------------------------------------------
# Tab 2: 4-stage array + TEA
# ---------------------------------------------------------------------------
with tab_array:
    st.subheader("4-Stage Staggered Parallel Pulse Array")
    st.write(
        "Chains 4 heater stages so a gas parcel sees four consecutive thermal impulses in one "
        "pass, driving the cascade toward complete decomposition (CH4 -> C(s) + 2 H2) instead of "
        "freezing at C2 products."
    )

    col_in, col_out = st.columns([1, 2])
    with col_in:
        arr_T_high = st.slider("T_high per stage (K)", 1600, 2400, 2000, step=50, key="arr_thigh")
        arr_pulse_on = st.number_input("Pulse ON (s)", 0.02, 0.5, 0.055, step=0.005, key="arr_on")
        arr_pulse_off = st.number_input("Pulse OFF (s)", 0.1, 3.0, 1.045, step=0.01, key="arr_off")
        arr_cycles = st.slider("Pulse cycles per stage", 1, 30, 5, key="arr_cycles")
        arr_preheat_C = st.slider("Inter-stage feed preheat (deg C)", 0, 800, 500, step=25, key="arr_preheat")
        arr_run = st.button("Run 4-stage array", type="primary")

    with col_out:
        if arr_run:
            arr_result = four_stage_ppa(
                T_high_K=arr_T_high,
                pulse_on_s=arr_pulse_on,
                pulse_off_s=arr_pulse_off,
                n_cycles_per_stage=arr_cycles,
                heater=DEFAULT_HEATER,
                params=DEFAULT_PARAMS,
                feed_preheat_K=273.15 + arr_preheat_C,
            )
            log_run(
                {
                    "mode": "array",
                    "T_high_K": arr_T_high,
                    "pulse_on_s": arr_pulse_on,
                    "pulse_off_s": arr_pulse_off,
                    "n_cycles_per_stage": arr_cycles,
                    "feed_preheat_K": 273.15 + arr_preheat_C,
                    "conversion_pct": round(arr_result.overall.conversion_pct, 2),
                    "H2_yield_mol_per_mol_CH4": round(arr_result.overall.H2_yield_mol_per_mol_CH4, 3),
                    "C2_sel_pct": round(arr_result.overall.selectivity_pct["C2"], 2),
                    "C6H6_sel_pct": round(arr_result.overall.selectivity_pct["C6H6"], 2),
                    "Coke_sel_pct": round(arr_result.overall.selectivity_pct["Coke"], 2),
                }
            )

            stage_df = pd.DataFrame(
                {
                    "Stage": [f"Stage {i+1}" for i in range(4)],
                    "Cumulative conversion (%)": [s.conversion_pct for s in arr_result.per_stage],
                    "H2 yield (mol/mol CH4)": [s.H2_yield_mol_per_mol_CH4 for s in arr_result.per_stage],
                    "Coke selectivity (%)": [s.selectivity_pct["Coke"] for s in arr_result.per_stage],
                }
            )
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            ax3.plot(stage_df["Stage"], stage_df["Cumulative conversion (%)"], "o-", label="Conversion (%)")
            ax3.plot(stage_df["Stage"], stage_df["Coke selectivity (%)"], "s-", label="Coke selectivity (%)")
            ax3.set_ylabel("%")
            ax3.legend()
            ax3.set_title("Progression through the 4 stages")
            st.pyplot(fig3)
            st.dataframe(stage_df, hide_index=True, use_container_width=True)

            st.divider()
            st.write("### TEA comparison vs. design targets")

            st.caption(
                "Tier 1 (below) is first-principles and does NOT depend on the kinetics fit -- exact "
                "stoichiometry and standard thermochemistry. Tier 2 uses the calibrated kinetics "
                "(RMSE ~14.5 percentage points against digitized paper data) and should be read as "
                "directional. See the Validation Status tab."
            )

            t1c1, t1c2 = st.columns(2)
            with t1c1:
                s = stoichiometric_ideal()
                st.metric(
                    "Stoichiometric ideal: kg CH4 / kg H2",
                    f"{s.kg_ch4_per_kg_h2:.2f}",
                    delta=f"{s.kg_ch4_per_kg_h2 - 4.21:+.2f} vs. design target 4.21",
                    delta_color="off",
                )
                st.metric(
                    "Stoichiometric ideal: kg C / kg H2",
                    f"{s.kg_c_per_kg_h2:.2f}",
                    delta=f"{s.kg_c_per_kg_h2 - 3.00:+.2f} vs. design target 3.00",
                    delta_color="off",
                )
            with t1c2:
                e = thermodynamic_energy_estimate(T_high_K=arr_T_high, T_preheat_K=273.15 + arr_preheat_C)
                st.metric(
                    "Thermodynamic minimum energy",
                    f"{e.kWh_per_kg_h2:.2f} kWh/kg H2",
                    delta=f"{e.kWh_per_kg_h2 - 12.5:+.2f} vs. design target 12.5",
                    delta_color="off",
                )

            mb = array_mass_balance(
                conversion_pct=arr_result.overall.conversion_pct,
                h2_yield_mol_per_mol_ch4=arr_result.overall.H2_yield_mol_per_mol_CH4,
                coke_selectivity_pct=arr_result.overall.selectivity_pct["Coke"],
                c2_plus_c6h6_selectivity_pct=arr_result.overall.selectivity_pct["C2"]
                + arr_result.overall.selectivity_pct["C6H6"],
            )
            st.write("**Tier 2 (kinetics-dependent) one-pass performance**")
            t2c1, t2c2, t2c3 = st.columns(3)
            t2c1.metric("One-pass conversion", f"{mb.conversion_pct:.1f} %", help="Design target: >75%")
            t2c2.metric("Still-gaseous C2/C6H6", f"{mb.unconverted_gaseous_hydrocarbon_selectivity_pct:.1f} %")
            if np.isfinite(mb.kg_ch4_fed_per_kg_h2_one_pass):
                t2c3.metric("kg CH4 fed / kg H2 (one pass)", f"{mb.kg_ch4_fed_per_kg_h2_one_pass:.2f}")
        else:
            st.info("Set array conditions on the left and click **Run 4-stage array**.")

# ---------------------------------------------------------------------------
# Tab 3: Parameter sweep -- the first consumer of mc2_sim.sweep, the building block for the
# orchestration/optimization/agent roadmap items (see README.md item 4).
# ---------------------------------------------------------------------------
with tab_sweep:
    st.subheader("Parameter sweep — response surface")
    st.write(
        "Runs a grid of scenarios at once and plots the result as a heatmap over two chosen "
        "parameters — the same idea as Dong22's own Fig. 3f response surface (which they generated "
        "via Bayesian-optimization active learning). This is the building block the rest of the "
        "orchestration roadmap (parameter sweeps, optimization loops, agent-driven exploration) "
        "sits on top of."
    )

    SWEEP_AXES = {
        "Single-Stage PHQ": {
            "T_high_K": (1200.0, 2400.0, 1800.0),
            "pulse_on_s": (0.005, 0.5, 0.02),
            "pulse_off_s": (0.1, 3.0, 1.08),
            "tau_s": (0.1, 100.0, float(fitted["tau_4sccm_s"])),
        },
        "4-Stage Array": {
            "T_high_K": (1600.0, 2400.0, 2000.0),
            "pulse_on_s": (0.02, 0.5, 0.055),
            "pulse_off_s": (0.1, 3.0, 1.045),
            "n_cycles_per_stage": (1.0, 30.0, 5.0),
            "feed_preheat_K": (298.0, 1373.0, 773.0),
        },
    }
    METRIC_LABELS = {
        "conversion_pct": "Conversion (%)",
        "H2_yield_mol_per_mol_CH4": "H2 yield (mol/mol CH4)",
        "Coke_sel_pct": "Coke selectivity (%)",
        "C2_sel_pct": "C2 selectivity (%)",
    }

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        sweep_mode = st.radio("Reactor mode", list(SWEEP_AXES.keys()), key="sweep_mode")
        axes_spec = SWEEP_AXES[sweep_mode]
        axis_names = list(axes_spec.keys())

        x_param = st.selectbox("X axis parameter", axis_names, index=0, key="sweep_x_param")
        x_lo_default, x_hi_default, _ = axes_spec[x_param]
        x_lo, x_hi = st.slider(
            f"{x_param} range", x_lo_default, x_hi_default, (x_lo_default, x_hi_default), key="sweep_x_range"
        )
        x_points = st.slider("X grid points", 3, 12, 6, key="sweep_x_points")

        y_param = st.selectbox("Y axis parameter", axis_names, index=1, key="sweep_y_param")
        y_lo_default, y_hi_default, _ = axes_spec[y_param]
        y_lo, y_hi = st.slider(
            f"{y_param} range", y_lo_default, y_hi_default, (y_lo_default, y_hi_default), key="sweep_y_range"
        )
        y_points = st.slider("Y grid points", 3, 12, 6, key="sweep_y_points")

        metric = st.selectbox(
            "Metric to plot", list(METRIC_LABELS.keys()), format_func=lambda k: METRIC_LABELS[k], key="sweep_metric"
        )

        if x_param == y_param:
            st.warning("X and Y axis parameters must be different.")
            sweep_run = False
        else:
            total_runs = x_points * y_points
            st.caption(f"This will run {total_runs} scenarios.")
            sweep_run = st.button("Run sweep", type="primary")

    with col_plot:
        if sweep_run and x_param != y_param:
            fixed = {k: v[2] for k, v in axes_spec.items() if k not in (x_param, y_param)}
            axes = {
                x_param: np.linspace(x_lo, x_hi, x_points).tolist(),
                y_param: np.linspace(y_lo, y_hi, y_points).tolist(),
            }
            run_fn = (
                single_stage_run_fn(DEFAULT_HEATER, DEFAULT_PARAMS, fixed=fixed)
                if sweep_mode == "Single-Stage PHQ"
                else array_run_fn(DEFAULT_HEATER, DEFAULT_PARAMS, fixed=fixed)
            )

            progress_bar = st.progress(0.0, text="Running sweep...")

            def _on_progress(i: int, n: int) -> None:
                progress_bar.progress(i / n, text=f"Running sweep... {i}/{n}")

            sweep_df = run_sweep(axes, run_fn, progress_callback=_on_progress)
            progress_bar.empty()
            st.caption(f"Completed {sweep_df.attrs['n_runs']} runs in {sweep_df.attrs['elapsed_s']:.2f}s.")

            pivot = sweep_df.pivot(index=y_param, columns=x_param, values=metric)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                extent=[x_lo, x_hi, y_lo, y_hi],
                cmap="RdYlGn_r" if "Coke" in metric or metric == "conversion_pct" else "viridis",
            )
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_title(f"{METRIC_LABELS[metric]} response surface")
            fig.colorbar(im, ax=ax, label=METRIC_LABELS[metric])
            st.pyplot(fig)

            with st.expander("Raw sweep results"):
                st.dataframe(sweep_df, hide_index=True, use_container_width=True)
        else:
            st.info("Configure a grid on the left and click **Run sweep**.")

# ---------------------------------------------------------------------------
# Tab 4: Bayesian optimization -- directed search (mc2_sim.optimize) instead of a fixed grid.
# Reuses SWEEP_AXES/METRIC_LABELS defined in the Parameter Sweep tab above.
# ---------------------------------------------------------------------------
with tab_optimize:
    st.subheader("Bayesian optimization — directed search")
    st.write(
        "Instead of evaluating a fixed grid, this searches for the parameters that best hit a "
        "target objective using a Gaussian-process surrogate model (scikit-optimize) — the same "
        "active-learning approach Dong22 itself used (Fig. 3e-g) for its own process optimization. "
        "Useful when you have a specific target rather than wanting the whole surface."
    )

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        opt_mode = st.radio("Reactor mode", list(SWEEP_AXES.keys()), key="opt_mode")
        opt_axes_spec = SWEEP_AXES[opt_mode]
        opt_axis_names = list(opt_axes_spec.keys())

        search_params = st.multiselect(
            "Parameters to search over",
            opt_axis_names,
            default=opt_axis_names[:2],
            key="opt_search_params",
            help="Any parameter not selected here is held fixed at its default value.",
        )

        opt_ranges = {}
        for p in search_params:
            lo_default, hi_default, _ = opt_axes_spec[p]
            opt_ranges[p] = st.slider(f"{p} search range", lo_default, hi_default, (lo_default, hi_default), key=f"opt_range_{p}")

        objective_metric = st.selectbox(
            "Objective",
            list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS[k],
            key="opt_objective",
        )
        direction = st.radio("Direction", ["Maximize", "Minimize"], horizontal=True, key="opt_direction")
        n_calls = st.slider("Number of evaluations", 8, 60, 20, key="opt_n_calls")

        if not search_params:
            st.warning("Select at least one parameter to search over.")
            opt_run = False
        else:
            st.caption(f"This will run {n_calls} scenarios, directed by the optimizer.")
            opt_run = st.button("Run optimization", type="primary")

    with col_plot:
        if opt_run and search_params:
            fixed = {k: v[2] for k, v in opt_axes_spec.items() if k not in search_params}
            axes = {p: opt_ranges[p] for p in search_params}

            progress_bar = st.progress(0.0, text="Optimizing...")

            def _on_opt_progress(i: int, n: int) -> None:
                progress_bar.progress(i / n, text=f"Optimizing... {i}/{n}")

            opt_result = run_bayesian_optimization(
                mode="single_stage" if opt_mode == "Single-Stage PHQ" else "array",
                axes=axes,
                fixed=fixed,
                objective_metric=objective_metric,
                maximize=(direction == "Maximize"),
                heater=DEFAULT_HEATER,
                params=DEFAULT_PARAMS,
                n_calls=n_calls,
                progress_callback=_on_opt_progress,
            )
            progress_bar.empty()
            st.caption(f"Completed {opt_result.n_calls} evaluations in {opt_result.elapsed_s:.2f}s.")

            log_run(
                {
                    "mode": f"optimize:{opt_mode}",
                    "objective": objective_metric,
                    "direction": direction,
                    **{k: round(v, 4) for k, v in opt_result.best_params.items()},
                    "best_value": round(opt_result.best_value, 4),
                }
            )

            m1, m2 = st.columns(2)
            m1.metric(f"Best {METRIC_LABELS[objective_metric]}", f"{opt_result.best_value:.3f}")
            with m2:
                st.write("**Best parameters found**")
                for k, v in opt_result.best_params.items():
                    st.caption(f"{k} = {v:.4g}")

            maximize = direction == "Maximize"
            best_so_far = []
            running_best = None
            for v in opt_result.history[objective_metric]:
                running_best = v if running_best is None else (max(running_best, v) if maximize else min(running_best, v))
                best_so_far.append(running_best)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(range(1, len(best_so_far) + 1), best_so_far, "-", color="tab:blue", label="Best value so far")
            ax.scatter(
                range(1, len(opt_result.history) + 1), opt_result.history[objective_metric], color="tab:gray", s=15, label="Sampled value"
            )
            ax.set_xlabel("Evaluation number")
            ax.set_ylabel(METRIC_LABELS[objective_metric])
            ax.set_title("Optimization convergence")
            ax.legend()
            st.pyplot(fig)

            if len(search_params) == 2:
                p1, p2 = search_params
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sc = ax2.scatter(
                    opt_result.history[p1], opt_result.history[p2], c=opt_result.history[objective_metric], cmap="viridis", s=60
                )
                ax2.set_xlabel(p1)
                ax2.set_ylabel(p2)
                ax2.set_title(f"Sampled points, colored by {METRIC_LABELS[objective_metric]}")
                fig2.colorbar(sc, ax=ax2, label=METRIC_LABELS[objective_metric])
                st.pyplot(fig2)

            with st.expander("Raw optimization history"):
                st.dataframe(opt_result.history, hide_index=True, use_container_width=True)
        else:
            st.info("Configure the search on the left and click **Run optimization**.")

# ---------------------------------------------------------------------------
# Tab 5: SimOps Copilot -- Gemini-backed chat over the session's results, proposes (never runs
# unapproved) concrete follow-up simulations.
# ---------------------------------------------------------------------------
with tab_copilot:
    st.subheader("SimOps Copilot")
    st.write(
        "Ask about results from the runs above, or ask what to try next. The copilot can propose "
        "concrete follow-up simulations — it never runs anything without you clicking **Run this "
        "job** yourself."
    )

    def _get_gemini_key() -> str | None:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
            if key:
                return key
        except Exception:
            pass
        return os.environ.get("GEMINI_API_KEY")

    api_key = _get_gemini_key()

    if not api_key:
        st.warning(
            "No Gemini API key found. Add `GEMINI_API_KEY = \"...\"` to `.streamlit/secrets.toml` "
            "(git-ignored) or set it as an environment variable, then reload this page."
        )
    else:
        col_chat, col_log = st.columns([2, 1])

        with col_log:
            st.write("**Session run log**")
            if st.session_state.run_log:
                st.dataframe(pd.DataFrame(st.session_state.run_log), hide_index=True, use_container_width=True, height=300)
            else:
                st.caption("No simulations run yet this session. Run one from another tab, or ask the copilot to suggest one.")
            if st.button("Clear conversation"):
                st.session_state.chat_messages = []
                st.session_state.pending_jobs = []
                st.rerun()

        with col_chat:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if st.session_state.pending_jobs:
                st.write("##### Suggested simulations — review and run")
                for i, job in enumerate(st.session_state.pending_jobs):
                    with st.container(border=True):
                        st.write(f"**{job.mode}** — {job.rationale}")
                        details = f"T_high={job.T_high_K:.0f} K, on={job.pulse_on_s:.3f}s, off={job.pulse_off_s:.3f}s"
                        if job.mode == "single_stage" and job.tau_s is not None:
                            details += f", tau={job.tau_s:.1f}s"
                        if job.mode == "array":
                            if job.n_cycles_per_stage is not None:
                                details += f", cycles/stage={job.n_cycles_per_stage:.0f}"
                            if job.feed_preheat_K is not None:
                                details += f", preheat={job.feed_preheat_K:.0f}K"
                        st.caption(details)
                        col_run, col_skip = st.columns([1, 1])
                        if col_run.button("▶ Run this job", key=f"run_job_{i}", type="primary"):
                            with st.spinner("Running simulation..."):
                                result = execute_job(job, DEFAULT_HEATER, DEFAULT_PARAMS, fitted["tau_4sccm_s"])
                            log_run(result)
                            st.session_state.pending_jobs.pop(i)
                            st.session_state.chat_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Ran the {job.mode} job: conversion={result['conversion_pct']}%, "
                                    f"H2 yield={result['H2_yield_mol_per_mol_CH4']} mol/mol CH4, "
                                    f"selectivity C2/C6H6/Coke = {result['C2_sel_pct']}/{result['C6H6_sel_pct']}/{result['Coke_sel_pct']}%.",
                                }
                            )
                            st.rerun()
                        if col_skip.button("Dismiss", key=f"skip_job_{i}"):
                            st.session_state.pending_jobs.pop(i)
                            st.rerun()

            user_msg = st.chat_input("Ask about your results, or ask what to simulate next...")
            if user_msg:
                st.session_state.chat_messages.append({"role": "user", "content": user_msg})
                with st.spinner("Thinking..."):
                    reply = ask_copilot(
                        api_key,
                        user_msg,
                        st.session_state.chat_messages[:-1],
                        st.session_state.run_log,
                        fitted["tau_4sccm_s"],
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": reply.answer})
                st.session_state.pending_jobs.extend(reply.suggested_jobs)
                st.rerun()

# ---------------------------------------------------------------------------
# Tab 6: Validation status
# ---------------------------------------------------------------------------
with tab_validation:
    st.subheader("Validation status — read this before trusting a number above")

    st.markdown(
        """
| Piece | Status | Confidence |
|---|---|---|
| Thermal model (heater T_avg) | Matches paper's stated values within ~6% | **High** |
| Stoichiometric mass balance vs. design targets | Within 0.7-5.5% | **High** (exact physics) |
| Thermodynamic energy estimate vs. design targets | Within 2.4-10.9% depending on T_high | **High** (standard thermochemistry) |
| C2 selectivity trend vs. T_high / pulse duration | Right direction/shape, offset ~5-15 points | Medium |
| Conversion vs. T_high curve shape | Wrong curvature -- undershoots low T, overshoots high T | **Low** |
| 1273 K continuous baseline condition | Not reproduced by any tested configuration | **Low** -- likely a real limit of a 0D model |

**Overall kinetics fit: RMSE 14.5 percentage points** across 45 digitized data points from the
paper's Fig. 3 (down from 32.5 on the first calibration attempt), reached over 5 calibration
iterations detailed below.
        """
    )

    st.write("### Calibration history — what was tried, what went wrong, what fixed it")
    st.caption(
        "Every iteration below is a real, distinct debugging cycle from developing this engine, "
        "not a hypothetical. Expand any of them for the full story."
    )

    with st.expander("① Fixed activation energies, fit only pre-factors  →  RMSE 32.5, max error 92pp"):
        st.markdown(
            """
Held all 6 reaction activation energies at literature-informed defaults and fit only the 6
pre-exponential factors plus two effective residence times.

**What went wrong:** benzene (C6H6) selectivity collapsed to ~0% everywhere the fit tried,
against an observed 7-15%. The fixed activation energy for C6H6 → coke (250 kJ/mol) was *lower*
than for C2H2 → C6H6 formation (320 kJ/mol) — so any benzene formed decayed to coke almost as
fast as it formed, at *any* value of the pre-factors. No amount of pre-factor tuning could fix an
ordering problem. Separately, the fitted residence time for the 24 sccm flow condition ended up
*longer* than the 4 sccm condition's — physically backwards for a 6x faster flow rate.
"""
        )

    with st.expander("② Let activation energies float, physically order the two residence times  →  speed wall"):
        st.markdown(
            """
Freed all 6 activation energies (not just the pre-factors) and constrained the 24 sccm residence
time to be ≤ the 4 sccm one, matching the physical expectation that faster flow means shorter
residence.

**What went wrong:** the per-cycle ODE integration became intractable once the fit wanted
~50-90 pulse cycles per evaluation — a single calibration run ran for 30+ minutes without
finishing.

**The fix:** every reaction step in this model is first order, which means the whole kinetics
network is an exactly *linear* system at any fixed temperature. Instead of re-solving an ODE for
every single pulse cycle, the engine now solves ONE ODE for the state-transition matrix over a
single pulse period, then applies that matrix repeatedly (cheap matrix-vector multiplication).
A 100-second exposure (~90 pulse cycles) dropped from an unbounded slow crawl to 0.11 seconds.
"""
        )

    with st.expander("③ Down-weighted the hardest points (soft_l1 loss)  →  backfired into a degenerate fit"):
        st.markdown(
            """
Tried `soft_l1` loss to stop the optimizer from over-fixating on the two hardest-to-reconcile
data points (the 24 sccm baseline conditions).

**What went wrong:** the optimizer found a lazy "almost nothing reacts" solution — near-zero
conversion everywhere trivially satisfies the many low-conversion data points and defaults to
~100% C2 selectivity, while the down-weighting suppressed exactly the gradient pressure that
would have caught this shortcut (one condition was off by 57 percentage points and the optimizer
didn't care).

**The fix:** reverted to plain linear (L2) loss, which keeps every residual's full gradient and
removes the "cheat."
"""
        )

    with st.expander("④ Multi-start optimization + widened, physically-motivated parameter bounds  →  RMSE 17.0 → 16.5"):
        st.markdown(
            """
Added multiple different starting points to the search (to escape local minima) and widened the
activation-energy lower bound after back-of-envelope arithmetic on the data showed the observed
~10x conversion growth from 1200 K to 2000 K implies an apparent activation energy around 50-65
kJ/mol for the dominant low-severity channel — well below the 100 kJ/mol floor originally assumed
(a typical true bond-activation energy), and more consistent with a transport/boundary-layer-
limited apparent rate than a true chemical barrier.
"""
        )

    with st.expander("⑤ Two-channel CH4 activation + reweighted residuals  →  final RMSE 14.5"):
        st.markdown(
            """
Split methane activation into two parallel channels — a low-activation-energy "fast" pathway
alongside the original high-Ea one — specifically to let the model represent "the low-severity
part of the network is rate-limited differently than the high-severity part" without needing full
spatial (1D) resolution.

**Result:** tested with three different seeding strategies, including one built directly from the
transport-limited-channel arithmetic above. In every case, the optimizer pushed the new channel
back toward inactive. This is a genuine finding, not a bug — activating that channel enough to
help the temperature-curve shape would also disturb the (currently near-perfect) 100% C2
selectivity match at low temperatures more than it would help, so the optimizer correctly declines
the trade. It suggests the remaining curve-shape gap is a structural limit of a spatially-uniform
(0D) model — the paper's own Fig. 2g shows a steep temperature gradient near the heater that a
lumped model has no way to represent — rather than something more parameter-fitting can resolve.

**What did help:** reweighting conversion residuals 3x relative to selectivity-split residuals
(since conversion is what actually drives the H2-yield numbers this tool is meant to inform)
brought the honest, unweighted validation RMSE down to the current **14.5 percentage points**.

One condition — 1273 K continuous, 24 sccm flow — remains unfit (model predicts ~0% vs. actual
14% conversion) in every configuration tried, including one where it was excluded from the fit
entirely (which made the *overall* fit worse, so it was kept in).
"""
        )

    col_a, col_b = st.columns(2)
    parity_path = ROOT / "validation" / "output" / "parity_plot.png"
    fig3_path = ROOT / "examples" / "output" / "reproduce_fig3.png"
    with col_a:
        st.write("**Model vs. experiment parity plot**")
        if parity_path.exists():
            st.image(str(parity_path))
        else:
            st.warning("Run `python -m validation.compare` to generate this plot.")
    with col_b:
        st.write("**Fig. 3b-d reproduction**")
        if fig3_path.exists():
            st.image(str(fig3_path))
        else:
            st.warning("Run `python -m examples.reproduce_fig3` to generate this plot.")

    st.write("**Current fitted kinetics parameters**")
    st.json(fitted)
