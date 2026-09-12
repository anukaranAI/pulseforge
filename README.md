# PulseForge — Pulsed Joule-Heating (PHQ) Thermochemical Simulation Engine

*By anukaranAI. Built on the `mc2_sim` Python package (the engine); "PulseForge" is the product
name for the engine + Streamlit front end together (`app.py`).*

**© anukaranAI. All rights reserved.** This repository is source-available for demonstration and
review purposes. No license is granted to use, copy, modify, or redistribute this code without
explicit written permission from anukaranAI.

A first-order (reduced-order, lumped-parameter) Python simulation engine for **programmable
heating and quenching (PHQ)** thermochemical reactors: a porous carbon element is Joule-heated in
millisecond pulses (peak temperatures 1200–2400 K, ~10⁴ K/s heating/cooling rates) to drive
methane pyrolysis down a controlled reaction cascade.

It exists to do two things:

1. **Reproduce and validate against** Dong et al., *"Programmable heating and quenching for
   efficient thermochemical synthesis,"* *Nature* 605, 470–476 (2022) — the peer-reviewed paper
   this technique is based on.
2. **Extend that validated physics** to our own reactor design, which pushes the same mechanism
   further (longer residence, a 4-stage staggered pulse array) to drive the cascade all the way to
   **CH₄ → C(s) + 2 H₂** instead of freezing it at C₂ products.

## The physics, in one paragraph

A thin carbon-paper/felt heater sits in the gas flow path inside a quartz reactor. Electric current
pulses it on for milliseconds (e.g. 0.02 s) then off for about a second, reaching peak temperatures
up to ~2400 K while averaging far lower (e.g. T_avg ≈ 815 K at T_high = 2000 K) — because the pulse
is *shorter than the cyclization timescale*, methane pyrolysis (CH₄ → CH₃• → C₂H₆ → C₂H₄ → C₂H₂ →
C₆H₆ → coke) gets kinetically "frozen" partway down the cascade, giving high selectivity to
whichever product that partial point favors, at a fraction of the energy cost of continuous
heating. The published technique stops the cascade at C₂ products; our design instead uses
**longer** exposure (a 4-stage array, so gas sees four consecutive pulses) to drive the *same*
cascade all the way to solid carbon + hydrogen.

## Repository layout

```
mc2_sim/                    the simulation engine itself
├── config.py                typed scenario definitions (pydantic)
├── thermal.py                lumped heater energy-balance model
├── kinetics.py                lumped CH4-pyrolysis reaction network
├── reactor.py                  couples thermal + kinetics, pulsed & continuous modes
├── array.py                     chains reactor stages into our 4-stage staged-array design
├── tea.py                        mass/energy balance checks vs. our design targets
├── sweep.py                       parameter-sweep runner (grid search)
├── optimize.py                     Bayesian optimization (directed search)
└── copilot.py                       Gemini-backed chat assistant ("SimOps Copilot")

validation/                 calibration against real experimental data
├── data/                        digitized datasets, each with a provenance header
├── calibrate_thermal.py          fits thermal.py's heat-loss parameters
├── calibrate_kinetics.py          fits kinetics.py's Arrhenius parameters
├── compare.py                     runs the calibrated model against every data point, reports error
└── fitted_kinetics_params.json     the current best-fit parameters (output of calibrate_kinetics.py)

examples/                   runnable end-to-end scripts
├── reproduce_fig3.py             overlays model predictions on the paper's Fig. 3b-d
└── run_staged_array.py            runs the 4-stage array, checks it against our design's TEA targets

app.py                      Streamlit front end ("PulseForge") — see its own docstring for the tabs

tests/                      pytest unit tests (26 passing) — physics invariants, not calibration accuracy
```

## File-by-file: what each one does and why

### `mc2_sim/config.py`
Typed, validated (pydantic) data classes describing a scenario: `HeaterProperties` (thermal mass,
area, emissivity, convective coefficient), `PulseProgram` (T_high, on/off durations, ambient temp),
`FeedConditions` (flow rate, CH4 fraction, residence time), and `ReactorScenario` bundling them.
Pure data — no physics — so a scenario can be built identically from a CSV row, a calibration
routine, or a UI form.

### `mc2_sim/thermal.py`
A **lumped-capacitance energy balance** for the carbon heater:

```
C_th · dT/dt = P(t) − h·A·(T − T_env) − ε·σ·A·(T⁴ − T_env⁴)
```

(Joule heating in, convective + radiative loss out.) Because the heater's thermal mass is tiny
(<0.033 J/K per the paper), it heats and cools at ~10⁴ K/s — this is *why* PHQ works at all. Given a
target peak temperature and pulse duration, the module root-finds the electrical power needed
(`solve_power_for_target`), then iterates the on/off cycle to a **periodic steady state**
(`periodic_steady_state`) — the temperature the heater actually settles into cycle after cycle,
not just the first pulse. `HeaterProperties`' default area/convective-coefficient values are
*calibrated* (see `validation/calibrate_thermal.py`) against two T_avg values the paper states
explicitly in its figure captions (815 K and 891 K) — this piece validates to within ~6%.

### `mc2_sim/kinetics.py`
The **lumped reaction network**, tracked in mol-carbon (not mol-species) so carbon conservation and
selectivity bookkeeping are exact by construction:

```
          R1a (low Ea)  \
CH4 ---------------------+--> C2H6 --R2--> C2H4 --R3--> C2H2 --R4--> C6H6 --R5--> Coke
          R1b (high Ea) /                                        \--R6--> Coke  [direct]
```

Every step is first order (hence "first-order sim") — a deliberate simplification appropriate for
a reduced-order engineering model, not a mechanistic one. CH₄ activation is split into two parallel
channels (R1a/R1b) rather than one, added specifically because a single Arrhenius step couldn't
reproduce the *shape* of the conversion-vs-temperature curve (see Validation Status below).
H₂ co-product yield is tracked via each step's stoichiometry. The module also exposes
`rate_matrix`/`augmented_rhs` — because every step is first order, the whole network is a **linear**
time-varying ODE at any fixed temperature, which `reactor.py` exploits for speed (see next).

### `mc2_sim/reactor.py`
Couples a heater's temperature-vs-time profile into the kinetics ODE and integrates over a gas
parcel's residence time (the paper's own framing: a fluid element flowing past the heater
experiences a temperature-time history — PFR-in-space is the same as batch-in-time for that
element). Two modes: `run_pulsed_stage` (PHQ) and `run_continuous_stage` (conventional furnace
heating, for the paper's baseline comparisons).

The performance-critical trick: because the kinetics are linear, one full pulse period's evolution
is an exact linear map (a matrix) on the carbon-pool state. Instead of re-solving an ODE for every
one of potentially dozens of pulse cycles, `reactor.py` solves for that map **once**
(`_period_propagator`) and then just multiplies it in — turning what was originally a
multi-minute-per-scenario calculation into tens of milliseconds. This is what makes calibration
against ~50 data points (thousands of reactor evaluations) tractable at all.

### `mc2_sim/array.py`
Our **4-Stage Staggered Parallel Pulse Array** reactor design: chains N single-stage reactors so a
gas parcel that has partially reacted in stage 1 continues reacting in stage 2, 3, 4 (via
`inlet_state`). "Staggered" (each physical channel firing 250 ms out of phase) is a
power-electronics benefit — it smooths peak current draw — not a chemistry one, so the model
doesn't need to represent the phase relationship explicitly, only that the gas sees four
consecutive thermal impulses. Inter-stage heat recovery (feed preheated to 500 °C) is modeled by
raising every stage's quench-target temperature, which both reduces per-pulse electrical power and
raises the baseline temperature between pulses — both real, captured for free by the existing
thermal/kinetics coupling.

### `mc2_sim/tea.py`
Techno-economic sanity checks against our design targets (4.21 kg CH₄/kg H₂, 3.00 kg C/kg H₂,
12.5 kWh/kg H₂), split into two deliberately separate confidence tiers:

- **`stoichiometric_ideal()`** and **`thermodynamic_energy_estimate()`** — exact atomic mass
  balance and standard thermochemistry (reaction enthalpy + sensible heat − preheat credit). These
  do **not** depend on the calibrated kinetics fit at all; they check the design targets against
  textbook physics. This is the highest-confidence result in the whole project (all targets land
  within 1–6% of first-principles calculations).
- **`array_mass_balance()`** — uses the (imperfectly calibrated) kinetics engine to estimate actual
  per-pass conversion. Inherits Phase 1's calibration uncertainty; read as directional, not precise.

Energy demand is deliberately **not** estimated by scaling up the calibrated `HeaterThermalModel` —
that model's heater properties were fit to the paper's specific lab-scale sample, and there's no
given scale-up relationship to our industrial-throughput hardware. Using it directly would silently
conflate a lab sample's heating power with an industrial system's. Standard thermochemistry is
scale-independent and is the honest tool for that specific check.

### `mc2_sim/sweep.py`
Runs the reactor engine over a grid of parameter combinations (e.g. T_high × pulse duration) and
returns a tidy results table — the building block the rest of the orchestration roadmap
(response-surface visualization, Bayesian optimization, agent-driven exploration) sits on top of.
Deliberately generic: `run_sweep(axes, run_fn)` doesn't know or care whether `run_fn` drives the
single-stage reactor or the 4-stage array; `single_stage_run_fn`/`array_run_fn` are the two
ready-made builders for those cases.

Performance matters here specifically because a sweep calls the reactor engine dozens to hundreds
of times: both builders share a thermal-profile cache AND a kinetics-propagator cache (see
`reactor.compute_propagator`) across every call they make, keyed on the pulse program. Since a
sweep holds kinetics parameters fixed and often varies only one axis that actually changes the
pulse program (e.g. T_high vs. residence time, or vs. array cycle count), most grid points end up
sharing both a thermal solve and a propagator with a neighbor — cutting a 5×5 array sweep from 24s
to under 4s in testing. The one case that can't benefit (both swept axes change the pulse program,
so every grid point is genuinely unique) still completes correctly, just without the speedup.

The same caching problem existed one level up in `mc2_sim/array.py`: `four_stage_ppa`'s 4 identical
stages were each re-solving the same thermal profile independently before this was fixed — so the
fix benefits any array run, not just sweeps.

### `mc2_sim/optimize.py`
Bayesian optimization via scikit-optimize's `gp_minimize`: given a target objective (e.g. maximize
H2 yield, or conversion) and a search space (which parameters, and what range), it directs its
evaluations toward promising regions instead of covering a fixed grid — the same active-learning
approach Dong22 itself used for its own process optimization (Fig. 3e-g: initial sampling, GP
surrogate model, qEI acquisition function, iterate). `mc2_sim.sweep` is the fixed-grid counterpart
— useful for seeing the whole response surface; this module is for when you have a specific target
and the space is too large to grid-sweep exhaustively.

Deliberately reuses `mc2_sim.sweep`'s `single_stage_run_fn`/`array_run_fn` builders rather than
duplicating reactor-calling logic, so the optimizer's repeated evaluations inherit the same
thermal-profile/propagator caching whenever the search revisits a nearby pulse program.

### `mc2_sim/copilot.py`
The "SimOps Copilot" — a Gemini-backed chat assistant that answers questions about the session's
simulation results and proposes concrete follow-up runs. Deliberately **not** autonomous:
`ask_copilot` only ever returns `suggested_jobs` (structured, typed proposals); nothing executes
until the user clicks "Run this job" in the UI, which calls `execute_job`.

Uses Gemini's structured-output mode (`response_schema` bound to a Pydantic model) rather than
parsing free text, so `suggested_jobs` are always valid, directly-runnable parameter sets, not
prose the UI would have to interpret. The system prompt bakes in the same calibration caveats
documented in the Validation Status tab (RMSE ~14.5pp, the conversion-curve shape issue, etc.) so
the copilot doesn't overstate confidence when discussing results. Kept independent of Streamlit
(no `import streamlit`) so it's usable/testable standalone; `app.py` resolves the API key (from
`.streamlit/secrets.toml` or a `GEMINI_API_KEY` environment variable) and passes it in.

**Setup:** put your Gemini API key in `.streamlit/secrets.toml` (already `.gitignore`'d):
```toml
GEMINI_API_KEY = "your-key-here"
```
Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Never commit this file or
paste a real key into a chat/issue/commit message — treat it as compromised and regenerate it if
you ever do. For a deployed app (e.g. Streamlit Community Cloud), set this as a platform secret
instead of relying on the local file.

### `validation/data/*.csv`
Experimental data points, **digitized by visual inspection of the actual published figures**
(rendered at 600 dpi from the source PDF, axis-calibrated by eye — not OCR'd or guessed from the
text) rather than approximated. Each file's header documents its source, exact conditions, and an
honest digitization-uncertainty estimate. `dong22_fig3a/b/c/d_*.csv` cover Fig. 3's conversion and
selectivity curves; `dong22_tavg_anchors.csv` holds the two explicit T_avg values used to calibrate
`thermal.py`; `guo14_science_benchmark.csv` is an independent cross-check point from a different
peer-reviewed paper.

### `validation/calibrate_thermal.py`
Fits `HeaterProperties`' area and convective coefficient (thermal mass and emissivity held at
paper-stated/typical values) against the two T_avg anchor points, via `scipy.optimize.least_squares`.
Converges to within ~6% of both anchors with a purely radiative loss term.

### `validation/calibrate_kinetics.py`
Fits the reaction network's Arrhenius parameters (pre-factors and activation energies) plus two
effective gas residence times (4 sccm and 24 sccm flow conditions) against all the digitized Fig. 3
data, via multi-start `scipy.optimize.least_squares`. The file's own comments document five
iterations of real debugging across this project's development — a fixed-Ea assumption that made
one product branch structurally impossible to reproduce, an unconstrained residence-time ratio that
let the optimizer physically invert which flow condition should react faster, a loss function that
rewarded a degenerate "nothing reacts" solution, and finally a residual-reweighting and two-channel
kinetics extension that together brought validation RMSE from 32.5 down to 14.5 percentage points.

### `validation/compare.py`
Runs the current best-fit parameters against every digitized data point and reports the honest,
**unweighted** error (this is the source of truth for "how good is the fit," independent of
whatever weighting `calibrate_kinetics.py` used internally to find those parameters). Produces
`validation/output/parity_plot.png`.

### `validation/fitted_kinetics_params.json`
The current best-fit parameter set — output of `calibrate_kinetics.py`, consumed by
`examples/reproduce_fig3.py` and `examples/run_staged_array.py`.

### `examples/reproduce_fig3.py`
Overlays the calibrated model's predictions on the digitized Fig. 3b–d data points — the clearest
single visual for "does this match the paper." Produces `examples/output/reproduce_fig3.png`.

### `examples/run_staged_array.py`
Runs the 4-stage array at several severities and prints the two-tier TEA comparison described in
`tea.py` above — the main "does the engineering design hold together physically" deliverable.

### `app.py`
The Streamlit front end ("PulseForge") — six tabs covering single-stage runs, the staged array,
parameter sweeps, Bayesian optimization, the SimOps Copilot, and the validation status. See the
module's own docstring for a tab-by-tab breakdown.

### `tests/`
26 pytest unit tests covering **physics invariants that must hold regardless of calibration**:
carbon conservation, correct full-conversion stoichiometry (CH₄ → C + 2H₂), monotonicity of
conversion with temperature/time, thermal model hitting its target peak temperature, sweep/
optimization result shapes, and the first-principles TEA numbers landing near our design targets.
These test that the code is *correct*, not that the calibration is *tight* — that's a different,
ongoing question tracked in `validation/`.

## Validation status (honest summary)

| Piece | Status | Confidence |
|---|---|---|
| Thermal model (heater T_avg) | Matches paper's stated values within ~6% | High |
| Stoichiometric mass balance vs. design targets | Within 0.7–5.5% | High (exact physics) |
| Thermodynamic energy estimate vs. design targets | Within 2.4–10.9% depending on T_high | High (standard thermochemistry) |
| C2 selectivity trend vs. T_high / pulse duration | Right direction and shape, offset by ~5–15 points | Medium |
| Conversion vs. T_high curve shape | Wrong curvature — undershoots below ~1700 K, overshoots above | Low |
| One baseline condition (1273 K continuous) | Not reproduced by any tested configuration | Low — likely a real structural limit of a 0D (spatially-uniform) model |

Overall kinetics fit: **RMSE 14.5 percentage points** across 45 digitized data points (down from
32.5 on the first attempt). Five distinct calibration iterations, documented in
`validation/calibrate_kinetics.py`'s comments, progressively fixed real bugs; the remaining gap
looks like a genuine limitation of a spatially-uniform (0D) lumped model rather than something more
parameter-fitting would resolve — the paper's own Fig. 2g shows a steep spatial temperature gradient
near the heater that this model has no way to represent.

## Running it

```bash
pip install -r requirements.txt

# Re-run calibration (writes validation/fitted_kinetics_params.json)
python -m validation.calibrate_thermal
python -m validation.calibrate_kinetics

# Check validation quality
python -m validation.compare

# Reproduce the paper's Fig. 3b-d
python -m examples.reproduce_fig3

# Run the 4-stage array + TEA comparison
python -m examples.run_staged_array

# Run the test suite
python -m pytest tests/ -v

# Launch the Streamlit app
python -m streamlit run app.py
```

## Roadmap

1. ~~Simulation engine + validation against Dong22~~ — done, see Validation Status above.
2. ~~4-stage array + TEA check~~ — done.
3. ~~Streamlit front end (PulseForge, `app.py`)~~ — done: scenario builder (pulse program, array
   config) with live plots of the reactor's temperature/conversion/selectivity output, the TEA
   comparison, and the full calibration history inline (not just pointers to source files).
4. **Multi-run orchestration** — running many scenarios at scale, eventually agent-driven:
   - ~~4a. Batch/sweep runner (`mc2_sim/sweep.py`)~~ — done, with thermal-profile and
     kinetics-propagator caching shared across a sweep's runs (see file writeup above).
   - ~~4b. Response-surface visualization~~ — done, the "Parameter Sweep" tab in `app.py`,
     mirroring Dong22's own Fig. 3f.
   - ~~4c. Bayesian optimization loop (`mc2_sim/optimize.py`, "Bayesian Optimization" tab)~~ —
     done: directed search via `scikit-optimize`'s `gp_minimize` for parameters hitting a target
     objective, instead of evaluating a fixed grid. This is the same active-learning technique
     Dong22 itself describes (Fig. 3e-g). Reuses `mc2_sim.sweep`'s run_fn builders, so it inherits
     the same thermal/propagator caching whenever the search revisits a nearby pulse program.
   - ~~4d. AI agent layer (`mc2_sim/copilot.py`, "SimOps Copilot" tab)~~ — done for the
     copilot/chat piece: a Gemini-backed assistant that answers questions about session results
     and proposes concrete follow-up single_stage/array runs (structured, typed, directly
     runnable), which the user reviews and runs one at a time — never autonomous. Proposing
     *sweeps* or driving the optimization loop through the chat is a natural follow-up.
