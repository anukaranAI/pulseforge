"""SimOps Copilot: a Gemini-backed chat assistant that answers questions about PulseForge
simulation results and proposes concrete follow-up simulations ("SimOps jobs") for the user to
review and run.

Deliberately NOT autonomous: ask_copilot only ever *suggests* jobs (SuggestedJob objects); nothing
runs until the caller (app.py) explicitly executes an approved job via execute_job. This matches
the agreed design -- a copilot that proposes and waits for approval, not one that runs simulations
on its own.

Kept independent of Streamlit (no `import streamlit`) so it's usable/testable standalone -- the
caller resolves the API key (from st.secrets or an environment variable) and passes it in.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from mc2_sim.array import four_stage_ppa
from mc2_sim.config import FeedConditions, HeaterProperties, PulseProgram, ReactorScenario
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.reactor import run_pulsed_stage
from mc2_sim.thermal import HeaterThermalModel

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are the SimOps Copilot inside PulseForge, a pulsed Joule-heating (PHQ)
thermochemical reactor simulation engine built by anukaranAI. You help a materials/process
engineer interpret simulation results and plan follow-up simulations.

THE PHYSICAL SYSTEM: a porous carbon heater is Joule-heated in millisecond pulses (peak
temperature T_high_K, "on" duration pulse_on_s, "off" duration pulse_off_s) inside a quartz
reactor. Methane flows past it and pyrolyzes down a cascade: CH4 -> C2H6 -> C2H4 -> C2H2 -> C6H6 ->
solid coke, releasing H2 at each step. Two run modes:
  - single_stage: one heater, validated against Dong et al., Nature 605, 470-476 (2022).
  - array: our 4-stage staged array design, chaining 4 heater pulses onto the same gas in one pass
    with inter-stage feed preheat (feed_preheat_K), designed to drive the SAME cascade all the way
    to CH4 -> C(s) + 2 H2 instead of freezing partway.

IMPORTANT CALIBRATION CAVEATS -- always be upfront about these, never overstate precision:
  - The kinetics fit has RMSE ~14.5 percentage points against digitized experimental data (45
    points from the paper's Fig. 3). Treat any single run's numbers as directional, not precise.
  - Conversion tends to be UNDERESTIMATED below ~1700 K and OVERESTIMATED above ~1900-2000 K --
    the model's temperature sensitivity is sharper than the real system's.
  - The model under-predicts benzene (C6H6) selectivity (near 0% vs. an observed 7-15%).
  - A 1273 K continuous-heating baseline condition is not well reproduced by any configuration
    tried so far.
  - The mass-balance and energy TEA checks against our design targets (4.21 kg CH4/kg H2, 3.00 kg
    C/kg H2, 12.5 kWh/kg H2), by contrast, are first-principles and NOT subject to the kinetics
    fit's uncertainty -- you can speak about those with high confidence.

YOUR JOB, every turn:
1. Answer the user's question using the run log context provided, in plain, concise, technical
   language (this is a tool for an engineer, not a customer-facing chatbot).
2. When it would genuinely help, propose 1-3 concrete follow-up simulations as suggested_jobs --
   specific, runnable parameter sets (T_high_K, pulse_on_s, pulse_off_s, and mode-specific fields),
   never vague advice like "try a higher temperature" without a number. Each job needs a
   one-sentence rationale tying it to what's already been observed.
3. Only propose single_stage or array jobs -- parameter sweeps and optimization loops aren't
   available as copilot-suggested jobs yet, so don't suggest them as suggested_jobs (you can
   mention the idea in your answer text if relevant).
4. If the run log is empty, you can still answer general questions about the engine/design, but
   say so before suggesting jobs -- you have no results yet to reason from.
5. Never mention any external funding proposal, grant application, or third-party document this
   engine may have been built from -- present the reactor design and its targets as our own.
"""


class SuggestedJob(BaseModel):
    mode: Literal["single_stage", "array"]
    rationale: str = Field(description="One sentence: why this run, given what's been observed so far")
    T_high_K: float
    pulse_on_s: float
    pulse_off_s: float
    tau_s: Optional[float] = Field(default=None, description="single_stage only: effective residence time (s)")
    n_cycles_per_stage: Optional[float] = Field(default=None, description="array only: pulse cycles per stage")
    feed_preheat_K: Optional[float] = Field(default=None, description="array only: inter-stage feed preheat (K)")


class CopilotReply(BaseModel):
    answer: str
    suggested_jobs: list[SuggestedJob] = Field(default_factory=list)


def summarize_run_log(run_log: list[dict], max_entries: int = 15) -> str:
    """Compact text summary of recent runs for the copilot's context window. Most-recent-first,
    capped so the prompt doesn't grow unbounded over a long session."""

    if not run_log:
        return "(no simulations have been run in this session yet)"

    lines = []
    for i, entry in enumerate(reversed(run_log[-max_entries:])):
        lines.append(f"Run {len(run_log) - i}: {entry}")
    return "\n".join(lines)


def ask_copilot(
    api_key: str,
    user_message: str,
    chat_history: list[dict],
    run_log: list[dict],
    default_tau_4sccm_s: float,
) -> CopilotReply:
    """One turn of the copilot conversation. chat_history is a list of {"role": "user"|"assistant",
    "content": str} dicts (the caller's session state); this function does not mutate it.

    Returns a CopilotReply even on API failure (with an explanatory answer and no suggested jobs)
    rather than raising, so a transient Gemini API issue doesn't crash the whole app.

    Retries automatically on ServerError (Gemini's 5xx responses, e.g. 503 UNAVAILABLE for
    "currently experiencing high demand") with a short backoff -- these are common, transient,
    and unrelated to the API key or network being wrong, so silently retrying a couple of times
    before surfacing an error saves the user from manually resending the same message.
    """

    client = genai.Client(api_key=api_key)

    history_text = "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in chat_history)
    context = (
        f"Calibrated default effective residence time (4 sccm flow condition): "
        f"{default_tau_4sccm_s:.2f} s -- use this as tau_s for single_stage jobs unless there's a "
        f"specific reason to change it.\n\n"
        f"SIMULATION RUN LOG (most recent first):\n{summarize_run_log(run_log)}\n\n"
        f"CONVERSATION SO FAR:\n{history_text}\n\n"
        f"NEW USER MESSAGE: {user_message}"
    )

    max_attempts = 3
    backoff_s = 2.0
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=CopilotReply,
                ),
            )
            return response.parsed
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(backoff_s)
                backoff_s *= 2
        except Exception as exc:  # noqa: BLE001 -- non-server errors (bad key, bad request, etc.)
            # aren't transient, so fail immediately rather than retrying.
            return CopilotReply(
                answer=f"Copilot request failed ({type(exc).__name__}: {exc}). Check your API key and network, then try again.",
                suggested_jobs=[],
            )

    return CopilotReply(
        answer=(
            f"Gemini's servers are still reporting high demand after {max_attempts} attempts "
            f"({last_exc}). This is on Google's side, not PulseForge's -- wait a minute and try again."
        ),
        suggested_jobs=[],
    )


def execute_job(job: SuggestedJob, heater: HeaterProperties, params: ArrheniusParams, default_tau_4sccm_s: float) -> dict:
    """Runs one approved SuggestedJob through the actual reactor engine and returns a flat dict of
    outputs suitable for appending to the run log / displaying in the UI."""

    if job.mode == "single_stage":
        tau = job.tau_s if job.tau_s is not None else default_tau_4sccm_s
        pulse = PulseProgram(T_high_K=job.T_high_K, pulse_on_s=job.pulse_on_s, pulse_off_s=job.pulse_off_s)
        scenario = ReactorScenario(
            name="copilot-job",
            heater=heater,
            pulse=pulse,
            feed=FeedConditions(flow_rate_sccm=1.0, residence_time_s=tau),
        )
        cycle = HeaterThermalModel(heater).periodic_steady_state(pulse)
        result = run_pulsed_stage(scenario, params, cycle_profile=cycle)
        return {
            "mode": "single_stage",
            "rationale": job.rationale,
            "T_high_K": job.T_high_K,
            "pulse_on_s": job.pulse_on_s,
            "pulse_off_s": job.pulse_off_s,
            "tau_s": tau,
            "conversion_pct": round(result.conversion_pct, 2),
            "H2_yield_mol_per_mol_CH4": round(result.H2_yield_mol_per_mol_CH4, 3),
            "C2_sel_pct": round(result.selectivity_pct["C2"], 2),
            "C6H6_sel_pct": round(result.selectivity_pct["C6H6"], 2),
            "Coke_sel_pct": round(result.selectivity_pct["Coke"], 2),
        }

    n_cycles = int(job.n_cycles_per_stage) if job.n_cycles_per_stage is not None else 5
    preheat = job.feed_preheat_K if job.feed_preheat_K is not None else 773.0
    arr = four_stage_ppa(
        T_high_K=job.T_high_K,
        pulse_on_s=job.pulse_on_s,
        pulse_off_s=job.pulse_off_s,
        n_cycles_per_stage=n_cycles,
        heater=heater,
        params=params,
        feed_preheat_K=preheat,
    )
    overall = arr.overall
    return {
        "mode": "array",
        "rationale": job.rationale,
        "T_high_K": job.T_high_K,
        "pulse_on_s": job.pulse_on_s,
        "pulse_off_s": job.pulse_off_s,
        "n_cycles_per_stage": n_cycles,
        "feed_preheat_K": preheat,
        "conversion_pct": round(overall.conversion_pct, 2),
        "H2_yield_mol_per_mol_CH4": round(overall.H2_yield_mol_per_mol_CH4, 3),
        "C2_sel_pct": round(overall.selectivity_pct["C2"], 2),
        "C6H6_sel_pct": round(overall.selectivity_pct["C6H6"], 2),
        "Coke_sel_pct": round(overall.selectivity_pct["Coke"], 2),
    }
