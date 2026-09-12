from mc2_sim.config import HeaterProperties
from mc2_sim.copilot import SuggestedJob, execute_job
from mc2_sim.kinetics import ArrheniusParams


def test_execute_job_single_stage_uses_default_tau_when_unspecified():
    heater = HeaterProperties()
    params = ArrheniusParams()
    job = SuggestedJob(mode="single_stage", rationale="test", T_high_K=1800.0, pulse_on_s=0.02, pulse_off_s=1.08)
    result = execute_job(job, heater, params, default_tau_4sccm_s=15.0)
    assert result["mode"] == "single_stage"
    assert result["tau_s"] == 15.0
    assert 0.0 <= result["conversion_pct"] <= 100.0
    assert result["C2_sel_pct"] + result["C6H6_sel_pct"] + result["Coke_sel_pct"] == 0 or abs(
        result["C2_sel_pct"] + result["C6H6_sel_pct"] + result["Coke_sel_pct"] - 100.0
    ) < 0.1


def test_execute_job_single_stage_respects_explicit_tau():
    heater = HeaterProperties()
    params = ArrheniusParams()
    job = SuggestedJob(
        mode="single_stage", rationale="test", T_high_K=1800.0, pulse_on_s=0.02, pulse_off_s=1.08, tau_s=5.0
    )
    result = execute_job(job, heater, params, default_tau_4sccm_s=15.0)
    assert result["tau_s"] == 5.0


def test_execute_job_array_uses_defaults_when_unspecified():
    heater = HeaterProperties()
    params = ArrheniusParams()
    job = SuggestedJob(mode="array", rationale="test", T_high_K=2000.0, pulse_on_s=0.055, pulse_off_s=1.045)
    result = execute_job(job, heater, params, default_tau_4sccm_s=15.0)
    assert result["mode"] == "array"
    assert result["n_cycles_per_stage"] == 5
    assert result["feed_preheat_K"] == 773.0


def test_execute_job_array_respects_explicit_values():
    heater = HeaterProperties()
    params = ArrheniusParams()
    job = SuggestedJob(
        mode="array",
        rationale="test",
        T_high_K=2000.0,
        pulse_on_s=0.055,
        pulse_off_s=1.045,
        n_cycles_per_stage=10.0,
        feed_preheat_K=600.0,
    )
    result = execute_job(job, heater, params, default_tau_4sccm_s=15.0)
    assert result["n_cycles_per_stage"] == 10
    assert result["feed_preheat_K"] == 600.0
