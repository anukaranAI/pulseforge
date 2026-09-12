from mc2_sim.config import HeaterProperties
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.optimize import run_bayesian_optimization


def test_optimization_returns_history_with_one_row_per_call():
    heater = HeaterProperties()
    params = ArrheniusParams()
    result = run_bayesian_optimization(
        mode="single_stage",
        axes={"T_high_K": (1200.0, 2400.0)},
        fixed={"pulse_on_s": 0.02, "pulse_off_s": 1.08, "tau_s": 10.0},
        objective_metric="conversion_pct",
        maximize=True,
        heater=heater,
        params=params,
        n_calls=8,
    )
    assert len(result.history) == 8
    assert result.n_calls == 8
    assert "T_high_K" in result.best_params


def test_optimization_finds_a_high_value_when_maximizing():
    heater = HeaterProperties()
    params = ArrheniusParams()
    result = run_bayesian_optimization(
        mode="single_stage",
        axes={"T_high_K": (1200.0, 2400.0)},
        fixed={"pulse_on_s": 0.055, "pulse_off_s": 1.045, "tau_s": 20.0},
        objective_metric="conversion_pct",
        maximize=True,
        heater=heater,
        params=params,
        n_calls=12,
    )
    # best found should be at least as good as the best individually-sampled point
    assert result.best_value >= result.history["conversion_pct"].max() - 1e-6


def test_optimization_progress_callback_fires_n_calls_times():
    calls = []
    heater = HeaterProperties()
    params = ArrheniusParams()
    run_bayesian_optimization(
        mode="single_stage",
        axes={"T_high_K": (1200.0, 2400.0)},
        fixed={"pulse_on_s": 0.02, "pulse_off_s": 1.08, "tau_s": 10.0},
        objective_metric="conversion_pct",
        maximize=True,
        heater=heater,
        params=params,
        n_calls=6,
        progress_callback=lambda i, n: calls.append((i, n)),
    )
    assert len(calls) == 6
    assert calls[-1] == (6, 6)


def test_optimization_array_mode_works():
    heater = HeaterProperties()
    params = ArrheniusParams()
    result = run_bayesian_optimization(
        mode="array",
        axes={"T_high_K": (1800.0, 2400.0)},
        fixed={"pulse_on_s": 0.055, "pulse_off_s": 1.045, "n_cycles_per_stage": 5, "feed_preheat_K": 773.0},
        objective_metric="H2_yield_mol_per_mol_CH4",
        maximize=True,
        heater=heater,
        params=params,
        n_calls=6,
    )
    assert len(result.history) == 6
