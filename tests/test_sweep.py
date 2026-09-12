from mc2_sim.config import HeaterProperties
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.sweep import array_run_fn, cartesian_grid, run_sweep, single_stage_run_fn


def test_cartesian_grid_produces_all_combinations():
    grid = cartesian_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(grid) == 6
    assert {"a": 1, "b": 10} in grid
    assert {"a": 2, "b": 30} in grid


def test_run_sweep_single_stage_produces_one_row_per_combo():
    heater = HeaterProperties()
    params = ArrheniusParams()
    run_fn = single_stage_run_fn(heater, params, fixed={"pulse_on_s": 0.02, "pulse_off_s": 1.08, "tau_s": 5.0})
    df = run_sweep({"T_high_K": [1600, 1800, 2000]}, run_fn)
    assert len(df) == 3
    assert "conversion_pct" in df.columns
    # conversion should increase monotonically with T_high, matching the underlying physics
    assert df.sort_values("T_high_K")["conversion_pct"].is_monotonic_increasing


def test_run_sweep_array_produces_one_row_per_combo():
    heater = HeaterProperties()
    params = ArrheniusParams()
    run_fn = array_run_fn(heater, params, fixed={"pulse_on_s": 0.055, "pulse_off_s": 1.045, "n_cycles_per_stage": 3})
    df = run_sweep({"T_high_K": [1800, 2000]}, run_fn)
    assert len(df) == 2
    assert "H2_yield_mol_per_mol_CH4" in df.columns


def test_run_sweep_progress_callback_fires_for_every_combo():
    calls = []
    heater = HeaterProperties()
    params = ArrheniusParams()
    run_fn = single_stage_run_fn(heater, params, fixed={"tau_s": 5.0})
    run_sweep({"T_high_K": [1600, 1800]}, run_fn, progress_callback=lambda i, n: calls.append((i, n)))
    assert calls == [(1, 2), (2, 2)]
