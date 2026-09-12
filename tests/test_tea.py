import pytest

from mc2_sim.tea import stoichiometric_ideal, thermodynamic_energy_estimate


def test_stoichiometric_ideal_matches_design_target_within_10pct():
    s = stoichiometric_ideal()
    assert s.kg_ch4_per_kg_h2 == pytest.approx(4.21, rel=0.10)
    assert s.kg_c_per_kg_h2 == pytest.approx(3.00, rel=0.10)


def test_thermodynamic_energy_estimate_matches_design_target_within_15pct():
    e = thermodynamic_energy_estimate(T_high_K=2000.0, T_preheat_K=773.0)
    assert e.kWh_per_kg_h2 == pytest.approx(12.5, rel=0.15)


def test_energy_estimate_increases_with_peak_temperature():
    e_low = thermodynamic_energy_estimate(T_high_K=1800.0, T_preheat_K=773.0)
    e_high = thermodynamic_energy_estimate(T_high_K=2200.0, T_preheat_K=773.0)
    assert e_high.kWh_per_kg_h2 > e_low.kWh_per_kg_h2


def test_energy_estimate_decreases_with_more_preheat():
    e_no_preheat = thermodynamic_energy_estimate(T_high_K=2000.0, T_preheat_K=298.0)
    e_preheated = thermodynamic_energy_estimate(T_high_K=2000.0, T_preheat_K=773.0)
    assert e_preheated.kWh_per_kg_h2 < e_no_preheat.kWh_per_kg_h2
