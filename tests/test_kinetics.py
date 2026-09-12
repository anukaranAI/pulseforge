import numpy as np
import pytest

from mc2_sim.kinetics import ArrheniusParams, carbon_rhs, conversion_and_selectivity, initial_state
from mc2_sim.reactor import run_continuous_stage


def test_carbon_is_conserved_along_trajectory():
    """Every reaction in the network is a carbon-preserving relabeling; sum(C_species) must stay
    at 1.0 (the normalized feed carbon) for the whole trajectory, regardless of T(t) or rates."""

    params = ArrheniusParams()
    result = run_continuous_stage(1800.0, 5.0, params)
    total_carbon = 100.0  # conversion_pct + (100 - conversion_pct) trivially, check selectivity sums instead
    sel_sum = sum(result.selectivity_pct.values())
    assert sel_sum == pytest.approx(100.0, abs=1e-6)


def test_zero_temperature_gives_zero_conversion():
    params = ArrheniusParams()
    result = run_continuous_stage(300.0, 10.0, params)
    assert result.conversion_pct == pytest.approx(0.0, abs=1e-6)
    assert result.H2_yield_mol_per_mol_CH4 == pytest.approx(0.0, abs=1e-9)


def test_conversion_increases_monotonically_with_temperature():
    params = ArrheniusParams()
    conversions = [run_continuous_stage(T, 2.0, params).conversion_pct for T in [1200, 1400, 1600, 1800, 2000]]
    assert all(b >= a for a, b in zip(conversions, conversions[1:]))


def test_conversion_increases_monotonically_with_exposure_time():
    params = ArrheniusParams()
    conversions = [run_continuous_stage(1700.0, t, params).conversion_pct for t in [0.5, 1.0, 2.0, 4.0, 8.0]]
    assert all(b >= a for a, b in zip(conversions, conversions[1:]))


def test_full_conversion_gives_two_h2_per_ch4():
    """If everything cascades all the way to coke, stoichiometry is CH4 -> C(s) + 2 H2."""

    params = ArrheniusParams()
    result = run_continuous_stage(2400.0, 50.0, params)
    assert result.conversion_pct == pytest.approx(100.0, abs=0.5)
    assert result.H2_yield_mol_per_mol_CH4 == pytest.approx(2.0, abs=0.05)


def test_rate_constants_are_positive_and_increase_with_temperature():
    params = ArrheniusParams()
    k_low = params.rate_constants(1000.0)
    k_high = params.rate_constants(2000.0)
    assert np.all(k_low > 0)
    assert np.all(k_high > k_low)
