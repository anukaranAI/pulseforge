import pytest

from mc2_sim.config import HeaterProperties, PulseProgram
from mc2_sim.thermal import HeaterThermalModel


def test_periodic_steady_state_hits_target_peak_temperature():
    heater = HeaterProperties()
    model = HeaterThermalModel(heater)
    pulse = PulseProgram(T_high_K=1800.0, pulse_on_s=0.02, pulse_off_s=1.08)
    cycle = model.periodic_steady_state(pulse)
    assert cycle.T_peak_K == pytest.approx(1800.0, abs=1.0)


def test_cycle_cools_back_toward_env_temperature():
    heater = HeaterProperties()
    model = HeaterThermalModel(heater)
    pulse = PulseProgram(T_high_K=2000.0, pulse_on_s=0.02, pulse_off_s=1.08, T_env_K=300.0)
    cycle = model.periodic_steady_state(pulse)
    # off-phase should bring it most of the way back to ambient given >>tau off-time
    assert cycle.T_K[-1] < 0.6 * cycle.T_peak_K


def test_higher_target_temperature_requires_more_power():
    heater = HeaterProperties()
    model = HeaterThermalModel(heater)
    pulse_low = PulseProgram(T_high_K=1400.0, pulse_on_s=0.02, pulse_off_s=1.08)
    pulse_high = PulseProgram(T_high_K=2000.0, pulse_on_s=0.02, pulse_off_s=1.08)
    P_low = model.solve_power_for_target(300.0, pulse_low)
    P_high = model.solve_power_for_target(300.0, pulse_high)
    assert P_high > P_low


def test_longer_pulse_duration_needs_less_power_for_same_peak():
    heater = HeaterProperties()
    model = HeaterThermalModel(heater)
    pulse_short = PulseProgram(T_high_K=1800.0, pulse_on_s=0.02, pulse_off_s=1.08)
    pulse_long = PulseProgram(T_high_K=1800.0, pulse_on_s=0.11, pulse_off_s=0.99)
    P_short = model.solve_power_for_target(300.0, pulse_short)
    P_long = model.solve_power_for_target(300.0, pulse_long)
    assert P_long < P_short
