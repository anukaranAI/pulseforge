"""Lumped-capacitance thermal model of the pulsed Joule-heated carbon element.

Physics (Dong et al., Nature 2022, Supplementary Discussion 2): the carbon paper/felt heater has
very low heat capacity, so it can be treated as a single lumped thermal mass exchanging heat with
the surrounding gas/walls by convection and radiation:

    C_th * dT/dt = P(t) - h*A*(T - T_env) - eps*sigma*A*(T^4 - T_env^4)

where P(t) is the electrical (I^2*R) power switched on/off according to the pulse program. Because
the heater's thermal time constant is milliseconds while the "off" period is ~1 s, the element
quenches back to (approximately) the ambient/feed-gas temperature between pulses -- this is exactly
the "rapid heating and cooling" (~1e4 K/s) the paper reports.

Given a target peak temperature T_high and an "on" duration, the required heater power is not known
a priori (it depends on the loss terms), so we solve for it by root-finding: find the constant P_on
such that integrating the energy balance for pulse_on_s starting from the cycle's starting
temperature lands exactly on T_high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from mc2_sim.config import HeaterProperties, PulseProgram

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 K^4)


@dataclass
class CycleProfile:
    """Temperature history over one pulse period, sampled densely enough for ODE coupling."""

    t_s: np.ndarray
    T_K: np.ndarray
    P_on_W: float
    T_start_K: float
    T_peak_K: float


class HeaterThermalModel:
    def __init__(self, heater: HeaterProperties):
        self.heater = heater

    def _q_loss(self, T: float, T_env: float) -> float:
        h = self.heater
        conv = h.convective_coeff_w_per_m2k * h.area_m2 * (T - T_env)
        rad = h.emissivity * STEFAN_BOLTZMANN * h.area_m2 * (T**4 - T_env**4)
        return conv + rad

    def _rhs(self, t: float, y: np.ndarray, P: float, T_env: float) -> list[float]:
        (T,) = y
        dTdt = (P - self._q_loss(T, T_env)) / self.heater.thermal_mass_j_per_k
        return [dTdt]

    def _integrate(
        self, T0: float, duration_s: float, P: float, T_env: float, n_points: int = 40
    ) -> tuple[np.ndarray, np.ndarray]:
        t_eval = np.linspace(0.0, duration_s, n_points)
        sol = solve_ivp(
            self._rhs,
            (0.0, duration_s),
            [T0],
            args=(P, T_env),
            t_eval=t_eval,
            method="Radau",
            rtol=1e-7,
            atol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Thermal ODE integration failed: {sol.message}")
        return sol.t, sol.y[0]

    def _peak_temperature(self, T0: float, pulse: PulseProgram, P_on: float, T_env: float) -> float:
        _, T = self._integrate(T0, pulse.pulse_on_s, P_on, T_env, n_points=2)
        return float(T[-1])

    def solve_power_for_target(self, T0: float, pulse: PulseProgram) -> float:
        """Find the constant on-phase power that brings the heater from T0 to pulse.T_high_K
        by the end of pulse_on_s. Uses pulse.T_env_K for the loss terms -- self-contained, does
        not depend on any other method having run first."""

        target = pulse.T_high_K
        T_env = pulse.T_env_K

        def objective(P_on: float) -> float:
            return self._peak_temperature(T0, pulse, P_on, T_env) - target

        # Bracket: P=0 undershoots (or matches only if target<=T0); scale upper bound generously
        # from a first-order estimate of the power needed to raise a mass of thermal_mass_j_per_k
        # by (target - T0) in pulse_on_s, then pad it to also cover the loss terms.
        naive = self.heater.thermal_mass_j_per_k * max(target - T0, 1.0) / max(pulse.pulse_on_s, 1e-6)
        P_hi = max(naive * 20.0, 1.0)
        while objective(P_hi) < 0:
            P_hi *= 4.0
            if P_hi > 1e9:
                raise RuntimeError("Could not bracket a feasible heater power for target T_high.")
        return brentq(objective, 0.0, P_hi, xtol=1e-6, rtol=1e-8)

    def periodic_steady_state(
        self, pulse: PulseProgram, tol_K: float = 0.5, max_iter: int = 50, n_points_per_phase: int = 40
    ) -> CycleProfile:
        """Iterate cycle-start temperature to a fixed point (periodic steady state), then return
        the full on+off temperature history for one converged period."""

        T_env = pulse.T_env_K
        T_start = T_env
        P_on = self.solve_power_for_target(T_start, pulse)

        for _ in range(max_iter):
            t_on, T_on = self._integrate(T_start, pulse.pulse_on_s, P_on, T_env, n_points_per_phase)
            t_off, T_off = self._integrate(T_on[-1], pulse.pulse_off_s, 0.0, T_env, n_points_per_phase)
            T_start_new = T_off[-1]
            if abs(T_start_new - T_start) < tol_K:
                T_start = T_start_new
                break
            T_start = T_start_new
            P_on = self.solve_power_for_target(T_start, pulse)
        else:
            raise RuntimeError("Periodic steady state did not converge within max_iter cycles.")

        # Re-integrate once more at the converged T_start for a clean, consistent output profile.
        t_on, T_on = self._integrate(T_start, pulse.pulse_on_s, P_on, T_env, n_points_per_phase)
        t_off, T_off = self._integrate(T_on[-1], pulse.pulse_off_s, 0.0, T_env, n_points_per_phase)

        t_full = np.concatenate([t_on, t_off[1:] + t_on[-1]])
        T_full = np.concatenate([T_on, T_off[1:]])
        return CycleProfile(t_s=t_full, T_K=T_full, P_on_W=P_on, T_start_K=T_start, T_peak_K=float(T_on[-1]))
