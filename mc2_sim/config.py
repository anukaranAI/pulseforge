"""Typed configuration objects for a PHQ (pulsed heating/quenching) reactor scenario.

These describe *what to run*, independent of *how it's solved* (thermal.py / kinetics.py /
reactor.py). Keeping them as plain, validated data objects means a scenario can be constructed
from a CSV row, a Bayesian-optimization proposal, or a hand-written script identically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeaterProperties(BaseModel):
    """Lumped thermal/electrical properties of the porous carbon heater element.

    Defaults correspond to the carbon paper used in Dong et al., Nature 2022 (heat capacity
    <0.033 J/K enabling ~1e4 K/s heating/cooling rates; Supplementary Discussion 2).
    """

    thermal_mass_j_per_k: float = Field(
        0.03, gt=0, description="Lumped heat capacity of the heater element, m*cp (J/K)."
    )
    # area_m2 and convective_coeff_w_per_m2k below are fitted (validation/calibrate_thermal.py)
    # against the two T_avg anchor points Dong22 states in its Fig. 1b / Fig. 4b captions
    # (2000 K/0.02s-on -> T_avg~815 K; 1400 K/0.11s-on -> T_avg~891 K). The fit lands within
    # ~5-6% of both anchors with a purely radiative loss term (fitted h ~ 0) -- a known
    # simplification of this lumped model, documented in validation/calibrate_thermal.py output.
    area_m2: float = Field(
        8.052e-4, gt=0, description="Effective heater surface area for convective/radiative loss (m^2)."
    )
    emissivity: float = Field(0.9, ge=0, le=1, description="Radiative emissivity of the carbon element.")
    convective_coeff_w_per_m2k: float = Field(
        0.0, ge=0, description="Effective convective heat transfer coefficient to the flowing gas (W/m^2/K)."
    )


class PulseProgram(BaseModel):
    """The programmed electrical heating waveform for one stage."""

    T_high_K: float = Field(..., gt=0, description="Target peak heater temperature (K).")
    T_env_K: float = Field(300.0, gt=0, description="Ambient / feed-gas temperature the heater quenches toward (K).")
    pulse_on_s: float = Field(..., gt=0, description="Duration the heater is powered on per cycle (s).")
    pulse_off_s: float = Field(..., gt=0, description="Duration the heater is unpowered per cycle (s).")

    @property
    def period_s(self) -> float:
        return self.pulse_on_s + self.pulse_off_s

    @property
    def duty_cycle(self) -> float:
        return self.pulse_on_s / self.period_s


class FeedConditions(BaseModel):
    """Gas-phase feed to a single reactor stage."""

    flow_rate_sccm: float = Field(..., gt=0, description="Total volumetric flow rate (sccm).")
    ch4_mol_frac: float = Field(0.75, ge=0, le=1, description="Mole fraction of CH4 in the feed.")
    inert_mol_frac: float = Field(0.25, ge=0, le=1, description="Mole fraction of inert diluent (e.g. Ar).")
    residence_time_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Gas residence time near the heater per cycle. If None, reactor.py estimates it from "
            "flow_rate_sccm and the heater/reactor geometry."
        ),
    )


class ReactorScenario(BaseModel):
    """A complete, self-contained single-stage PHQ scenario."""

    name: str
    heater: HeaterProperties = HeaterProperties()
    pulse: PulseProgram
    feed: FeedConditions
    n_cycles: int = Field(30, ge=1, description="Number of pulse cycles to integrate to reach periodic steady state.")
