"""Techno-economic / mass-and-energy-balance checks against our design targets:
4.21 kg CH4/kg H2, 3.00 kg C/kg H2, 12.5 kWh/kg H2.

Deliberately split into two independence tiers:

1. stoichiometric_ideal() and thermodynamic_energy_estimate() are EXACT/first-principles: atomic
   mass balance and standard thermochemistry (reaction enthalpy + sensible heat - preheat credit).
   Neither depends on mc2_sim's calibrated kinetics fit at all -- they're checking the design
   targets against textbook physics, and are the high-confidence half of this module.

2. array_mass_balance() uses the (imperfectly calibrated -- see validation/) reactor/kinetics
   engine to estimate ACTUAL per-pass conversion and H2 yield for a given array configuration.
   This inherits Phase 1's calibration uncertainty and should be read as directional, not precise.

Energy demand is NOT estimated by scaling up mc2_sim's calibrated HeaterThermalModel: that model's
HeaterProperties (area, thermal mass) were fitted to Dong22's specific lab-scale heater element,
and there is no scale-up relationship given anywhere (paper or elsewhere) connecting that lab
sample to our industrial-throughput hardware. Using it directly for an absolute kWh/kg H2 figure
would silently conflate a small lab sample's heating power with an industrial system's.
Standard thermochemistry, which is scale-independent, is the honest tool for that specific check.
"""

from __future__ import annotations

from dataclasses import dataclass

# Molar masses (g/mol)
M_CH4 = 16.043
M_H2 = 2.016
M_C = 12.011

# Standard enthalpy of formation (kJ/mol) at 298 K
DHF_CH4 = -74.8
DHF_C_GRAPHITE = 0.0
DHF_H2 = 0.0

# Approximate constant heat capacities (J/mol/K), first-order engineering estimates spanning the
# relevant temperature ranges (NOT full Shomate/NIST polynomials -- a documented simplification).
# CP_CH4_LOW covers 298-773 K (ambient to our design's 500 degC preheat target); CP_C_GRAPHITE and
# CP_H2 cover 298 K up to ~2000 K, where graphite's heat capacity in particular rises substantially
# with temperature (consistent with ref. 19 in Dong22, Butland & Maddison 1973, cited there for the
# same reason -- graphite Cp is strongly temperature-dependent over this range).
CP_CH4_LOW_J_MOL_K = 46.0
CP_H2_J_MOL_K = 30.0
CP_C_GRAPHITE_J_MOL_K = 18.0

T_REF_K = 298.0
KJ_PER_KWH = 3600.0


@dataclass
class StoichiometricIdeal:
    kg_ch4_per_kg_h2: float
    kg_c_per_kg_h2: float


def stoichiometric_ideal() -> StoichiometricIdeal:
    """Exact atomic mass balance for CH4 -> C(s) + 2 H2 at 100% conversion and 100% selectivity
    (no intermediate C2/coke losses) -- the theoretical best case, independent of any kinetics."""

    moles_h2_per_mole_ch4 = 2.0
    mass_h2_per_mole_ch4 = moles_h2_per_mole_ch4 * M_H2
    return StoichiometricIdeal(
        kg_ch4_per_kg_h2=M_CH4 / mass_h2_per_mole_ch4,
        kg_c_per_kg_h2=M_C / mass_h2_per_mole_ch4,
    )


@dataclass
class ThermodynamicEnergyEstimate:
    kJ_per_mol_ch4: float
    kWh_per_kg_h2: float


def thermodynamic_energy_estimate(T_high_K: float, T_preheat_K: float = 298.0) -> ThermodynamicEnergyEstimate:
    """First-principles minimum electrical energy demand for CH4 -> C(s) + 2 H2, complete
    conversion and selectivity, no heat losses beyond what's implied by not recovering the hot
    product stream's own heat (only the stated feed preheat credit is applied). This is a LOWER
    BOUND (a real process also has thermal losses, incomplete heat exchanger effectiveness, and
    incomplete conversion/selectivity, all of which raise the real energy demand above this).

    Energy path (enthalpy is a state function, so this path is exact given the reaction/species and
    average-Cp assumptions): react CH4 at 298 K (releasing/absorbing DH_rxn), then heat the C(s) and
    H2 products from 298 K to T_high_K; credit back the sensible heat the incoming CH4 feed already
    received from ambient up to T_preheat_K via our design's inter-stage heat exchanger.
    """

    dH_rxn = (DHF_C_GRAPHITE + 2 * DHF_H2) - DHF_CH4  # kJ/mol CH4, endothermic (positive)

    sensible_heat_products_kJ = (
        CP_C_GRAPHITE_J_MOL_K * (T_high_K - T_REF_K) + 2 * CP_H2_J_MOL_K * (T_high_K - T_REF_K)
    ) / 1000.0
    preheat_credit_kJ = CP_CH4_LOW_J_MOL_K * (T_preheat_K - T_REF_K) / 1000.0

    kJ_per_mol_ch4 = dH_rxn + sensible_heat_products_kJ - preheat_credit_kJ
    kJ_per_mol_h2 = kJ_per_mol_ch4 / 2.0  # 2 mol H2 per mol CH4 at complete conversion
    kWh_per_kg_h2 = (kJ_per_mol_h2 / M_H2) / KJ_PER_KWH * 1000.0  # kJ/mol -> kJ/g -> kJ/kg -> kWh/kg

    return ThermodynamicEnergyEstimate(kJ_per_mol_ch4=kJ_per_mol_ch4, kWh_per_kg_h2=kWh_per_kg_h2)


@dataclass
class ArrayMassBalance:
    conversion_pct: float
    h2_yield_mol_per_mol_ch4: float
    kg_ch4_fed_per_kg_h2_one_pass: float
    coke_selectivity_pct: float
    unconverted_gaseous_hydrocarbon_selectivity_pct: float  # C2 + C6H6, still needs more passes


def array_mass_balance(conversion_pct: float, h2_yield_mol_per_mol_ch4: float, coke_selectivity_pct: float, c2_plus_c6h6_selectivity_pct: float) -> ArrayMassBalance:
    """One-pass mass balance from an mc2_sim ArrayResult -- inherits the kinetics fit's
    uncertainty (see module docstring). kg_ch4_fed_per_kg_h2_one_pass is NOT directly comparable to
    the 4.21 kg/kg design target (which assumes >=95% overall conversion via recycle); it's the
    one-pass number, useful for judging how many recycle loops a given severity would need."""

    if h2_yield_mol_per_mol_ch4 <= 1e-9:
        kg_ch4_per_kg_h2 = float("inf")
    else:
        mass_h2_per_mol_ch4_fed = h2_yield_mol_per_mol_ch4 * M_H2
        kg_ch4_per_kg_h2 = M_CH4 / mass_h2_per_mol_ch4_fed

    return ArrayMassBalance(
        conversion_pct=conversion_pct,
        h2_yield_mol_per_mol_ch4=h2_yield_mol_per_mol_ch4,
        kg_ch4_fed_per_kg_h2_one_pass=kg_ch4_per_kg_h2,
        coke_selectivity_pct=coke_selectivity_pct,
        unconverted_gaseous_hydrocarbon_selectivity_pct=c2_plus_c6h6_selectivity_pct,
    )
