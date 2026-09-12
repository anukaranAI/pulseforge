"""Phase 2: run our 4-stage staggered pulse array (mc2_sim.array) and compare against our design's
target TEA numbers (mc2_sim.tea), using the Phase 1 calibrated kinetics.

Reads clearly as two confidence tiers -- see tea.py's module docstring:
  - Stoichiometry and thermodynamic energy: first-principles, kinetics-independent, high confidence.
  - Per-pass conversion / H2 yield / recycle-loop estimate: uses the Phase 1 kinetics fit
    (RMSE ~14.5 percentage points against digitized Dong22 data -- see validation/), directional
    only. Flagged explicitly in the printed output.

Run: python -m examples.run_staged_array
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mc2_sim.array import four_stage_ppa
from mc2_sim.config import HeaterProperties
from mc2_sim.kinetics import ArrheniusParams
from mc2_sim.tea import array_mass_balance, stoichiometric_ideal, thermodynamic_energy_estimate

FITTED_PARAMS_PATH = Path(__file__).parent.parent / "validation" / "fitted_kinetics_params.json"


def load_fitted_params() -> ArrheniusParams:
    d = json.loads(FITTED_PARAMS_PATH.read_text())
    return ArrheniusParams(log10_A=np.array(d["log10_A"]), Ea_kJ_per_mol=np.array(d["Ea_kJ_per_mol"]))


if __name__ == "__main__":
    params = load_fitted_params()
    heater = HeaterProperties()

    print("=" * 78)
    print("TIER 1 -- first-principles checks (independent of the kinetics fit)")
    print("=" * 78)

    s = stoichiometric_ideal()
    print(f"\nStoichiometric ideal (100% conversion, 100% selectivity to C(s)+2H2):")
    print(f"  {s.kg_ch4_per_kg_h2:.3f} kg CH4/kg H2   (design target 4.21 -- {abs(s.kg_ch4_per_kg_h2-4.21)/4.21*100:.1f}% off)")
    print(f"  {s.kg_c_per_kg_h2:.3f} kg C/kg H2      (design target 3.00 -- {abs(s.kg_c_per_kg_h2-3.00)/3.00*100:.1f}% off)")

    print(f"\nThermodynamic minimum energy (reaction enthalpy + sensible heat - preheat credit):")
    for T_high in [1800, 2000, 2200]:
        e = thermodynamic_energy_estimate(T_high_K=T_high, T_preheat_K=773.0)
        diff = abs(e.kWh_per_kg_h2 - 12.5) / 12.5 * 100
        print(f"  T_high={T_high}K, 500C preheat: {e.kWh_per_kg_h2:.2f} kWh/kg H2  (design target 12.5 -- {diff:.1f}% off)")

    print()
    print("=" * 78)
    print("TIER 2 -- per-pass reactor performance (uses Phase 1 kinetics fit, RMSE ~14.5pp)")
    print("=" * 78)

    for T_high, n_cycles in [(1800, 5), (2000, 5), (2200, 5), (2000, 10)]:
        result = four_stage_ppa(
            T_high_K=T_high,
            pulse_on_s=0.055,
            pulse_off_s=1.045,
            n_cycles_per_stage=n_cycles,
            heater=heater,
            params=params,
            feed_preheat_K=773.0,
        )
        mb = array_mass_balance(
            conversion_pct=result.overall.conversion_pct,
            h2_yield_mol_per_mol_ch4=result.overall.H2_yield_mol_per_mol_CH4,
            coke_selectivity_pct=result.overall.selectivity_pct["Coke"],
            c2_plus_c6h6_selectivity_pct=result.overall.selectivity_pct["C2"] + result.overall.selectivity_pct["C6H6"],
        )
        print(
            f"\nT_high={T_high}K, {n_cycles} cycles/stage x 4 stages: "
            f"{mb.conversion_pct:.1f}% one-pass conversion, "
            f"H2 yield {mb.h2_yield_mol_per_mol_ch4:.3f} mol/mol CH4 fed"
        )
        print(
            f"  Selectivity: {mb.coke_selectivity_pct:.1f}% to solid coke (target H2 co-product), "
            f"{mb.unconverted_gaseous_hydrocarbon_selectivity_pct:.1f}% still gaseous C2/C6H6 (needs more passes)"
        )
        if np.isfinite(mb.kg_ch4_fed_per_kg_h2_one_pass):
            print(f"  One-pass basis: {mb.kg_ch4_fed_per_kg_h2_one_pass:.2f} kg CH4 fed per kg H2 produced this pass")

    print()
    print("Design target: >75% single-pass conversion, >95% overall (with recycle)")
    print(
        "NOTE: per-pass numbers above carry Phase 1's calibration uncertainty -- read as directional "
        "(does severity/pass-count roughly trend the right way), not as a precision prediction."
    )
