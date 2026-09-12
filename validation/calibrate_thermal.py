"""Calibrate HeaterProperties (area, convective coefficient) against the T_avg anchor points
Dong22 states explicitly in its figure captions (Fig. 1b, Fig. 4b/c). thermal_mass_j_per_k and
emissivity are held at the paper's stated/typical values; area and h are the two free parameters
since the paper does not state exact heater geometry.

Run: python -m validation.calibrate_thermal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mc2_sim.config import HeaterProperties, PulseProgram
from mc2_sim.thermal import HeaterThermalModel

DATA_PATH = Path(__file__).parent / "data" / "dong22_tavg_anchors.csv"


def _predict_T_avg(log_area: float, log_h: float, anchors: pd.DataFrame) -> np.ndarray:
    heater = HeaterProperties(
        thermal_mass_j_per_k=0.03,
        area_m2=10.0**log_area,
        emissivity=0.9,
        convective_coeff_w_per_m2k=10.0**log_h,
    )
    model = HeaterThermalModel(heater)
    preds = []
    for _, row in anchors.iterrows():
        pulse = PulseProgram(
            T_high_K=row.T_high_K, pulse_on_s=row.pulse_on_s, pulse_off_s=row.pulse_off_s
        )
        cycle = model.periodic_steady_state(pulse)
        T_avg = float(np.trapezoid(cycle.T_K, cycle.t_s) / pulse.period_s)
        preds.append(T_avg)
    return np.array(preds)


def calibrate() -> HeaterProperties:
    anchors = pd.read_csv(DATA_PATH, comment="#")

    def residuals(x: np.ndarray) -> np.ndarray:
        return _predict_T_avg(x[0], x[1], anchors) - anchors.T_avg_K.values

    x0 = np.array([np.log10(1e-4), np.log10(50.0)])
    result = least_squares(residuals, x0, method="lm")

    fitted = HeaterProperties(
        thermal_mass_j_per_k=0.03,
        area_m2=10.0 ** result.x[0],
        emissivity=0.9,
        convective_coeff_w_per_m2k=10.0 ** result.x[1],
    )
    return fitted, result


if __name__ == "__main__":
    fitted, result = calibrate()
    anchors = pd.read_csv(DATA_PATH, comment="#")
    preds = _predict_T_avg(np.log10(fitted.area_m2), np.log10(fitted.convective_coeff_w_per_m2k), anchors)

    print("Fitted HeaterProperties:")
    print(f"  area_m2                 = {fitted.area_m2:.4e}")
    print(f"  convective_coeff_w_per_m2k = {fitted.convective_coeff_w_per_m2k:.4f}")
    print(f"  thermal_mass_j_per_k    = {fitted.thermal_mass_j_per_k} (fixed, paper-stated)")
    print(f"  emissivity              = {fitted.emissivity} (fixed)")
    print()
    print("Anchor fit:")
    for (_, row), pred in zip(anchors.iterrows(), preds):
        print(
            f"  T_high={row.T_high_K:.0f} K, {row.pulse_on_s}s on/{row.pulse_off_s}s off: "
            f"target T_avg={row.T_avg_K:.0f} K, model T_avg={pred:.1f} K"
        )
