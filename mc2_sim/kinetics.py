"""Lumped CH4-pyrolysis reaction network.

Follows the cascade described in Dong et al. (Nature 2022), extended with a direct-to-coke route:

              R1a (low Ea)  \\
    CH4 ---------------------+--> C2H6 --R2--> C2H4 --R3--> C2H2 --R4--> C6H6 --R5--> Coke
              R1b (high Ea) /                                        \\--R6--> Coke  [direct]

R6 (C2H2 straight to solid carbon) is negligible under the paper's short-pulse/quench conditions
but becomes the dominant path under our staged-array design's longer-residence, higher-T_high
regime, where the cascade is deliberately driven to completion (CH4 -> C(s) + 2 H2) rather than
frozen at C2.

CH4 activation is split into two parallel first-order channels (R1a, R1b) rather than one. A first
version of this model used a single CH4->C2H6 step; calibrating it against the digitized Dong22
Fig. 3b conversion-vs-T_high data (validation/calibrate_kinetics.py) consistently produced the
wrong curve SHAPE no matter the fitted (A, Ea): real conversion rises gradually and keeps rising
mildly from 1200-2000 K, while a single Arrhenius step forces a sharp low-T plateau followed by a
runaway threshold, undershooting badly below ~1700 K and overshooting badly above it. A low-Ea
channel (weak T-sensitivity, contributes a gentle baseline across the whole range) running in
parallel with a high-Ea channel (only turns on at high T) is the standard way a lumped/engineering
kinetics model represents "the low-severity part of the reaction network is a different rate-
limiting step than the high-severity part" without resolving actual sub-mechanisms or spatial
temperature gradients (which the paper's Fig. 2g shows exist near the heater, and which a 0D
lumped model like this one can't otherwise capture).

Every step is modelled as first order in its precursor's carbon-atom content (hence "first-order
sim"): tracking species in mol-C rather than mol-species makes carbon conservation and selectivity
bookkeeping exact and trivial (selectivity_i = C_i / (C_fed - C_CH4_remaining)), at the cost of
lumping away the true reaction orders of the underlying radical chemistry -- an explicit, documented
simplification appropriate for a reduced-order/engineering-scale model, not a mechanistic one.

H2 co-product is tracked in mol H2 per mol CH4 fed, using the stoichiometry of each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAS_CONSTANT_J_PER_MOL_K = 8.314462618

SPECIES = ["CH4", "C2H6", "C2H4", "C2H2", "C6H6", "Coke"]
N_SPECIES = len(SPECIES)

# index map for readability
I_CH4, I_C2H6, I_C2H4, I_C2H2, I_C6H6, I_COKE = range(N_SPECIES)

# Which species each reaction consumes/produces (carbon-atom basis, so every row is a simple
# -1/+1 carbon transfer) and how much H2 (mol H2 per mol-C converted) each step releases.
_REACTION_FROM_TO = [
    (I_CH4, I_C2H6),  # R1a: CH4 -> C2H6     (low-Ea / fast-onset channel)
    (I_CH4, I_C2H6),  # R1b: CH4 -> C2H6     (high-Ea / high-T channel)
    (I_C2H6, I_C2H4),  # R2: C2H6 -> C2H4     (C2H6 -> C2H4 + H2)
    (I_C2H4, I_C2H2),  # R3: C2H4 -> C2H2     (C2H4 -> C2H2 + H2)
    (I_C2H2, I_C6H6),  # R4: C2H2 -> C6H6     (3 C2H2 -> C6H6, no H2: same C:H ratio)
    (I_C6H6, I_COKE),  # R5: C6H6 -> Coke     (C6H6 -> 6 C(s) + 3 H2)
    (I_C2H2, I_COKE),  # R6: C2H2 -> Coke     (C2H2 -> 2 C(s) + H2)  [our design's "full cascade" route]
]
_H2_PER_MOLC_CONVERTED = [0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
N_REACTIONS = len(_REACTION_FROM_TO)


@dataclass
class ArrheniusParams:
    """log10(A [1/s]) and activation energy (kJ/mol) for each of the 7 reaction steps (R1a, R1b,
    R2-R6; see module docstring for why CH4 activation is split into two parallel channels).

    Defaults are order-of-magnitude literature-informed starting points for non-catalytic gas-phase
    CH4 pyrolysis steps (C-H homolysis / dehydrogenation-cascade type activation energies), meant to
    be refined by validation.calibrate against the digitized Dong22 Fig. 3 dataset -- they are a
    starting point, not a claimed fit.
    """

    log10_A: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 13.0, 12.5, 12.5, 11.5, 10.5, 9.5])
    )
    Ea_kJ_per_mol: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 365.0, 280.0, 300.0, 320.0, 250.0, 340.0])
    )

    def rate_constants(self, T_K: float) -> np.ndarray:
        A = 10.0**self.log10_A
        Ea = self.Ea_kJ_per_mol * 1000.0
        return A * np.exp(-Ea / (GAS_CONSTANT_J_PER_MOL_K * T_K))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.log10_A, self.Ea_kJ_per_mol])

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "ArrheniusParams":
        n = N_REACTIONS
        return cls(log10_A=np.asarray(x[:n]), Ea_kJ_per_mol=np.asarray(x[n:]))


def carbon_rhs(t: float, C: np.ndarray, T_of_t, params: ArrheniusParams) -> np.ndarray:
    """dC/dt for the 6 carbon-pool species (mol-C, normalized so sum(C) == 1 at t=0) plus H2
    (mol H2 per mol CH4 fed) appended as the 7th state.

    T_of_t: callable t -> temperature (K), typically an interpolant of a CycleProfile repeated
    periodically (see reactor.py).
    """

    C_species = C[:N_SPECIES]
    T = T_of_t(t)
    k = params.rate_constants(T)

    dC = np.zeros(N_SPECIES)
    dH2 = 0.0
    for i, (src, dst) in enumerate(_REACTION_FROM_TO):
        rate = k[i] * C_species[src]
        dC[src] -= rate
        dC[dst] += rate
        dH2 += _H2_PER_MOLC_CONVERTED[i] * rate

    return np.concatenate([dC, [dH2]])


def rate_matrix(T_K: float, params: ArrheniusParams) -> np.ndarray:
    """The 6x6 linear operator K such that dC/dt = K @ C (every step is first order, so the whole
    carbon network is a linear time-varying ODE at fixed T -- this is what makes the period
    propagator in reactor.py possible: no need to re-solve an ODE for every pulse cycle, just
    apply this matrix repeatedly)."""

    k = params.rate_constants(T_K)
    K = np.zeros((N_SPECIES, N_SPECIES))
    for i, (src, dst) in enumerate(_REACTION_FROM_TO):
        K[dst, src] += k[i]
        K[src, src] -= k[i]
    return K


def h2_rate_vector(T_K: float, params: ArrheniusParams) -> np.ndarray:
    """v such that dH2/dt = v . C (H2 production is also linear in the carbon-pool state)."""

    k = params.rate_constants(T_K)
    v = np.zeros(N_SPECIES)
    for i, (src, _dst) in enumerate(_REACTION_FROM_TO):
        v[src] += _H2_PER_MOLC_CONVERTED[i] * k[i]
    return v


def augmented_rhs(t: float, Y_flat: np.ndarray, T_of_t, params: ArrheniusParams) -> np.ndarray:
    """RHS for propagating a (N_SPECIES+1) x N_SPECIES matrix Y forward: the top N_SPECIES rows
    carry the carbon-subsystem's fundamental solution (columns = images of the 6 unit basis
    vectors), the last row accumulates the corresponding H2 output. Used by reactor.py to compute
    a whole pulse period's state-transition operator in one ODE solve, rather than one solve per
    individual reaction step."""

    Y = Y_flat.reshape(N_SPECIES + 1, N_SPECIES)
    T = T_of_t(t)
    K = rate_matrix(T, params)
    v = h2_rate_vector(T, params)
    dC = K @ Y[:N_SPECIES, :]
    dH2 = v @ Y[:N_SPECIES, :]
    return np.vstack([dC, dH2[None, :]]).ravel()


def initial_state(ch4_carbon_fraction: float = 1.0) -> np.ndarray:
    """State vector [C_CH4, C_C2H6, C_C2H4, C_C2H2, C_C6H6, C_Coke, H2], all CH4, no H2 yet."""

    state = np.zeros(N_SPECIES + 1)
    state[I_CH4] = ch4_carbon_fraction
    return state


def conversion_and_selectivity(state: np.ndarray) -> dict[str, float]:
    """Given a final state vector, compute CH4 conversion (%) and product selectivities (% of
    converted carbon) -- the same quantities reported in Dong22 Fig. 3."""

    C = state[:N_SPECIES]
    C_ch4_remaining = C[I_CH4]
    C_fed = 1.0
    converted = C_fed - C_ch4_remaining
    conversion_pct = 100.0 * converted / C_fed

    if converted <= 1e-12:
        selectivity = {"C2": 0.0, "C6H6": 0.0, "Coke": 0.0}
    else:
        selectivity = {
            "C2": 100.0 * (C[I_C2H6] + C[I_C2H4] + C[I_C2H2]) / converted,
            "C6H6": 100.0 * C[I_C6H6] / converted,
            "Coke": 100.0 * C[I_COKE] / converted,
        }

    h2_yield = state[N_SPECIES]  # mol H2 per mol CH4 fed
    return {"conversion_pct": conversion_pct, **selectivity, "H2_yield_mol_per_mol_CH4": h2_yield}
