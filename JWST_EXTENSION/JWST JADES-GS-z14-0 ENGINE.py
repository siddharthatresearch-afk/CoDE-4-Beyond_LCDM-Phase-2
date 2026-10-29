#!/usr/bin/env python3
#!/usr/bin/env python3
"""JWST/JADES-GS-z14-0 cosmology sensitivity audit.

This is a standalone calculation: it imports no earlier project scripts and
does not use cached pipeline outputs. The legacy assumptions from the supplied
preliminary script are retained for an apples-to-apples audit. Additional
control cases isolate the epsilon modification and the choice to hold Omega_m
fractions fixed while changing H0.

The code computes distances, cosmic ages, equality scales, a linear-growth ODE,
and a correctly dimensioned gas Jeans mass for an explicitly stated gas state.
It does not turn those quantities into a formation-time prediction without a
specified halo population, star-formation history, and efficiency model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, solve_ivp


# -------------------------- Explicit inputs -------------------------------
C_KMS = 299_792.458
MPC_KM = 3.0856775814913673e19
MPC_CM = 3.0856775814913673e24
SEC_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
GYR_PER_H0_UNIT = MPC_KM / SEC_PER_GYR  # multiply by 1/H0 to obtain Gyr
G_CGS = 6.67430e-8
K_B_CGS = 1.380649e-16
M_P_G = 1.67262192369e-24
M_SUN_G = 1.98847e33
RHO_CRIT_H2_G_CM3 = 1.87847e-29

# Values used in the supplied preliminary analysis.
Z_GS14 = 14.1793                 # ALMA [O III] spectroscopic redshift
H0_LCDM = 67.4                   # km/s/Mpc
OM_LCDM = 0.315
OR_LCDM = 9.4e-5
OB_LCDM = 0.049
H0_CODE4_LOCAL = 72.86           # supplied preliminary CoDE4 normalization
OM_CODE4_LEGACY = 0.315           # fixed fraction in the supplied script
OR_CODE4_LEGACY = 9.4e-5
OB_CODE4_LEGACY = 0.049
EPSILON = 0.05
BETA = 0.05
STELLAR_AGE_ASSUMPTIONS_MYR = (20.0, 90.0)
REFERENCE_LOG10_STELLAR_MASS = 8.6  # illustrative published SED-fit central value
GAS_TEMPERATURE_K = 1.0e4         # explicit illustrative neutral-gas assumption
MEAN_MOLECULAR_WEIGHT = 1.22

# Active CoDE4 engine values, included as explicit scenarios rather than
# importing or executing another pipeline. The two H0 normalizations are
# distinct outputs of that engine: local ladder refit and theta*-calibrated.
H0_CODE4_ACTIVE_LOCAL = 72.9331
H0_CODE4_ACTIVE_THETA = 66.3528
OM_CODE4_ACTIVE = 0.32917
OMEGA_B_H2_BBN = 0.02241
OMEGA_R_H2 = 4.183e-5


@dataclass(frozen=True)
class Cosmology:
    name: str
    H0: float
    Om: float
    Or: float
    Ob: float
    epsilon: float = 0.0
    beta: float = 0.0

    @property
    def Ode(self) -> float:
        return 1.0 - self.Om - self.Or

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def omega_m(self) -> float:
        return self.Om * self.h**2

    @property
    def omega_b(self) -> float:
        return self.Ob * self.h**2

    def E2(self, z: float | np.ndarray) -> float | np.ndarray:
        z = np.asarray(z, dtype=float)
        if np.any(z < 0.0):
            raise ValueError("Redshift must be non-negative.")
        f = z / (1.0 + z) ** 2
        modifier = 1.0 + self.epsilon * f * (1.0 - 0.15 / (1.0 + z))
        radiation_modifier = 1.0 + self.epsilon * z / (1.0 + z) ** 3
        value = (self.Om * (1.0 + z) ** 3
                 + self.Or * (1.0 + z) ** 4 * radiation_modifier
                 + self.Ode * modifier)
        if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError("Cosmology produced non-positive or non-finite E(z)^2.")
        return float(value) if value.ndim == 0 else value

    def E(self, z: float | np.ndarray) -> float | np.ndarray:
        return np.sqrt(self.E2(z))

    def H(self, z: float | np.ndarray) -> float | np.ndarray:
        return self.H0 * self.E(z)


def comoving_distance_mpc(cosmo: Cosmology, z: float) -> float:
    integral = quad(lambda x: C_KMS / cosmo.H(x), 0.0, z,
                    epsabs=0.0, epsrel=1e-11, limit=300)[0]
    return float(integral)


def luminosity_distance_mpc(cosmo: Cosmology, z: float) -> float:
    return (1.0 + z) * comoving_distance_mpc(cosmo, z)  # flat geometry


def cosmic_age_myr(cosmo: Cosmology, z: float) -> float:
    """Age since a->0, integrating dt/dln(a)=1/H(a) in Gyr."""
    x_target = -np.log1p(z)

    def integrand(log_a: float) -> float:
        z_here = np.expm1(-log_a)
        return 1.0 / float(cosmo.E(z_here))

    dimensionless_age = quad(integrand, -32.0, x_target,
                             epsabs=0.0, epsrel=1e-10, limit=400)[0]
    return 1000.0 * GYR_PER_H0_UNIT / cosmo.H0 * dimensionless_age


def equality_scales(cosmo: Cosmology) -> tuple[float, float, float]:
    """Return z_eq, k_eq [1/Mpc], and k_eq [h/Mpc]."""
    z_eq = cosmo.Om / cosmo.Or - 1.0
    k_eq = (cosmo.H0 / C_KMS) * np.sqrt(2.0 * cosmo.Om * (1.0 + z_eq))
    return float(z_eq), float(k_eq), float(k_eq / cosmo.h)


def equality_mass_msun(cosmo: Cosmology) -> float:
    """Matter mass in a sphere of radius pi/k_eq, using consistent cgs units."""
    _, k_eq_mpc, _ = equality_scales(cosmo)
    rho_m0 = cosmo.Om * RHO_CRIT_H2_G_CM3 * cosmo.h**2
    radius_cm = np.pi * MPC_CM / k_eq_mpc
    return float((4.0 / 3.0) * np.pi * rho_m0 * radius_cm**3 / M_SUN_G)


def gas_jeans_mass_msun(cosmo: Cosmology, z: float, temperature_k: float,
                        mean_molecular_weight: float) -> float:
    """Isothermal baryonic Jeans mass; depends on the declared gas state.

    M_J = (pi^(5/2)/6) c_s^3 / (G^(3/2) rho_b^(1/2));
    c_s^2 = k_B T/(mu m_p). This is a gas Jeans scale, not a galaxy/halo mass.
    """
    if temperature_k <= 0.0 or mean_molecular_weight <= 0.0:
        raise ValueError("Gas temperature and mean molecular weight must be positive.")
    cs = np.sqrt(K_B_CGS * temperature_k / (mean_molecular_weight * M_P_G))
    rho_b = cosmo.Ob * RHO_CRIT_H2_G_CM3 * cosmo.h**2 * (1.0 + z) ** 3
    mass_g = (np.pi ** 2.5 / 6.0) * cs**3 / (G_CGS**1.5 * np.sqrt(rho_b))
    return float(mass_g / M_SUN_G)


def growth_solution(cosmo: Cosmology):
    """Solve the same scale-independent ln(a) growth ODE used by the CoDE engine."""
    a_initial = 1e-6
    a_eq = cosmo.Or / cosmo.Om
    step = 1e-4

    def rhs(x, y):
        a = np.exp(x)
        z = 1.0 / a - 1.0
        e2 = float(cosmo.E2(z))
        x_plus = min(x + step, 0.0)
        x_minus = max(x - step, np.log(a_initial))
        e2_plus = float(cosmo.E2(np.exp(-x_plus) - 1.0))
        e2_minus = float(cosmo.E2(np.exp(-x_minus) - 1.0))
        dln_e = 0.5 * (np.log(e2_plus) - np.log(e2_minus)) / (x_plus - x_minus)
        omega_m_a = cosmo.Om * a**-3 / e2
        growth_coupling = 1.0 + cosmo.beta * z / (1.0 + z) ** 2
        return [y[1], -(2.0 + dln_e) * y[1]
                + 1.5 * omega_m_a * growth_coupling * y[0]]

    y_initial = [1.0 + 1.5 * a_initial / a_eq, 1.5 * a_initial / a_eq]
    solution = solve_ivp(rhs, [np.log(a_initial), 0.0], y_initial,
                         method="DOP853", rtol=1e-9, atol=1e-13,
                         dense_output=True)
    if not solution.success:
        raise RuntimeError(f"Growth ODE failed for {cosmo.name}: {solution.message}")
    normalization = 1.5 / a_eq
    return solution, normalization


def growth_ratio(solution, normalization, z: float) -> float:
    a = 1.0 / (1.0 + z)
    return float(solution.sol(np.log(a))[0] / normalization)


def make_cosmologies() -> list[Cosmology]:
    lcdm = Cosmology("LCDM baseline", H0_LCDM, OM_LCDM, OR_LCDM, OB_LCDM)
    # Isolates the CoDE4 modifier while holding the old script's H0 and fractions fixed.
    epsilon_only = Cosmology("CoDE4 epsilon only", H0_LCDM, OM_LCDM, OR_LCDM,
                             OB_LCDM, EPSILON, BETA)
    # Exact legacy H0/fraction choices from the submitted preliminary analysis.
    legacy = Cosmology("CoDE4 legacy local-H0 fractions", H0_CODE4_LOCAL,
                       OM_CODE4_LEGACY, OR_CODE4_LEGACY, OB_CODE4_LEGACY,
                       EPSILON, BETA)
    # Control: change H0 but preserve LCDM physical matter, baryon, and radiation densities.
    om_phys = lcdm.omega_m / (H0_CODE4_LOCAL / 100.0) ** 2
    ob_phys = lcdm.omega_b / (H0_CODE4_LOCAL / 100.0) ** 2
    or_phys = (OR_LCDM * lcdm.h**2) / (H0_CODE4_LOCAL / 100.0) ** 2
    fixed_physical = Cosmology("CoDE4 fixed physical densities", H0_CODE4_LOCAL,
                               om_phys, or_phys, ob_phys, EPSILON, BETA)
    active_local = active_engine_case("CoDE4 active local-ladder H0", H0_CODE4_ACTIVE_LOCAL)
    active_theta = active_engine_case("CoDE4 active theta*-calibrated H0", H0_CODE4_ACTIVE_THETA)
    return [lcdm, epsilon_only, legacy, fixed_physical, active_local, active_theta]


def active_engine_case(name: str, h0: float) -> Cosmology:
    """Build a current-engine scenario from explicit fitted/input values."""
    h = h0 / 100.0
    return Cosmology(name, h0, OM_CODE4_ACTIVE, OMEGA_R_H2 / h**2,
                     OMEGA_B_H2_BBN / h**2, EPSILON, BETA)


def legacy_timescale_reproduction(lcdm: Cosmology, code4: Cosmology,
                                  dl_ratio_sq: float, age_lcdm: float,
                                  age_code4: float) -> None:
    """Reproduce the old heuristic for audit only; it is not a physical prediction."""
    k_eq_l_h = 0.01675
    k_eq_c_h = 0.01566
    h_l, h_c = lcdm.h, code4.h
    k_l, k_c = k_eq_l_h * h_l, k_eq_c_h * h_c
    rho_l = lcdm.Om * RHO_CRIT_H2_G_CM3 * h_l**2
    rho_c = code4.Om * RHO_CRIT_H2_G_CM3 * h_c**2
    m_eq_l = (4.0/3.0) * np.pi * rho_l * (np.pi * MPC_CM / k_l) ** 3 / M_SUN_G
    m_eq_c = (4.0/3.0) * np.pi * rho_c * (np.pi * MPC_CM / k_c) ** 3 / M_SUN_G
    # This ratio is the old missing-sound-speed Jeans expression squared.
    old_jeans_compaction = (rho_l / rho_c)
    old_growth_boost = 1.0103
    old_required_lcdm_myr = 600.0
    old_required_code4 = (old_required_lcdm_myr * dl_ratio_sq**1.5
                          * old_jeans_compaction / (old_growth_boost * (m_eq_c / m_eq_l)))
    old_gap_l = old_required_lcdm_myr - (age_lcdm - 90.0)
    old_gap_c = old_required_code4 - (age_code4 - 90.0)
    gap_reduction = 1.0 - old_gap_c / old_gap_l
    print("\n[LEGACY FORMULA REPRODUCTION - AUDIT ONLY]")
    print(f"Legacy heuristic required time:              {old_required_code4:.2f} Myr")
    print(f"Legacy LambdaCDM residual gap:                {old_gap_l:.2f} Myr")
    print(f"Legacy CoDE4 residual gap:                    {old_gap_c:.2f} Myr")
    print(f"Legacy gap reduction from formula:            {gap_reduction:.4f} (unitless fraction)")
    print("This is not a derived collapse-time prediction: the mass exponent, growth multiplier, and equality-scale multiplier were stipulated, not inferred from a calibrated galaxy-formation model.")


def main() -> None:
    models = make_cosmologies()
    baseline = models[0]
    baseline_dl = luminosity_distance_mpc(baseline, Z_GS14)
    baseline_age = cosmic_age_myr(baseline, Z_GS14)
    reference_mass = 10.0 ** REFERENCE_LOG10_STELLAR_MASS
    growth = {c.name: growth_solution(c) for c in models}

    print("=" * 86)
    print("JADES-GS-z14-0 - COSMOLOGY, MASS-SCALING, AND AGE AUDIT")
    print("No previous Python pipeline is imported; all calculations are in this file.")
    print(f"Target redshift: {Z_GS14:.4f}; stellar ages shown are assumptions, not measured formation times.")
    print("=" * 86)
    print("\n[MODEL INPUTS]")
    print(f"Legacy CoDE4 H0 input: {H0_CODE4_LOCAL:.2f} km/s/Mpc; epsilon={EPSILON:.3f}; beta={BETA:.3f}")
    print(f"Legacy fixed Omega_m fractions: LCDM={OM_LCDM:.5f}, CoDE4={OM_CODE4_LEGACY:.5f}")
    print(f"Active engine scenarios: Omega_m={OM_CODE4_ACTIVE:.5f}, omega_b(BBN)={OMEGA_B_H2_BBN:.5f};")
    print(f"  H0 local-ladder={H0_CODE4_ACTIVE_LOCAL:.4f} and theta*-calibrated={H0_CODE4_ACTIVE_THETA:.4f} km/s/Mpc are shown separately.")
    print(f"Gas Jeans illustration: T={GAS_TEMPERATURE_K:.0f} K, mu={MEAN_MOLECULAR_WEIGHT:.2f}; assumed, not fitted to GS-z14-0")
    print(f"Illustrative reference stellar mass: 10^{REFERENCE_LOG10_STELLAR_MASS:.1f} Msun; mass is SED-model dependent")

    print("\n[DISTANCE, AGE, AND THE D_L^2 MASS RESCALING]")
    print(f"{'Model':38s} {'H0':>8s} {'Omega_m':>9s} {'D_L [Mpc]':>13s} {'Age [Myr]':>12s} {'M*/M*_LCDM':>14s}")
    print("-" * 100)
    rows = {}
    for c in models:
        dl = luminosity_distance_mpc(c, Z_GS14)
        age = cosmic_age_myr(c, Z_GS14)
        mass_ratio = (dl / baseline_dl) ** 2
        rows[c.name] = (dl, age, mass_ratio)
        print(f"{c.name:38s} {c.H0:8.3f} {c.Om:9.5f} {dl:13.3f} {age:12.3f} {mass_ratio:14.6f}")
    print("Mass scaling assumes identical observed flux, SED, lensing, dust, and IMF treatment; it is not a refit of the JWST photometry.")
    for c in models[1:]:
        _, _, ratio = rows[c.name]
        scaled_mass = reference_mass * ratio
        print(f"{c.name}: illustrative M* = {scaled_mass:.3e} Msun (from {reference_mass:.3e} Msun under LCDM)")
    print("The epsilon-only case isolates the modifier. The legacy local-H0 case also changes the physical matter density because Omega_m is held fixed.")
    print("The fixed-physical-density control shows why changing H0 while holding Omega_m fixed is not an innocuous normalization change.")

    print("\n[AVAILABLE COSMIC TIME UNDER STELLAR-AGE ASSUMPTIONS]")
    for age_assumption in STELLAR_AGE_ASSUMPTIONS_MYR:
        print(f"Assumed stellar population age = {age_assumption:.1f} Myr")
        for c in (baseline, models[2], models[3], models[4], models[5]):
            age = rows[c.name][1]
            formation_start = age - age_assumption
            print(f"  {c.name:38s}: cosmic age={age:8.3f} Myr; implied onset time={formation_start:8.3f} Myr after Big Bang")
    print("An onset time is not a formation-timescale requirement; a star-formation history is needed for that.")

    print("\n[MATTER-RADIATION EQUALITY AND LINEAR GROWTH]")
    for c in models:
        z_eq, k_eq_mpc, k_eq_h = equality_scales(c)
        m_eq = equality_mass_msun(c)
        sol, norm = growth[c.name]
        d_target = growth_ratio(sol, norm, Z_GS14)
        d_today = growth_ratio(sol, norm, 0.0)
        print(f"{c.name}: z_eq={z_eq:.3f}, k_eq={k_eq_mpc:.6e} 1/Mpc, k_eq={k_eq_h:.6e} h/Mpc")
        print(f"  Equality mass scale={m_eq:.4e} Msun; D(z_target)/D(0)={d_target/d_today:.8f}; D(z_target)={d_target:.8e}")
    d_l = growth_ratio(*growth[baseline.name], Z_GS14)
    d_c = growth_ratio(*growth[models[2].name], Z_GS14)
    print(f"Legacy-normalized CoDE4/LCDM linear growth at z={Z_GS14:.4f}: {d_c/d_l:.8f}")
    print("This is computed from the growth ODE; it replaces the hard-coded 1.0103 multiplier.")

    print("\n[GAS JEANS SCALE - EXPLICIT THERMAL ASSUMPTION]")
    jeans = {}
    for c in (baseline, models[2], models[3], models[4], models[5]):
        jeans[c.name] = gas_jeans_mass_msun(c, Z_GS14, GAS_TEMPERATURE_K, MEAN_MOLECULAR_WEIGHT)
        print(f"{c.name}: M_J={jeans[c.name]:.4e} Msun")
    print(f"Legacy fixed-fraction Jeans-mass ratio CoDE4/LCDM: {jeans[models[2].name]/jeans[baseline.name]:.6f}")
    print("The previous expression omitted c_s^3 and therefore was not an absolute Jeans mass. This corrected scale is only illustrative because gas temperature and density structure are unspecified.")

    dl_ratio_sq = rows[models[2].name][2]
    legacy_timescale_reproduction(baseline, models[2], dl_ratio_sq,
                                  baseline_age, rows[models[2].name][1])
    print("\n[INTERPRETATION]")
    print("The D_L^2 rescaling can lower an inferred stellar mass under the same SED assumptions. It does not, by itself, halve a cosmological formation-time tension.")
    print("In the legacy fixed-Omega_m comparison, the higher H0 makes the z~14 universe younger; the epsilon modification is small there.")
    print("A defensible abundance/tension claim needs halo abundance, survey volume/completeness, stellar-to-halo efficiency, and a self-consistent star-formation history.")
    print("=" * 86)


if __name__ == "__main__":
    main()
