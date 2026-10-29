#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code4_engine.py  --  integrated CoDE4 observational comparison engine
=========================================================

What changed relative to CoDE4 (all of these were holes):

  1. NO matter power spectrum existed.  Now: primordial A_s,n_s  x  Eisenstein-Hu
     transfer function  x  model growth D(a)  ->  P(k,z), and sigma8 is COMPUTED
     from the integral over P(k) instead of being hard-coded to 0.811.
  2. Physical densities.  Omega_gamma, Omega_r scale as 1/h^2 and (optionally) omega_b,
     omega_c are held fixed while H0 is solved (CoDE4 held Omega fractions fixed, which
     silently changes omega_m = Omega_m h^2 and omega_b when H0 moves).
  3. Recombination is solved (Peebles 3-level atom + Saha for He / early H) so z_*,
     the optical depth, tau_reion, and the diffusion-damping scale k_D are computed
     with the MODEL's H(z).  CoDE4 used z=1100 and a "damping test" that compared H(z) to
     itself and printed a hard-coded "Consistent".
  4. Growth ODE solved with an adaptive 8th-order integrator in ln a, starting in the
     radiation era with the Meszaros initial condition (CoDE4: forward Euler, delta=a
     initial condition that is wrong before equality, growth[0] left at 0).
  5. D_V bug fixed.  CoDE4 used D_A where D_M = (1+z) D_A belongs (D_V was too small).
  6. BAO, SH0ES H0, compressed Planck CMB scales, and Pantheon+ supernova distances
     are reported as model predictions and absolute differences from reference values.

KNOWN LIMITS (read these before quoting any number):
  * Transfer function: Eisenstein & Hu (1998) zero-baryon-oscillation fit, built from the
     model's omega_m, omega_b, h.  Approximate sigma8 accuracy; no BAO wiggles; no
    neutrino mass; no dark-energy perturbations.  The model's modifications enter through
    H(z) and the growth source term only.
  * Recombination: Peebles 3-level atom (+ fudge 1.14).  Not RECFAST/HyRec.
  * This is NOT a Boltzmann code: no C_l(TT,TE,EE), no lensing potential.  For those,
    patch H(z) and G(z) into CLASS or CAMB.  What you get here are the compressed CMB
    quantities (theta_*, l_A, R, z_*, r_*, r_d, k_D, tau).
  * Observational numbers hard-coded below were typed from the cited papers; the table is
    at the top of the data section so you can audit/update it.

INTEGRATED DIAGNOSTICS ADDED:
  * SH0ES Hubble-flow corrected scale and forward H0 prediction.
  * Point-by-point absolute differences for all 13 DESI DR2 BAO observables.
  * Full Pantheon+ Hubble-flow supernova distance shifts; the released covariance is
    used only to remove a common magnitude intercept.

Usage:
  python code4_engine.py                 # full observational comparison
"""
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp, simpson, cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================================
# 0. CONSTANTS
# ======================================================================
C_KMS      = 299792.458              # km/s
C_CMS      = 2.99792458e10           # cm/s
MPC_CM     = 3.0856775814914e24      # cm
T_CMB      = 2.7255                  # K
OMEGA_G_H2 = 2.4728e-5               # photon density, T=2.7255 K
N_EFF      = 3.046
OMEGA_R_H2 = OMEGA_G_H2 * (1 + 0.22711 * N_EFF)   # photons + massless neutrinos
GYR_KMS_MPC = 977.7922               # (1/H0)[Gyr] = 977.7922 / H0[km/s/Mpc]
RHO_CRIT_H2_MSUN_MPC3 = 2.77536627e11
RHO_CRIT_H2_G_CM3     = 1.87847e-29
M_H_G      = 1.673533e-24
SIGMA_T    = 6.6524587e-25           # cm^2
K_B_EV     = 8.617333262e-5
SAHA_PREF  = 2.4147e15               # (2 pi m_e k /h^2)^{3/2}  [cm^-3 K^-3/2]
Y_P        = 0.245                   # primordial helium mass fraction
F_HE       = Y_P / (4 * (1 - Y_P))   # n_He / n_H
DELTA_C    = 1.686

# ======================================================================
# 1. OBSERVATIONAL DATA  (audit / update here)
# ======================================================================
# Planck 2018 TT,TE,EE+lowE+lensing (Table 2 of Planck 2018 VI)
PLANCK = dict(H0=(67.36, 0.54), omega_b=(0.02237, 0.00015), omega_c=(0.1200, 0.0012),
              omega_m=(0.1430, 0.0011), Om=(0.3153, 0.0073), ns=0.9649,
              As=2.099e-9, kp=0.05, tau=(0.0544, 0.0073), sigma8=(0.8111, 0.0060),
              S8=(0.832, 0.013), zstar=(1089.92, 0.25), rstar=(144.43, 0.26),
              theta100=(1.04110, 0.00031), zdrag=(1059.94, 0.30), rdrag=(147.09, 0.26),
              age=(13.797, 0.023), zeq=(3402.0, 26.0))
# CMB distance priors, Chen, Huang & Wang 2019 (Planck 2018): R, l_A, omega_b + correlations
DIST_PRIOR_MEAN = np.array([1.7502, 301.471, 0.02236])
DIST_PRIOR_SIG  = np.array([0.0046, 0.090, 0.00015])
DIST_PRIOR_CORR = np.array([[1.00, 0.46, -0.66], [0.46, 1.00, -0.33], [-0.66, -0.33, 1.00]])
# DESI DR2 BAO (arXiv:2503.14738, as tabulated in many follow-ups): z, DM/rd, DH/rd, sig, sig, corr
DESI_DR2 = [  # z,     DM/rd,  sDM,   DH/rd,  sDH,   rho
    (0.510, 13.588, 0.167, 21.863, 0.425, -0.459),
    (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
    (0.934, 21.576, 0.152, 17.641, 0.193, -0.416),
    (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
    (1.484, 30.512, 0.760, 12.817, 0.516, -0.500),
    (2.330, 38.988, 0.531,  8.632, 0.101, -0.431)]
DESI_BGS = (0.295, 7.942, 0.075)          # z, DV/rd, sigma

# Official DESI DR2 all-tracer Gaussian vector and covariance (13 measurements).
# Order: BGS DV; five (DM,DH) pairs; Ly-alpha (DH,DM).
DESI_BAO_MEAN = np.array([
    7.94167639, 13.58758434, 21.86294686, 17.35069094, 19.45534918,
    21.57563956, 17.64149464, 27.60085612, 14.17602155, 30.51190063,
    12.81699964, 8.631545674846294, 38.988973961958784,
])
def desi_bao_prediction(c, q):
    """Return DESI's 13-observable DR2 data-vector ordering."""
    rd = q["r_drag"]
    zb = DESI_BGS[0]
    dv = (zb * comoving_distance(c, zb) ** 2 * C_KMS / c.H(zb)) ** (1 / 3)
    values = [dv / rd]
    for z, *_ in DESI_DR2[:5]:
        values.extend((comoving_distance(c, z) / rd, C_KMS / c.H(z) / rd))
    z_lya = DESI_DR2[-1][0]
    values.extend((C_KMS / c.H(z_lya) / rd, comoving_distance(c, z_lya) / rd))
    return np.asarray(values, dtype=float)


# RSD f*sigma8 (approximate literature values; fiducial-cosmology dependent; correlations ignored)
RSD = [(0.067, 0.423, 0.055, "6dFGS"), (0.15, 0.490, 0.145, "SDSS-MGS"),
       (0.38, 0.497, 0.045, "BOSS DR12"), (0.51, 0.458, 0.038, "BOSS DR12"),
       (0.61, 0.436, 0.034, "BOSS DR12"), (0.70, 0.473, 0.044, "eBOSS LRG"),
       (1.48, 0.462, 0.045, "eBOSS QSO")]
S8_DATA = [("Planck 2018", 0.832, 0.013), ("DES Y3 3x2pt", 0.776, 0.017),
           ("KiDS-Legacy", 0.815, 0.019), ("DES+KiDS-1000", 0.790, 0.016)]
H0_SHOES = (73.04, 1.04)
# Independent baryon constraint: primordial D/H mapped through standard BBN.
# Pettini & Cooke 2026 quote omega_b = 0.02241 +/- 0.00031 from D/Hp
# = (2.510 +/- 0.028)e-5 using LINX with PArthENoPE nuclear rates.
OMEGA_B_BBN = (0.02241, 0.00031)
# Unanchored Pantheon+ SN-only fit under the CoDE4 background (eps=0.05).
# The fit uses m_b_corr, excludes Cepheid calibrators, profiles the magnitude
# intercept, and has no SH0ES H0 or CMB likelihood input.
CODE4_OMEGA_M = 0.32917
CODE4_OMEGA_M_PROFILE68 = (0.31109, 0.34771)
THETA_OBS = PLANCK["theta100"][0] * 1e-2

# ======================================================================
# 2. COSMOLOGY (background)
# ======================================================================
@dataclass(frozen=True)
class Cosmo:
    name: str
    H0: float            # km/s/Mpc
    Om: float            # Omega_m  (b + c)
    Ob: float            # Omega_b
    eps: float = 0.0     # CoDE early/DE-scaling modification
    beta: float = 0.0    # CoDE growth coupling
    ns: float = PLANCK["ns"]
    As: float = PLANCK["As"]
    z_reion: float = 7.67
    legacy: bool = False   # True = CoDE4 radiation bookkeeping (Omega_r0=9e-5, Omega_gamma0=5.38e-5 held fixed)

    @property
    def h(self):        return self.H0 / 100.0
    @property
    def omega_m(self):  return self.Om * self.h ** 2
    @property
    def omega_b(self):  return self.Ob * self.h ** 2
    @property
    def omega_c(self):  return self.omega_m - self.omega_b
    @property
    def Or(self):       return 9e-5 if self.legacy else OMEGA_R_H2 / self.h ** 2
    @property
    def omega_g(self):  return 5.38e-5 * self.h ** 2 if self.legacy else OMEGA_G_H2
    @property
    def Ode(self):      return 1.0 - self.Om - self.Or     # flat by construction

    # ---- model modifications (identical functional forms to CoDE4) ----
    @staticmethod
    def f_of_z(z):      return z / (1 + z) ** 2
    def C_of_z(self, z): return 1 + self.eps * self.f_of_z(z) * (1 - 0.15 / (1 + z))
    def G_ratio(self, z): return 1 + self.beta * self.f_of_z(z)

    def E2(self, z):
        z = np.asarray(z, dtype=float)
        rad = self.Or * (1 + z) ** 4 * (1 + self.eps * z / (1 + z) ** 3)
        return self.Om * (1 + z) ** 3 + rad + self.Ode * self.C_of_z(z)

    def E(self, z):     return np.sqrt(self.E2(z))
    def H(self, z):     return self.H0 * self.E(z)                       # km/s/Mpc
    def H_per_s(self, z): return self.H(z) * 1e5 / MPC_CM                # 1/s

    @classmethod
    def physical(cls, name, H0, wb, wc, **kw):
        h2 = (H0 / 100.0) ** 2
        return cls(name, H0, (wb + wc) / h2, wb / h2, **kw)

    @classmethod
    def fractions(cls, name, H0, Om=CODE4_OMEGA_M, Ob=0.049, **kw):
        return cls(name, H0, Om, Ob, **kw)


def comoving_distance(c, z):
    """D_M(z) = D_C in a flat universe, Mpc."""
    return quad(lambda zz: C_KMS / c.H(zz), 0, z, epsabs=0, epsrel=1e-11, limit=300)[0]


def sound_horizon(c, z):
    """r_s(z) = int_z^inf c_s/H dz', Mpc (integrated in ln(1+z))."""
    Rf = 3 * c.omega_b / (4 * c.omega_g)

    def integrand(lx):
        x = np.exp(lx)
        cs = C_KMS / np.sqrt(3 * (1 + Rf / x))
        return cs / c.H(x - 1) * x
    return quad(integrand, np.log1p(z), np.log(1e10), epsabs=0, epsrel=1e-10, limit=400)[0]


def z_drag_EH(c):
    wm, wb = c.omega_m, c.omega_b
    b1 = 0.313 * wm ** -0.419 * (1 + 0.607 * wm ** 0.674)
    b2 = 0.238 * wm ** 0.223
    return 1291 * wm ** 0.251 / (1 + 0.659 * wm ** 0.828) * (1 + b1 * wb ** b2)


def z_star_HS(c):
    wm, wb = c.omega_m, c.omega_b
    g1 = 0.0783 * wb ** -0.238 / (1 + 39.5 * wb ** 0.763)
    g2 = 0.560 / (1 + 21.1 * wb ** 1.81)
    return 1048 * (1 + 0.00124 * wb ** -0.738) * (1 + g1 * wm ** g2)


def age_Gyr(c):
    f = lambda lx: 1.0 / c.E(np.exp(lx) - 1)
    return GYR_KMS_MPC / c.H0 * quad(f, 0, np.log(1e10), epsabs=0, epsrel=1e-11, limit=300)[0]


def z_equality(c):   return c.Om / c.Or - 1
def k_equality(c):   return (c.H0 / C_KMS) * np.sqrt(2 * c.Om * (1 + z_equality(c)))   # Mpc^-1


# ======================================================================
# 3. RECOMBINATION (Peebles 3-level atom, Saha before z=1700)
# ======================================================================
def _nH0(c):  return (1 - Y_P) * c.omega_b * RHO_CRIT_H2_G_CM3 / M_H_G          # cm^-3 today


def _saha_xe(z, c):
    """Full H + He Saha equilibrium, returns x_e = n_e/n_H."""
    T = T_CMB * (1 + z)
    kT = K_B_EV * T
    nH = _nH0(c) * (1 + z) ** 3
    S = lambda E: SAHA_PREF * T ** 1.5 * np.exp(-E / kT)
    SH, S1, S2 = S(13.6057), S(24.5874), S(54.4178)

    def F(xe):
        ne = xe * nH
        xp = SH / (ne + SH)
        r1 = 4 * S1 / ne
        r2 = S2 / ne
        N = 1 + r1 + r1 * r2
        return xe - (xp + F_HE * (r1 + 2 * r1 * r2) / N)
    return brentq(F, 1e-12, 1 + 2 * F_HE + 1e-3, xtol=1e-14, rtol=1e-13)


def _peebles(c, z0=1700.0):
    H0s = lambda z: c.H_per_s(z)
    nH0 = _nH0(c)
    alpha_fudge = 1.14
    Lam2s, lam_a, B1, B2 = 8.2245809, 1.21567e-5, 13.6057, 13.6057 / 4

    def rhs(z, y):
        x = min(max(y[0], 1e-12), 1.0)
        T = T_CMB * (1 + z)
        kT = K_B_EV * T
        nH = nH0 * (1 + z) ** 3
        t4 = T / 1e4
        aB = alpha_fudge * 4.309e-13 * t4 ** -0.6166 / (1 + 0.6703 * t4 ** 0.53)
        bB = aB * SAHA_PREF * T ** 1.5 * np.exp(-B2 / kT)
        H = float(H0s(z))
        K = lam_a ** 3 / (8 * np.pi * H)
        Cf = (1 + K * Lam2s * nH * (1 - x)) / (1 + K * (Lam2s + bB) * nH * (1 - x))
        dxdt = Cf * (bB * (1 - x) * np.exp(-(B1 - B2) / kT) - aB * nH * x * x)
        return [dxdt / ((1 + z) * H)]          # dx/dz = -(1/((1+z)H)) * (-dxdt) ... sign handled below
    # dx/dt = Cf*(ionisation - recombination);  dz/dt = -(1+z)H  =>  dx/dz = -dxdt/((1+z)H)
    def rhs_z(z, y):
        return [-rhs(z, y)[0]]

    xe0 = _saha_xe(z0, c)
    xp0 = min(xe0 - F_HE * 0.0, 1.0)          # He is neutral to <1e-8 at z<1700
    sol = solve_ivp(rhs_z, [z0, 0.0], [xp0], method="Radau", rtol=1e-8, atol=1e-12,
                    dense_output=True)
    return sol


@lru_cache(maxsize=64)
def recombination(c):
    """Ionization history, optical depth, z_*, k_D (all with the model's H(z))."""
    sol = _peebles(c)
    z_lo = np.linspace(0, 3000, 4001)
    z_hi = np.logspace(np.log10(3000.0), 6.5, 700)[1:]
    zg = np.concatenate([z_lo, z_hi])
    xe = np.empty_like(zg)
    xe_nr = np.empty_like(zg)                     # same, without reionisation
    y_re = (1 + c.z_reion) ** 1.5
    dy = 1.5 * np.sqrt(1 + c.z_reion) * 0.5
    for i, z in enumerate(zg):
        if z < 1700.0:
            xr = float(sol.sol(z)[0])
            s_re = 0.5 * (1 + np.tanh((y_re - (1 + z) ** 1.5) / dy))
            xe_nr[i] = xr
            xe[i] = xr + (1 + F_HE - xr) * s_re
        else:
            xe_nr[i] = xe[i] = _saha_xe(z, c)
    nH0 = _nH0(c)
    pref = nH0 * SIGMA_T * C_CMS * (1 + zg) ** 2 / c.H_per_s(zg)
    tau = cumulative_trapezoid(xe * pref, zg, initial=0.0)
    tau_nr = cumulative_trapezoid(xe_nr * pref, zg, initial=0.0)
    z_star = float(np.interp(1.0, tau, zg))
    tau_reion = float(np.interp(30.0, zg, tau) - np.interp(30.0, zg, tau_nr))
    Rf0 = 3 * c.omega_b / (4 * c.omega_g)
    tau_d = cumulative_trapezoid(xe_nr * pref * (1 + zg) / Rf0, zg, initial=0.0)   # Compton drag depth
    z_drag = float(np.interp(1.0, tau_d, zg))
    # diffusion damping scale at z_*
    Rf = 3 * c.omega_b / (4 * c.omega_g)
    m = zg >= z_star
    zz = zg[m]
    R = Rf / (1 + zz)
    ne_sigT = xe[m] * nH0 * (1 + zz) ** 3 * SIGMA_T * MPC_CM            # Mpc^-1
    integ = (C_KMS / c.H(zz)) * (1 + zz) / (6 * (1 + R) * ne_sigT) * (R ** 2 / (1 + R) + 16.0 / 15.0)
    kD = 1.0 / np.sqrt(np.trapezoid(integ, zz))
    return dict(z=zg, xe=xe, tau=tau, z_star=z_star, z_drag=z_drag, tau_reion=tau_reion, kD=kD)


# ======================================================================
# 4. CMB COMPRESSED QUANTITIES + H0 SOLVER
# ======================================================================
def cmb_quantities(c):
    rec = recombination(c)
    zs = rec["z_star"]
    DM = comoving_distance(c, zs)
    rs = sound_horizon(c, zs)
    zd = rec["z_drag"]
    rd = sound_horizon(c, zd)
    theta = rs / DM
    lA = np.pi / theta
    R = np.sqrt(c.Om) * c.H0 * DM / C_KMS
    r_star = (c.Or * c.h ** 2 / c.omega_m) * (1 + zs)                        # rho_r/rho_m at z_*
    phi1 = 0.267 * (r_star / 0.3) ** 0.1
    ell1 = lA * (1 - phi1)
    thetaD = 1.0 / (rec["kD"] * DM)                                     # damping angle
    return dict(z_star=zs, z_star_HS=z_star_HS(c), DM_star=DM, r_star=rs, theta=theta,
                lA=lA, R=R, ell1=ell1, z_drag=zd, z_drag_EH=z_drag_EH(c), r_drag=rd, kD=rec["kD"], thetaD=thetaD,
                tau_reion=rec["tau_reion"])


def solve_H0(builder, target=None, lo=55.0, hi=95.0):
    """Find H0 such that theta_* = target.  Raises if there is no root (no silent non-convergence)."""
    target = THETA_OBS if target is None else target
    f = lambda H0: cmb_quantities(builder(H0))["theta"] - target
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise RuntimeError(f"theta_* = theta_obs has no root in H0 in [{lo},{hi}] (f={flo:.2e},{fhi:.2e})")
    H0 = brentq(f, lo, hi, xtol=1e-5, rtol=1e-9)
    return builder(H0)



# ======================================================================
# 5. GROWTH  (ln a ODE, radiation-era start, scale-independent)
# ======================================================================
@lru_cache(maxsize=64)
def growth_solution(c):
    ai = 1e-6
    a_eq = c.Or / c.Om
    d = 1e-4

    def rhs(x, y):
        a = np.exp(x)
        z = 1 / a - 1
        e2 = float(c.E2(z))
        dlnE = 0.5 * (np.log(float(c.E2(np.exp(-(x + d)) - 1))) - np.log(float(c.E2(np.exp(-(x - d)) - 1)))) / (2 * d)
        Oma = c.Om * a ** -3 / e2
        return [y[1], -(2 + dlnE) * y[1] + 1.5 * Oma * float(c.G_ratio(z)) * y[0]]

    y0 = [1 + 1.5 * ai / a_eq, 1.5 * ai / a_eq]       # Meszaros growing mode
    sol = solve_ivp(rhs, [np.log(ai), 0.0], y0, method="DOP853", rtol=1e-10, atol=1e-14,
                    dense_output=True)
    norm = 1.5 / a_eq                                   # makes D -> a deep in matter domination
    return sol, norm


def D_of_a(c, a):
    sol, norm = growth_solution(c)
    return sol.sol(np.log(a))[0] / norm


def f_of_a(c, a):
    sol, _ = growth_solution(c)
    y = sol.sol(np.log(a))
    return y[1] / y[0]


def Omega_m_a(c, a):
    return c.Om * a ** -3 / c.E2(1 / a - 1)


# ======================================================================
# 6. MATTER POWER SPECTRUM
# ======================================================================
def transfer_EH_nowiggle(k, c):
    """Eisenstein & Hu 1998 (eqs 26,28-31), k in Mpc^-1."""
    h, wm, wb = c.h, c.omega_m, c.omega_b
    fb = c.Ob / c.Om
    th2 = (T_CMB / 2.7) ** 2
    s = 44.5 * np.log(9.83 / wm) / np.sqrt(1 + 10 * wb ** 0.75)
    ag = 1 - 0.328 * np.log(431 * wm) * fb + 0.38 * np.log(22.3 * wm) * fb ** 2
    Geff = c.Om * h * (ag + (1 - ag) / (1 + (0.43 * k * s) ** 4))
    q = (k / h) * th2 / Geff
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q ** 2)


class MatterPower:
    """P(k, z) = 2 pi^2/k^3 * (4/25) (k/H0c)^4/Om^2 * T^2 * A_s (k/kp)^(ns-1) * D(z)^2."""
    def __init__(self, c, amp_scale=1.0):
        self.c = c
        self.amp_scale = amp_scale
        self.H0c = c.H0 / C_KMS                      # Mpc^-1
        self.lnk = np.linspace(np.log(1e-5), np.log(1e3), 6000)
        self.k = np.exp(self.lnk)
        self.g0 = D_of_a(c, 1.0)
        self.Delta2_0 = self._delta2(self.k)

    def _delta2(self, k):
        c = self.c
        prim = c.As * (k / PLANCK["kp"]) ** (c.ns - 1)
        T = transfer_EH_nowiggle(k, c)
        return self.amp_scale * (4 / 25) * (k / self.H0c) ** 4 / c.Om ** 2 * T ** 2 * prim * self.g0 ** 2

    def Pk(self, k, z=0.0):
        a = 1 / (1 + z)
        Dr = D_of_a(self.c, a) / self.g0
        return 2 * np.pi ** 2 / k ** 3 * self._delta2(np.asarray(k, float)) * Dr ** 2

    def sigma(self, R_mpc, z=0.0):
        x = self.k * R_mpc
        W = np.where(x < 1e-3, 1 - x ** 2 / 10, 3 * (np.sin(x) - x * np.cos(x)) / np.where(x == 0, 1, x) ** 3)
        s2 = simpson(self.Delta2_0 * W ** 2, x=self.lnk)
        return np.sqrt(s2) * D_of_a(self.c, 1 / (1 + z)) / self.g0

    def sigma8(self, z=0.0):
        return self.sigma(8.0 / self.c.h, z)

    def sigma_M(self, M_msun, z=0.0):
        rho_m = self.c.Om * RHO_CRIT_H2_MSUN_MPC3 * self.c.h ** 2
        R = (3 * M_msun / (4 * np.pi * rho_m)) ** (1 / 3)
        return self.sigma(R, z)

    def k_peak(self):
        P = 2 * np.pi ** 2 / self.k ** 3 * self.Delta2_0
        i = np.argmax(P)
        return self.k[i], P[i]


# ======================================================================
# 10A. INTEGRATED SH0ES + DARK-SECTOR DIAGNOSTICS
# ======================================================================
# These constants intentionally match the two snippets supplied for integration.
# The direct comparison preserves their fixed Omega_r convention; the full Cosmo
# engine uses its own (normally h-dependent) radiation bookkeeping.
SHOES_OR = 9.2e-5
SHOES_EPS = 0.05
SHOES_H0_MEASURED = 73.04
SHOES_H0_SIGMA = 1.04
SHOES_Q0_FIDUCIAL = -0.55
SHOES_J0_FIDUCIAL = 1.0
SHOES_ZGRID = np.linspace(0.023, 0.15, 400)
SHOES_WEIGHTS = np.ones_like(SHOES_ZGRID)

DARK_SECTOR_REGIMES = {
    "Perfectly Local (Right Now)": 0.0,
    "SH0ES Near Limit": 0.023,
    "SH0ES Mid-Range": 0.05,
    "SH0ES Far Limit": 0.15,
    "Deep Supernovae Regime": 0.5,
    "Galaxy Cluster Regime": 1.0,
    "Cosmic Dawn (Early Galaxies)": 10.0,
}


def E_model(z, Om, eps=SHOES_EPS, Or=SHOES_OR):
    """Dimensionless E(z) from the supplied SH0ES/dark-sector snippets.

    This helper deliberately retains the snippets' fixed Or=9.2e-5 convention.
    For results from the full engine use ``c.E(z)`` instead.
    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0):
        raise ValueError("Redshift z must be non-negative.")
    if not (0.0 < Om < 1.0) or Or < 0.0:
        raise ValueError("Require 0 < Om < 1 and Or >= 0.")
    f = z / (1.0 + z) ** 2
    C = 1.0 + eps * f * (1.0 - 0.15 / (1.0 + z))
    rad = Or * (1.0 + z) ** 4 * (1.0 + eps * z / (1.0 + z) ** 3)
    E2 = Om * (1.0 + z) ** 3 + rad + (1.0 - Om - Or) * C
    if np.any(~np.isfinite(E2)) or np.any(E2 <= 0):
        raise ValueError("The supplied parameters produce non-positive or non-finite E(z)^2.")
    result = np.sqrt(E2)
    return float(result) if result.ndim == 0 else result


def f_exact(z, Om, eps=SHOES_EPS, Or=SHOES_OR):
    """Exact luminosity-distance shape f(z)=(1+z)/z * integral(1/E dz)."""
    z = float(z)
    if z <= 0:
        raise ValueError("f_exact requires z > 0; use the z->0 limit f=1 at z=0.")
    integral = quad(lambda x: 1.0 / E_model(x, Om, eps, Or), 0.0, z,
                    epsabs=0.0, epsrel=1e-12, limit=200)[0]
    return (1.0 + z) * integral / z


def f_cosmographic(z, q0=SHOES_Q0_FIDUCIAL, j0=SHOES_J0_FIDUCIAL):
    """Second-order cosmographic luminosity-distance shape from the supplied code."""
    z = np.asarray(z, dtype=float)
    return (1.0 + 0.5 * (1.0 - q0) * z
            - (1.0 / 6.0) * (1.0 - q0 - 3.0 * q0**2 + j0) * z**2)


def shoes_correction_factor(Om, eps=SHOES_EPS, zgrid=SHOES_ZGRID,
                            weights=SHOES_WEIGHTS, Or=SHOES_OR,
                            q0=SHOES_Q0_FIDUCIAL, j0=SHOES_J0_FIDUCIAL):
    """Weighted mean of exact/model-to-cosmographic distance-shape ratios."""
    zgrid = np.asarray(zgrid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if zgrid.ndim != 1 or weights.shape != zgrid.shape or zgrid.size == 0:
        raise ValueError("zgrid and weights must be non-empty 1-D arrays of equal shape.")
    if np.any(zgrid <= 0) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Require zgrid > 0 and non-negative weights with positive total weight.")
    ratios = np.array([f_exact(z, Om, eps, Or) / f_cosmographic(z, q0, j0) for z in zgrid])
    if np.any(~np.isfinite(ratios)):
        raise ValueError("SH0ES distance ratio contains a non-finite value.")
    return float(np.average(ratios, weights=weights))


def compute_shoes_h0(Om, eps=SHOES_EPS, H0_measured=SHOES_H0_MEASURED,
                     zgrid=SHOES_ZGRID, weights=SHOES_WEIGHTS, Or=SHOES_OR):
    """Return the supplied snippet's weighted SH0ES-inferred H0 correction."""
    return float(H0_measured * shoes_correction_factor(Om, eps, zgrid, weights, Or))


def f_exact_for_cosmo(z, c):
    """Distance shape evaluated with a model instance from the full Cosmo engine."""
    z = float(z)
    if z <= 0:
        raise ValueError("f_exact_for_cosmo requires z > 0.")
    integral = quad(lambda x: 1.0 / float(c.E(x)), 0.0, z,
                    epsabs=0.0, epsrel=1e-12, limit=200)[0]
    return (1.0 + z) * integral / z


def compute_shoes_h0_for_cosmo(c, H0_measured=SHOES_H0_MEASURED,
                               zgrid=SHOES_ZGRID, weights=SHOES_WEIGHTS):
    """Apply the same SH0ES weighting to a fitted model from the full engine."""
    zgrid = np.asarray(zgrid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if zgrid.ndim != 1 or weights.shape != zgrid.shape or zgrid.size == 0:
        raise ValueError("zgrid and weights must be non-empty 1-D arrays of equal shape.")
    if np.any(zgrid <= 0) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Require zgrid > 0 and non-negative weights with positive total weight.")
    ratios = np.array([
        f_exact_for_cosmo(z, c) / f_cosmographic(z)
        for z in zgrid
    ])
    return float(H0_measured * np.average(ratios, weights=weights))


def predict_shoes_apparent_h0(c, zgrid=SHOES_ZGRID, weights=SHOES_WEIGHTS):
    """Forward-predict the H0 a fiducial SH0ES-style fit would infer.

    Unlike compute_shoes_h0_for_cosmo, this does not use measured SH0ES H0
    as an input. The model global H0 sets its distance scale, and the low-z
    distance-shape ratio maps that scale to the fiducial cosmographic fit.
    """
    zgrid = np.asarray(zgrid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if zgrid.ndim != 1 or weights.shape != zgrid.shape or zgrid.size == 0:
        raise ValueError("zgrid and weights must be non-empty 1-D arrays of equal shape.")
    if np.any(zgrid <= 0) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Require zgrid > 0 and non-negative weights with positive total weight.")
    ratios = np.array([f_exact_for_cosmo(z, c) / f_cosmographic(z) for z in zgrid])
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("SH0ES forward distance ratio must be finite and positive.")
    return float(c.H0 / np.average(ratios, weights=weights))

def check_dark_sector_fraction(z, Om, eps=SHOES_EPS, Or=SHOES_OR):
    """Return unit fractions (dark energy, matter, radiation) using supplied E(z)."""
    z = float(z)
    if z < 0:
        raise ValueError("Redshift z must be non-negative.")
    # Evaluate the same three components used by E_model; fractions sum to one.
    f = z / (1.0 + z) ** 2
    C = 1.0 + eps * f * (1.0 - 0.15 / (1.0 + z))
    rad = Or * (1.0 + z) ** 4 * (1.0 + eps * z / (1.0 + z) ** 3)
    mat = Om * (1.0 + z) ** 3
    de = (1.0 - Om - Or) * C
    total = mat + rad + de
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Total E(z)^2 must be positive and finite.")
    return de / total, mat / total, rad / total


def dark_sector_fraction_for_cosmo(z, c):
    """Return unit fractions (dark energy, matter, radiation) for a Cosmo instance."""
    z = float(z)
    if z < 0:
        raise ValueError("Redshift z must be non-negative.")
    mat = c.Om * (1.0 + z) ** 3
    rad = c.Or * (1.0 + z) ** 4 * (1.0 + c.eps * z / (1.0 + z) ** 3)
    de = c.Ode * c.C_of_z(z)
    total = mat + rad + de
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Total E(z)^2 must be positive and finite for {c.name} at z={z}.")
    return de / total, mat / total, rad / total


def _print_dark_sector_table(label, fraction_fn):
    print(f"\n  {label}")
    print(f"  {'Cosmic regime':<30} | {'z':>7} | {'Dark energy':>14} | {'Matter':>10} | {'Radiation':>12} | {'Sum':>8}")
    print("  " + "-" * 96)
    for name, z in DARK_SECTOR_REGIMES.items():
        de_pct, mat_pct, rad_pct = fraction_fn(z)
        print(f"  {name:<30} | {z:7.3f} | {de_pct:14.6f} | {mat_pct:10.6f} | {rad_pct:12.6f} | {de_pct + mat_pct + rad_pct:8.6f}")


def run_integrated_diagnostics(engine_models=None, include_supplied=True):
    """Run supplied diagnostics and optionally report models from the full engine."""
    if include_supplied:
        print("\n" + "=" * 100 + "\n  INTEGRATED SH0ES + DARK-SECTOR DIAGNOSTICS\n" + "=" * 100, flush=True)
        print("  Direct comparison preserves the snippets' fixed Or=9.2e-5 and uniform-in-z weights.", flush=True)
        print("  Note: the Om=0.3155 case is the supplied parameter proxy; fixed physical densities are handled by Cosmo.physical().", flush=True)
        print(f"\n  {'Comparison':<46} {'Omega_m':>9} {'eps':>7} {'factor':>10} {'H0 inferred':>14}")
        print("  " + "-" * 91)
        direct_cases = [
            ("CoDE4 SN-only unanchored fit", CODE4_OMEGA_M, SHOES_EPS),
            ("CoDE5 supplied Om proxy (not a density solve)", 0.3155, SHOES_EPS),
        ]
        for label, Om, eps in direct_cases:
            factor = shoes_correction_factor(Om, eps)
            h0_local = SHOES_H0_MEASURED * factor
            print(f"  {label:<46} {Om:9.4f} {eps:7.3f} {factor:10.6f} {h0_local:14.3f}")

        _print_dark_sector_table(
            "Dark-sector fractions (fixed-Or SH0ES-shape helper; CoDE4 SN-only Omega_m)",
            lambda z: check_dark_sector_fraction(z, Om=CODE4_OMEGA_M, eps=SHOES_EPS, Or=SHOES_OR),
        )

    if engine_models:
        print("\n  Applying the same distance weighting to models solved by the full Cosmo engine:", flush=True)
        print(f"  {'Engine model':<40} {'Omega_m':>9} {'Omega_r':>10} {'H0 engine':>11} {'SH0ES-corrected':>17}")
        print("  " + "-" * 91)
        for c in engine_models:
            h0_local = compute_shoes_h0_for_cosmo(c)
            print(f"  {c.name:<40} {c.Om:9.5f} {c.Or:10.6g} {c.H0:11.3f} {h0_local:17.3f}")
        for c in engine_models:
            _print_dark_sector_table(
                f"Dark-sector fractions from the full engine: {c.name}",
                lambda z, model=c: dark_sector_fraction_for_cosmo(z, model),
            )


def legacy_decomposition(eps, beta, theta_target):
    """Why did CoDE4 report H0 = 72.9?  Replicates its solver and then changes one bookkeeping choice at a time."""
    print("\n" + "=" * 100 + "\n  WHERE CoDE4's H0 = 72.9 CAME FROM\n" + "=" * 100)
    th_obs4 = 0.01041

    def codee4_theta(H0):
        """CoDE4's own numerics: z*=1100, trapezoid on 20000-pt log grid / 5000-pt linear grid, fixed Omega_r0, Omega_gamma0."""
        c = Cosmo.fractions("legacy", H0, CODE4_OMEGA_M, 0.049, eps=eps, beta=beta, legacy=True)
        Rf = 3 * 0.049 / (4 * 5.38e-5)
        z = np.logspace(np.log10(1100), 7, 20000)
        rs = np.trapezoid(C_KMS / np.sqrt(3 * (1 + Rf / (1 + z))) / c.H(z), z)
        zz = np.linspace(0, 1100, 5000)
        return rs / np.trapezoid(C_KMS / c.H(zz), zz)

    print("  (1) theta_*(H0) with CoDE4's bookkeeping (Omega_m, Omega_b, Omega_r0, Omega_gamma0 all fixed fractions):")
    for H0 in (55, 65, 72.86, 85, 100):
        print(f"        H0 = {H0:6.2f}   theta_* = {codee4_theta(H0):.7f}")
    print("      -> theta_* is the SAME number for every H0 (the whole problem just rescales as 1/H0).")
    print("         The H0 'solution' therefore has nothing to converge to.")

    # (2) replicate the CoDE4 damped iteration
    H0g, converged = 70.0, False
    for i in range(60):
        corr = th_obs4 / codee4_theta(H0g)
        H0n = H0g * (1 + 0.35 * (corr - 1))
        if abs(H0n - H0g) < 1e-3:
            H0g, converged = H0n, True
            break
        H0g = H0n
    print(f"  (2) CoDE4 loop replicated (start 70, damping 0.35, 60 iterations, tol 1e-3):  H0 = {H0g:.2f}   converged = {converged}")
    print("      CoDE4 returns the last iterate silently when it hits the iteration cap; each pass multiplies H0 by")
    print(f"      1 + 0.35*(theta_obs/theta - 1) = {1 + 0.35 * (th_obs4 / codee4_theta(70.0) - 1):.6f},  so H0 = 70 * that^60.  (The 'converged' message never printed in your run.)")

    # (3) physical radiation, still z*=1100
    b_phy = lambda H0: Cosmo.fractions("phys-rad", H0, CODE4_OMEGA_M, 0.049, eps=eps, beta=beta, legacy=False)
    f = lambda H0: sound_horizon(b_phy(H0), 1100.0) / comoving_distance(b_phy(H0), 1100.0) - th_obs4
    H_b = brentq(f, 55, 95, xtol=1e-5)
    print(f"  (3) photon + radiation densities scaled physically (omega fixed), z*=1100, theta_obs=0.01041:  H0 = {H_b:.2f}")
    # (4) + solved z*
    H_d = solve_H0(lambda H0: Cosmo.fractions("phys-rad", H0, CODE4_OMEGA_M, 0.049, eps=eps, beta=beta), theta_target).H0
    print(f"  (4) ... and z_* from the recombination solver (this pipeline, fixed Omega fractions):         H0 = {H_d:.2f}")
    return H0g, H_b, H_d

# ======================================================================
# 11. MAIN - METRICS ONLY
# ======================================================================
def _pantheon_plus_paths():
    from pathlib import Path
    import urllib.request

    engine_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if not (engine_dir / "pantheon_plus_data").is_dir() and (engine_dir / "outputs").is_dir():
        engine_dir = engine_dir / "outputs"
    data_dir = engine_dir / "pantheon_plus_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    table = data_dir / "Pantheon+SH0ES.dat"
    cov = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    readme = data_dir / "README.txt"
    for path, remote in ((table, "Pantheon%2BSH0ES.dat"), (cov, "Pantheon%2BSH0ES_STAT%2BSYS.cov"), (readme, "README")):
        if not path.is_file():
            print(f"Fetching official Pantheon+ release file: {path.name}")
            urllib.request.urlretrieve(base + remote, path)
    return table, cov


def shoes_full_ladder_local_refit(Om=CODE4_OMEGA_M, eps=SHOES_EPS, Or=SHOES_OR):
    """Refit the official three-rung SH0ES matrix with the CoDE Hubble-flow shape.

    The released L/y/C matrices encode the baseline cosmographic Hubble-flow
    relation. This replaces only its 277 Hubble-flow RHS terms and refits all
    ladder parameters with the full covariance. No CMB prior enters.

    Algebra: with A=L.T, the generalized least-squares solution is
      q_hat = (A.T C^-1 A)^-1 A.T C^-1 y_model.
    The released parameter convention maps the final ladder component to
      H0 = 10**(q_hat[46]/5).
    The CoDE distance-shape correction is
      delta_mu(z) = 5 log10[f_CoDE(z)/f_fid(z)],
    where f(z)=(1+z) integral_0^z dz'/E(z')./z.

    The public matrix lacks Hubble-flow SN identifiers, so the matching to
    Pantheon+ rows is reconstructed by sorting their fiducial RHS values.
    Returned residual diagnostics expose the quality of that reconstruction.
    """
    from pathlib import Path
    import urllib.request
    from scipy.linalg import cho_factor, cho_solve

    engine_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if not (engine_dir / "shoes_ladder_data").is_dir() and (engine_dir / "outputs").is_dir():
        engine_dir = engine_dir / "outputs"
    data_dir = engine_dir / "shoes_ladder_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/SH0ES_Data/"
    lfs = "https://media.githubusercontent.com/media/PantheonPlusSH0ES/DataRelease/main/SH0ES_Data/"
    names = {"L": "alll_shoes_ceph_topantheonwt6.0_112221.fits",
             "y": "ally_shoes_ceph_topantheonwt6.0_112221.fits",
             "C": "allc_shoes_ceph_topantheonwt6.0_112221.fits"}
    paths = {key: data_dir / name for key, name in names.items()}
    for key, path in paths.items():
        if not path.is_file():
            urllib.request.urlretrieve((lfs if key == "C" else raw) + names[key], path)

    def read_fits_primary(path):
        with open(path, "rb") as stream:
            header = {}
            while True:
                block = stream.read(2880)
                if len(block) != 2880:
                    raise ValueError(f"Truncated FITS header: {path}")
                ended = False
                for pos in range(0, 2880, 80):
                    card = block[pos:pos + 80].decode("ascii", "ignore")
                    if card.startswith("END"):
                        ended = True
                        break
                    if card[8:10] == "= ":
                        key = card[:8].strip()
                        raw_value = card[10:30].split("/")[0].strip()
                        try:
                            header[key] = int(raw_value)
                        except ValueError:
                            header[key] = raw_value.strip("' ")
                if ended:
                    break
            if header.get("BITPIX") != -32:
                raise ValueError(f"Expected float32 FITS data in {path}")
            shape = tuple(header[f"NAXIS{i}"] for i in range(header["NAXIS"], 0, -1))
            count = int(np.prod(shape))
            data = np.frombuffer(stream.read(count * 4), dtype=">f4", count=count)
            if data.size != count:
                raise ValueError(f"Truncated FITS data array: {path}")
            return data.astype(np.float64).reshape(shape)

    L = read_fits_primary(paths["L"])
    y = read_fits_primary(paths["y"]).reshape(-1)
    C = read_fits_primary(paths["C"])
    if L.shape != (47, 3492) or y.shape != (3492,) or C.shape != (3492, 3492):
        raise ValueError(f"Unexpected SH0ES matrix shapes: L={L.shape}, y={y.shape}, C={C.shape}")
    hf_columns = np.flatnonzero(np.isclose(L[46], -1.0) & np.isclose(L[42], 1.0))
    if hf_columns.size != 277:
        raise ValueError(f"Expected 277 Hubble-flow rows; found {hf_columns.size}")

    pantheon_path, _ = _pantheon_plus_paths()
    pantheon = np.genfromtxt(pantheon_path, names=True, dtype=None, encoding="utf-8")
    hf = pantheon[np.asarray(pantheon["USED_IN_SH0ES_HF"], dtype=int) == 1]
    if len(hf) != hf_columns.size:
        raise ValueError("Pantheon+ and SH0ES Hubble-flow sample sizes do not agree")
    z = np.asarray(hf["zHD"], dtype=float)
    mb = np.asarray(hf["m_b_corr"], dtype=float)
    y_fid = mb - 5.0 * np.log10(C_KMS * z * f_cosmographic(z)) - 25.0
    matrix_order = np.argsort(y[hf_columns])
    pantheon_order = np.argsort(y_fid)
    match_resid = y[hf_columns[matrix_order]] - y_fid[pantheon_order]
    match_offset = float(np.median(match_resid))
    match_centered = match_resid - match_offset
    if float(np.std(match_centered)) > 0.01:
        raise ValueError("SH0ES Hubble-flow row matching failed its residual check")

    shape_ratio = np.array([
        f_exact(zz, Om, eps, Or) / f_cosmographic(zz)
        for zz in z[pantheon_order]
    ])
    y_model = y.copy()
    y_model[hf_columns[matrix_order]] -= 5.0 * np.log10(shape_ratio)

    # GLS solution for q in y ~= L.T @ q, with the full public covariance.
    A = L.T
    factor = cho_factor(C, lower=True, check_finite=False)
    solved = cho_solve(factor, np.column_stack((A, y, y_model)), check_finite=False)
    normal = A.T @ solved[:, :A.shape[1]]
    normal_cov = np.linalg.inv(normal)
    q_base = normal_cov @ (A.T @ solved[:, A.shape[1]])
    q_model = normal_cov @ (A.T @ solved[:, A.shape[1] + 1])
    residual_base = y - A @ q_base
    residual_model = y_model - A @ q_model
    hf_mask = np.zeros(y.size, dtype=bool)
    hf_mask[hf_columns] = True
    h0_base = float(10.0 ** (q_base[46] / 5.0))
    h0_model = float(10.0 ** (q_model[46] / 5.0))
    return {"h0_baseline": h0_base, "h0_model": h0_model,
            "delta_h0": h0_model - h0_base,
            "q46_baseline": float(q_base[46]), "q46_model": float(q_model[46]),
            "n_hubble_flow": int(hf_columns.size),
            "match_offset_mag": match_offset,
            "match_rms_mag": float(np.sqrt(np.mean(match_centered**2))),
            "match_max_abs_mag": float(np.max(np.abs(match_centered))),
            "hf_rms_baseline_mag": float(np.sqrt(np.mean(residual_base[hf_mask]**2))),
            "hf_rms_model_mag": float(np.sqrt(np.mean(residual_model[hf_mask]**2))),
            "omega_m_fixed": float(Om), "epsilon_fixed": float(eps)}


def pantheon_plus_residuals(model):
    """Summarize Pantheon+ magnitude residuals after profiling one offset.

    Distances are evaluated on one cumulative grid, avoiding a separate adaptive
    quadrature for every supernova and avoiding unnecessary transformed-distance arrays.
    """
    import csv
    from pathlib import Path

    table_path, covariance_path = _pantheon_plus_paths()
    data = np.genfromtxt(table_path, names=True, dtype=None, encoding="utf-8")
    required = {"zHD", "zHEL", "MU_SH0ES", "IS_CALIBRATOR", "CID", "IDSURVEY"}
    if data.size == 0 or not required.issubset(data.dtype.names):
        raise ValueError("Pantheon+ data table is missing required columns.")
    with open(covariance_path, "r", encoding="utf-8") as stream:
        n_cov = int(stream.readline().strip())
        cov_all = np.loadtxt(stream, dtype=float)
    if cov_all.size == n_cov * n_cov:
        cov_all = cov_all.reshape((n_cov, n_cov))
    if n_cov != len(data) or cov_all.shape != (len(data), len(data)):
        raise ValueError("Pantheon+ table and full covariance dimensions do not match.")

    zhd = np.asarray(data["zHD"], dtype=float)
    zhel = np.asarray(data["zHEL"], dtype=float)
    mu_obs = np.asarray(data["MU_SH0ES"], dtype=float)
    keep = (zhd > 0.01) & (np.asarray(data["IS_CALIBRATOR"], dtype=int) == 0) & np.isfinite(mu_obs)
    indexes = np.flatnonzero(keep)
    if len(indexes) < 2:
        raise ValueError("Pantheon+ Hubble-flow selection contains too few supernovae.")

    zhd, zhel, mu_obs = zhd[keep], zhel[keep], mu_obs[keep]
    zgrid = np.linspace(0.0, float(zhd.max()) + 1e-5, 60001)
    dc_grid = C_KMS / model.H0 * cumulative_trapezoid(1.0 / model.E(zgrid), zgrid, initial=0.0)
    dc = np.interp(zhd, zgrid, dc_grid)
    mu_model = 5.0 * np.log10((1.0 + zhel) * dc * 1.0e5)
    residual = mu_model - mu_obs
    cov = cov_all[np.ix_(indexes, indexes)]
    ones = np.ones(len(indexes))
    inv_cov_ones = np.linalg.solve(cov, ones)
    offset = float(inv_cov_ones @ residual / (ones @ inv_cov_ones))
    delta_mu = residual - offset
    out_path = Path(table_path).parent.parent / "pantheon_plus_residuals.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["CID", "IDSURVEY", "zHD", "zHEL", "MU_SH0ES", "mu_model_before_offset", "delta_mu_after_offset_mag"])
        for row, zh, zhe, muo, mum, dmu in zip(data[keep], zhd, zhel, mu_obs, mu_model, delta_mu):
            writer.writerow([row["CID"], row["IDSURVEY"], f"{zh:.8g}", f"{zhe:.8g}", f"{muo:.8f}", f"{mum:.8f}", f"{dmu:.8f}"])

    return {"n_total": len(data), "n_used": len(indexes), "z_min": float(zhd.min()), "z_max": float(zhd.max()),
            "median_mag": float(np.median(delta_mu)), "mean_mag": float(np.mean(delta_mu)),
            "rms_mag": float(np.sqrt(np.mean(delta_mu ** 2))),
            "csv_path": out_path}


def fit_code4_omega_m_from_pantheon():
    """Fit Omega_m from released, non-calibrator Pantheon+ distances in this file.

    A free magnitude intercept removes absolute calibration, so this fit does not
    use SH0ES H0 or a CMB likelihood. The fit is conditional on eps=0.05 and on
    Pantheon+'s released bias-corrected magnitudes/covariance.
    """
    # Reuse local copies when available; otherwise fetch the official release
    # files into pantheon_plus_data/ beside the engine (or the notebook cwd).
    table, cov_path = _pantheon_plus_paths()
    data = np.genfromtxt(table, names=True, dtype=None, encoding="utf-8")
    with cov_path.open(encoding="utf-8") as stream:
        n_cov = int(stream.readline().strip())
        cov_all = np.loadtxt(stream, dtype=float).reshape(n_cov, n_cov)
    if n_cov != len(data):
        raise ValueError("Pantheon+ table/covariance dimensions do not match.")
    keep = ((np.asarray(data["IS_CALIBRATOR"], dtype=int) == 0)
            & (np.asarray(data["zHD"], dtype=float) > 0.01))
    idx = np.flatnonzero(keep)
    zhd = np.asarray(data["zHD"][keep], dtype=float)
    zhel = np.asarray(data["zHEL"][keep], dtype=float)
    mb = np.asarray(data["m_b_corr"][keep], dtype=float)
    cov = cov_all[np.ix_(idx, idx)]
    cf = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones_like(mb)
    ci_one = cho_solve(cf, ones, check_finite=False)
    denom = float(ones @ ci_one)
    zgrid = np.linspace(0.0, float(zhd.max()) + 1e-5, 60001)
    hfit = 70.0  # only sets tiny physical radiation density; intercept absorbs H0
    h2 = (hfit / 100.0) ** 2

    def profile(om, epsilon):
        c = Cosmo.fractions("Pantheon+ SN-only profile", hfit, om,
                            OMEGA_B_BBN[0] / h2, eps=epsilon, beta=0.05)
        integral = cumulative_trapezoid(1.0 / np.asarray(c.E(zgrid)), zgrid, initial=0.0)
        dshape = np.interp(zhd, zgrid, integral)
        model_shape = 5.0 * np.log10((1.0 + zhel) * dshape)
        resid = mb - model_shape
        ci_resid = cho_solve(cf, resid, check_finite=False)
        intercept = float(ones @ ci_resid / denom)
        centered = resid - intercept * ones
        return float(centered @ ci_resid), intercept

    def fit(epsilon):
        opt = minimize_scalar(lambda om: profile(om, epsilon)[0], bounds=(0.05, 0.80),
                              method="bounded", options={"xatol": 1e-10})
        return float(opt.x), profile(float(opt.x), epsilon)[1]

    om_best, intercept = fit(0.05)
    return {"omega_m": om_best, "intercept": intercept, "n": len(zhd),
            "z_min": float(zhd.min()), "z_max": float(zhd.max()),
            "epsilon": 0.05, "fit": profile}


def main(argv=()):
    """Report raw predictions and absolute differences for the integrated datasets."""
    fit_result = fit_code4_omega_m_from_pantheon()
    om, eps, beta = fit_result["omega_m"], 0.05, 0.05
    wb = OMEGA_B_BBN[0]
    reference = Cosmo.physical("Planck 2018 reference", PLANCK["H0"][0], PLANCK["omega_b"][0], PLANCK["omega_c"][0])
    qref = cmb_quantities(reference)
    theta_target = qref["theta"]
    model = solve_H0(lambda h: Cosmo.fractions("CoDE4 BBN-DH omega_b model", h, om, wb/(h/100.0)**2, eps=eps, beta=beta), theta_target)
    q = cmb_quantities(model)

    print("\n" + "=" * 78)
    print("CoDE4 ENGINE - RAW PREDICTIONS AND ABSOLUTE DIFFERENCES")
    print("Comparison tables show model, reference, and absolute differences.")
    print("=" * 78)
    print("\n[MODEL INPUT AND INDEPENDENT OMEGA_M FIT]")
    print(f"Pantheon+ non-calibrator SNe used:            {fit_result['n']}")
    print(f"Pantheon+ redshift range:                     {fit_result['z_min']:.5f} to {fit_result['z_max']:.5f}")
    print(f"CoDE4 Omega_m fitted in this run:             {om:.5f} (epsilon fixed at {eps:.3f})")
    print("Fit uses m_b_corr, full released covariance, and a profiled free magnitude intercept.")
    print("No SH0ES H0 target, Planck prior, or BAO likelihood enters this Omega_m fit.")
    print("Pantheon+ bias-correction simulations used their published fiducial cosmology.")

    local_fit = shoes_full_ladder_local_refit(Om=om, eps=eps, Or=SHOES_OR)
    print("\n[SH0ES THREE-RUNG LADDER: MODEL EQUATIONS AND REFIT]")
    print("For each Hubble-flow SN: d_L(z)=(1+z)(c/H0) integral_0^z dz'/E(z').")
    print("Its model-vs-fiducial magnitude correction is Delta_mu=5 log10[f_CoDE(z)/f_fid(z)].")
    print("Only the 277 Hubble-flow RHS entries are corrected; anchor, Cepheid, and calibrator rows stay as released.")
    print("With A=L^T and the released full covariance C, solve q=(A^T C^-1 A)^-1 A^T C^-1 y_model.")
    print("The released ladder parameterization gives H0=10^(q[46]/5); this refit has no CMB prior and no H0 target.")
    print(f"Fitted ladder component q[46], baseline:       {local_fit['q46_baseline']:.8f}")
    print(f"Fitted ladder component q[46], CoDE:           {local_fit['q46_model']:.8f}")
    print("\n[LOCAL HUBBLE CONSTANT - FULL SH0ES THREE-RUNG REFIT]")
    print(f"Baseline SH0ES matrix H0 recovery:            {local_fit['h0_baseline']:.4f} km/s/Mpc")
    print(f"CoDE local-ladder H0 fit:                      {local_fit['h0_model']:.4f} km/s/Mpc")
    print(f"Absolute change from baseline ladder fit:      {local_fit['delta_h0']:+.4f} km/s/Mpc")
    print("H0 is fitted from the released anchor-Cepheid-calibrator-SN-Hubble-flow ladder; no CMB prior enters.")
    print(f"Held-fixed local expansion-shape inputs:       Omega_m={local_fit['omega_m_fixed']:.5f}, eps={local_fit['epsilon_fixed']:.4f}")
    print("Omega_m source: unanchored Pantheon+ SN-only CoDE4 fit; no SH0ES H0 or CMB prior used in that fit.")
    print(f"Hubble-flow match zero-point offset:             {local_fit['match_offset_mag']:+.4f} mag")
    print(f"Hubble-flow match residual after offset removal:  RMS={local_fit['match_rms_mag']:.4f} mag; max={local_fit['match_max_abs_mag']:.4f} mag")
    print("The SH0ES matrix omits SN identifiers; Pantheon+ rows are associated by matching the fiducial RHS values.")
    print(f"Hubble-flow residual RMS, baseline -> CoDE:       {local_fit['hf_rms_baseline_mag']:.4f} -> {local_fit['hf_rms_model_mag']:.4f} mag")
    print(f"Forward SH0ES-style H0 from model distance scale: {predict_shoes_apparent_h0(model):.4f} km/s/Mpc (inherits the model's CMB-calibrated global H0)")
    print(f"Legacy weighted correction applied to 73.04 input: {compute_shoes_h0(om, eps):.4f} km/s/Mpc (SH0ES-anchored diagnostic)")

    cmb_rows = [
        ("Global H0 from theta* calibration", model.H0, PLANCK["H0"][0]),
        ("100 theta* proxy", 100*q["theta"], PLANCK["theta100"][0]),
        ("Shift parameter R", q["R"], DIST_PRIOR_MEAN[0]),
        ("Acoustic scale l_A", q["lA"], DIST_PRIOR_MEAN[1]),
        ("Recombination redshift z*", q["z_star"], PLANCK["zstar"][0]),
        ("Sound horizon r_s(z*) [Mpc]", q["r_star"], PLANCK["rstar"][0]),
        ("Comoving distance D_M(z*) [Mpc]", q["DM_star"], qref["DM_star"]),
        ("Drag sound horizon r_d [Mpc]", q["r_drag"], PLANCK["rdrag"][0]),
        ("Physical baryon density omega_b", model.omega_b, DIST_PRIOR_MEAN[2]),
        ("Diffusion scale k_D [1/Mpc]", q["kD"], qref["kD"]),
        ("Damping angle theta_D", q["thetaD"], qref["thetaD"]),
        ("First acoustic peak proxy ell_1", q["ell1"], qref["ell1"]),
    ]
    print("\n[COMPRESSED CMB SCALES - PLANCK 2018 REFERENCES]")
    print(f"{'Observable':38s} {'Model':>16s} {'Planck reference':>18s} {'Difference':>14s}")
    for label, value, observed in cmb_rows:
        print(f"{label:38s} {value:16.6g} {observed:18.6g} {value-observed:+14.6g}")
    print("These are compressed background scales, not Planck TT/TE/EE power-spectrum predictions.")
    print("The global H0 shown above is obtained by matching the model to the Planck acoustic-angle target.")

    predictions = desi_bao_prediction(model, q)
    labels = ["BGS z=0.295 D_V/r_d", "LRG1 z=0.510 D_M/r_d", "LRG1 z=0.510 D_H/r_d",
              "LRG2 z=0.706 D_M/r_d", "LRG2 z=0.706 D_H/r_d", "LRG3+ELG1 z=0.934 D_M/r_d",
              "LRG3+ELG1 z=0.934 D_H/r_d", "ELG2 z=1.321 D_M/r_d", "ELG2 z=1.321 D_H/r_d",
              "QSO z=1.484 D_M/r_d", "QSO z=1.484 D_H/r_d", "Ly-alpha z=2.330 D_H/r_d",
              "Ly-alpha z=2.330 D_M/r_d"]
    print("\n[BAO - DESI DR2 MEASUREMENTS]")
    print(f"{'Observable':38s} {'Model':>13s} {'DESI reference':>16s} {'Difference':>14s}")
    for label, value, observed in zip(labels, predictions, DESI_BAO_MEAN):
        print(f"{label:38s} {value:13.6f} {observed:16.6f} {value-observed:+14.6f}")
    print("DESI is the Dark Energy Spectroscopic Instrument; this is its 13-observable DR2 BAO vector.")

    print("\n[SUPERNOVA DISTANCE SHAPE - PANTHEON+]")
    sn = pantheon_plus_residuals(model)
    print(f"Pantheon+ release table rows:                {sn['n_total']}")
    print(f"Non-calibrator Hubble-flow SNe used:          {sn['n_used']}  (zHD={sn['z_min']:.4f}-{sn['z_max']:.4f})")
    print(f"Median offset-removed magnitude residual:      {sn['median_mag']:+.5f} mag")
    print(f"Mean offset-removed magnitude residual:        {sn['mean_mag']:+.5f} mag")
    print(f"RMS offset-removed magnitude residual:         {sn['rms_mag']:.5f} mag")
    print(f"Per-supernova residuals saved to:              {sn['csv_path']}")
    print("The released full covariance removes one common magnitude intercept; no fit score is formed.")
    print("This same Pantheon+ Hubble-flow sample supplied Omega_m, so this is an in-sample shape check, not a holdout validation.")

    print("\n[BBN AND MATTER-RADIATION EQUALITY]")
    print(f"Physical baryon density omega_b:               {model.omega_b:.7f}")
    print(f"BBN D/H reference omega_b:                     {OMEGA_B_BBN[0]:.7f}")
    print(f"BBN H^2(model)/H^2(standard) at 1 MeV:         {model.E2(1.0e6/(K_B_EV*T_CMB)-1) / replace(model, eps=0.0, beta=0.0).E2(1.0e6/(K_B_EV*T_CMB)-1):.12e}")
    bbn_ratio = model.E2(1.0e6/(K_B_EV*T_CMB)-1) / replace(model, eps=0.0, beta=0.0).E2(1.0e6/(K_B_EV*T_CMB)-1)
    print(f"BBN implied Delta N_eff proxy:                  {(43.0/7.0)*(bbn_ratio-1):+.8e}")
    print(f"Matter-radiation equality z_eq:                 {z_equality(model):.5f}")
    print(f"Matter-radiation equality k_eq:                 {k_equality(model):.8e} 1/Mpc")

    print("\n[MATTER POWER, GROWTH, RSD, AND WEAK LENSING]")
    mp_model = MatterPower(model)
    mp_ref = MatterPower(reference)
    sigma8_raw = mp_model.sigma8()
    sigma8_ref_raw = mp_ref.sigma8()
    amplitude_calibration = PLANCK["sigma8"][0] / sigma8_ref_raw
    sigma8 = sigma8_raw * amplitude_calibration
    kpk, ppk = mp_model.k_peak()
    print(f"Matter-power turnover k_peak:                  {kpk:.8e} 1/Mpc ({kpk/model.h:.8e} h/Mpc)")
    print(f"Matter-power raw peak P(k_peak):               {ppk:.8e} Mpc^3")
    print(f"sigma8 raw pipeline output:                    {sigma8_raw:.8f}")
    print(f"sigma8 calibrated to Planck amplitude:          {sigma8:.8f}")
    print("Amplitude calibration uses Planck sigma8 for normalization; shape/growth remain model-derived.")
    print(f"Raw growth amplitude D(a=1), M-normalized:      {D_of_a(model, 1.0):.8e}")
    for zt in (0.0, 0.5, 1.0, 2.0, 10.0):
        growth_ratio = D_of_a(model, 1/(1+zt)) / D_of_a(model, 1.0)
        print(f"Growth amplitude D(z={zt:g})/D(0):                 {growth_ratio:.8f}")
    zfit = np.linspace(0.0, 1.0, 60)
    afit = 1.0 / (1.0 + zfit)
    lnf = np.log([f_of_a(model, a) for a in afit])
    lnOm = np.log([Omega_m_a(model, a) for a in afit])
    gamma = float(np.sum(lnf*lnOm) / np.sum(lnOm**2))
    print(f"Growth-index gamma fit over z=[0,1]:           {gamma:.6f}")
    print(f"Growth-index distance from 0.55, in 0.05 units: {abs(gamma-0.55)/0.05:.6f}")
    s8 = sigma8 * np.sqrt(model.Om / 0.3)
    print(f"S8 = sigma8*sqrt(Omega_m/0.3):                 {s8:.6f}")
    for zt, observed, _sigma, dataset in RSD:
        a = 1.0 / (1.0 + zt)
        fs8 = f_of_a(model, a) * sigma8 * D_of_a(model, a) / D_of_a(model, 1.0)
        print(f"f*sigma8 z={zt:.3f} ({dataset}):                    {fs8:.6f} (reference {observed:.6f}; difference {fs8-observed:+.6f})")
    for zt in (0.0, 0.5, 1.0, 2.0):
        a = 1.0 / (1.0 + zt)
        dl = 1e-4
        phi = lambda aa: float(model.G_ratio(1/aa-1)) * D_of_a(model, aa) / aa
        dphi = (phi(a*np.exp(dl))-phi(a*np.exp(-dl))) / (2*dl)
        print(f"ISW potential-decay proxy dPhi/dln(a), z={zt:.1f}: {dphi:+.8e}")
    for name, observed, _unc in S8_DATA:
        print(f"S8 comparison {name}: model {s8:.5f}, reference {observed:.5f}, difference {s8-observed:+.5f}")

    print("\n[AGE OF THE UNIVERSE]")
    print(f"Age today:                                       {age_Gyr(model):.6f} Gyr")
    print("\n" + "=" * 78)
    return 0


def cli(argv):
    allowed = {"--full", "--metrics-only"}
    unknown = set(argv) - allowed
    if unknown:
        raise SystemExit("Unsupported option(s): " + ", ".join(sorted(unknown)))
    return main(argv)


if __name__ == "__main__":
    # Jupyter injects its own ``-f <kernel.json>`` arguments into sys.argv.
    # They are runtime plumbing, not user-supplied engine options.
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(cli(sys.argv[1:]))


