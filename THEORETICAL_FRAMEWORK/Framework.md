# CoDE-4 Theoretical Framework

## Overview

CoDE-4 (Cosmological Dynamical Evolution Framework – Version 4) is an exploratory phenomenological cosmology framework designed to investigate bounded late-time perturbative modifications to standard cosmology while preserving near-standard early-universe behavior.

The framework was constructed around the idea that cosmological expansion may admit weak late-time dynamical deformation without destroying the highly constrained geometric consistency of:
- recombination physics,
- acoustic peak structure,
- matter-radiation equality,
- and large-scale CMB observables.

Rather than functioning as a fully fundamental theory of gravity, CoDE-4 currently operates as a phenomenological expansion framework intended to study:
- late-time expansion deformation,
- weakly dynamical dark-energy behavior,
- perturbative structure-growth evolution,
- and compressed-observable cosmological consistency.

The framework attempts to preserve the successful early-universe predictions of standard cosmology while allowing controlled late-time departures from a pure cosmological constant.

---

# Core Design Philosophy

## 1. Near-Standard Early-Universe Recovery

Any viable late-time cosmological modification must preserve:
- recombination geometry,
- acoustic peak consistency,
- matter-radiation equality,
- and CMB distance constraints.

For this reason, all CoDE-4 deformation sectors are constructed to asymptotically recover standard cosmological behavior at high redshift.

---

## 2. Bounded Perturbative Evolution

The framework was intentionally designed to avoid:
- divergent vacuum behavior,
- singular late-time evolution,
- runaway expansion sectors,
- and unstable asymptotic growth.

All deformation terms remain bounded across cosmic evolution.

---

## 3. Smooth Late-Time Expansion Deformation

The framework allows mild low-redshift modification of the effective vacuum sector in order to explore potential late-time cosmological deviations while preserving early-universe consistency.

This includes exploration of:
- Hubble tension behavior,
- weak dynamical dark-energy evolution,
- and perturbative structure-growth effects.

---

## 4. Preserved Acoustic Geometry

A major construction goal of CoDE-4 is preservation of near-standard acoustic structure despite introducing effective vacuum deformation.

This requirement strongly constrains:
- the modifier structure,
- asymptotic behavior,
- and allowed perturbative evolution.

---

# Central Modifier Structure

The primary CoDE-4 deformation function is written as:

$$
C(z) = 1 + \epsilon \left( \frac{z}{(1+z)^2} \right) \left( 1 - \frac{0.15}{1+z} \right)
$$

This modifier was constructed phenomenologically to satisfy several simultaneous conditions:
- weak low-redshift enhancement,
- suppression at high redshift,
- bounded asymptotic behavior,
- smooth perturbative evolution,
- and preservation of early-universe consistency.

The structure naturally produces:
- localized late-time deformation,
- near-standard high-redshift recovery,
- and smooth continuous expansion behavior.

---

# Modified Friedmann Expansion

The standard Friedmann equation is modified through an effective vacuum deformation sector:

$$
H^2(z) = H_0^2 \left[ \Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda C(z) \right]
$$

where:
- $\Omega_m$ represents the matter density parameter,
- $\Omega_r$ represents the radiation density parameter,
- $\Omega_\Lambda$ represents the effective vacuum density parameter,
- and $C(z)$ represents the perturbative deformation sector.

The matter and radiation sectors remain standard while the deformation acts entirely through the effective vacuum contribution.

This construction helps preserve:
- early structure formation,
- recombination behavior,
- and acoustic consistency.

---

# Effective Dark-Energy Interpretation

The deformation may be interpreted phenomenologically as an evolving effective dark-energy sector:

$$
\rho_X(z) = \rho_\Lambda C(z)
$$

Using the cosmological continuity equation in terms of redshift:

$$
\frac{d\rho_X}{dz} = \frac{3(1+w_{\mathrm{eff}})\rho_X}{1+z}
$$

The effective equation-of-state evolution is rigorously reconstructed as:

$$
w_{\mathrm{eff}}(z) = -1 + \frac{1+z}{3C(z)} \frac{dC}{dz}
$$

The resulting evolution remains close to a cosmological-constant-like state while permitting weak late-time dynamical behavior.

Current reconstructed estimates approximately yield:
- $w_0 \approx -0.986$
- $w_a \approx -0.052$

placing CoDE-4 within weakly dynamical dark-energy territory while remaining close to standard cosmological evolution.

---

# Asymptotic Structure

An essential property of CoDE-4 is asymptotic recovery of standard cosmological behavior.

As:

$$
z \to \infty
$$

the framework satisfies:

$$
C(z) \to 1
$$

leading to:

$$
H(z) \to H_{\mathrm{standard}}(z)
$$

This asymptotic behavior is critical for preserving:
- recombination physics,
- acoustic geometry,
- matter-radiation equality,
- and early structure formation consistency.

At low redshift, the deformation remains smooth and bounded without generating divergent cosmological behavior.

---

# Perturbation Framework

The framework also admits a perturbative structure-growth interpretation.

Linear matter perturbations $\delta \equiv \delta\rho_m/\rho_m$ evolve with respect to scale factor $a$ (where $' \equiv d/da$) according to:

$$
\delta'' + \left( \frac{3}{a} + \frac{H'}{H} \right) \delta' - \frac{3\Omega_m H_0^2}{2a^5 H^2} \delta = 0
$$

The modified expansion evolution influences:
- clustering amplitudes,
- weak-lensing structure growth,
- growth-index evolution,
- and late-time perturbation behavior.

This provides the primary connection between CoDE-4 and observables such as:
- $S_8$,
- $f\sigma_8$,
- and weak-lensing diagnostics.

---

# Compressed Observable Chi-Square Framework

A preliminary compressed-observable chi-square analysis has been implemented to evaluate phenomenological consistency against several cosmological observables.

Included observables currently include:
- $H_0$
- CMB shift parameter $R$
- acoustic peak scale $l_1$
- weak-lensing parameter $S_8$
- matter-radiation equality epoch $z_{eq}$

The current framework demonstrates:
- strong geometric consistency,
- preserved acoustic structure,
- successful late-time expansion behavior,
- and moderate residual weak-lensing tension.

The dominant residual contribution currently arises from late-time clustering observables.

---

# Stability Interpretation

The perturbative structure of CoDE-4 remains bounded throughout cosmic evolution and avoids:
- finite-redshift singularities,
- unstable asymptotic growth,
- divergent vacuum behavior,
- and runaway expansion sectors.

The framework therefore behaves as a smooth perturbative deformation around standard cosmological evolution.

---

# Phenomenological Interpretation

Overall, CoDE-4 currently behaves as:
- a weakly dynamical late-time cosmological framework,
- with perturbative vacuum deformation,
- near-standard early-universe recovery,
- preserved acoustic geometry,
- bounded asymptotic behavior,
- and preliminary compressed-observable statistical consistency.

The framework remains exploratory and phenomenological in nature and is not yet presented as a complete fundamental cosmological theory.

---

# Current Limitations

The present implementation does not yet include:
- covariance-matrix likelihood analyses,
- Bayesian evidence comparison,
- full MCMC parameter estimation,
- perturbation evolution through Boltzmann solvers,
- or full large-scale structure likelihood integration.

These remain important future directions for further validation.

---

# Future Directions

Planned future extensions include:
- covariance-aware chi-square analyses,
- Cobaya MCMC integration,
- CLASS/CAMB compatibility,
- expanded BAO and supernova datasets,
- improved perturbation evolution,
- and Bayesian comparison against standard cosmological models.
