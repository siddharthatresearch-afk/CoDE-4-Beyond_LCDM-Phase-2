<img width="5370" height="2955" alt="image" src="https://github.com/user-attachments/assets/c38f056f-9b69-4926-b064-aded57b9a4ac" />
# CoDE-4 Cosmology Engine

## Computational Dark-Energy Framework for Cross-Epoch Cosmological Consistency

---

# Overview

**CoDE-4** (Cosmological Dynamical Engine v4) is an exploratory computational cosmology framework designed to investigate whether mild late-time modifications to cosmic expansion can remain approximately consistent with multiple observational probes while reducing the inferred Hubble tension.

The project focuses on:
- late-time expansion dynamics,
- cosmic growth evolution,
- geometric consistency tests,
- large-scale structure behavior,
- and cross-epoch observational comparisons.

Rather than proposing a fully fundamental theory of gravity, CoDE-4 currently operates as a phenomenological proof-of-concept framework intended for computational experimentation and cosmological analysis.

The framework numerically evaluates:
- expansion history,
- structure growth,
- acoustic scales,
- weak-lensing parameters,
- supernova distances,
- BAO scales,
- matter-radiation equality,
- and Integrated Sachs-Wolfe behavior.

---

# Scientific Motivation

Modern cosmology faces several observational tensions between early-universe and late-universe measurements.

Most notably:

## Hubble Tension

Measurements from:
- the Planck CMB inference pipeline suggest

$$
H_0 \approx 67.4 \,\mathrm{km\,s^{-1}\,Mpc^{-1}}
$$

while local distance-ladder observations (SH0ES) suggest

$$
H_0 \approx 73 \,\mathrm{km\,s^{-1}\,Mpc^{-1}}
$$

creating a discrepancy approaching:

$$
\sim 5\sigma
$$

CoDE-4 explores whether a mild redshift-dependent modification to dark-energy scaling can alter late-time expansion while preserving approximate consistency with:
- CMB geometry,
- acoustic scales,
- structure growth,
- and distance observables.

---

# Core Model

The framework modifies the effective dark-energy contribution using the phenomenological scaling function:

$$
C(z)=1+\epsilon\left(\frac{z}{(1+z)^2}\right)\left(1-\frac{0.15}{1+z}\right)
$$

where:
- ε controls modification strength,
- the correction decays naturally at both early and late times,
- and the model remains close to standard ΛCDM at high redshift.

The modified expansion history becomes:

$$
H(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_r(1+z)^4+\Omega_{DE}C(z)}
$$

This allows the framework to explore small departures from ΛCDM while preserving approximate early-universe consistency.

---

# Features

## Expansion History Solver
- Modified Hubble evolution
- Radiation, matter, and dark-energy sectors
- Self-consistent H₀ iteration loop

## Structure Growth Solver
- Numerical second-order growth evolution
- Dynamic fσ₈(z)
- Growth-index analysis

## CMB Consistency Checks
- Acoustic peak position
- Angular scale consistency
- Damping-tail deviation analysis
- Shift parameter evaluation

## Large-Scale Structure
- Matter power-spectrum turnover
- Equality redshift analysis
- Sigma-8 evolution

## Distance Probes
- Type Ia supernova distance modulus
- BAO distance scales
- Comoving distance calculations

## Late-Time Dynamics
- ISW potential evolution
- Weak-lensing consistency checks
- Hubble-tension comparison analysis

---

# Numerical Components

The repository contains:

| File | Purpose |
|---|---|
| `CoDE4_engine.py` | Main cosmology engine and numerical solvers |
| `plots.py` | Publication-style visualization suite |
| `result.py` | Validation and observational consistency tests |
| `VALIDATION_SUMMARY` | Numerical output summary |
| `README.md` | Project documentation |

---

# Observational Tests Included

The framework currently evaluates:

- Big Bang Nucleosynthesis expansion consistency
- CMB damping-tail stability
- First acoustic peak position
- fσ₈(z) structure growth
- ISW potential decay
- CMB shift parameter
- Growth-index gamma evolution
- Matter-radiation equality
- Matter power-spectrum turnover
- Weak-lensing S₈
- Supernova distance modulus
- BAO distance scales
- Cosmic age estimation
- Raw high-redshift growth evolution
- Hubble-tension comparison metrics

---

# Example Outputs

Current representative outputs include:

| Observable | Result |
|---|---|
| Derived H₀ | 72.86 km/s/Mpc |
| Sound Horizon rₛ | 133.3 Mpc |
| Acoustic Peak l₁ | 220.72 |
| Equality Redshift | z_eq ≈ 3499 |
| S₈ | 0.8396 |
| Age of Universe | 12.75 Gyr |

---

# Visualizations

The framework generates:
- Hubble-tension comparison plots
- Dark-energy modifier evolution
- Structure-growth evolution
- ISW decay behavior
- CMB acoustic peak structure
- Matter power-spectrum turnover
- BAO consistency plots
- Pantheon supernova comparisons
- Multi-panel summary figures

All plots are exported in:
- PNG
- PDF
- SVG

publication-ready formats.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd CoDE-4-Beyond_LCDM-Phase-2
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# Running the Engine

Run the primary cosmology engine:

```bash
python CoDE4_engine.py
```

Generate plots:

```bash
python plots.py
```

Run validation suite:

```bash
python result.py
```

---

# Scientific Interpretation

CoDE-4 should currently be interpreted as:
- a phenomenological exploratory framework,
- a computational cosmology prototype,
- and a proof-of-concept late-time modification study.

The purpose of the project is to investigate whether mild expansion-history modifications can simultaneously preserve approximate observational consistency across multiple cosmological sectors.

Development is actively ongoing, and the framework is expected to receive substantial updates during this period.



The current release should therefore be considered an evolving exploratory research framework rather than a finalized cosmological model.

# Major Revisions
A major update has taken place in our engine.py and jwst engine further updates on the rest of repository will role in stages

### 📥 Pre-downloading Datasets

Run the following command in your terminal to create and execute the download script:

```bash
import os

try:
    print('Downloading Pantheon+ datasets...')
    table_path, cov_path = _pantheon_plus_paths()
    print(f'Pantheon+ files ready: {table_path.name}, {cov_path.name}')
    
    print('\nDownloading SH0ES ladder fits files...')
    _ = shoes_full_ladder_local_refit(Om=0.3, eps=0.05)
    print('SH0ES ladder files successfully downloaded and verified!')
except Exception as e:
    print(f'Encountered error during download: {e}')


python3 download_data.py
```



