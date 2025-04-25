# Data Generator

---

## 3.1 Data generator

In order to ensure feasibility of the proposed experiment, a synthetic data generator was created to simulate the expected detector readings from a physically logical standpoint.

This approach allows us to:

- Create labelled datasets that mimic expected detector readings.
- Introduce controlled noise and beam variations to test model robustness.
- Ensure that our ML models can generalize to real-world beamline data.

---

## 3.2 Working Principle

The generator simulates events as follows:

- **Beam Core Generation**:  
Particles forming the main Gaussian-shaped core of the beam are generated according to a 2D Gaussian distribution in the transverse (x, y) plane.

- **Halo Particle Injection**:  
Particles belonging to the beam halo are generated with a different radial distribution — typically broader and heavier-tailed — to mimic the halo effect caused by scattering or beam instabilities.

- **Noise Addition**:  
Random electronic or detector noise is added to simulate realistic imperfect readings.

- **Label Assignment**:  
Each generated point is labeled either as:
  - Core (Label 0), or
  - Halo (Label 1).

- **Parameter Variations**:  
The generator can dynamically adjust parameters such as:
  - Beam width (standard deviation of the Gaussian core)
  - Halo intensity (fraction of particles belonging to the halo)
  - Noise level (electronic noise simulation)

---

## 3.3 Mathematical Formulation

### 3.3.1 Core Particles

The (x, y) positions of core beam particles follow a normal distribution centered at the beam axis:

\[
x_{\mathrm{core}}, y_{\mathrm{core}} \sim \mathcal{N}(0, \sigma_{\mathrm{core}}^2)
\quad \text{(Eq. 9)}
\]

Where:
- \( \sigma_{\mathrm{core}} \) is the standard deviation of the beam core, controlling its width.

---

### 3.3.2 Halo Particles

Halo particles, formed by scattering, are modeled with a radial Cauchy distribution for distance and a uniform angular distribution:

\[
r_{\mathrm{halo}} \sim \text{Cauchy}(\gamma); \quad \theta \sim \text{Uniform}(0,2\pi)
\quad \text{(Eq. 10)}
\]

The x and y coordinates of halo particles are calculated from their radial distance and angle:

\[
x_{\mathrm{halo}} = r_{\mathrm{halo}}\cos(\theta), \quad y_{\mathrm{halo}} = r_{\mathrm{halo}}\sin(\theta)
\quad \text{(Eq. 11)}
\]

Where:
- \( \gamma \) controls the spread of the halo, determining how far particles scatter.
- \( \theta \) is the angle in the x-y plane, uniformly distributed to ensure circular symmetry.

---

### 3.3.3 Noise Addition

To account for detector imperfections, small Gaussian noise is added to each coordinate:

\[
x_{\mathrm{final}} = x + \epsilon_x, \quad y_{\mathrm{final}} = y + \epsilon_y
\quad \text{(Eq. 12)}
\]

Where:
- \( \epsilon_x, \epsilon_y \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2) \) represent independent noise terms.
- \( \sigma_{\mathrm{noise}} \) is the standard deviation of the noise, reflecting detector resolution limits.

---

## 3.4 Sample Generated Data

| Event ID | S1 Time (ns) | S2 Time (ns) | DWC1 X (mm) | DWC1 Y (mm) | DWC2 X (mm) | DWC2 Y (mm) | DWC3 X (mm) | DWC3 Y (mm) | DWC4 X (mm) | DWC4 Y (mm) | Halo Q1 | Halo Q2 | Halo Q3 | Halo Q4 | Cherenkov 1 | Cherenkov 2 | CALO Energy (MeV) | Particle Type | True Energy (GeV) | Beam Stability | Halo Pattern |
|:--------:|:------------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|:-------:|:-------:|:-------:|:------------:|:------------:|:----------------:|:-------------:|:----------------:|:--------------:|:------------:|
| 1 | 0.003 | 0.029 | 2.1 | -1.8 | 2.5 | -1.5 | 2.6 | -1.7 | 2.7 | -1.6 | 120 | 90 | 110 | 85 | 1 | 0 | 5.1 | Electron | 5.0 | Stable | Symmetric |
| 2 | 0.002 | 0.030 | -0.5 | 1.3 | -0.7 | 1.8 | -0.8 | 1.7 | -0.9 | 1.6 | 45 | 120 | 70 | 140 | 0 | 1 | 2.8 | Pion | 8.0 | Unstable | Skewed |

(Only 2 sample rows shown; full dataset contains thousands of events.)

---

## 3.5 Adjustable Parameters

| Parameter | Description | Typical Values |
|:----------|:------------|:---------------|
| Core Standard Deviation \( \sigma_{\mathrm{core}} \) | Width of core beam spread | 1–3 mm |
| Halo Intensity | Fraction of halo particles | 5%–15% |
| Halo Spread \( \gamma \) | Broadness of halo radial distribution | 3–7 mm |
| Noise Standard Deviation \( \sigma_{\mathrm{noise}} \) | Detector/electronic noise magnitude | 0.1–0.5 mm |
| Number of Particles | Total number of events per sample | 5,000–50,000 particles |

---
