# Scientific Background & Methodology

---

## 2.1 Scientific Background

### 2.1.1 Beam Halo Formation

Beam halos primarily originate from **Multiple Coulomb Scattering (MCS)** where charged particles interact electromagnetically with the nuclei and electrons of materials like thin foils or collimators. MCS causes particles to deviate from their original path, forming a diffuse halo. The root-mean-square (RMS) scattering angle, \( \theta_0 \), is given by the Highland formula:

When a beam hits thin foils or internal collimators, some particles are deflected at small angles, forming a scattered formation leading to halo creation.  
The root-mean-square (RMS) scattering angle, \( \theta_0 \), is approximately given by:

\[
\theta_0 \approx \frac{13.6 \text{ MeV}}{\beta pc} \cdot z \cdot \sqrt{\frac{x}{X_0}} \left( 1 + 0.038 \ln{\left(\frac{x}{X_0}\right)} \right) \quad (\text{Eq. 1})
\]

Where:
- \( \beta = \frac{v}{c} \) is the particle’s velocity normalized to the speed of light
- \( p \) is the momentum (MeV/c)
- \( z \) is the particle’s charge
- \( x \) is the thickness of the material
- \( X_0 \) is the radiation length of the material

This equation shows that the scattering angle depends on the particle type (via β, p, z) and material properties (x, X₀), leading to distinct halo patterns for different beams and foils.


---

### 2.1.2 Role of Halo Detectors in Beam Diagnostics


Unlike typical beamline setups that filter out halo particles, our experiment intentionally studies them to understand the beam’s properties.

In this project, four scintillation detectors forming a hollow square pattern downstream of a target region — creating a **Halo Detector (HD) array**, placed downstream of a scattering region (an Al/C/Cu foil to ensure halo creation). 

Each panel measures the intensity of scattered particles using an Analog-to-Digital Converter (ADC), which converts light signals from particle hits into digital values.

By comparing ADC values across the four panels, we can calculate the halo’s shape and asymmetry, revealing information about the beam’s composition and energy. 
 
This approach is like interpreting a shadow: although the halo detector provides coarse data compared to high-resolution trackers, its patterns are rich enough to diagnose beam conditions — while the resolution may relatively be low, shadows can still reveal properties of the object casting them.

---

## 2.2 Methodology

### 2.2.1 Research Question

> **Can meaningful insights be extracted from beam halo structures using machine learning?**

Specifically, can we:
- Predict beam composition and energy using only halo features?
- Detect or anticipate beam instabilities from halo time evolution?
- Classify halo shape patterns and uncover hidden structures?

---

### 2.2.2 Experimental Tasks Breakdown

| Task | Goal | Method |
|:----:|:----|:------|
| **Task 1** | Beam Composition Classification | Use halo quadrant asymmetries + PID detectors (Cherenkov, ToF, Calorimeter) to classify particle types. |
| **Task 2** | Beam Energy Regression | Correlate energy with angular spread and calorimeter response. |
| **Task 3** | Beam Stability Analysis | Track time-series variance in halo signals across spills to detect instability. |
| **Task 4** | Halo Shape Classification | Cluster events based on quadrant signal ratios to categorize shape profiles (symmetric, skewed, diffuse). |

---

### 2.2.3 Mathematical Tools Employed

- **Momentum-Energy Relation** (for mass estimation via ToF and Cherenkov thresholds):

\[
E^2 = (pc)^2 + (mc^2)^2 \quad (\text{Eq. 2})
\]

Where \( E \) is the total energy, \( p \) is momentum, \( m \) is rest mass, and \( c \) is the speed of light.

- **Cherenkov Threshold Condition**:

\[
\beta > \frac{1}{n} \quad (\text{Eq. 3})
\]

Where \( \beta = \frac{v}{c} \) is the particle’s velocity relative to light, and \( n \) is the refractive index of the Cherenkov detector’s gas. Particles faster than this threshold emit light, identifying their type.

- **Halo Asymmetry Metrics** — used to capture geometric deviation:

\[
A_x = \frac{(Q_1 + Q_4) - (Q_2 + Q_3)}{Q_1 + Q_2 + Q_3 + Q_4} \quad (\text{Eq. 4})
\]

\[
A_y = \frac{(Q_1 + Q_2) - (Q_3 + Q_4)}{Q_1 + Q_2 + Q_3 + Q_4} \quad (\text{Eq. 5})
\]

Where \( Q_1, Q_2, Q_3, Q_4 \) are the ADC values from the four halo detector panels, representing particle hit intensities. \( A_x \) and \( A_y \) measure left-right and top-bottom asymmetries.

- **Time-of-Flight (ToF) Mass Estimation** (from S1 and S2 timing):

\[
m = \frac{p}{c} \sqrt{\left(\frac{c \Delta t}{L}\right)^2 - 1} \quad (\text{Eq. 6})
\]

Where \( \Delta t \) is time-of-flight, and \( L \) is the distance between S1 and S2 (in cm).

---

### 2.2.4 Detector Integration Strategy

| Detector | Role | Method |
|:--------|:-----|:------|
| **S1 & S2 (Scintillators)** | Start/stop for ToF | Fast timing (~200 ps) |
| **DWC1–4 (Delay Wire Chambers)** | Pre/post tracking | Linear track fitting, angular deviation measurement |
| **Cherenkov Detectors (C1, C2)** | PID via velocity thresholding | Tunable pressure settings |
| **CALO (Lead Glass Calorimeter)** | Energy measurement | Used in energy regression and PID cross-checks |
| **Halo Detector (Custom)** | Coarse angular halo information | 4-quadrant ADC vector input to AI model |

---

