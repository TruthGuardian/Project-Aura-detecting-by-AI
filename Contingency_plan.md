# Contingency (Backup) Plan

---

## 6.1 Identified Risks and Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
|:----|:------|:--------------------|
| **Tracking Detector Failure (DWC1–4)** | Loss of fine particle trajectory measurements | Focus on Time-of-Flight (ToF) and Cherenkov PID features. Use S1–S2 timing, halo asymmetries, and CALO signals to proceed with composition classification (Task 1) and halo-based tasks (Task 4). |
| **Halo Detector Malfunction (Partial or Complete)** | Loss of direct halo quadrant signals | Shift emphasis to DWC scattering angles and Cherenkov PID for classification. If needed, reframe Tasks 1 & 2 around core beam parameters only. |
| **S1 or S2 Scintillator Failure** | Loss of trigger and/or ToF measurement | Use only tracking and PID detectors for event reconstruction. Rely on Halo Quadrants + DWC angular spread for analysis. |
| **Cherenkov Detector Failure (C1 or C2)** | Incomplete PID (Particle ID) | Reinforce classification models using ToF and calorimeter features. Compensate particle separation loss through halo asymmetry patterns if statistically viable. |
| **Beam Composition Unexpected** | Different mixture than assumed (e.g., wrong proton/electron ratios) | Retrain machine learning models on early collected data. Adjust experimental goals dynamically to focus on available particle types. |
| **Beam Energy Instability or Low Statistics** | Reduced event quality and statistical power | Prioritize data collection for baseline runs (no foil, standard beam) and carbon foil runs first. Limit variations to maximize usable datasets for Tasks 1 and 2. |
| **DAQ Issues or Synchronization Problems** | Incomplete datasets or timing errors | Use redundant signals (S1 and Halo hits) to validate events. Perform offline corrections if timing skew is systematic. |
| **Beam Alignment Offsets** | Beam shifted from center, affecting halo symmetry | Correct for beam offsets offline during data reconstruction. If impossible, use relative quadrant differences instead of absolute patterns. |
| **Magnet (if requested) Not Available** | No direct momentum measurement | Focus on relative energy spread via halo width and calorimeter response instead of exact momentum estimation. Omit magnet-based tasks if necessary. |
| **Limited Beamtime or Interruptions** | Fewer datasets than planned | Strict run prioritization:  
  1. Baseline runs (no foil)  
  2. Carbon foil runs  
  3. Aluminum foil runs  
  4. Copper foil or collimator runs  
Only attempt different energies if main conditions are satisfied. |

---

## 6.2 Strategy

- **Prioritize critical physics goals** first (Tasks 1 and 4: composition classification and halo pattern recognition).
- **Remain flexible**: focus dynamically on the most robust detectors available at any time.
- **Minimize dependence** on heavily correlated multi-detector events if failures occur.

---


