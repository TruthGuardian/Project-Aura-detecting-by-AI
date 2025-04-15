# Project Aura – Beam Halo Composition via Machine Learning

This project explores how beam halo signals—typically treated as noise—can reveal meaningful information about beam composition. Using simulated beamline data and machine learning, we predict the fractions of particle types (protons, pions, muons, etc.) in a given beam based solely on halo detector responses. The project was developed for the CERN Beamline for Schools (BL4S) competition by the Halo Hunters team.

---

## Code Overview

### `DataGenerator.py`
Beamline simulation code. Defines the `BL4SHaloCompositionGenerator` class which:
- Generates particle beams with configurable composition
- Simulates scattering (MCS), magnetic deflection, and detector responses
- Models a 4-quadrant halo detector (hit flags and ADCs)
- Outputs labeled datasets for ML training (including beam composition)

### `model.py`
Machine learning model code. Defines the `BeamHaloCompositionModel` class which:
- Builds a multi-input neural network using Keras
- Processes features from halo hits, ADCs, and derived asymmetries
- Uses an attention mechanism to emphasize important halo features
- Trains to predict particle composition from halo data
- Includes tools for evaluation and visualization (correlation plots, heatmaps, etc.)

---

## Workflow

1. Use `BL4SHaloCompositionGenerator` to generate a synthetic dataset
2. Initialize and train a `BeamHaloCompositionModel` on the dataset
3. Evaluate predictions and visualize halo-feature correlations

Example:
```python
gen = BL4SHaloCompositionGenerator()
df = gen.generate_dataset(n_runs=5, n_events_per_run=200)

model = create_beam_composition_model(data_generator=gen)
results = train_and_evaluate_composition_model(model, gen, epochs=30)
