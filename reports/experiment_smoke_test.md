# BayesFlow Smoke Test

Date: 2026-07-07

## Goal

Verify that the full offline BayesFlow pipeline runs end-to-end:

- load generated gravitational-wave dataset
- split train/validation data
- train a BayesFlow workflow
- save the trained approximator checkpoint
- generate loss and posterior diagnostic figures

## Configuration

- Dataset: `data/gw_dataset_smoke_20x512.npz`
- Dataset shape: `X = (20, 512)`, `theta = (20, 6)`
- Parameter order: `m1, m2, chi1, chi2, distance, inclination`
- Epochs: `1`
- Batch size: `4`
- Model output: `models/bayesflow_model/model.keras`

## Outputs

- Training loss figure: `figures/training_loss.png`
- Posterior example figure: `figures/posterior_example.png`
- Model checkpoint: `models/bayesflow_model/model.keras`

## Result

The smoke test completed successfully. The model checkpoint was written and the
posterior diagnostic figure contains samples for all six parameters with the true
validation value marked in each panel.

This run is a pipeline validation, not a final scientific training run. A larger
experiment should increase the number of samples, number of epochs, or reduce the
waveform length/model size to keep training practical on CPU.
