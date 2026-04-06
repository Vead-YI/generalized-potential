# generalized-potential

A minimal toy project for understanding the core program architecture behind `mir-group/allegro-pol` without depending on DFT or realistic material physics.

The central idea is to learn a single generalized potential

```text
U(R, E)
```

from which we obtain

```text
energy   U
forces   F = -dU/dR
response P = -dU/dE
```

using automatic differentiation.

This repository is deliberately small. The goal is not to reproduce the paper's physical targets, but to internalize the software pattern:

- one model predicts a scalar energy-like quantity
- geometry and external field both enter that model
- force and field response are derived from the same scalar potential
- training uses joint supervision on energy, forces, and response

## Why this project exists

`allegro-pol` extends an interatomic potential so that electric-response quantities can be obtained from the same differentiable energy model. For learning that architecture, the full NequIP/Allegro stack is heavier than necessary.

This toy project keeps the parts that matter most:

- a generalized potential `U(R, E)`
- a dataset with `positions`, `species`, `electric_field`, `total_energy`, `forces`, and `response`
- a model interface `forward(batch) -> total_energy`
- autograd-based derivation of forces and response
- training and evaluation scripts that operate on those quantities

It intentionally simplifies away:

- real polarization physics
- Berry-phase subtleties
- Born charges and polarizability as production targets
- equivariant message passing
- simulator integration and model compilation

## Toy physics model

The current system is a 2D three-particle toy molecule with species `[A, B, A]` and effective charges `[+1, -2, +1]`.

The analytic potential is

```text
U(R, E) = U_geom(R) - E · M(R) - 1/2 α(R) ||E||^2
```

with

```text
U_geom(R) = Σ_{i<j} [ 1/2 k (d_ij - l0)^2 + A exp(-d_ij^2 / (2σ^2)) ]
M(R)      = Σ_i q_i r_i
α(R)      = α0 + α1 Σ_{i<j} exp(-d_ij^2 / (2ρ^2))
```

where:

- `R` is the full set of particle coordinates
- `E` is the external electric field
- `d_ij = ||r_i - r_j||`
- `M(R)` is a dipole-like effective response quantity
- `α(R)` is a simple geometry-dependent scalar response coefficient

From this definition:

- `forces = -dU/dR`
- `response = -dU/dE = M(R) + α(R) E`

The analytic labels are generated directly in [physics.py](/Users/vead/.qclaw/workspace/generalized%20potential/src/toy_allegro_pol/physics.py), and the same structure is what the learned model tries to emulate.

## Project layout

```text
generalized-potential/
  README.md
  configs/default.yaml
  data/processed/
  scripts/
    data_gen.py
    train.py
    eval.py
    infer.py
  src/toy_allegro_pol/
    physics.py
    dataset.py
    model.py
    train_utils.py
    eval_utils.py
  outputs/
```

Main responsibilities:

- [configs/default.yaml](/Users/vead/.qclaw/workspace/generalized%20potential/configs/default.yaml): all tunable parameters for physics, sampling, model, and training
- [scripts/data_gen.py](/Users/vead/.qclaw/workspace/generalized%20potential/scripts/data_gen.py): generates train/val/test `.npz` datasets from the analytic toy potential
- [scripts/train.py](/Users/vead/.qclaw/workspace/generalized%20potential/scripts/train.py): trains the neural model and saves checkpoints
- [scripts/eval.py](/Users/vead/.qclaw/workspace/generalized%20potential/scripts/eval.py): computes regression metrics, parity plots, finite-difference checks, and field-sweep plots
- [scripts/infer.py](/Users/vead/.qclaw/workspace/generalized%20potential/scripts/infer.py): runs single-sample inference
- [src/toy_allegro_pol/physics.py](/Users/vead/.qclaw/workspace/generalized%20potential/src/toy_allegro_pol/physics.py): analytic toy model and exact labels
- [src/toy_allegro_pol/model.py](/Users/vead/.qclaw/workspace/generalized%20potential/src/toy_allegro_pol/model.py): minimal learned generalized-potential model
- [src/toy_allegro_pol/train_utils.py](/Users/vead/.qclaw/workspace/generalized%20potential/src/toy_allegro_pol/train_utils.py): autograd and loss assembly
- [src/toy_allegro_pol/eval_utils.py](/Users/vead/.qclaw/workspace/generalized%20potential/src/toy_allegro_pol/eval_utils.py): parity metrics, finite differences, and plotting helpers

## Data format

Each split is stored as a compressed `.npz` file with fields:

- `positions`: shape `[n_samples, n_atoms, dim]`
- `species`: shape `[n_samples, n_atoms]`
- `electric_field`: shape `[n_samples, dim]`
- `total_energy`: shape `[n_samples]`
- `forces`: shape `[n_samples, n_atoms, dim]`
- `response`: shape `[n_samples, dim]`

The current default setup uses:

- `dim = 2`
- `n_atoms = 3`
- `dtype = float64`

`float64` is intentional because finite-difference and autograd consistency checks are part of the workflow.

## Model structure

The learned model is intentionally simple but keeps the key API:

```python
total_energy = model(batch)
```

Then forces and response are derived through autograd:

```python
forces = -dU/dR
response = -dU/dE
```

The current network is not Allegro. It is a small MLP-based energy model with:

- an atomwise channel
- a pairwise channel
- explicit conditioning on external field
- a scalar total-energy output

This is enough to preserve the most important architectural idea from `allegro-pol`: the response quantities are not predicted by independent heads; they are derivatives of the same scalar potential.

## Quick start

### 1. Install dependencies

This repository is intentionally lightweight. At minimum, you need:

```bash
pip install torch numpy pyyaml matplotlib
```

### 2. Generate the toy dataset

```bash
python scripts/data_gen.py --config configs/default.yaml
```

This writes:

- `data/processed/toy_dataset_train.npz`
- `data/processed/toy_dataset_val.npz`
- `data/processed/toy_dataset_test.npz`

### 3. Train the model

```bash
python scripts/train.py --config configs/default.yaml
```

The training script saves checkpoints to:

- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/last.pt`

### 4. Run evaluation

```bash
python scripts/eval.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --output-dir outputs/eval
```

This produces:

- `metrics.json`
- energy parity plot
- force parity plot
- response parity plot
- energy-vs-field sweep plot
- response-vs-field sweep plot

It also runs a finite-difference vs autograd consistency check.

### 5. Run single-sample inference

Using a dataset sample:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --index 0
```

Or using a custom JSON input:

```json
{
  "positions": [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.8]],
  "species": [0, 1, 0],
  "electric_field": [0.2, -0.1]
}
```

then

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --input-json path/to/sample.json
```

## Evaluation outputs

The evaluation script reports:

- energy regression metrics
- force regression metrics
- response regression metrics
- finite-difference consistency errors

Two especially important checks are:

1. `force_fd_max_abs_err`
2. `response_fd_max_abs_err`

These compare autograd-derived quantities to finite differences of the learned energy. If those are small, the differentiable structure is behaving consistently.

## How this maps to allegro-pol

This repository mirrors the most important conceptual blocks:

- `positions + external field -> total energy`
- `total energy -> autograd -> force`
- `total energy -> autograd -> field response`
- `joint loss on energy, force, response`

In `allegro-pol`, the internal feature extractor is much richer and more physically structured. Here, the feature extractor is intentionally simple, but the data flow is the same.

Rough analogy:

- `src/toy_allegro_pol/model.py` plays the role of a simplified energy backbone
- `src/toy_allegro_pol/train_utils.py` plays the role of the autograd wrapper
- `scripts/eval.py` is the toy analogue of parity and response checks

## Current limitations

This is still a learning scaffold, not a production model.

Current simplifications:

- fixed number of particles
- 2D system
- no periodic boundary conditions
- no long-range electrostatics
- no explicit orientation degrees of freedom
- no true polarization/Born-charge physics
- no equivariant architecture
- no simulator integration

## Good next steps

Natural extensions from here:

- add ablation modes for `energy` only vs `energy + force` vs `energy + force + response`
- make the number of particles variable
- add orientation degrees of freedom
- replace the toy response with a more ER-like coarse-grained dipole mechanism
- move from this MLP backbone toward a more graph-like local interaction model

## Development notes

Useful command sequence during iteration:

```bash
python scripts/data_gen.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --epochs 2
python scripts/infer.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --index 0
python scripts/eval.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --output-dir outputs/eval-smoke
```

That sequence is enough to exercise the full loop:

```text
analytic toy physics
-> dataset
-> learned generalized potential
-> autograd force / response
-> evaluation and inference
```
