# SimpleNN and DeepSets: Layer-by-Layer Implementation Guide

This document captures exactly how the two non-GNN benchmark architectures are implemented in this repository, so another agent can reason about behavior without re-reading all source code.

Source of truth:
- `code/benchmark_models.py`
- Classes: `SimpleNNModel`, `SimpleNN`, `FourierFeatures`, `DeepSetsModel`, `DeepSets`

## 1. Scope and Intended Use

- `SimpleNNModel`: dense feedforward baseline that consumes full syndrome arrays directly.
- `DeepSetsModel`: set-based model that converts fired detectors into coordinate sets and applies permutation-invariant pooling.
- Both are binary classifiers with sigmoid outputs and BCE training loss.

---

## 2. SimpleNN (Array-Based Baseline)

### 2.1 Input/Output Contract

- Input tensor: `x` with shape `[batch_size, num_detectors]`
- Output tensor: `y_hat` with shape `[batch_size, 1]`
- Semantic output: probability of logical error in `[0, 1]`

### 2.2 Constructor Parameters

`SimpleNNModel(in_channels, hidden_dims=(256, 512, 1024), dropout=0.0)`

- `in_channels`: detector count for the chosen code distance.
- `hidden_dims`: tuple controlling layer count and width.
- `dropout`: applied after SiLU in each hidden block only if `dropout > 0`.

### 2.3 Exact Layer Stack (Built Dynamically)

For each hidden width `h` in `hidden_dims`, append:
1. `Linear(prev_dim, h)`
2. `LayerNorm(h)`
3. `SiLU()`
4. `Dropout(dropout)` only when `dropout > 0`

After final hidden block append:
1. `Linear(last_hidden_dim, 1)`
2. `Sigmoid()`

All layers are wrapped in a single `torch.nn.Sequential`.

### 2.4 Forward Pass

- `forward(x)` returns `self.layers(x)` directly.
- No residuals, no skip-connections, no batch norm.

### 2.5 Initialization

- Every `Linear` weight: Kaiming normal (`kaiming_normal_`, nonlinearity=`relu`).
- Every `Linear` bias: zeros.

### 2.6 Wrapper Behavior (`SimpleNN`)

- Auto-selects `cuda` if available.
- Infers `num_detectors` from stim-generated circuit for distance `d`.
- Rebuilds model if detector count changes.
- Training uses:
  - `Adam(lr=...)`
  - `BCELoss()`
  - optional gradient clipping via `clip_grad_norm_` when `max_grad_norm` is not `None`
- Accuracy threshold used in evaluation: `0.5`.

### 2.7 Training Data Path

- Calls `SurfaceCodeSampler.sample(...)` to get:
  - `detections`: syndrome vectors
  - `labels`: binary targets
- For each batch:
  - `pred = model(X)`
  - `loss = BCE(pred, y.unsqueeze(1))`

---

## 3. DeepSets (Coordinate-Based, Size-Invariant)

### 3.1 High-Level Pipeline

1. Dense syndrome vector -> fired detector indices
2. Fired detector indices -> normalized `(x, y, t)` coordinates
3. Optional Fourier feature encoding
4. Per-element MLP (`phi`)
5. Masked pooling across set elements (sum or mean)
6. Add learnable null token
7. Set-level MLP (`rho`) + sigmoid

### 3.2 Input/Output Contract

At model level (`DeepSetsModel.forward`):
- `coords`: `[batch, max_fired, 3]`
- `counts` (optional): `[batch]`, number of valid items in each sample
- Output: `[batch, 1]`

At wrapper level (`DeepSets.predict`):
- Input detections: `[batch, num_detectors]` or `[num_detectors]`
- Output after squeeze: `[batch]`

### 3.3 Fourier Feature Encoder (`FourierFeatures`)

Constructor:
`FourierFeatures(in_features, mapping_size, scale=10.0, learnable=False)`

- Parameter `B`: shape `[in_features, mapping_size // 2]`
- Initialized as Gaussian `N(0, scale^2)`.
- `requires_grad = learnable`.

Forward:
1. `xp = 2*pi*(x @ B)`
2. return `concat(sin(xp), cos(xp))`

Output dimension is `mapping_size`.

### 3.4 DeepSets Constructor Defaults

`DeepSetsModel(in_features=3, phi_hidden=(128, 128), rho_hidden=(256, 128), pool='sum', dropout=0.0, use_fourier_features=True, fourier_dim=64, fourier_scale=5.0)`

- If Fourier enabled: phi input dim = `fourier_dim`.
- Else: phi input dim = `in_features` (3).
- Pooling mode: `'sum'` or `'mean'`.

### 3.5 Phi Network (Elementwise MLP)

For each hidden width `h` in `phi_hidden`, append:
1. `Linear(prev_dim, h)`
2. `LayerNorm(h)`
3. `SiLU()`
4. `Dropout(dropout)` only if `dropout > 0`

Stored as `self.phi = Sequential(...)`.
Final phi output width saved as `self.phi_out_dim`.

### 3.6 Null Token

- `self.null_token`: learnable parameter of shape `[1, phi_out_dim]`.
- Added after pooling to every sample representation.
- Initialized to small Gaussian noise (`std=0.01`).

### 3.7 Rho Network (Set-Level MLP)

For each hidden width `h` in `rho_hidden`, append:
1. `Linear(prev_dim, h)`
2. `LayerNorm(h)`
3. `SiLU()`
4. `Dropout(dropout)` only if `dropout > 0`

Then append final:
1. `Linear(last_hidden_dim, 1)`
2. `Sigmoid()`

Stored as `self.rho = Sequential(...)`.

### 3.8 DeepSets Forward (Exact Step Order)

1. Feature encoding:
   - If Fourier enabled: `x = fourier(coords)`
   - Else: `x = coords`
2. Element embedding: `h = phi(x)`
3. If `counts` is provided:
   - Build boolean mask for valid positions per sample.
   - Zero padded positions in `h`.
   - Pool:
     - `'mean'`: `sum(h, dim=1) / clamp(counts, min=1)`
     - `'sum'`: `sum(h, dim=1)`
4. If `counts` is not provided:
   - `'sum'`: `h.sum(dim=1)`
   - `'mean'`: `h.mean(dim=1)`
5. Add null token: `pooled = pooled + null_token`
6. Classify: `rho(pooled)`

### 3.9 Coordinate Conversion (`DeepSets._syndromes_to_coords`)

Given detections `[batch, num_detectors]`:
1. Threshold fired detectors with `detections > 0.5`.
2. Count fired per sample -> `fired_counts`.
3. Convert detector indices to cached normalized coordinates.
4. Pack into dense padded tensor `[batch, max_fired, 3]`.
5. Return `(coords, fired_counts)`.

Edge case:
- If no detector fires in any sample, returns:
  - `coords = zeros([batch, 1, 3])`
  - `counts = zeros([batch])`

### 3.10 Coordinate Source and Normalization

- Coordinates are extracted from stim detector error model instructions.
- Uses first up to 3 detector coordinate args from stim (`x, y, t`).
- Each coordinate dimension is min-max normalized to `[0, 1]` for a given distance `d`.
- Cached per distance in `coord_cache`.

### 3.11 Training Loop (`DeepSets.train_from_data`)

Per batch:
1. slice detections/labels
2. convert detections -> `(coords, counts)`
3. `pred = model(coords, counts)`
4. `loss = BCE(pred, labels.unsqueeze(-1))`
5. backward
6. optional gradient clipping
7. optimizer step

Optimizer/loss:
- `Adam(lr=...)`
- `BCELoss()`

---

## 4. Practical Comparison (For Agent Handoff)

### 4.1 What Changes with Code Distance `d`

- `SimpleNN`: input width `num_detectors` changes with `d`; model may be rebuilt for new width.
- `DeepSets`: model width is independent of detector count; only set size changes, handled by masking/pooling.

### 4.2 Invariance and Inductive Bias

- `SimpleNN`: no permutation invariance over detector order.
- `DeepSets`: permutation invariant after pooling (`sum` or `mean`).

### 4.3 Output Semantics

- Both output sigmoid probability of logical error and commonly threshold at `0.5`.

---

## 5. Known Implementation Notes

1. `SimpleNN.load(...)` currently reconstructs using keys `hidden_dim` and `num_layers`, but saved config stores `hidden_dims` and `dropout`. This is likely a mismatch bug in load path.
2. `DeepSets.load(...)` rebuilds model without passing `use_fourier_features` and `fourier_dim` from config. This can mismatch architecture when loading checkpoints trained with Fourier settings.
3. Both wrappers save configs and timestamps in checkpoint dictionaries with `state_dict`.

---

## 6. Minimal Equations (Aligned with Current Code)

### 6.1 SimpleNN

`y_hat = sigmoid(Linear_o(f_L(...f_2(f_1(x)))))`, where each hidden block is
`f_l(h) = Dropout(SiLU(LayerNorm(Linear_l(h))))` (dropout optional).

### 6.2 DeepSets

`y_hat = rho(Pool({phi(gamma(c_i))}_{i=1..n}) + t0)`,
where `gamma` is optional Fourier encoding, `Pool` is `sum` or `mean`, and `rho` ends in sigmoid.

---

## 7. Verified Findings Bullets for the Slide (Top-Left Box)

Use these three bullets if you want claims that are aligned with saved results.

1. Distance-7 benchmark screening selected GraphSAGE as the best-performing GNN among tested architectures, with mean accuracy 0.89231 versus GIN 0.88171, GAT 0.81574, and GCN 0.81481.
2. Based on that ranking, the later GNN development path is GraphSAGE-centric: tuning artifacts, best-config export, and downstream comparison tables are all built around GraphSAGE runs.
3. GraphSAGE hyperparameter tuning on d=7 found a best configuration at 5 layers, hidden size 256, learning rate 3e-4, dropout 0.1, and max aggregation, reaching validation accuracy 0.93904 and test accuracy 0.93884 after 10 epochs.

### Evidence for the Three Bullets

- Architecture ranking values are in `code/benchmarks/test_summary_d7_2026-02-01_11-55-45.csv`.
- Best tuning configuration and metrics are in `code/gSAGE/tuning/best_model_config.json`.
- Full tuning sweep is in `code/gSAGE/tuning/combined_results.csv` and `code/gSAGE/tuning/results.json`.

### Accuracy Formatting for Presentation

If you want percentages in the slide table, these are the consistent rounded values:
- GraphSAGE: 89.2%
- GIN: 88.2%
- GAT: 81.6%
- GCN: 81.5%

### Important Scope Note

These bullets describe model-selection among GNN architectures, not absolute best decoder overall. The same benchmark file shows MWPM with higher accuracy than learned models on that test setup.

---

## 8. Hyperparameter Tuning Method: Random Sampling vs Random-Walk

This section clarifies exactly what the tuning procedure is in the GraphSAGE pipeline.

### 8.1 What the Current Pipeline Actually Does

The current implementation is a seeded discrete random search (plus distributed execution), not a true random-walk method.

Why:
1. Candidate configs are generated by independent random draws from each hyperparameter list.
2. There is no neighborhood operator around a current best point.
3. There is no acceptance/rejection rule (e.g., Metropolis criterion) that defines a random walk.

### 8.2 Search Space and Fixed Parameters

Searched hyperparameters (discrete choices):
- `num_layers`: [2, 3, 4, 5]
- `hidden_dim`: [64, 128, 256, 512]
- `learning_rate`: [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
- `dropout`: [0.0, 0.1, 0.2, 0.3]
- `aggr`: ['mean', 'max']

Fixed parameters during tuning:
- `batch_size`, `epochs`, `distance`, `in_channels`

### 8.3 Generation Procedure (As Coded)

The generation logic in the tuning notebook does the following:
1. Set a random seed (`SEED = 42`) for reproducibility.
2. For each config id `i`, sample each hyperparameter independently using `random.choice(...)`.
3. Save all sampled configs to `configs.json`.

Formally, if the search space dimensions are `S_1 ... S_k`, each config is sampled as:

`h_i = (u_{i1}, ..., u_{ik}),  with  u_{ij} ~ Uniform(S_j)`

This is random search over a finite discrete domain.

### 8.4 Distributed Worker Execution

Training is parallelized by assigning config ids to workers via modulo partitioning:
- Worker `w` trains ids with `id % 5 == (w - 1)`.
- `CUSTOM_CONFIGS` can override worker assignment and force an explicit id order.

Resume behavior:
- Each worker writes incremental results to `results/worker_<id>.json`.
- On restart, already-completed ids are skipped.

### 8.5 Selection Rule

After all runs finish, results are aggregated (`results.json` and `combined_results.csv`) and the best configuration is exported (`best_model_config.json`).

Observed best saved configuration:
- layers: 5
- hidden_dim: 256
- learning_rate: 3e-4
- dropout: 0.1
- aggregation: max
- val/test accuracy: 0.93904 / 0.93884

### 8.6 Artifact Consistency Note

There is a bookkeeping mismatch between artifacts:
1. Notebook text and worker logic describe a 50-config plan (`id` 0..49).
2. Current `configs.json` snapshot in this repo records 25 configs (`n_configs: 25`, ids 0..24).

Interpretation: the method is unchanged, but this repository snapshot appears to include a reduced/partial config set.

### 8.7 If You Want a True Random-Walk Tuner

A true random-walk variant would need these additional elements:
1. Start from a seed config.
2. Propose a neighboring config by perturbing one or two dimensions per step.
3. Accept/reject moves based on improvement (or probabilistic acceptance).
4. Continue for `T` steps and keep the best visited state.

Without those elements, the current process should be described as random search, not random walk.
