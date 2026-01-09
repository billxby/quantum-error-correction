# Quantum Error Correction - GNN Decoder

This project implements a Graph Neural Network (GNN) decoder for surface code quantum error correction, based on the approach described in [Lange et al.](https://arxiv.org/abs/2307.01241).

## Overview

The decoder takes syndrome measurements from a surface code and predicts logical errors (bit-flips and phase-flips) using graph neural networks.

---

## Implementations

### 1. Dense GNN (v1) - Original Implementation

**Location:** Cells 7-14 in `gnn_attempt_fixed2.ipynb`

| Component | Description |
|-----------|-------------|
| `SyndromeGraph` | Fixed graph with ALL detectors as nodes |
| `GATConvWithEdgeType` | Graph attention with discrete edge type embeddings (spatial=0, temporal=1) |
| `SyndromeDecoder` | 4-layer GAT, single output head (λ_Z only) |
| `GNNTrainer` | Training loop with BCEWithLogitsLoss |

**Characteristics:**
- ❌ Fixed node count regardless of error rate
- ❌ Scales with code size, not error rate
- ✅ Simple to understand
- ❌ Single output head

---

### 2. Sparse GNN (v2) - Thesis-Inspired Implementation

**Location:** Cells 15-22 in `gnn_attempt_fixed2.ipynb`

| Component | Description |
|-----------|-------------|
| `DEMGraph` | Sparse graph - only TRIGGERED detectors become nodes |
| `GATConvContinuous` | Graph attention with continuous edge weights from DEM |
| `SparseDecoder` | 4-layer GAT with residual connections, TWO output heads |
| `SparseGNNTrainer` | Training with masked two-head loss |

**Key Improvements:**

#### 1. Sparse Graph Construction
```python
# Only triggered detectors become nodes
triggered = detections[b].nonzero()  # e.g., 3 nodes instead of 75
```
- At p=0.5%, typical d=5 has ~3-5 triggered detectors vs 75 total
- Inference scales linearly with error rate

#### 2. DEM Edge Weights
```python
# Edge weight = -log(probability) from detector error model
dem = circuit.detector_error_model(decompose_errors=True)
weight = -np.log(probability)  # Higher weight = less likely
```
- Uses physics-informed edge probabilities
- Continuous projection via `nn.Linear(1, hidden_dim)` instead of discrete embedding

#### 3. Two Output Heads
```python
self.head_Z = nn.Sequential(...)  # Predicts λ_Z (bit-flip)
self.head_X = nn.Sequential(...)  # Predicts λ_X (phase-flip)
```
- Joint prediction captures Y-error correlations
- Masked loss for missing labels: `if ~isnan(label): compute_loss()`

#### 4. Residual Connections
```python
x = x + F.silu(layer(x, edge_index, edge_weight))  # Preserves original features
```

#### 5. Empty Graph Handling
```python
if num_triggered == 0:
    return self.empty_graph_embedding  # Learned parameter
```

---

## File Structure

```
learning/
├── gnn_attempt_fixed2.ipynb   # Main notebook with both implementations
├── TRAINING_LOG.md            # Training results log
├── README.md                  # This file
└── weights/
    ├── nn_decoder_d3_p0.005.pt       # Dense v1 weights
    ├── nn_decoder_d5_p0.005.pt
    ├── nn_decoder_d7_p0.005.pt
    ├── sparse_gnn_d3_p0.005_*.pt     # Sparse v2 weights
    └── ...
```

---

## Usage

### Quick Test (Sparse v2)
```python
# Run cell 21 in the notebook
sparse_trainer = SparseGNNTrainer(p=0.005, d=3, device=device)
detections, labels = sparse_trainer.sample_data(10000)
loss, acc = sparse_trainer.train_epoch(detections, labels)
```

### Full Training
```python
# Uncomment and run cell 22
trained_models, results = run_sparse_training()
```

### Evaluate Against MWPM
```python
mwpm_acc = get_mwpm_accuracy(p=0.005, d=5, num_shots=100000)
gnn_acc = trainer.evaluate(num_samples=100000)
print(f"GNN beats MWPM: {gnn_acc > mwpm_acc}")
```

---

## Architecture Comparison

| Feature | Dense v1 | Sparse v2 |
|---------|----------|-----------|
| Nodes | All detectors | Triggered only |
| Edge weights | Discrete types (0,1) | Continuous (-log p) |
| Output heads | 1 (Z only) | 2 (Z and X) |
| Residual connections | No | Yes |
| Empty graph handling | N/A | Learned embedding |
| Parameters | ~500K | ~500K |
| Inference scaling | O(d² × rounds) | O(error_rate × volume) |

---

## Performance Tuning

If sparse v2 underperforms:

1. **Edge weights not helping?**
   - Try: `weight = dem_weight * sigmoid(learned_scale)`
   - Or: Concatenate DEM weight with learned edge embedding

2. **Overfitting?**
   - Increase dropout in attention layers (currently 0.1)
   - Add dropout before output heads

3. **Slow convergence?**
   - Learning rate warmup
   - Larger batch size (512+)

4. **Scale to larger codes (d≥9)?**
   - Increase layers (6-8)
   - Sparse advantage grows with code size

---

## References

- [Lange et al. (2023) - Data-driven decoding of quantum error correcting codes using graph neural networks](https://arxiv.org/abs/2307.01241)
- [Stim - Fast stabilizer circuit simulator](https://github.com/quantumlib/Stim)
- [PyMatching - MWPM decoder](https://github.com/oscarhiggott/PyMatching)

---

## Training Log

Results are automatically logged to `TRAINING_LOG.md` with columns:
- Date, Distance, Error rate, Training samples
- GNN accuracy, MWPM accuracy, Difference
- Training time, Status, Notes
