# GNN Quantum Error Correction - Training Log

## Current Architecture

### Node Features (5 total)
| Index | Feature | Range | Type | Description |
|-------|---------|-------|------|-------------|
| 0 | syndrome | -1, +1 | Dynamic | Detection measurement value |
| 1 | stabilizer_type | 0, 1 | Static | X vs Z stabilizer (checkerboard pattern) |
| 2 | dist_north | 0-1 | Static | Normalized distance from north boundary |
| 3 | dist_west | 0-1 | Static | Normalized distance from west boundary |
| 4 | time_round | 0-1 | Static | Normalized time round (0 at t=0, 1 at t=d-1) |

**Graph Structure:**
- Spatial edges (type=0): Adjacent detectors at same time round
- Temporal edges (type=1): Same spatial position, consecutive time rounds
- Edge type embedding: 2 types, learned embeddings

### Model Architecture
```
Input: 5 node features
↓
GATConvWithEdgeType (1→hidden): 4 heads, concat
BatchNorm1d(hidden_dim)
SiLU activation
↓
[Repeat 3 more times]
GATConvWithEdgeType (hidden→hidden): 4 heads, concat
BatchNorm1d(hidden_dim)
SiLU activation
↓
GlobalMeanPool (aggregate all nodes)
↓
Linear(hidden_dim, 64) → SiLU → Linear(64, 1)
↓
Output: Logical error prediction (sigmoid for probability)
```

### Core Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 128 | Dimension after first layer |
| num_layers | 4 | Total attention layers |
| num_heads | 4 | Heads per attention layer |
| learning_rate | 3e-4 | Adam optimizer |
| batch_size | 256 | Training batch size |
| optimizer | Adam | Default settings |
| loss | BCEWithLogitsLoss | Weighted for class imbalance |
| num_edge_types | 2 | Spatial (0) and Temporal (1) |
| activation | SiLU | Between layers |

---

## Training Performance Log

Add new rows as training completes. Format: keep distances in order (3, 5, 7, ...) for easy comparison.

| Date | Distance | Error Rate | Train Size | GNN Accuracy | MWPM Accuracy | Difference | Training Time | Epochs | Notes |
|------|----------|------------|-----------|--------------|---------------|-----------|---------------|--------|-------|
| | | | | | | | | | |

### Log Entry Template
When adding a new row, use this format:
- **Date**: YYYY-MM-DD
- **Distance**: Code distance d
- **Error Rate**: Physical error rate p
- **Train Size**: Total samples used (e.g., 1M, 10M)
- **GNN Accuracy**: Final test accuracy on fresh data
- **MWPM Accuracy**: Baseline matching accuracy
- **Difference**: (GNN Acc - MWPM Acc) * 100 (%)
- **Training Time**: Total wall-clock training time
- **Epochs**: Total epochs trained
- **Notes**: Any observations (e.g., "beat MWPM", "convergence issues", "hardware used")

---

## Previous Attempts

### Baseline (Before Rich Features)
- **Date**: 2026-01-04
- **Architecture**: 1 input feature (syndrome only)
- **Status**: Completed
- **Note**: Added 4 static node features to improve performance

### Current Implementation (With Rich Features)
- **Date**: 2026-01-04
- **Changes**:
  - Added stabilizer_type feature
  - Added distance from boundaries features
  - Added normalized time_round feature
  - Total features: 5 (was 1)
- **Expected Impact**: Significant improvement due to explicit geometric/temporal information
- **Status**: Ready for training

---

## Quick Reference: Running Training

```python
# Initialize trainer for a specific distance
trainer = GNNTrainer(p=0.005, d=3, device=device)

# Train until beating MWPM baseline
trainer, train_size = train_until_beat_mwpm(
    p=0.005, d=3, device=device,
    max_train_size=10**8, chunk_size=10**7
)

# Save the trained model
trainer.save(f"gnn_decoder_d3_p0.005.pt")
```

---

## Analysis & Observations

### Feature Importance Hypothesis
- **stabilizer_type**: Should help distinguish X vs Z error propagation patterns
- **boundary distances**: Detectors near edges behave differently; errors propagate differently at boundaries
- **time_round**: Explicit temporal ordering should improve learning of error dynamics
- **Expected improvement**: 5-15% accuracy boost over baseline (speculation)

### Known Considerations
- Models trained separately per distance d (no transfer learning yet)
- Class imbalance handled with pos_weight in BCEWithLogitsLoss
- All features normalized to 0-1 range for stable training
- Batch normalization after each layer helps with mixed feature scales

---

## Future Improvements

- [ ] Try transfer learning across distances
- [ ] Ablation study: remove one feature at a time to measure importance
- [ ] Hyperparameter tuning: learning rate, hidden_dim, num_layers
- [ ] Try different GNN architectures (GCN, GraphSAGE, etc.)
- [ ] Add error type information (if available in detector_error_model)
