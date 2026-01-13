# Plan: Graph Attention for QEC Decoder

**TL;DR:** Attention can be implemented in two ways—either by recomputing distances from node features during attention, or by incorporating the precomputed edge weights into the attention mechanism. Both are viable, with different tradeoffs. You already have GAT examples in `learning/gnn_attempt_fixed2.ipynb` to reference.

---

## Current Data Flow

| Data | Location | Content |
|------|----------|---------|
| **Node features** | `x` [N, 5] | `[is_x, is_z, d_north, d_west, d_time]` — spatial info IS here |
| **Edge weights** | `edge_attr` [E, 1] | `(supremum_distance)^(-2)` — derived from node spatial features |

**Key insight:** Since `d_north`, `d_west`, `d_time` are in node features, attention mechanisms **can recompute pairwise distances dynamically** during message passing. The `edge_attr` is essentially a precomputed cache.

---

## Implementation Steps

1. **Choose attention strategy:** Either (A) use edge weights as attention bias/features, or (B) let attention learn spatial relationships directly from node coordinates—no changes to `SparseGraph` needed either way.

2. **Implement attention layer:** Create a `GATConv` variant that computes attention scores from source/target node features, optionally incorporating `edge_attr` as a learned bias term (see `learning/gnn_attempt_fixed2.ipynb#GATConvContinuous` for an existing pattern).

3. **Replace `GCNConv` layers:** Swap `GCNConv` for the new attention layer in `GCNModel`, keeping the same interface (`forward(x, edge_index, edge_attr)`).

4. **Keep pooling and classification head unchanged:** Global mean pooling and MLP classifier work identically regardless of attention vs GCN.

---

## Design Decisions to Make

### 1. Use edge weights in attention or not?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A** | Ignore `edge_attr`, let attention learn from raw coordinates in `x` | Simpler, attention learns what matters | May need to relearn distance relationships |
| **B** | Project `edge_attr` into attention space as bias | Leverages precomputed physics | Adds complexity, may over-constrain |
| **C** | Store raw distances in `edge_attr` instead of inverse-square | More flexible for attention to transform | Requires `SparseGraph` modification |

### 2. Multi-head attention?

Standard for GAT; increases expressivity but also parameters—typically 4-8 heads works well.

### 3. Residual connections?

Your existing sparse v2 implementation uses them (`x = x + self.res_proj(x_in)`)—worth including for training stability.

---

## Attention Architecture Sketch

```python
class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, use_edge_weights=True):
        super().__init__(aggr='add')
        self.heads = heads
        self.out_channels = out_channels
        self.use_edge_weights = use_edge_weights

        # Linear projections for Q, K, V
        self.lin_src = Linear(in_channels, heads * out_channels)
        self.lin_dst = Linear(in_channels, heads * out_channels)

        # Attention parameter
        self.att = Parameter(torch.zeros(1, heads, 2 * out_channels))

        # Optional: project edge weight into attention space
        if use_edge_weights:
            self.edge_proj = Linear(1, heads)

    def forward(self, x, edge_index, edge_attr=None):
        H, C = self.heads, self.out_channels

        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)

        # Propagate with attention
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        return out.view(-1, H * C)

    def message(self, x_i, x_j, edge_attr, index):
        # Compute attention: a^T [x_i || x_j]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        # Optionally add edge weight bias
        if self.use_edge_weights and edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # [E, heads]
            alpha = alpha + edge_bias

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)  # Normalize per-node

        return alpha.unsqueeze(-1) * x_j
```

---

## Questions to Resolve

- [ ] Should attention be computed from raw spatial coords or transformed node embeddings?
- [ ] How many attention heads? (4, 8?)
- [ ] Include edge weights as bias, as separate features, or ignore?
- [ ] Add residual connections between layers?
- [ ] Dropout on attention weights?
