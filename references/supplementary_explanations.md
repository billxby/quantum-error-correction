# Supplementary Figure Explanations (Copy/Paste)

## GraphSAGE: strong at fixed distance, not yet fault-tolerant scaling
GraphSAGE is a learned graph neural network decoder. The plot shows logical error rate $p_L$ (decoder failure rate) versus physical error rate $p$ (noise per operation) across distances $d=3,5,7,9,11,13$. In a fault-tolerant regime, higher distance should reduce $p_L$ at fixed $p$. Here, GraphSAGE does not consistently show that separation. It performs competitively at each fixed distance but lacks the expected scaling trend across distance.

**Definitions:** $p$ = physical error rate; $p_L$ = logical error rate; $d$ = code distance.

**Note:** $p_L=0$ points are omitted on log scale.

**Meaning:** GraphSAGE is accurate for fixed-$d$ decoding but does not yet achieve fault-tolerant scaling.

---

## MWPM: clear threshold and correct scaling below it
MWPM (Minimum-Weight Perfect Matching) is the standard physics-based baseline for surface-code decoding. Its curves show stronger distance separation at lower $p$, indicating fault-tolerant behavior. The threshold region is around $p \approx 7.5\times10^{-3}$, where curves for adjacent distances cross; below this, higher $d$ reduces $p_L$.

**Definition:** Threshold $p_{th}$ = the $p$ value where increasing $d$ starts to suppress $p_L$.

**Note:** For low $p$ and high $d$, some MWPM runs have zero observed errors in 20k shots, so those $p_L=0$ points are omitted on log scale.

**Meaning:** MWPM achieves the expected scaling below threshold, which is why it remains the gold-standard baseline.

---

## $\Lambda$ directly measures whether distance helps
$\Lambda = p_L(d_1)/p_L(d_2)$ compares adjacent distances (e.g., $d=3$ vs $d=5$). If $\Lambda < 1$, higher distance suppresses errors (good). If $\Lambda > 1$, scaling fails. The dotted line marks $\Lambda=1$. In these results, MWPM shows $\Lambda>1$ below threshold (good suppression), while GraphSAGE does not consistently keep $\Lambda<1$ across $p$.

**Definition:** $\Lambda$ = suppression factor between distances; it measures scaling directly.

**Meaning:** This is the clearest diagnostic of fault-tolerant behavior: MWPM passes; GraphSAGE does not yet.

---

## Improvements / Future Work (applies to all three)
- Add **soft information** (confidence-weighted detector events), not just binary syndromes.
- Train **at circuit level** with realistic noise correlations.
- Use **multi-distance or curriculum training** to encourage scaling.
- Try **hybrid decoding**: fast GNN for most cases, MWPM for rare or ambiguous cases.
