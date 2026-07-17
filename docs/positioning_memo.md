# Positioning Comparison Memo

## Executive Summary

My CoT-HMM approach differs fundamentally from two concurrent 2026 works, occupying a distinct niche in the efficiency-validity tradeoff space.

---

## Method Comparison Table

| Aspect | My Method (CoT-HMM) | HHMM Causal (ICLR 2026) | uPRM (arXiv:2605.10158) |
|--------|---------------------|-------------------------|--------------------------|
| Architecture | Ensemble of Categorical HMMs on rule-based tags | Hierarchical HMM + causal intervention vectors | Distilled neural reward model (teacher-student) |
| Training Data | 50-100 correct traces only | Large-scale with interventions | Preference datasets + ProcessBench |
| Compute | CPU seconds (no GPU) | GPU hours, multi-GPU for intervention | GPU days for distillation |
| Inference Speed | ~150ms per trace | ~1-2 sec per trace | ~100ms (GPU) / ~500ms (CPU) |
| Interpretable | ✓ Full transition matrices | ✓ Partial | ✗ Neural black box |
| Causal Validation | ✗ Correlation only | ✓ Interventional steering | ✓ RL training with test-time gains |
| Main Metric | r=0.21, p=0.008 | 15% accuracy gain from steering | 73% ProcessBench accuracy |

---

## Key Differentiators

### What They Do That I Don't

1. **HHMM Causal (fr9t7r43am)**:
   - Uses hierarchical hidden Markov models with causal intervention vectors
   - CAN CONVERT failing reasoning into correct reasoning (not just score)
   - Requires labeled intervention data
   - Multi-GPU training for intervention optimization

2. **Unsupervised Process Reward Models (2605.10158)**:
   - Neural distillation approach with larger scale
   - RL validation shows ACTUAL test-time accuracy gains (not just correlation)
   - Trained on ProcessBench scale datasets
   - Reports 73% accuracy on ProcessBench benchmark

### What Makes My Contribution Valuable

1. **Minimal Data Requirement**: Works with 50-100 correct examples vs 1000+ for others
2. **Near-Zero Compute**: CPU-only inference (~150ms) vs GPU-required for competitors
3. **Full Interpretability**: Every transition probability is inspectable
4. **Fast Iteration**: Model trains in seconds, enabling rapid experimentation

---

## Positioning Statement

While concurrent work has demonstrated more ambitious capabilities, our approach provides a genuinely lightweight alternative for chain-of-thought verification in resource-constrained settings. We explicitly do NOT claim causal intervention capability or RL-validated downstream gains in our current draft. Our contribution is the classical probabilistic modeling approach with near-zero compute requirements — a non-trivial contribution given the field's trend toward ever-larger neural verifiers.

---

## Related Work Placement

This positions our work for Track A venues (KI, Big Data Mining and Analytics) which value efficiency and interpretability contributions. Track B would require running the RL validation scaffold to add causal/interventional validation.