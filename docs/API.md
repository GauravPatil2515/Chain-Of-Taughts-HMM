# API Reference

This document provides an overview of the main modules and their functions.

---

## `src.hmm.trainer`

### `train_hmm(sequences, n_components=6, n_iter=100, random_state=42)`

Train a Categorical HMM on tag sequences.

**Parameters:**

- `sequences` (List[List[str]]): List of tag sequences
- `n_components` (int): Number of hidden states (default: 6)
- `n_iter` (int): Maximum training iterations (default: 100)
- `random_state` (int): Random seed for reproducibility

**Returns:**

- `hmmlearn.hmm.CategoricalHMM`: Trained model

**Example:**

```python
from src.hmm.trainer import train_hmm

sequences = [
    ["PR", "CA", "CA", "CO"],
    ["PR", "EQ", "CA", "IN", "CO"],
]
model = train_hmm(sequences, n_components=5)
```

---

## `src.hmm.scorer`

### `class HMMScorer`

Score CoT sequences using a trained HMM.

**Methods:**

#### `__init__(model_path, anomaly_threshold=0.05)`

Initialize scorer with a trained model.

#### `score(tag_sequence) -> CoTScore`

Compute all metrics for a tag sequence.

**Returns:**

- `log_likelihood`: Raw log-likelihood
- `normalized_ll`: Per-step log-likelihood (main metric)
- `anomaly_count`: Number of low-probability transitions
- `viterbi_path`: Most likely hidden state sequence

**Example:**

```python
from src.hmm.scorer import HMMScorer

scorer = HMMScorer("models/phase5_hmm.pkl")
result = scorer.score(["PR", "CA", "EQ", "CO"])
print(f"Score: {result.normalized_ll:.2f}")
```

---

## `src.tagging.rule_based_tagger`

### `tag_step(step_text) -> str`

Classify a reasoning step into one of 8 semantic tags.

**Tags:**

| Tag | Meaning |
|-----|---------|
| PR | Problem Restatement |
| IN | Inference |
| CA | Calculation |
| EQ | Equation |
| CO | Conclusion |
| RF | Reflection |
| VE | Verification |
| DV | Diversion |

**Example:**

```python
from src.tagging.rule_based_tagger import tag_step

print(tag_step("First, let's identify what we know."))  # "PR"
print(tag_step("2 + 3 = 5"))                            # "CA"
print(tag_step("Therefore, the answer is 42."))         # "CO"
```

---

## `src.evaluation.baselines`

Heuristic baselines for comparison:

- `LengthBaseline`: Scores based on CoT length
- `TagDiversityBaseline`: Ratio of unique tags
- `StructureBaseline`: Checks for expected elements
- `TransitionBaseline`: Bigram transition probabilities
- `RandomBaseline`: Uniform random scores

---

## `src.analysis.qualitative`

### `compute_transition_matrix(tag_sequences) -> np.ndarray`

Calculate empirical transition probabilities between tags.

### `plot_transition_matrices(correct_seqs, incorrect_seqs, output_path)`

Generate side-by-side heatmap comparison.

### `extract_qualitative_examples(df, output_path)`

Extract representative examples for paper.

---

## Command-Line Usage

```bash
# Train HMM
python -m src.hmm.trainer --input data.json --output model.pkl

# Score CoTs
python -m src.hmm.scorer --model model.pkl --input data.json --output scored.json

# Full evaluation
python run_baseline_eval.py --data data.json --output results/
```
