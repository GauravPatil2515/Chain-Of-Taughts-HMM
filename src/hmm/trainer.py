"""
HMM Trainer
Train Hidden Markov Model on tagged CoT sequences.
"""

import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Tuple
from hmmlearn import hmm


# Tag vocabulary
TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(TAGS)}


def load_tagged_data(file_path: str) -> Tuple[List[List[str]], List[bool]]:
    """Load tagged CoT data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sequences = []
    correctness = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        if len(tag_seq) >= 2:  # Need at least 2 steps for HMM
            sequences.append(tag_seq)
            correctness.append(item.get("is_correct", False))
    
    return sequences, correctness


def encode_sequences(sequences: List[List[str]]) -> Tuple[np.ndarray, List[int]]:
    """Convert tag sequences to integer arrays for HMM."""
    encoded = []
    lengths = []
    
    for seq in sequences:
        encoded_seq = [TAG_TO_ID.get(tag, TAG_TO_ID["IN"]) for tag in seq]
        encoded.extend(encoded_seq)
        lengths.append(len(encoded_seq))
    
    return np.array(encoded).reshape(-1, 1), lengths


def train_hmm(
    sequences: List[List[str]],
    n_components: int = 6,
    n_iter: int = 100,
    random_state: int = 42
) -> hmm.CategoricalHMM:
    """
    Train HMM on tag sequences.
    
    Args:
        sequences: List of tag sequences
        n_components: Number of hidden states
        n_iter: Max training iterations
        random_state: Random seed
    
    Returns:
        Trained HMM model
    """
    X, lengths = encode_sequences(sequences)
    
    n_symbols = len(TAGS)
    
    # CategoricalHMM is correct for discrete categorical observations
    model = hmm.CategoricalHMM(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
        verbose=True,
        n_features=n_symbols  # Number of possible emission symbols
    )
    
    model.fit(X, lengths)
    
    return model


def analyze_model(model: hmm.CategoricalHMM) -> Dict:
    """Analyze trained HMM parameters."""
    analysis = {
        "n_hidden_states": model.n_components,
        "n_observations": len(TAGS),
        "converged": model.monitor_.converged,
        "n_iterations": model.monitor_.n_iter,
    }
    
    # Initial state distribution
    analysis["initial_distribution"] = {
        f"State_{i}": float(p) for i, p in enumerate(model.startprob_)
    }
    
    # Transition matrix
    trans = model.transmat_
    analysis["transition_matrix"] = trans.tolist()
    
    # Most likely transitions
    top_transitions = []
    for i in range(model.n_components):
        for j in range(model.n_components):
            if trans[i, j] > 0.1:  # Only significant transitions
                top_transitions.append({
                    "from": f"State_{i}",
                    "to": f"State_{j}",
                    "probability": float(trans[i, j])
                })
    analysis["top_transitions"] = sorted(
        top_transitions, key=lambda x: x["probability"], reverse=True
    )[:10]
    
    # Emission matrix
    emit = model.emissionprob_
    analysis["emission_matrix"] = emit.tolist()
    
    # State-tag associations
    state_tags = []
    for i in range(model.n_components):
        best_tag_idx = np.argmax(emit[i])
        state_tags.append({
            "state": f"State_{i}",
            "most_likely_tag": TAGS[best_tag_idx],
            "probability": float(emit[i, best_tag_idx])
        })
    analysis["state_tag_associations"] = state_tags
    
    return analysis


def train_and_save(
    input_path: str,
    output_path: str,
    n_components: int = 6,
    correct_only: bool = True
):
    """Train HMM and save model + analysis."""
    print(f"Loading data from {input_path}")
    sequences, correctness = load_tagged_data(input_path)
    
    print(f"Total sequences: {len(sequences)}")
    
    # Filter to correct only if specified
    if correct_only:
        sequences = [seq for seq, correct in zip(sequences, correctness) if correct]
        print(f"Training on correct sequences only: {len(sequences)}")
    
    if len(sequences) < 5:
        raise ValueError(f"Not enough sequences for training: {len(sequences)}")
    
    print(f"\nTraining HMM with {n_components} hidden states...")
    model = train_hmm(sequences, n_components=n_components)
    
    # Save model
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Analyze and save analysis
    analysis = analyze_model(model)
    analysis_path = model_path.with_suffix(".analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {analysis_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Hidden states: {analysis['n_hidden_states']}")
    print(f"Converged: {analysis['converged']}")
    print(f"Iterations: {analysis['n_iterations']}")
    print("\nState-Tag Associations:")
    for assoc in analysis["state_tag_associations"]:
        print(f"  {assoc['state']} → {assoc['most_likely_tag']} ({assoc['probability']:.2f})")
    
    return model, analysis


def main():
    parser = argparse.ArgumentParser(description="Train HMM on tagged CoTs")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with tagged CoTs")
    parser.add_argument("--output", type=str, default="models/hmm_model.pkl",
                       help="Output path for trained model")
    parser.add_argument("--n-components", type=int, default=6,
                       help="Number of hidden states")
    parser.add_argument("--include-incorrect", action="store_true",
                       help="Include incorrect CoTs in training")
    
    args = parser.parse_args()
    
    train_and_save(
        input_path=args.input,
        output_path=args.output,
        n_components=args.n_components,
        correct_only=not args.include_incorrect
    )


if __name__ == "__main__":
    main()
