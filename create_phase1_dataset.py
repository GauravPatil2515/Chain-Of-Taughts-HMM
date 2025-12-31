"""
Create Phase 1 dataset manually using rule-based CoT generation
This avoids the transformers dependency issues
"""

import json
from pathlib import Path
from datetime import datetime
import random

# Simple CoT templates for different problem types
def generate_math_cot(question: str, answer: str, should_be_correct: bool = True) -> tuple:
    """Generate a simple math CoT using templates."""
    
    # Extract numbers from question
    import re
    numbers = re.findall(r'\d+', question)
    
    if should_be_correct:
        # Generate reasoning that leads to correct answer
        steps = [
            f"Let me break this down step by step.",
            f"First, I'll identify the key numbers: {', '.join(numbers[:3]) if len(numbers) >= 3 else ', '.join(numbers)}.",
            f"Next, I need to perform the calculation.",
            f"Working through the arithmetic carefully.",
            f"Therefore, the answer is {answer}."
        ]
        predicted = answer
    else:
        # Generate faulty reasoning
        steps = [
            f"Looking at this problem, I'll calculate quickly.",
            f"Taking the first two numbers I see.",
            f"Performing a basic operation.",
            f"The answer appears to be {random.choice(numbers) if numbers else '42'}."
        ]
        predicted = random.choice(numbers) if numbers else "42"
    
    cot = " ".join(steps)
    is_correct = (predicted.strip() == answer.strip()) if should_be_correct else False
    
    return cot, predicted, is_correct


def generate_science_cot(question: str, answer: str, choices, should_be_correct: bool = True) -> tuple:
    """Generate science reasoning CoT."""
    
    # Handle both dict and list formats for choices
    if isinstance(choices, dict):
        choice_strs = [f"{k}) {v[:30]}..." for k, v in list(choices.items())[:2]]
    else:
        choice_strs = [f"{chr(65+i)}) {c[:30]}..." for i, c in enumerate(choices[:2])]
    
    if should_be_correct:
        steps = [
            f"Let me analyze this science question carefully.",
            f"The question asks about a specific concept.",
            f"Considering the choices provided: {', '.join(choice_strs)}",
            f"Based on scientific principles, the correct answer is {answer}."
        ]
        predicted = answer
    else:
        if isinstance(choices, dict):
            wrong_choice = random.choice([k for k in choices.keys() if k != answer])
        else:
            wrong_choice = random.choice([c for c in ['A', 'B', 'C', 'D'] if c != answer])
        steps = [
            f"Looking at this question quickly.",
            f"The most obvious answer seems to be {wrong_choice}.",
            f"Therefore, I'll go with {wrong_choice}."
        ]
        predicted = wrong_choice
    
    cot = " ".join(steps)
    is_correct = (predicted == answer) if should_be_correct else False
    
    return cot, predicted, is_correct


def create_dataset():
    """Create Phase 1 dataset with rule-based CoTs."""
    
    print("="*70)
    print("Phase 1 Dataset Creation - Rule-Based Approach")
    print("="*70)
    
    # Create 60% correct, 40% incorrect distribution
    random.seed(42)
    
    # Load GSM8K
    print("\n[1/2] Processing GSM8K (100 samples)...")
    with open("data/raw/gsm8k.json", "r") as f:
        gsm8k = json.load(f)
    
    gsm8k_results = []
    correct_target = int(len(gsm8k) * 0.6)  # 60 correct, 40 incorrect
    
    for i, item in enumerate(gsm8k):
        should_be_correct = i < correct_target
        cot, predicted, is_correct = generate_math_cot(
            item["question"], 
            item["answer"],
            should_be_correct
        )
        
        gsm8k_results.append({
            "id": f"gsm8k_rule_{item['id']}",
            "dataset": "gsm8k",
            "domain": "math",
            "question": item["question"],
            "ground_truth": item["answer"],
            "model": "rule-based-generator",
            "cot_raw": cot,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
    
    # Shuffle to mix correct/incorrect
    random.shuffle(gsm8k_results)
    
    # Save
    Path("data/cot_outputs").mkdir(parents=True, exist_ok=True)
    with open("data/cot_outputs/gsm8k_rule_100.json", "w") as f:
        json.dump(gsm8k_results, f, indent=2)
    
    gsm8k_correct = sum(1 for r in gsm8k_results if r["is_correct"])
    print(f"✓ Generated {len(gsm8k_results)} samples")
    print(f"  Correct: {gsm8k_correct} ({gsm8k_correct/len(gsm8k_results)*100:.1f}%)")
    print(f"  Saved to: data/cot_outputs/gsm8k_rule_100.json")
    
    # Load ARC
    print("\n[2/2] Processing ARC-Challenge (100 samples)...")
    arc_path = "data/raw/arc_challenge.json"
    if not Path(arc_path).exists():
        print(f"  Skipping ARC - file not found")
        arc_results = []
        arc_correct = 0
    else:
        with open(arc_path, "r") as f:
            arc = json.load(f)
        
        arc_results = []
        arc_correct_target = int(len(arc) * 0.6)
        
        for i, item in enumerate(arc):
            should_be_correct = i < arc_correct_target
            cot, predicted, is_correct = generate_science_cot(
                item["question"],
                item["answer"],
                item.get("choices", []),
                should_be_correct
            )
            
            arc_results.append({
                "id": f"arc_rule_{item['id']}",
                "dataset": "arc_challenge",
                "domain": "science",
                "question": item["question"],
                "choices": item.get("choices", []),
                "ground_truth": item["answer"],
                "model": "rule-based-generator",
                "cot_raw": cot,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "timestamp": datetime.now().isoformat()
            })
        
        random.shuffle(arc_results)
        
        with open("data/cot_outputs/arc_rule_100.json", "w") as f:
            json.dump(arc_results, f, indent=2)
        
        arc_correct = sum(1 for r in arc_results if r["is_correct"])
        print(f"✓ Generated {len(arc_results)} samples")
        print(f"  Correct: {arc_correct} ({arc_correct/len(arc_results)*100:.1f}%)")
        print(f"  Saved to: data/cot_outputs/arc_rule_100.json")
    
    # Merge datasets
    print("\n[3/3] Merging datasets...")
    all_results = gsm8k_results + arc_results
    random.shuffle(all_results)
    
    with open("data/cot_outputs/phase1_merged_200.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    total_correct = sum(1 for r in all_results if r["is_correct"])
    
    print(f"\n{'='*70}")
    print("Phase 1 Dataset Created!")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_results)}")
    print(f"Correct: {total_correct} ({total_correct/len(all_results)*100:.1f}%)")
    print(f"Incorrect: {len(all_results)-total_correct} ({(len(all_results)-total_correct)/len(all_results)*100:.1f}%)")
    print(f"\nDatasets:")
    print(f"  - GSM8K: {len(gsm8k_results)} (Math)")
    print(f"  - ARC: {len(arc_results)} (Science)")
    print(f"\nMerged file: data/cot_outputs/phase1_merged_200.json")
    print(f"\n⚠️  Note: These are rule-based CoTs, not from LLMs.")
    print(f"    They work for testing the pipeline but won't be")
    print(f"    as realistic as actual model-generated CoTs.")
    print(f"\n✅ Ready for Phase 2: Tagging!")


if __name__ == "__main__":
    create_dataset()
