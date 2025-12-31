"""Balance correct/incorrect ratio in dataset"""
import json
import random
import sys

def balance_data(input_path, output_path, target_correct_ratio=0.6, seed=42):
    """Balance correct/incorrect ratio"""
    random.seed(seed)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct = [d for d in data if d['is_correct']]
    incorrect = [d for d in data if not d['is_correct']]
    
    print(f"Original: {len(correct)} correct ({len(correct)/len(data)*100:.1f}%), "
          f"{len(incorrect)} incorrect ({len(incorrect)/len(data)*100:.1f}%)")
    
    # Calculate target counts
    if len(correct) / len(data) > target_correct_ratio + 0.15:
        # Too many correct, downsample
        target_correct = int(len(data) * target_correct_ratio)
        correct = random.sample(correct, min(target_correct, len(correct)))
        print(f"Downsampled correct to {len(correct)}")
    
    elif len(incorrect) / len(data) > (1 - target_correct_ratio) + 0.15:
        # Too many incorrect, downsample
        target_incorrect = int(len(data) * (1 - target_correct_ratio))
        incorrect = random.sample(incorrect, min(target_incorrect, len(incorrect)))
        print(f"Downsampled incorrect to {len(incorrect)}")
    else:
        print("No balancing needed - ratio is acceptable")
    
    balanced = correct + incorrect
    random.shuffle(balanced)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced, f, indent=2, ensure_ascii=False)
    
    print(f"\nBalanced: {len(correct)} correct ({len(correct)/len(balanced)*100:.1f}%), "
          f"{len(incorrect)} incorrect ({len(incorrect)/len(balanced)*100:.1f}%)")
    print(f"Total: {len(balanced)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python balance_dataset.py <input.json> <output.json> [target_ratio]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    
    balance_data(input_path, output_path, target_ratio)
