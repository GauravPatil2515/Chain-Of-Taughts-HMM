import json
import random

def generate_dummy_data(output_path, n=20):
    data = []
    
    templates = [
        "I need to find the total sum. First, I have 5 apples. Then I add 3 apples. So, 5 + 3 = 8. The answer is 8.",
        "The problem asks for the velocity. We know v = d / t. Given d = 100, t = 10. Thus, 100 / 10 = 10. Final Answer: 10 m/s.",
        "Let x be the number. We have 2x = 10. So x = 5. The total is 5.",
        # Incorrect looking structure (no conclusion)
        "I think maybe it is 5. But wait, 5 is too small. Let me try 6.",
    ]
    
    for i in range(n):
        is_correct = random.choice([True, True, False]) # bias towards correct
        cot = random.choice(templates)
        
        item = {
            "question": f"Dummy question {i}",
            "cot_raw": cot,
            "answer_raw": "8" if "8" in cot else "10" if "10" in cot else "5",
            "ground_truth": "8" if "8" in cot else "10" if "10" in cot else "5",
            "is_correct": is_correct
        }
        data.append(item)
        
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {n} dummy samples to {output_path}")

if __name__ == "__main__":
    generate_dummy_data("data/cot_outputs/dummy_20.json")
