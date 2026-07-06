"""Clean generated data by removing errors and very poor samples"""
import json
import sys

def clean_data(input_path, output_path, min_cot_length=20):
    """Remove samples with errors or very short CoTs"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Filter criteria
    cleaned = []
    removed = []
    
    for item in data:
        cot = item.get('cot_raw', '')
        
        # Remove if:
        # 1. Empty or very short CoT
        # 2. Contains error message
        # 3. Predicted answer is empty
        
        if len(cot) < min_cot_length:
            removed.append((item['id'], 'too_short'))
            continue
        
        if '[Generation Error' in cot or '[Error:' in cot:
            removed.append((item['id'], 'error'))
            continue
        
        if not item.get('predicted_answer', '').strip():
            removed.append((item['id'], 'no_answer'))
            continue
        
        cleaned.append(item)
    
    # Save cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    # Report
    print(f"Original: {original_count}")
    print(f"Cleaned: {len(cleaned)}")
    print(f"Removed: {len(removed)} ({len(removed)/original_count*100:.1f}%)")
    
    if removed:
        print("\nRemoval reasons:")
        from collections import Counter
        reasons = Counter(r[1] for r in removed)
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
    
    return len(cleaned)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_batch.py <input.json> <output.json> [min_cot_length]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    min_length = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    clean_data(input_path, output_path, min_length)
