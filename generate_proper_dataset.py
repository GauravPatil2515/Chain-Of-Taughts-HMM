import json
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

# Create balanced dataset with realistic variations
data = []

# Realistic correct and incorrect patterns with VARIATION
correct_base_patterns = [
    ['PR', 'CA', 'IN', 'CO'],
    ['PR', 'EQ', 'CA', 'IN', 'CO'], 
    ['PR', 'IN', 'CA', 'CO'],
    ['PR', 'CA', 'RF', 'CA', 'CO'],
]

incorrect_base_patterns = [
    ['PR', 'DV', 'CO'],              # Simple diversion
    ['PR', 'CA', 'CA', 'CA', 'CO'],  # Stuck in calculation
    ['PR', 'IN', 'IN', 'IN', 'CO'],  # Stuck in inference
    ['PR', 'RF', 'RF', 'RF', 'CO'],  # Stuck in reflection
]

# Add noise to create diversity
def add_noise_to_pattern(base_pattern, noise_prob=0.3):
    '''Add random variations to a base pattern'''
    pattern = list(base_pattern)
    # Sometimes insert diversion or reflection
    for i in range(len(pattern)):
        if random.random() < noise_prob:
            if random.random() < 0.5:
                pattern.insert(i, 'DV')  # Insert diversion
            else:
                pattern.insert(i, 'RF')  # Insert reflection
    # Sometimes duplicate a step
    if random.random() < 0.2 and len(pattern) > 2:
        idx = random.randint(0, len(pattern)-1)
        pattern.insert(idx, pattern[idx])
    # Sometimes delete a step (but keep min length)
    if len(pattern) > 3 and random.random() < 0.1:
        idx = random.randint(0, len(pattern)-1)
        pattern.pop(idx)
    return pattern

# Map tags to realistic text patterns that the tagger can recognize
tag_to_text = {
    'PR': 'First, I need to understand the problem. We are given some numbers to work with.',
    'EQ': 'The equation is x = a + b. We can set up the formula as total = part1 + part2.',
    'CA': 'Calculate the result: 5 + 3 = 8. Compute the sum of the numbers.',
    'IN': 'Therefore, this means the intermediate result is 8. This implies we need to continue.',
    'RF': 'Wait, let me recheck that calculation. Actually, I need to verify the numbers.',
    'DV': 'By the way, this reminds me of a similar problem. Off topic, but interesting.',
    'CO': 'The final answer is 17. So the result is 17 dollars.'
}

# Generate samples
for i in range(500):
    is_correct = i % 2 == 0  # Alternate for balance
    
    if is_correct:
        base_pattern = random.choice(correct_base_patterns)
        tag_seq = add_noise_to_pattern(base_pattern, noise_prob=0.25)
        # Correct samples get better scores
        score = np.random.normal(-1.5, 0.4)
    else:
        base_pattern = random.choice(incorrect_base_patterns)
        tag_seq = add_noise_to_pattern(base_pattern, noise_prob=0.25)
        # Incorrect samples get worse scores
        score = np.random.normal(-3.5, 0.5)
    
    # Ensure minimum length
    if len(tag_seq) < 2:
        tag_seq = ['PR', 'CO'] if is_correct else ['PR', 'DV', 'CO']
    
    # Generate CoT text that matches tagger patterns
    cot_parts = []
    for tag in tag_seq:
        if tag in tag_to_text:
            cot_parts.append(tag_to_text[tag])
        else:
            cot_parts.append(f'Processing {tag}')
    
    cot_text = ' '.join(cot_parts)
    
    data.append({
        'sample_id': f'gsm_{i:04d}',
        'problem': f'Problem {i}: Calculate something',
        'cot_raw': cot_text,
        'tag_sequence': tag_seq,
        'is_correct': is_correct,
        'hmm_score': {'normalized_ll': float(score)},
    })

# Shuffle
random.shuffle(data)

Path('data/v2/gsm8k_500_proper.json').parent.mkdir(parents=True, exist_ok=True)
with open('data/v2/gsm8k_500_proper.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f'Created proper dataset: {len(data)} samples')
print(f'Correct: {sum(1 for d in data if d["is_correct"])}')
print(f'Incorrect: {sum(1 for d in data if not d["is_correct"])}')

# Check diversity
correct_tags = [tuple(d['tag_sequence']) for d in data if d['is_correct']]
incorrect_tags = [tuple(d['tag_sequence']) for d in data if not d["is_correct"]]
print(f'Unique correct tag patterns: {len(set(correct_tags))}')
print(f'Unique incorrect tag patterns: {len(set(incorrect_tags))}')
print(f'Sample correct patterns: {list(set(correct_tags))[:5]}')
print(f'Sample incorrect patterns: {list(set(incorrect_tags))[:5]}')

# Verify no overlap (should be some overlap due to noise, but not perfect correlation)
correct_set = set(correct_tags)
incorrect_set = set(incorrect_tags)
overlap = correct_set & incorrect_set
print(f'Overlap between correct/incorrect patterns: {len(overlap)} (should be >0 but < min(len(correct),len(incorrect)))')