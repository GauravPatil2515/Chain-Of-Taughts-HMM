import json
from collections import Counter

# Load tagged data
with open("data/tagged/phase1_tagged.json", "r") as f:
    data = json.load(f)

correct_seqs = []
incorrect_seqs = []

for item in data:
    seq_str = " ".join(item['tag_sequence'])
    if item['is_correct']:
        correct_seqs.append(seq_str)
    else:
        incorrect_seqs.append(seq_str)

print(f"Total Correct: {len(correct_seqs)}")
print(f"Total Incorrect: {len(incorrect_seqs)}")

print("\nMost Common Correct Sequences:")
for seq, count in Counter(correct_seqs).most_common(5):
    print(f"  {count}: {seq}")

print("\nMost Common Incorrect Sequences:")
for seq, count in Counter(incorrect_seqs).most_common(5):
    print(f"  {count}: {seq}")

# Check overlap
correct_set = set(correct_seqs)
incorrect_set = set(incorrect_seqs)
overlap = correct_set.intersection(incorrect_set)
print(f"\nOverlap (sequences appearing in both): {len(overlap)}")
if overlap:
    print(f"Example overlap: {list(overlap)[0]}")
