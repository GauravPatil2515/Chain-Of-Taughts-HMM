import json

with open("data/tagged/real_cot_tagged.json", "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
correct_count = sum(1 for d in data if d.get("is_correct"))
print(f"Total Correct: {correct_count}")

print("\nSample Audits:")
for i, d in enumerate(data[:10]):
    print(f"ID: {d.get('id')} | GT: {d.get('ground_truth')} | Pred: {d.get('predicted_answer')} | Correct: {d.get('is_correct')}")
