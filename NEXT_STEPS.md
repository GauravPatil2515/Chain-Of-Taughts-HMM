# Phase 1 ✅ → Phase 2 🚀 Quick Start

## What You Have Now

✅ **200 samples** in `data/cot_outputs/phase1_merged_200.json`  
✅ **60% correct, 40% incorrect** balance  
✅ **Math + Science** domains  
✅ **All quality checks passed**  

---

## Next Command (Run This Now!)

```bash
python -m src.tagging.rule_based_tagger data/cot_outputs/phase1_merged_200.json data/tagged/phase1_tagged.json
```

**This will:**
- Tag all 200 CoTs with semantic labels (PR, DV, EQ, CA, etc.)
- Take ~2-5 minutes
- Create `data/tagged/phase1_tagged.json`

---

## After Tagging, Run:

```bash
# Verify tagging worked
python -c "import json; d=json.load(open('data/tagged/phase1_tagged.json')); print(f'✅ Tagged {len(d)} samples'); print(f'Sample: {d[0][\"question\"][:50]}...'); print(f'Tags: {d[0].get(\"tags\", [])}') if d else print('❌ No data')"

# Train HMM on correct samples
python -m src.hmm.trainer --input data/tagged/phase1_tagged.json --output models/phase1_hmm.pkl --n-components 5

# Run baseline evaluation
python run_baseline_eval.py
```

---

## Important Notes

⚠️ **Rule-based CoTs** - Good for testing, not for publication  
📊 **200 samples** - Minimum viable, 300+ recommended for paper  
🎯 **Next milestone:** Complete tagging + HMM training today  

---

## Full Documentation

- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - What we accomplished
- [PHASE1_COMPLETION_REPORT.md](reports/PHASE1_COMPLETION_REPORT.md) - Detailed analysis
- [PHASE1_DETAILED_PLAN.md](docs/PHASE1_DETAILED_PLAN.md) - Original plan

---

**Ready? Run the tagging command above! 👆**
