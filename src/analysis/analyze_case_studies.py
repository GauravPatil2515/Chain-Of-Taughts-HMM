"""
Phase 5: Deep Analysis - Case Study Selection and Analysis
Selects representative samples and analyzes them in detail.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_results():
    """Load evaluation results and tagged data."""
    with open("experiments/results/phase1_baseline/baseline_comparison.json", "r") as f:
        results = json.load(f)
    
    with open("data/tagged/phase1_tagged.json", "r") as f:
        tagged_data = json.load(f)
    
    return results, tagged_data


def select_case_studies(results: dict, tagged_data: list, n_cases: int = 8) -> Dict[str, List[dict]]:
    """
    Select representative case studies across different categories.
    
    Categories:
    1. High HMM score, correct
    2. Low HMM score, correct
    3. High HMM score, incorrect
    4. Low HMM score, incorrect
    5. High baseline score, low HMM score (baseline wins)
    6. Low baseline score, high HMM score (HMM wins)
    7. Agreement (all methods agree)
    8. Disagreement (methods disagree)
    """
    
    # Extract scores from list of dicts
    all_scores_list = results['all_scores']
    hmm_scores = np.array([item['hmm_score'] for item in all_scores_list])
    combined_scores = np.array([item['combined_score'] for item in all_scores_list])
    diversity_scores = np.array([item['diversity_score'] for item in all_scores_list])
    correctness = np.array([item['is_correct'] for item in tagged_data])
    
    cases = {}
    
    # Category 1: High HMM, Correct
    high_hmm_correct = np.where((hmm_scores > np.percentile(hmm_scores, 75)) & correctness)[0]
    if len(high_hmm_correct) > 0:
        idx = high_hmm_correct[np.argmax(hmm_scores[high_hmm_correct])]
        cases['high_hmm_correct'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx])
        }
    
    # Category 2: Low HMM, Correct
    low_hmm_correct = np.where((hmm_scores < np.percentile(hmm_scores, 25)) & correctness)[0]
    if len(low_hmm_correct) > 0:
        idx = low_hmm_correct[np.argmin(hmm_scores[low_hmm_correct])]
        cases['low_hmm_correct'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx])
        }
    
    # Category 3: High HMM, Incorrect
    high_hmm_incorrect = np.where((hmm_scores > np.percentile(hmm_scores, 75)) & ~correctness)[0]
    if len(high_hmm_incorrect) > 0:
        idx = high_hmm_incorrect[np.argmax(hmm_scores[high_hmm_incorrect])]
        cases['high_hmm_incorrect'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx])
        }
    
    # Category 4: Low HMM, Incorrect
    low_hmm_incorrect = np.where((hmm_scores < np.percentile(hmm_scores, 25)) & ~correctness)[0]
    if len(low_hmm_incorrect) > 0:
        idx = low_hmm_incorrect[np.argmin(hmm_scores[low_hmm_incorrect])]
        cases['low_hmm_incorrect'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx])
        }
    
    # Category 5: Baseline wins (high combined, low HMM)
    baseline_advantage = combined_scores - hmm_scores
    baseline_wins = np.where(baseline_advantage > np.percentile(baseline_advantage, 75))[0]
    if len(baseline_wins) > 0:
        idx = baseline_wins[np.argmax(baseline_advantage[baseline_wins])]
        cases['baseline_wins'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx]),
            'advantage': float(baseline_advantage[idx])
        }
    
    # Category 6: HMM wins (high HMM, low combined)
    hmm_advantage = hmm_scores - combined_scores
    hmm_wins = np.where(hmm_advantage > np.percentile(hmm_advantage, 75))[0]
    if len(hmm_wins) > 0:
        idx = hmm_wins[np.argmax(hmm_advantage[hmm_wins])]
        cases['hmm_wins'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx]),
            'advantage': float(hmm_advantage[idx])
        }
    
    # Category 7: Perfect agreement (all high or all low)
    all_scores = np.vstack([hmm_scores, combined_scores, diversity_scores])
    score_std = np.std(all_scores, axis=0)
    agreement = np.where(score_std < np.percentile(score_std, 25))[0]
    if len(agreement) > 0:
        idx = agreement[0]
        cases['agreement'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx]),
            'score_std': float(score_std[idx])
        }
    
    # Category 8: High disagreement
    disagreement = np.where(score_std > np.percentile(score_std, 75))[0]
    if len(disagreement) > 0:
        idx = disagreement[np.argmax(score_std[disagreement])]
        cases['disagreement'] = {
            'index': int(idx),
            'sample': tagged_data[idx],
            'hmm_score': float(hmm_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'diversity_score': float(diversity_scores[idx]),
            'is_correct': bool(correctness[idx]),
            'score_std': float(score_std[idx])
        }
    
    return cases


def analyze_case(case_name: str, case_data: dict) -> dict:
    """Perform detailed analysis of a single case study."""
    
    sample = case_data['sample']
    
    analysis = {
        'case_name': case_name,
        'id': sample['id'],
        'dataset': sample['dataset'],
        'domain': sample['domain'],
        'question_preview': sample['question'][:100] + "..." if len(sample['question']) > 100 else sample['question'],
        'cot_preview': sample['cot_raw'][:150] + "..." if len(sample['cot_raw']) > 150 else sample['cot_raw'],
        'ground_truth': sample['ground_truth'],
        'predicted_answer': sample['predicted_answer'],
        'is_correct': case_data['is_correct'],
        'scores': {
            'hmm': case_data['hmm_score'],
            'combined': case_data['combined_score'],
            'diversity': case_data['diversity_score']
        },
        'steps': {
            'count': len(sample['steps']),
            'tags': [step['tag'] for step in sample['steps']],
            'tag_distribution': {}
        }
    }
    
    # Tag distribution
    from collections import Counter
    tag_counts = Counter(analysis['steps']['tags'])
    analysis['steps']['tag_distribution'] = dict(tag_counts)
    
    # Step details
    analysis['step_details'] = [
        {
            'text': step['text'],
            'tag': step['tag'],
            'confidence': step['confidence']
        }
        for step in sample['steps']
    ]
    
    return analysis


def generate_case_study_report(cases: dict) -> str:
    """Generate a detailed markdown report of all case studies."""
    
    report = ["# Phase 5: Case Study Analysis\n"]
    report.append(f"**Date:** 2025-12-27\n")
    report.append(f"**Total Cases:** {len(cases)}\n")
    report.append("\n---\n\n")
    
    for case_name, case_data in cases.items():
        analysis = analyze_case(case_name, case_data)
        
        report.append(f"## Case {len([r for r in report if r.startswith('## Case')])}: {case_name.replace('_', ' ').title()}\n\n")
        
        # Basic info
        report.append(f"**ID:** `{analysis['id']}`  \n")
        report.append(f"**Dataset:** {analysis['dataset']} ({analysis['domain']})  \n")
        report.append(f"**Correctness:** {'✅ Correct' if analysis['is_correct'] else '❌ Incorrect'}  \n\n")
        
        # Question
        report.append(f"### Question\n")
        report.append(f"```\n{analysis['question_preview']}\n```\n\n")
        
        # CoT
        report.append(f"### Chain of Thought\n")
        report.append(f"```\n{analysis['cot_preview']}\n```\n\n")
        
        # Answer
        report.append(f"**Ground Truth:** `{analysis['ground_truth']}`  \n")
        report.append(f"**Predicted:** `{analysis['predicted_answer']}`  \n\n")
        
        # Scores
        report.append(f"### Scores\n\n")
        report.append(f"| Method | Score |\n")
        report.append(f"|--------|-------|\n")
        report.append(f"| HMM | {analysis['scores']['hmm']:.6f} |\n")
        report.append(f"| Combined Baseline | {analysis['scores']['combined']:.4f} |\n")
        report.append(f"| Diversity Baseline | {analysis['scores']['diversity']:.4f} |\n\n")
        
        # Steps
        report.append(f"### Step Analysis\n\n")
        report.append(f"**Total Steps:** {analysis['steps']['count']}  \n")
        report.append(f"**Tag Sequence:** {' → '.join(analysis['steps']['tags'])}  \n\n")
        
        report.append(f"**Tag Distribution:**\n")
        for tag, count in analysis['steps']['tag_distribution'].items():
            report.append(f"- {tag}: {count}\n")
        report.append("\n")
        
        # Detailed steps
        report.append(f"**Step-by-Step Breakdown:**\n\n")
        for i, step in enumerate(analysis['step_details'], 1):
            report.append(f"{i}. **[{step['tag']}]** (conf: {step['confidence']:.2f}): {step['text']}\n")
        
        report.append("\n")
        
        # Interpretation
        report.append(f"### Interpretation\n\n")
        
        if 'high_hmm' in case_name:
            report.append(f"This case has a **high HMM score** ({analysis['scores']['hmm']:.6f}). ")
            if analysis['is_correct']:
                report.append("The HMM correctly identifies this as plausible reasoning.")
            else:
                report.append("⚠️ The HMM incorrectly scores this incorrect reasoning highly (false positive).")
        elif 'low_hmm' in case_name:
            report.append(f"This case has a **low HMM score** ({analysis['scores']['hmm']:.6f}). ")
            if not analysis['is_correct']:
                report.append("The HMM correctly identifies this as implausible reasoning.")
            else:
                report.append("⚠️ The HMM incorrectly scores this correct reasoning lowly (false negative).")
        
        if 'baseline_wins' in case_name:
            report.append(f"\n\n**Baseline advantage:** The combined baseline outperforms HMM by {case_data.get('advantage', 0):.4f}. ")
            report.append(f"This suggests simple heuristics work better for this type of reasoning.")
        elif 'hmm_wins' in case_name:
            report.append(f"\n\n**HMM advantage:** The HMM outperforms baselines by {case_data.get('advantage', 0):.6f}. ")
            report.append(f"This shows HMM captures complex sequential patterns that heuristics miss.")
        
        report.append("\n\n---\n\n")
    
    return "".join(report)


def main():
    print("="*70)
    print("Phase 5: Case Study Analysis")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading results and data...")
    results, tagged_data = load_results()
    print(f"   Loaded {len(tagged_data)} samples")
    
    # Select cases
    print("\n[2/4] Selecting representative case studies...")
    cases = select_case_studies(results, tagged_data, n_cases=8)
    print(f"   Selected {len(cases)} case studies")
    for case_name in cases.keys():
        print(f"      - {case_name.replace('_', ' ').title()}")
    
    # Save cases
    print("\n[3/4] Saving case study data...")
    output_dir = Path("experiments/results/phase5_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "case_studies.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)
    print(f"   Saved to: {output_dir / 'case_studies.json'}")
    
    # Generate report
    print("\n[4/4] Generating case study report...")
    report = generate_case_study_report(cases)
    
    with open(output_dir / "CASE_STUDIES_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   Saved to: {output_dir / 'CASE_STUDIES_REPORT.md'}")
    
    print("\n" + "="*70)
    print("Case Study Analysis Complete!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   - case_studies.json (raw data)")
    print(f"   - CASE_STUDIES_REPORT.md (detailed analysis)")
    print(f"\n📊 {len(cases)} representative cases selected and analyzed")


if __name__ == "__main__":
    main()
