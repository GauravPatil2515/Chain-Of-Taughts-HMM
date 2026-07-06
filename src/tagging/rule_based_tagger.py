"""
Rule-Based Tagger
Tags each CoT step with a semantic role using pattern matching.
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from .step_segmenter import segment_and_clean


# Tag definitions with patterns (priority order)
TAG_PATTERNS = {
    "CO": {  # Conclusion - highest priority
        "description": "Final answer or conclusion",
        "patterns": [
            r"(?:the\s+)?(?:final\s+)?answer\s+is",
            r"final\s+answer[:\s]",
            r"therefore.*(?:is|=)\s*\d+",
            r"in\s+conclusion",
            r"the\s+result\s+is",
            r"so\s+the\s+(?:total|answer|result)",
            r"\\boxed",
            r"answer:\s*",
        ]
    },
    "VE": {  # Verification
        "description": "Checking or validating",
        "patterns": [
            r"let'?s?\s+(?:verify|check|confirm)",
            r"to\s+(?:verify|check|confirm)",
            r"checking[:\s]",
            r"verification[:\s]",
            r"this\s+(?:makes\s+sense|is\s+correct)",
            r"substituting\s+back",
        ]
    },
    "CA": {  # Calculation
        "description": "Arithmetic computation",
        "patterns": [
            r"\d+\s*[×x*]\s*\d+\s*=\s*\d+",
            r"\d+\s*[+]\s*\d+\s*=\s*\d+",
            r"\d+\s*[-−]\s*\d+\s*=\s*\d+",
            r"\d+\s*[÷/]\s*\d+\s*=\s*\d+",
            r"=\s*\d+(?:\.\d+)?(?:\s*(?:dollars?|kg|m|cm|km|hours?|minutes?))?",
            r"calculate\s+",
            r"computed\s+as",
        ]
    },
    "DV": {  # Define Variable
        "description": "Variable definition",
        "patterns": [
            r"let\s+\w+\s*=",
            r"let'?s?\s+(?:call|say|denote|define)",
            r"suppose\s+(?:that\s+)?",
            r"assume\s+(?:that\s+)?",
            r"we\s+(?:can\s+)?define",
        ]
    },
    "EQ": {  # Equation Setup
        "description": "Setting up equation",
        "patterns": [
            r"(?:the\s+)?equation\s+is",
            r"(?:the\s+)?formula\s+is",
            r"we\s+(?:can\s+)?(?:write|set\s+up)",
            r"this\s+gives\s+us",
            r"[a-z]\s*=\s*[a-z\d\s\+\-\*/]+",  # algebraic equation
            r"\\\[.*\\\]",   # LaTeX display math
            r"\$.*\$",       # LaTeX inline math
        ]
    },
    "RF": {  # Recall Fact
        "description": "Retrieving knowledge",
        "patterns": [
            r"we\s+know\s+that",
            r"recall\s+that",
            r"(?:it\s+is\s+)?known\s+that",
            r"by\s+definition",
            r"the\s+(?:formula|rule)\s+(?:for|is)",
            r"according\s+to",
        ]
    },
    "PR": {  # Problem Restatement
        "description": "Restating the problem",
        "patterns": [
            r"(?:we\s+)?need\s+to\s+(?:find|calculate|determine)",
            r"the\s+(?:question|problem)\s+(?:asks|states|says)",
            r"(?:we\s+)?(?:are\s+)?given\s+(?:that)?",
            r"understand(?:ing)?\s+the\s+problem",
            r"let'?s?\s+(?:first\s+)?(?:understand|read)",
            r"i\s+need\s+to\s+(?:find|solve|calculate)",
            r"step\s+\d+[:\.]", # Step headers
        ]
    },
    "IN": {  # Inference (default)
        "description": "Logical inference",
        "patterns": [
            r"therefore",
            r"thus",
            r"hence",
            r"so\s*,",
            r"this\s+means",
            r"it\s+follows\s+that",
            r"which\s+(?:means|implies|gives)",
        ]
    },
}

# Priority order for tagging
TAG_PRIORITY = ["CO", "VE", "CA", "DV", "EQ", "RF", "PR", "IN"]


def tag_step(step: str) -> Tuple[str, float]:
    """
    Tag a single reasoning step.
    
    Returns:
        Tuple of (tag, confidence)
    """
    step_lower = step.lower()
    
    for tag in TAG_PRIORITY:
        patterns = TAG_PATTERNS[tag]["patterns"]
        for pattern in patterns:
            if re.search(pattern, step_lower):
                # Higher confidence if multiple patterns match
                matches = sum(1 for p in patterns if re.search(p, step_lower))
                confidence = min(0.5 + matches * 0.1, 1.0)
                return tag, confidence
    
    # Default to IN (Inference) with low confidence
    return "IN", 0.3


def tag_steps(steps: List[str]) -> List[str]:
    """
    Tag a list of pre-segmented reasoning steps.
    
    Returns:
        List of tags
    """
    return [tag_step(step)[0] for step in steps]


def tag_cot(cot_text: str) -> List[Dict]:
    """
    Segment and tag a full CoT trace.
    
    Returns:
        List of dicts with step text and tag info
    """
    steps = segment_and_clean(cot_text)
    
    tagged_steps = []
    for step in steps:
        tag, confidence = tag_step(step)
        tagged_steps.append({
            "text": step,
            "tag": tag,
            "confidence": confidence,
            "description": TAG_PATTERNS[tag]["description"]
        })
    
    return tagged_steps


def tag_cot_file(input_path: str, output_path: str) -> Dict:
    """Tag all CoTs in a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    tag_counts = {tag: 0 for tag in TAG_PRIORITY}
    
    for item in tqdm(data, desc="Tagging CoTs"):
        cot_text = item.get("cot_raw", "")
        tagged_steps = tag_cot(cot_text)
        
        # Count tags
        for step in tagged_steps:
            tag_counts[step["tag"]] += 1
        
        # Extract tag sequence
        tag_sequence = [s["tag"] for s in tagged_steps]
        
        results.append({
            **item,
            "steps": tagged_steps,
            "tag_sequence": tag_sequence,
            "num_steps": len(tagged_steps),
        })
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Stats
    total_steps = sum(tag_counts.values())
    stats = {
        "total_cots": len(results),
        "total_steps": total_steps,
        "tag_distribution": {
            tag: {"count": count, "percentage": count/total_steps*100 if total_steps > 0 else 0}
            for tag, count in tag_counts.items()
        },
        "avg_steps_per_cot": total_steps / len(results) if results else 0,
    }
    
    print(f"\nTagging complete!")
    print(f"Total CoTs: {stats['total_cots']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Avg steps/CoT: {stats['avg_steps_per_cot']:.1f}")
    print("\nTag distribution:")
    for tag, info in stats["tag_distribution"].items():
        print(f"  {tag}: {info['count']} ({info['percentage']:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Tag CoT steps")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with CoT outputs")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for tagged CoTs")
    
    args = parser.parse_args()
    tag_cot_file(args.input, args.output)


if __name__ == "__main__":
    main()
