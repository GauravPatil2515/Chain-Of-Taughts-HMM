"""
Semantic Tagger Module for Colab/Kaggle
"""

import re
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_DESCRIPTIONS = {
    "PR": "Problem restatement",
    "DV": "Diversion/aside",
    "EQ": "Equation setup",
    "CA": "Calculation",
    "IN": "Inference/derivation",
    "RF": "Reflection/correction",
    "VE": "Verification",
    "CO": "Conclusion"
}

TAG_PATTERNS = {
    "CO": [r"\b(final\s+)?answer\s+is\b", r"\btherefore.*\b\d+\b", r"\bthe\s+(result|answer)\s+is\b", r"\\boxed\{"],
    "VE": [r"\b(verify|checking|check|confirm|double-check)\b", r"\bthis\s+(makes\s+sense|is\s+correct)\b"],
    "PR": [r"\b(need\s+to\s+(find|calculate|determine))\b", r"\b(given\s+because)\b", r"\b(step\s+\d+[:\.])\b"],
    "EQ": [r"\b(equation\s+is|formula\s+is)\b", r"[a-z]\s*=\s*[a-z\d\s\+\-\*/]+", r"\$.*\$"],
    "CA": [r"\d+\s*[+×x*]\s*\d+\s*=\s*\d+", r"\d+\s*[-−]\s*\d+\s*=\s*\d+", r"(calculate|compute|multiply|add|subtract)\b"],
    "DV": [r"\b(by\s+the\s+way|reminds\s+me|off[- ]topic|tangent)\b", r"\b(weather|pizza)\b"],
    "RF": [r"\b(wait|recheck|reconsider|actually|correction|mistake)\b", r"\b(hold\s+on|oops)\b"],
    "IN": [r"\b(therefore|thus|hence|so\s*,)\b", r"\b(this\s+means|it\s+follows\s+that)\b"],
}


class SemanticTagger:
    """Rule-based + optional neural tagger for CoT steps."""
    
    def __init__(self, use_neural: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.use_neural = use_neural and SBERT_AVAILABLE
        self.model = None
        
        if self.use_neural:
            self.model = SentenceTransformer(model_name)
    
    def tag_step(self, step: str) -> Tuple[str, float]:
        """Tag a single reasoning step."""
        step_lower = step.lower().strip()
        
        for tag, patterns in TAG_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, step_lower, re.IGNORECASE):
                    matches = sum(1 for p in patterns if re.search(p, step_lower, re.IGNORECASE))
                    confidence = min(0.5 + matches * 0.15, 1.0)
                    return tag, confidence
        
        return "IN", 0.3
    
    def tag_steps(self, steps: List[str]) -> List[str]:
        """Tag multiple steps."""
        return [self.tag_step(step)[0] for step in steps]
    
    def segment(self, cot_text: str) -> List[str]:
        """Segment CoT text into steps."""
        steps = []
        current = []
        
        for line in cot_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.endswith('.') and len(line.split()) < 20:
                if current:
                    steps.append(' '.join(current))
                current = [line[:-1]]
            elif line.endswith(';') or line.endswith(','):
                current.append(line[:-1])
            else:
                current.append(line)
        
        if current:
            steps.append(' '.join(current))
        
        return steps if steps else [cot_text]
    
    def tag_cot(self, cot_text: str) -> List[Dict]:
        """Tag complete CoT text."""
        steps_text = self.segment(cot_text)
        results = []
        
        for step in steps_text:
            tag, conf = self.tag_step(step)
            results.append({
                'text': step,
                'tag': tag,
                'confidence': conf,
                'description': TAG_DESCRIPTIONS.get(tag, 'Unknown')
            })
        
        return results
    
    def tag_sequence(self, cot_text: str) -> List[str]:
        """Return just the tag sequence."""
        return [t['tag'] for t in self.tag_cot(cot_text)]


def tag_cot_text(cot_text: str) -> List[str]:
    """Simple function to tag CoT text."""
    tagger = SemanticTagger()
    return tagger.tag_sequence(cot_text)


if __name__ == "__main__":
    tagger = SemanticTagger()
    
    test_cot = """
    John has 5 apples. He gives away 2 apples. 5 - 2 = 3. So the answer is 3.
    """
    
    steps = tagger.segment(test_cot.strip())
    print(f"Steps: {steps}")
    
    tags = tagger.tag_steps(steps)
    print(f"Tags: {tags}")
    
    for tag, desc in zip(tags, TAG_DESCRIPTIONS.keys()):
        print(f"  {desc}: {tag}")