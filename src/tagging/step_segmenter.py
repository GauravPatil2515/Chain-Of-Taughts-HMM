"""
Step Segmenter
Splits Chain-of-Thought text into discrete reasoning steps.
"""

import re
from typing import List


def segment_by_sentences(text: str) -> List[str]:
    """Segment text by sentence boundaries."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short segments
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def segment_by_newlines(text: str) -> List[str]:
    """Segment text by newline characters."""
    lines = text.split('\n')
    return [l.strip() for l in lines if len(l.strip()) > 5]


def segment_by_numbered_steps(text: str) -> List[str]:
    """Segment text by numbered step patterns."""
    # Patterns: "1.", "Step 1:", "First,", etc.
    pattern = r'(?:^|\n)\s*(?:\d+[.):]\s*|Step\s+\d+[.:]\s*|(?:First|Second|Third|Fourth|Fifth|Then|Next|Finally)[,:]?\s*)'
    
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def segment_cot(text: str, method: str = "auto") -> List[str]:
    """
    Segment CoT text into reasoning steps.
    
    Args:
        text: Raw CoT text
        method: Segmentation method - "sentence", "newline", "numbered", "auto"
    
    Returns:
        List of reasoning step strings
    """
    if not text or not text.strip():
        return []
    
    if method == "sentence":
        return segment_by_sentences(text)
    elif method == "newline":
        return segment_by_newlines(text)
    elif method == "numbered":
        return segment_by_numbered_steps(text)
    elif method == "auto":
        # Auto-detect best method
        # Check for numbered patterns
        if re.search(r'(?:^|\n)\s*(?:\d+[.):]|Step\s+\d+[.:])', text, flags=re.IGNORECASE):
            steps = segment_by_numbered_steps(text)
            if len(steps) >= 2:
                return steps
        
        # Check for multiple newlines
        if text.count('\n') >= 2:
            steps = segment_by_newlines(text)
            if len(steps) >= 2:
                return steps
        
        # Fall back to sentence segmentation
        return segment_by_sentences(text)
    else:
        raise ValueError(f"Unknown method: {method}")


def clean_step(step: str) -> str:
    """Clean and normalize a step string."""
    # Remove leading Step numbers or numbers/bullets
    step = re.sub(r'^(?:Step\s+\d+[:\.]\s*|[\d\-\*•]+[.):\s]*)', '', step, flags=re.IGNORECASE)
    # Remove extra whitespace
    step = ' '.join(step.split())
    return step.strip()


def segment_and_clean(text: str, method: str = "auto") -> List[str]:
    """Segment and clean CoT text."""
    steps = segment_cot(text, method)
    return [clean_step(s) for s in steps if clean_step(s)]


if __name__ == "__main__":
    # Test example
    test_cot = """
    First, I need to understand the problem. We have 4 pens at $3 each.
    Step 1: Calculate the cost of pens: 4 × 3 = 12 dollars.
    Step 2: Add the notebook cost: 12 + 5 = 17 dollars.
    Therefore, the total is $17.
    """
    
    print("Auto segmentation:")
    for i, step in enumerate(segment_and_clean(test_cot)):
        print(f"  {i+1}. {step}")
