#!/usr/bin/env python3
"""
Unit tests for the semantic tagger module.
Tests determinism and correctness of tagging logic.
"""

import pytest
import json
from pathlib import Path
from src.tagging.rule_based_tagger import tag_steps, segment_and_clean


class TestStepSegmenter:
    """Tests for step segmentation logic."""

    def test_segment_simple_steps(self):
        """Test basic step segmentation by newlines."""
        text = "Step 1: First step\nStep 2: Second step\nStep 3: Third step"
        steps = segment_and_clean(text)
        assert len(steps) == 3
        assert steps[0] == "First step"
        assert steps[1] == "Second step"
        assert steps[2] == "Third step"

    def test_segment_numbered_steps(self):
        """Test segmentation with numbered format."""
        text = "1. Calculate the sum\n2. Verify the result\n3. Conclude"
        steps = segment_and_clean(text)
        assert len(steps) == 3

    def test_segment_empty_lines_filtered(self):
        """Test that empty lines are filtered out."""
        text = "Step 1: First\n\nStep 2: Second\n\n\nStep 3: Third"
        steps = segment_and_clean(text)
        assert len(steps) == 3


class TestRuleBasedTagger:
    """Tests for rule-based tagging logic."""

    def test_tag_conclusion(self):
        """Test CO tag for conclusion steps."""
        steps = ["Therefore, the answer is 42", "We conclude that x = 5"]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "CO" for t in tags)

    def test_tag_calculation(self):
        """Test CA tag for arithmetic operations."""
        steps = [
            "Calculate 2 + 2 = 4",
            "Multiply 5 by 3 to get 15",
            "The sum is 10 + 20 = 30"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "CA" for t in tags)

    def test_tag_equation(self):
        """Test EQ tag for equation setup."""
        steps = [
            "Set up the equation: x + 5 = 10",
            "Let y = 2x + 3",
            "We have the formula: a^2 + b^2 = c^2"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "EQ" for t in tags)

    def test_tag_problem_restatement(self):
        """Test PR tag for problem restatement."""
        steps = [
            "The problem asks us to find the total",
            "Given: 3 apples and 2 oranges",
            "We need to calculate the sum"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "PR" for t in tags)

    def test_tag_reflection(self):
        """Test RF tag for self-correction."""
        steps = [
            "Wait, let me recheck that",
            "Actually, I made a mistake",
            "Let me reconsider this step"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "RF" for t in tags)

    def test_tag_verification(self):
        """Test VE tag for verification steps."""
        steps = [
            "Verify that 2 + 2 = 4",
            "Check if the answer is correct",
            "Double-check the calculation"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "VE" for t in tags)

    def test_tag_diversion(self):
        """Test DV tag for off-topic content."""
        steps = [
            "By the way, the weather is nice",
            "This reminds me of a story",
            "I like pizza"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "DV" for t in tags)

    def test_tag_inference_default(self):
        """Test IN tag as default for reasoning steps."""
        steps = [
            "This implies that x > 0",
            "From this we can deduce",
            "It follows that the answer must be positive"
        ]
        tags = [tag_steps([s])[0] for s in steps]
        assert all(t == "IN" for t in tags)

    def test_tag_priority_order(self):
        """Test that higher priority tags take precedence."""
        # CO should take priority over IN
        step = "Therefore, we conclude the answer is 5"
        assert tag_steps([step])[0] == "CO"
        
        # CA should take priority over IN
        step = "Calculate 5 + 3 = 8"
        assert tag_steps([step])[0] == "CA"

    def test_deterministic_tagging(self):
        """Test that tagging is deterministic for same input."""
        steps = [
            "First, we calculate 2 + 2 = 4",
            "Then we verify the result",
            "Therefore, the answer is 4"
        ]
        tags1 = tag_steps(steps)
        tags2 = tag_steps(steps)
        assert tags1 == tags2

    def test_empty_step_handling(self):
        """Test handling of empty or whitespace-only steps."""
        tags = tag_steps(["", "   ", "\t\n"])
        assert all(t == "IN" for t in tags)  # defaults to IN


class TestIntegration:
    """Integration tests for the full tagging pipeline."""

    def test_full_pipeline_consistency(self):
        """Test that full pipeline produces consistent results."""
        cot_text = """Step 1: The problem asks us to find the sum of 5 and 3.
Step 2: Calculate 5 + 3 = 8.
Step 3: Verify that 5 + 3 equals 8.
Step 4: Therefore, the answer is 8."""
        
        steps = segment_and_clean(cot_text)
        tags = tag_steps(steps)
        
        expected_tags = ["PR", "CA", "VE", "CO"]
        assert tags == expected_tags

    def test_realistic_gsm8k_example(self):
        """Test with a realistic GSM8K-style reasoning chain."""
        cot_text = """Step 1: We need to find how many apples Sarah has.
Step 2: She starts with 10 apples.
Step 3: She gives away 3 apples to John.
Step 4: Calculate 10 - 3 = 7.
Step 5: Check: 7 + 3 = 10, correct.
Step 6: Therefore, Sarah has 7 apples left."""
        
        steps = segment_and_clean(cot_text)
        tags = tag_steps(steps)
        
        # Verify we get expected pattern
        assert "PR" in tags
        assert "CA" in tags
        assert "VE" in tags
        assert "CO" in tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])