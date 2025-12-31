# Tagging Guide: Semantic Roles for CoT Steps

This document provides detailed guidelines for tagging Chain-of-Thought reasoning steps with semantic roles.

---

## 1. Overview

Each step in a Chain-of-Thought trace is assigned one of **8 semantic tags** representing its role in the reasoning process.

---

## 2. Tag Definitions

### 2.1 `PR` - Problem Restatement

**Definition**: The model rephrases, summarizes, or acknowledges the question.

**Purpose**: Shows the model understands what is being asked.

**Example Patterns**:

- "We need to find..."
- "The question asks..."
- "We are given that..."
- "The problem states..."
- "I need to calculate..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "We need to find the total cost of all items" | `PR` | Restates the goal |
| "The question asks for the number of apples remaining" | `PR` | Acknowledges the question |
| "Given: 5 pens at $3 each" | `PR` | Summarizes given information |

---

### 2.2 `DV` - Define Variable

**Definition**: The model introduces a variable, symbol, or unknown quantity.

**Purpose**: Sets up the mathematical or logical framework.

**Example Patterns**:

- "Let x = ..."
- "Define n as..."
- "Suppose that..."
- "Let's call this..."
- "We'll denote..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "Let x be the number of apples" | `DV` | Defines a variable |
| "Suppose the speed is v km/h" | `DV` | Introduces an unknown |
| "Let's call the total cost T" | `DV` | Names a quantity |

---

### 2.3 `EQ` - Equation Setup

**Definition**: The model constructs an equation, formula, or mathematical relationship.

**Purpose**: Formalizes the problem mathematically.

**Example Patterns**:

- Contains "=" with variables on both sides
- "The equation is..."
- "We can write this as..."
- "This gives us..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "So the equation is: 3x + 5 = 20" | `EQ` | Forms an equation |
| "Total = price × quantity" | `EQ` | Sets up a formula |
| "Distance = Speed × Time" | `EQ` | Recalls a formula to apply |

---

### 2.4 `CA` - Calculation

**Definition**: The model performs explicit arithmetic or computation.

**Purpose**: Executes the mathematical operations.

**Example Patterns**:

- Contains numbers with operators: `+`, `-`, `×`, `÷`, `*`, `/`
- Results in a numeric answer
- "Calculating...", "This equals..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "4 × 3 = 12" | `CA` | Direct arithmetic |
| "20 - 5 = 15" | `CA` | Subtraction operation |
| "12 + 5 = 17, so the total is $17" | `CA` | Calculation with result |

**Detection Rule**:

```python
import re
def is_calculation(step):
    # Pattern: number operator number = number
    pattern = r'\d+\s*[×+\-÷*/]\s*\d+\s*=\s*\d+'
    return bool(re.search(pattern, step))
```

---

### 2.5 `IN` - Inference

**Definition**: The model draws a logical conclusion based on previous steps.

**Purpose**: Connects reasoning steps logically.

**Example Patterns**:

- "Therefore..."
- "Thus..."
- "Hence..."
- "So..."
- "This means..."
- "It follows that..."
- "Which implies..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "Therefore, x = 5" | `IN` | Concludes from previous steps |
| "So he has 7 apples left" | `IN` | Draws a conclusion |
| "This means the train will arrive at 3 PM" | `IN` | Logical deduction |

---

### 2.6 `RF` - Recall Fact

**Definition**: The model retrieves background knowledge, definitions, or known facts.

**Purpose**: Brings in external knowledge needed for reasoning.

**Example Patterns**:

- "We know that..."
- "Recall that..."
- "The formula for..."
- "By definition..."
- "A fact is that..."
- "It is known that..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "We know that the area of a circle is πr²" | `RF` | Recalls a formula |
| "Recall that there are 60 minutes in an hour" | `RF` | Retrieves a fact |
| "By definition, speed = distance/time" | `RF` | States a definition |

---

### 2.7 `VE` - Verification

**Definition**: The model checks, validates, or confirms a result.

**Purpose**: Ensures correctness of intermediate or final results.

**Example Patterns**:

- "Let's verify..."
- "Checking..."
- "To confirm..."
- "We can check that..."
- "Substituting back..."
- "This makes sense because..."

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "Let's verify: 3 × 5 = 15 ✓" | `VE` | Checks calculation |
| "Substituting x=5 back: 3(5)+5 = 20 ✓" | `VE` | Validates equation |
| "This makes sense since he started with more apples" | `VE` | Sanity check |

---

### 2.8 `CO` - Conclusion

**Definition**: The model states the final answer or result.

**Purpose**: Terminates the reasoning and provides the answer.

**Example Patterns**:

- "The answer is..."
- "Final answer:..."
- "Therefore, the result is..."
- "In conclusion..."
- "So the answer is..."
- Often appears at the end of the CoT

**Examples**:

| Step Text | Tag | Reasoning |
|-----------|-----|-----------|
| "The answer is 42" | `CO` | States final answer |
| "Final answer: $17" | `CO` | Explicit conclusion |
| "Therefore, there are 7 apples remaining" | `CO` | Concluding statement |

---

## 3. Tagging Priority Order

When a step matches multiple patterns, use this priority (highest first):

1. `CO` - Conclusion (if contains "answer is", "final answer")
2. `VE` - Verification (if contains "verify", "check", "confirm")
3. `CA` - Calculation (if contains explicit arithmetic)
4. `DV` - Define Variable (if contains "let x =", "define")
5. `EQ` - Equation Setup (if contains equation with "=")
6. `RF` - Recall Fact (if contains "we know", "recall")
7. `PR` - Problem Restatement (if contains "need to find", "given")
8. `IN` - Inference (default for logical statements)

---

## 4. Complete Example

### Input CoT

```
Question: A bookstore sells pens at $3 each. A student buys 4 pens and a notebook for $5. How much does he pay in total?

CoT Response:
First, I need to find the total cost for all items.
The student buys 4 pens at $3 each.
So the cost of pens is 4 × 3 = 12 dollars.
The notebook costs $5.
Adding these together: 12 + 5 = 17.
Therefore, the total amount paid is $17.
```

### Tagged Output

| Step | Tag | Reasoning |
|------|-----|-----------|
| "First, I need to find the total cost for all items" | `PR` | Restates goal |
| "The student buys 4 pens at $3 each" | `RF` | Recalls given info |
| "So the cost of pens is 4 × 3 = 12 dollars" | `CA` | Calculation |
| "The notebook costs $5" | `RF` | Recalls given info |
| "Adding these together: 12 + 5 = 17" | `CA` | Calculation |
| "Therefore, the total amount paid is $17" | `CO` | Final answer |

### Tag Sequence

```
["PR", "RF", "CA", "RF", "CA", "CO"]
```

---

## 5. Edge Cases and Ambiguities

### 5.1 Hybrid Steps

Some steps may contain multiple roles:

**Example**: "We know that speed = distance/time, so v = 100/2 = 50 km/h"

**Resolution**: Split or tag by dominant role. Here, the calculation dominates → `CA`

### 5.2 Missing Steps

If a CoT jumps directly from problem to answer:

**Example**: "The answer is 42"

**Tag Sequence**: `["CO"]` (just one tag)

**Note**: Short sequences may be flagged as anomalous by the HMM.

### 5.3 Repetitive Steps

If the model repeats the same reasoning:

**Example**:

- "4 × 3 = 12"
- "So 4 times 3 is 12"

**Resolution**: Tag both as `CA`. Repetition may indicate lower-quality reasoning.

---

## 6. Annotation Guidelines for Manual Tagging

When manually annotating CoTs:

1. **Read the entire CoT first** to understand context
2. **Segment into steps** (usually by sentence or numbered points)
3. **Apply tags using the priority order** (Section 3)
4. **Document ambiguous cases** for later review
5. **Use inter-annotator agreement** if possible (Cohen's Kappa ≥ 0.7)

---

## 7. Automated Tagging Implementation

See `src/tagging/rule_based_tagger.py` for the full implementation.

```python
# Simplified example
TAG_PATTERNS = {
    "CO": ["answer is", "final answer", "result is", "in conclusion"],
    "VE": ["verify", "check", "confirm", "makes sense"],
    "CA": [r'\d+\s*[×+\-÷*/]\s*\d+'],  # regex
    "DV": ["let x", "let ", "define", "suppose"],
    "EQ": ["equation", "formula", " = "],
    "RF": ["we know", "recall", "it is known"],
    "PR": ["need to find", "question asks", "given that", "we are given"],
    "IN": ["therefore", "thus", "hence", "so", "this means"]
}

def tag_step(step: str) -> str:
    step_lower = step.lower()
    for tag in ["CO", "VE", "CA", "DV", "EQ", "RF", "PR", "IN"]:
        patterns = TAG_PATTERNS[tag]
        for pattern in patterns:
            if pattern.startswith("r'"):  # regex
                if re.search(pattern[2:-1], step):
                    return tag
            elif pattern in step_lower:
                return tag
    return "IN"  # default
```

---

## 8. Quality Metrics for Tagging

| Metric | Target | Description |
|--------|--------|-------------|
| Inter-Annotator Agreement | κ ≥ 0.7 | Cohen's Kappa between annotators |
| Tag Coverage | 100% | Every step gets exactly one tag |
| Pattern Precision | ≥ 85% | Correct tags on pattern-matched steps |
| Default Rate | ≤ 20% | Steps falling to default `IN` tag |

---

## 9. Extending the Tag Set

To add new tags:

1. Define the semantic role clearly
2. Identify distinguishing patterns
3. Determine priority relative to existing tags
4. Update `config/config.yaml`
5. Update `src/tagging/rule_based_tagger.py`
6. Re-test on sample CoTs
