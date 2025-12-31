# Phase 5: Case Study Analysis
**Date:** 2025-12-27
**Total Cases:** 4

---

## Case 0: Baseline Wins

**ID:** `arc_rule_arc_0050`  
**Dataset:** arc_challenge (science)  
**Correctness:** ✅ Correct  

### Question
```
Suppose 20 g of liquid hydrogen peroxide is heated so it completely breaks down into liquid water an...
```

### Chain of Thought
```
Let me analyze this science question carefully. The question asks about a specific concept. Considering the choices provided: A) more than 20 g becaus...
```

**Ground Truth:** `D`  
**Predicted:** `D`  

### Scores

| Method | Score |
|--------|-------|
| HMM | 0.000000 |
| Combined Baseline | 0.4773 |
| Diversity Baseline | 1.2071 |

### Step Analysis

**Total Steps:** 4  
**Tag Sequence:** IN → PR → IN → CO  

**Tag Distribution:**
- IN: 2
- PR: 1
- CO: 1

**Step-by-Step Breakdown:**

1. **[IN]** (conf: 0.30): Let me analyze this science question carefully.
2. **[PR]** (conf: 0.60): The question asks about a specific concept.
3. **[IN]** (conf: 0.30): Considering the choices provided: A) more than 20 g because of the ..., B) more than 20 g because there a...
4. **[CO]** (conf: 0.60): Based on scientific principles, the correct answer is D.

### Interpretation



**Baseline advantage:** The combined baseline outperforms HMM by 0.4773. This suggests simple heuristics work better for this type of reasoning.

---

## Case 1: Hmm Wins

**ID:** `gsm8k_rule_gsm8k_0066`  
**Dataset:** gsm8k (math)  
**Correctness:** ❌ Incorrect  

### Question
```
There are four schools competing at a basketball tournament. Each school has sent a girlsÃ¢â‚¬â„¢ ba...
```

### Chain of Thought
```
Looking at this problem, I'll calculate quickly. Taking the first two numbers I see. Performing a basic operation. The answer appears to be 5.
```

**Ground Truth:** `48`  
**Predicted:** `5`  

### Scores

| Method | Score |
|--------|-------|
| HMM | 0.000000 |
| Combined Baseline | 0.1039 |
| Diversity Baseline | 0.4024 |

### Step Analysis

**Total Steps:** 4  
**Tag Sequence:** IN → IN → IN → IN  

**Tag Distribution:**
- IN: 4

**Step-by-Step Breakdown:**

1. **[IN]** (conf: 0.30): Looking at this problem, I'll calculate quickly.
2. **[IN]** (conf: 0.30): Taking the first two numbers I see.
3. **[IN]** (conf: 0.30): Performing a basic operation.
4. **[IN]** (conf: 0.30): The answer appears to be 5.

### Interpretation



**HMM advantage:** The HMM outperforms baselines by -0.103864. This shows HMM captures complex sequential patterns that heuristics miss.

---

## Case 2: Agreement

**ID:** `gsm8k_rule_gsm8k_0072`  
**Dataset:** gsm8k (math)  
**Correctness:** ❌ Incorrect  

### Question
```
Tommy is fundraising for his charity by selling brownies for $3 a slice and cheesecakes for $4 a sli...
```

### Chain of Thought
```
Looking at this problem, I'll calculate quickly. Taking the first two numbers I see. Performing a basic operation. The answer appears to be 23.
```

**Ground Truth:** `221`  
**Predicted:** `43`  

### Scores

| Method | Score |
|--------|-------|
| HMM | 0.000000 |
| Combined Baseline | 0.1039 |
| Diversity Baseline | 0.4024 |

### Step Analysis

**Total Steps:** 4  
**Tag Sequence:** IN → IN → IN → IN  

**Tag Distribution:**
- IN: 4

**Step-by-Step Breakdown:**

1. **[IN]** (conf: 0.30): Looking at this problem, I'll calculate quickly.
2. **[IN]** (conf: 0.30): Taking the first two numbers I see.
3. **[IN]** (conf: 0.30): Performing a basic operation.
4. **[IN]** (conf: 0.30): The answer appears to be 23.

### Interpretation



---

## Case 3: Disagreement

**ID:** `arc_rule_arc_0045`  
**Dataset:** arc_challenge (science)  
**Correctness:** ✅ Correct  

### Question
```
The best way to separate salt from water is with the use of
```

### Chain of Thought
```
Let me analyze this science question carefully. The question asks about a specific concept. Considering the choices provided: A) oil...., B) heat.... ...
```

**Ground Truth:** `B`  
**Predicted:** `B`  

### Scores

| Method | Score |
|--------|-------|
| HMM | 0.000000 |
| Combined Baseline | 0.4264 |
| Diversity Baseline | 1.2071 |

### Step Analysis

**Total Steps:** 4  
**Tag Sequence:** IN → PR → IN → CO  

**Tag Distribution:**
- IN: 2
- PR: 1
- CO: 1

**Step-by-Step Breakdown:**

1. **[IN]** (conf: 0.30): Let me analyze this science question carefully.
2. **[PR]** (conf: 0.60): The question asks about a specific concept.
3. **[IN]** (conf: 0.30): Considering the choices provided: A) oil...., B) heat....
4. **[CO]** (conf: 0.60): Based on scientific principles, the correct answer is B.

### Interpretation



---

