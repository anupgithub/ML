# Attention Mechanism - Toy Example (Fox Sentence)

## Document Flow
This document walks through a simplified and intuitive explanation of how attention works in Transformer models, using the sentence:

> "The quick brown fox jumps"

We start by understanding the idea of attention and how it results in a weighted combination of previous words. Then we explore how that weighted sum (context vector) leads to a prediction of the next word.

Later in the document, we explain how the model actually **calculates those attention weights** using Q (Query), K (Key), and V (Value) projections. Finally, we walk through what happens during forward and backward passes in training, and how the model gradually learns to improve attention and word prediction.

This markdown summarizes the end-to-end intuition and mechanics behind attention using the toy sentence:

> "The quick brown fox jumps"

---

## What is Attention?

**Attention gives a weight to each word in a sequence when deciding what matters for the current word being processed. ( Don't worry about how its calculated at this point) **

In Transformer terms:
- If you're at word *t*, and you're using self-attention, then:
  - You look at all previous and current words (in unidirectional models like GPT), or **all words in the sentence** (in bidirectional models like BERT).
  - The attention mechanism computes **weights (a.k.a. attention scores)** for each of those words.
- These weights are used to compute a **weighted sum of all the word embeddings** — this gives the model a focused context that captures what’s relevant for the current word.
  - In this toy example, V vectors are directly the **word embeddings**, so the weighted sum effectively becomes a **weighted sum of all the word embeddings**.
 (typically derived from word embeddings).
  - **This weighted sum becomes the 'context vector' for word *t*** — it's the distilled context that the model uses to process or generate word *t*.

### Visualizing It:
For the sentence:
> "The quick brown fox jumps"

When processing **"jumps"**, attention might assign:
- 0.1 weight to "The"
- 0.05 to "quick"
- 0.1 to "brown"
- 0.75 to "fox"

Because **“fox” is most relevant** to understanding or predicting “jumps”.

**Attention gives a weight to each word in a sequence when deciding what matters for the current word being processed.**

When the model is generating or analyzing a word (e.g., "jumps"), it uses attention to decide how much to focus on each previous word (like “fox”, “brown”, etc.). 

---

## 1. **Simplified Attention Weights (Toy Example)**
For the sentence: "The quick brown fox jumps" — we want to predict the next word after "fox" (target: "jumps").

Assume we’re computing attention from the perspective of **“jumps”**, and attending to all previous words.

| Word   | Embedding (2D)<br><sub><i>(Think of this as a list of features or properties of the word — e.g., noun-ness, verb-ness, animateness, etc.)</i></sub> | Attention weights assigned by 'jumps' to each of these previous words. |
|--------|------------------|------------------------|
| The    | [0.1, 0.2]       | 0.10                   |
| quick  | [0.4, 0.1]       | 0.05                   |
| brown  | [0.3, 0.5]       | 0.10                   |
| fox    | [0.8, 0.6]       | 0.75                   |

## 2. **Weighted Sum (Context Vector)**
We compute the context vector for “jumps” by multiplying each embedding by its attention weight:

```python
context = 0.10 * [0.1, 0.2] + 
          0.05 * [0.4, 0.1] + 
          0.10 * [0.3, 0.5] + 
          0.75 * [0.8, 0.6] 
```

Now compute the math step-by-step:

```python
= [0.01, 0.02]         # from "The"
+ [0.02, 0.005]        # from "quick"
+ [0.03, 0.05]         # from "brown"
+ [0.6, 0.45]          # from "fox"
```

Final sum:
```python
= [0.01 + 0.02 + 0.03 + 0.6, 0.02 + 0.005 + 0.05 + 0.45]
= [0.66, 0.525]
```

**This is the context vector for “jumps”**, built using attention to past tokens. It's enriched mostly by "fox", which had the highest attention weight (0.75).
We compute the context vector for “jumps” by multiplying each embedding by its attention weight:

```python
context = 0.10 * [0.1, 0.2] + 
          0.05 * [0.4, 0.1] + 
          0.10 * [0.3, 0.5] + 
          0.75 * [0.8, 0.6] 
```

This gives a 2D context vector that reflects what “jumps” should mean in this context.

---

## 3. **What Does the Context Vector Mean?**
The context vector by itself (like `[0.66, 0.525]`) **has no interpretable meaning** for humans.

But during training, the model **learns to shape these vectors** so they point toward areas in space associated with correct words (e.g., "jumps").

---

## 4. **How It Maps to Word Prediction**
Let’s walk through a **complete example** where the model is given a 3-word input and must predict the next word.

### Input Sequence:
> "The quick brown fox"

The model is now trying to predict the next word. Let’s say the real next word is **"jumps"**, but initially the model might guess something else (like "pizza").

### Step 1: Get Embeddings
Assume the embeddings are:
```python
"The"    : [0.1, 0.2]
"quick"  : [0.4, 0.1]
"brown"  : [0.3, 0.5]
"fox"    : [0.8, 0.6]
```

### Step 2: Compute Attention
Assume we’ve already computed attention weights for the previous words:

| Word   | Embedding       | Attention Weight (α) |
|--------|------------------|------------------------|
| The    | [0.1, 0.2]       | 0.1                    |
| quick  | [0.4, 0.1]       | 0.05                   |
| brown  | [0.3, 0.5]       | 0.1                    |
| fox    | [0.8, 0.6]       | 0.75                   |

### Step 3: Compute Weighted Sum (Context Vector)
```python
context = 0.1 * [0.1, 0.2] + 
          0.05 * [0.4, 0.1] + 
          0.1 * [0.3, 0.5] + 
          0.75 * [0.8, 0.6]
```
```python
= [0.01, 0.02] + [0.02, 0.005] + [0.03, 0.05] + [0.6, 0.45]
= [0.66, 0.525]
```

### Step 4: Output Projection
We pass this context vector through a vocab projection matrix (Wout) to get logits:
```python
Wout = [
  [0.5, 0.2],  # "apple"
  [0.1, 0.6],  # "run"
  [0.7, 0.4],  # "fox"
  [0.2, 0.9],  # "jumps"
  [0.3, 0.8]   # "eats"
]
```
```python
logits = [0.29, 0.32] @ Wout.T
```
Compute dot products:
```python
"apple" = 0.5*0.29 + 0.2*0.32 = 0.145 + 0.064 = 0.209
"run"   = 0.1*0.29 + 0.6*0.32 = 0.029 + 0.192 = 0.221
"fox"   = 0.7*0.29 + 0.4*0.32 = 0.203 + 0.128 = 0.331
"jumps" = 0.2*0.29 + 0.9*0.32 = 0.058 + 0.288 = 0.346
"eats"  = 0.3*0.29 + 0.8*0.32 = 0.087 + 0.256 = 0.343
```

### Step 5: Softmax → Prediction
```python
softmax([0.209, 0.221, 0.331, 0.346, 0.343]) ≈ [0.16, 0.17, 0.21, 0.24, 0.22]
```

**What does softmax do?**
Softmax takes a list of numbers (logits) and turns them into probabilities that sum to 1. It does this by:
1. Exponentiating each number: `exp(xi)`
2. Dividing by the sum of all exponentiated values

Example:
```python
softmax([2.0, 1.0, 0.1])
= [exp(2.0), exp(1.0), exp(0.1)] / sum([exp(2.0), exp(1.0), exp(0.1)])
≈ [7.39, 2.71, 1.10] / 11.2
≈ [0.66, 0.24, 0.10]
```

This means the highest logit (2.0) gets the highest probability (0.66), and smaller values get proportionally lower probabilities.
Predicted word = **"jumps"** (highest score — just above "eats")

### Step 6: Backpropagation
- Ground truth = "jumps"
- Model predicted = "eats"
- Loss is calculated
- Backprop updates:
  - Attention weights
  - Embeddings
  - Wq/Wk/Wv
  - Wout

Over time, this pushes the model to make the context vector point more toward the "jumps" region.

---
### Step 4: Output Projection (Revised and Simplified)
We pass the context vector `[0.66, 0.525]` through a vocab projection matrix (Wout) to get logits:

```python
Wout = [
  [0.5, 0.2],  # "apple"
  [0.1, 0.6],  # "run"
  [0.7, 0.4],  # "fox"
  [0.2, 0.9],  # "jumps"
  [0.3, 0.8]   # "eats"
]
```

Compute dot products:
```python
"apple" = 0.5*0.66 + 0.2*0.525 = 0.33 + 0.105 = 0.435
"run"   = 0.1*0.66 + 0.6*0.525 = 0.066 + 0.315 = 0.381
"fox"   = 0.7*0.66 + 0.4*0.525 = 0.462 + 0.21  = 0.672
"jumps" = 0.2*0.66 + 0.9*0.525 = 0.132 + 0.472 = 0.604
"eats"  = 0.3*0.66 + 0.8*0.525 = 0.198 + 0.42  = 0.618
```

Then apply softmax:
```python
softmax([0.435, 0.381, 0.672, 0.604, 0.618]) ≈ [0.17, 0.16, 0.24, 0.21, 0.22]
```

### Step 5: Prediction
Initially, the model might predict **"eats"**, because its score is slightly higher than "jumps".

After training (i.e., after backpropagation), the weights will adjust so that the score for **"jumps"** becomes higher — making the correct prediction.
The context vector is passed through a **vocab projection layer**:
```python
logits = context @ Wout
```
Then softmax is applied, and the model selects the word with the highest probability.

Initially, the model might predict **"eats"**, but after backpropagation, it adjusts to predict **"jumps"**.

---

## 5. **Role of Query (Q), Key (K), Value (V)**
To compute attention weights:

- **Q** (query): what the word ("jumps") is looking for
- **K** (key): what each word offers
- **V** (value): what each word will contribute if chosen

The scores are calculated as:
```python
score_i = Q · K_i
weight_i = softmax(score_i)
```

Then:
```python
context = sum(weight_i * V_i)
```

---

## 6. **What Are Wq, Wk, Wv?**
These are **learnable projection matrices**:
```python
Q = Wq @ embedding
K = Wk @ embedding
V = Wv @ embedding
```
- They **start with random values**
- Over training, they are updated via backpropagation
- Each learns to emphasize different semantic roles

---

## 7. **How Do Wq, Wk, Wv Converge?**
They converge through training:
- If attention misfires (looks at wrong word), gradients flow into Wq and Wk
- If the content from a word misleads the output, gradients flow into Wv

Each matrix is updated **based on its specific role** in attention computation.

---

## 8. **What Does Weighted Sum Indicate?**
The weighted sum (context vector) is the model’s **summary of relevant past information**, shaped by attention.

It is what the model “thinks” is important for the current word (e.g., “jumps”) based on previous words.

---

## 9. **Forward Pass Overview**
1. Get embeddings for input tokens
2. Compute Q, K, V using Wq, Wk, Wv
3. Compute attention scores using Q · K
4. Apply softmax to get weights (α)
5. Compute context = weighted sum of V
6. Project context to vocab logits → predict next word

---

## 10. **Backward Pass Overview**
1. Compute loss (e.g., cross entropy between "eats" vs. true word "jumps")
2. Backpropagate through:
   - Output projection layer
   - Context vector
   - V vectors → update Wv and embeddings
   - Attention scores → update Q/K → Wq/Wk

Each component learns **based on how much it contributed to the prediction error**.

---

This toy example and breakdown is based on the “fox” scenario we walked through. You can edit or expand this later for multi-head attention, classification, or BERT-style models.
