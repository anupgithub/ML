# Attention Mechanism - Toy Example (Fox Sentence)

This markdown summarizes the end-to-end intuition and mechanics behind attention using the toy sentence:

> "The quick brown fox jumps"

We'll walk through embeddings, attention weights, weighted sum, word prediction, and how Q/K/V and backpropagation all fit together.

---

## 1. **Simplified Attention Weights (Toy Example)**
For the sentence: "The quick brown fox jumps" — we want to predict the next word after "fox" (target: "jumps").

Assume we’re computing attention from the perspective of **“jumps”**, and attending to all previous words.

| Word   | Embedding (2D)  | Attention Weight (α) |
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

This gives a 2D context vector that reflects what “jumps” should mean in this context.

---

## 3. **What Does the Context Vector Mean?**
The context vector by itself (like `[0.66, 0.525]`) **has no interpretable meaning** for humans.

But during training, the model **learns to shape these vectors** so they point toward areas in space associated with correct words (e.g., "jumps").

---

## 4. **How It Maps to Word Prediction**
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
