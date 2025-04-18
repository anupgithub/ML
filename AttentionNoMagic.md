# No-Magic Explanation of Attention Score Calculation (Q · K)

When we talk about "attention weights assigned by 'jumps' to each of the previous words," we often introduce labels like **Query (Q)** and **Key (K)**. But these are just names — the actual mechanism doesn't inherently understand what a "query" or a "key" is.

Instead, what's happening is this:

---

## 🔢 Learnable Vector Projections

Each word in a sentence is represented by an **embedding vector** — a learned, fixed-size list of numbers (e.g., `[0.3, 0.8]`) that captures properties of the word.

To perform attention, we apply two **separate learnable matrices** to each embedding:

```python
Q = Wq @ embedding   # Treating the embedding as a column vector
K = Wk @ embedding
```

- `Wq` and `Wk` are weight matrices with **random values at first**
- Over training, they are updated through **gradient descent**

These projections create two new vector spaces:
- One for comparing from the perspective of the current word (Q)
- One for being compared against (K)

---

## ⚙️ The Actual Score Calculation

### The Math (Concrete Example)
You have a vector `Q_jumps` (produced by `Wq @ embedding_jumps`)

You have a set of vectors `[K_The, K_quick, K_brown, K_fox]` (each from `Wk @ embedding_word`)

You compute:

```python
scores = [Q_jumps · K_The,
          Q_jumps · K_quick,
          Q_jumps · K_brown,
          Q_jumps · K_fox]
```

These scores have no built-in semantics — they just reflect how aligned these learned vector projections are.

✅ The truth:  
While we call them “query” and “key,” they’re just two separate projections of the input embeddings.

There is:
- No “real query”
- No “real key”
- Just a learned direction and comparison function


To determine how much one word should "attend to" another, we compute the **dot product** of the two projected vectors:

```python
score = dot(Q_jumps, K_fox)
```

This score reflects how **aligned** the two vectors are — in other words, how much the model thinks the current word should be influenced by the other word.

But again, there's **no built-in semantics** here. The model doesn't "know":
- Which word is a subject or object
- Which word is acting on which

It just sees:
> “Here's a vector. Here's another. Take their dot product.”

---

## 🎯 Why Does This Work?

It works because of **training pressure**. Over time:
- The model learns to adjust `Wq` and `Wk`
- So that the dot products (i.e., attention scores) are **high** when a word should be attended to
- And **low** when it shouldn't

That pressure comes from the final prediction task — e.g., if the model gets the next word wrong, it adjusts every piece of the path that led to that prediction.

So if `Q_jumps` pointed too much at `K_quick` instead of `K_fox`, gradients flow to reduce that alignment next time.

---

## 🧠 The Bigger Picture

This setup doesn't work because we told the model what Q and K mean. It works because:
- Q and K are **used in different parts of the attention computation**
- Gradients flow differently through each
- That causes their learned projections to **naturally specialize**

So you can think of Q and K as:
> “Two differently transformed versions of the same input embedding — one used to compare, one to be compared.”

Their alignment becomes the attention score.

And that’s it — no magic. Just math and training loops.
