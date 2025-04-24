# 📘 Refresher: Chain Rule Intuition (12th-grade style)

The chain rule says:

> If a quantity A depends on B, and B depends on C, then the rate of change of A with respect to C is:
>
> \( rac{dA}{dC} = rac{dA}{dB} \cdot rac{dB}{dC} \)

## 👟 Real-world example

- Suppose your **height** depends on **weight**:
  \( 	ext{height} = 2 \cdot 	ext{weight} \)
- And your **shoe size** depends on **height**:
  \( 	ext{shoe\_size} = 0.25 \cdot 	ext{height} \)

Now, what's the rate of change of shoe size with respect to weight?

We use the chain rule:

\( rac{d(	ext{shoe\_size})}{d(	ext{weight})} = rac{d(	ext{shoe\_size})}{d(	ext{height})} \cdot rac{d(	ext{height})}{d(	ext{weight})} = 0.25 \cdot 2 = 0.5 \)

So, even though there's no direct formula from weight to shoe size, we can trace it through height. And if we actually substitute the expressions and resolve the composition directly (\( 	ext{shoe\_size} = 0.25 \cdot (2 \cdot 	ext{weight}) = 0.5 \cdot 	ext{weight} \)), the derivative comes out to be the same (0.5), which confirms the chain rule's correctness.

\[
\frac{d(\text{shoe\_size})}{d(\text{weight})} = \frac{d(0.5 \cdot \text{weight})}{d(\text{weight})} = 0.5
\]

---

# 🧐 Minimal Neural Network Example: Gradient Descent using Chain Rule

## Network
- Input: \(x = 2\)
- Weight1: \(w_1 = 3\)
- Weight2: \(w_2 = 4\)
- Ground truth: \(y_{	ext{true}} = 10\)
- Loss function: \(L = (	ext{prediction} - y)^2\)

## Forward pass
- Layer 1 output: \(a_1 = w_1 \cdot x = 3 \cdot 2 = 6\)
- Layer 2 output: \(a_2 = w_2 \cdot a_1 = 4 \cdot 6 = 24\)
- Loss: \(L = (24 - 10)^2 = 196\)

## Backward pass (backpropagation)

### Gradient for w2 (last layer)
- \( \frac{dL}{da_2} = 2 \cdot (a_2 - y) = 2 \cdot (24 - 10) = 28 \)
- We are calculating how the loss changes with respect to \(w_2\) (last layer) and \(w_1\) (first layer), so we can apply gradient descent updates to both.
- \( \frac{da_2}{dw_2} = a_1 = 6 \)
- \( \frac{dL}{dw_2} = 28 \cdot 6 = 168 \)

### Gradient for w1 (first layer)
- \( \frac{dL}{da_2} = 28 \)
- \( \frac{da_2}{da_1} = w_2 = 4 \)
- \( \frac{da_1}{dw_1} = x = 2 \)
- \( \frac{dL}{dw_1} = 28 \cdot 4 \cdot 2 = 224 \)

## Gradient Descent Updates
- Learning rate \( \eta = 0.01 \)
- \( w_2 \leftarrow w_2 - 0.01 \cdot 168 = 2.32 \)
- \( w_1 \leftarrow w_1 - 0.01 \cdot 224 = 0.76 \)

## 💡 Intuition
- Each layer “passes back” how sensitive it is to change
- Earlier layers get more complex expressions (chain rule!)
- It’s like measuring how changing early knobs affects the final result