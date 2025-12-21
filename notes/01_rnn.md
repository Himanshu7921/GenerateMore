# 01:  Recurrent Neural Networks (RNN)

## Context

Recurrent Neural Networks (RNNs) represent the first serious attempt to model **sequential data** using neural networks by introducing **stateful computation over time**.

Unlike MLP-based language models, RNNs do not treat context as a fixed window. Instead, they assume that **past information can be compressed into a hidden state and propagated forward**.

This file documents:
- the assumptions RNNs are built on,
- what fundamentally failed,
- and why gated architectures became necessary.

---

## Core Assumptions

### 1. Sequence Can Be Modeled Recursively

RNNs assume that a sequence can be modeled via a recursive state update:

$$
h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

This implies:
- All past information relevant to prediction at time \( t \) is contained in \( h_{t-1} \)
- The sequence has a **Markovian structure in hidden space**, even if the data itself is not Markovian

This is a **strong compression assumption**.

---

### 2. Fixed-Size Hidden State Is Sufficient

RNNs assume:
- A fixed-dimensional vector \( h_t \in \mathbb{R}^d \) can represent arbitrarily long histories
- Increasing model depth or training time compensates for limited memory capacity

This implicitly assumes **no need for explicit memory allocation or selective retention**.

---

### 3. Gradient-Based Learning Can Discover Long-Term Dependencies

Training relies on Backpropagation Through Time (BPTT):

$$
\frac{\partial \mathcal{L}}{\partial h_t}
= \sum_{k=t}^{T}
\left(
\frac{\partial \mathcal{L}}{\partial h_k}
\prod_{i=t+1}^{k}
\frac{\partial h_i}{\partial h_{i-1}}
\right)
$$

This assumes:
- Gradients can flow stably through repeated matrix multiplications
- Optimization dynamics will naturally preserve long-term information

This assumption turns out to be **mathematically fragile**.

---

## What Failed

### 1. Vanishing and Exploding Gradients

The recurrent Jacobian term:

$$
\frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^\top \cdot \text{diag}(\phi'( \cdot ))
$$

is applied **repeatedly** during BPTT.

If:
- $(W_{hh}) < 1$ → gradients vanish
- $(W_{hh}) > 1$ → gradients explode

As sequence length increases:

$$
\left\| \prod_{i} \frac{\partial h_i}{\partial h_{i-1}} \right\|
\rightarrow 0 \quad \text{or} \quad \infty
$$

This makes learning **long-range dependencies statistically improbable**, not just difficult.

---

### 2. Hidden State Becomes a Bottleneck

Because all past information is compressed into \( h_t \):

- Frequent patterns dominate
- Rare but important events are overwritten
- Long-term dependencies are sacrificed for short-term optimization

The hidden state behaves like a **lossy memory buffer**, not a true memory.

---

### 3. Training Loss Is Misleading

RNNs can achieve:
- low training loss
- fluent short-range generation

while still failing catastrophically during long-form inference.

This reveals a **training–inference mismatch**:
- BPTT optimizes local gradients
- Inference requires global memory consistency

---

### 4. No Mechanism for Selective Memory Retention

Vanilla RNNs have:
- no gates
- no explicit control over what to keep or forget

Every timestep applies the same update rule regardless of semantic importance.

This makes **intentional memory preservation impossible**.

---

## Why the Next Model Was Needed

The failures above are not implementation bugs — they are **structural**.

Any successor model must address:

### 1. Gradient Preservation

We need an **additive path** through time so that gradients can flow without repeated multiplication:

$$
c_t = c_{t-1} + \text{controlled update}
$$

instead of:

$$
h_t = f(h_{t-1})
$$

This insight directly motivates gated recurrence.

---

### 2. Explicit Memory Control

A better model must:
- decide what information to retain
- decide what to forget
- protect important signals from being overwritten

This requires **learned gating mechanisms**, not implicit dynamics.

---

### 3. Separation of Memory and Output

RNNs entangle:
- memory storage
- output representation

A more robust architecture must **decouple memory from exposure**, allowing internal state to persist even when not directly visible.

---

## Summary

Vanilla RNNs introduced the correct *idea* — recurrence — but made three fatal assumptions:

1. Fixed-size hidden state is sufficient for arbitrary history
2. Multiplicative dynamics are stable over long horizons
3. Optimization alone can discover memory preservation

These assumptions fail mathematically and empirically.

As a result:
- RNNs excel at short-range patterns
- RNNs fail at long-term dependency modeling
- Scaling depth or data does not fix the core issue

The next architectural step must introduce **explicit, learnable control over information flow**.

That step leads directly to **gated recurrent models**.

---

## References to Derivations

### Backpropagation Through Time derivation: 

  ![BPTT Derivation](../../math/rnn/rnn_bptt_derivation_01.png)
  ![BPTT Derivation](../../math/rnn/rnn_bptt_derivation_02.png)
  ![BPTT Derivation](../../math/rnn/rnn_bptt_derivation_03.png)

---



### Recurrent state update formulation:  

  ![Recurrent state update](../../math/rnn/rnn_derivation_01.png)
  ![Recurrent state update](../../math/rnn/rnn_derivation_02.png)
