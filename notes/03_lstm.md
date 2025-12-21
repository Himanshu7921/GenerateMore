# 03: Long Short-Term Memory (LSTM)

## Context

Long Short-Term Memory (LSTM) networks were introduced to address a problem that **persisted even after gated recurrence**:

> *How can memory be preserved reliably over long horizons **without being constantly overwritten by output demands**?*

While GRUs introduced gating, they still **entangle memory storage with output representation**.  
LSTMs break this entanglement by introducing an **explicit memory cell** with controlled access.

This file documents:
- the assumptions LSTMs introduce,
- what GRUs still fail to solve,
- and why LSTM-style memory separation was necessary.

---

## Core Assumptions

### 1. Memory Must Be Explicit and Protected

LSTMs assume that memory should **not be identical to the output state**.

They introduce a dedicated cell state:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

This assumes:
- Memory should persist independently of output pressure
- Updates should be **additive**, not purely multiplicative
- Long-term storage must be protected from frequent overwrites

This is a **structural departure** from GRUs.

---

### 2. Additive Memory Flow Is Essential for Gradient Stability

Unlike RNNs and GRUs, LSTMs enforce an additive recurrence path:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

If:
$$
f_t \approx 1
$$

then:
- Gradients pass through time **unchanged**
- Vanishing gradients are structurally mitigated

This assumes:
- Gates can learn to maintain near-identity behavior
- Stability is achieved by design, not optimization luck

---

### 3. Memory and Output Serve Different Roles

LSTMs assume:
- Memory storage (\(c_t\)) and output representation (\(h_t\)) must be decoupled

Output is computed as:

$$
h_t = o_t \odot \tanh(c_t)
$$

This allows:
- Memory to persist silently
- Output to expose only a controlled view of memory
- Hierarchical temporal abstraction

---

## What GRUs Failed to Solve (Motivation for LSTM)

### 1. Memory–Output Entanglement

In GRUs:

$$
h_t \rightarrow \text{memory} + \text{output}
$$

This means:
- Memory is constantly visible to the loss
- Optimization pressure corrupts long-term storage
- Scaling depth amplifies interference

LSTMs resolve this by separating:
- **storage** (\(c_t\))
- **exposure** (\(h_t\))

---

### 2. Insufficient Gradient Guarantees at Scale

GRU gradient stability depends on learned gate behavior.

In contrast, LSTMs provide a **guaranteed additive path**:

$$
c_t = c_{t-1} \quad \text{(when } f_t = 1, i_t = 0 \text{)}
$$

This makes long-term retention a **default mode**, not an emergent property.

---

### 3. Limited Multi-Timescale Memory

GRUs assume:
- One state vector can represent all temporal dependencies

LSTMs allow:
- Slow-moving memory in \(c_t\)
- Fast-changing representations in \(h_t\)

This supports **multi-timescale modeling**.

---

## Architectural Mechanics (Structurally)

LSTMs introduce four gates:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1}) \\
f_t &= \sigma(W_f x_t + U_f h_{t-1}) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1}) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1})
\end{aligned}
$$

These gates jointly control:
- what to write
- what to forget
- what to expose

Importantly, **forgetting is explicit**, not implicit.

---

## What LSTMs Still Fail To Solve

LSTMs are powerful — but not perfect.

---

### 1. Increased Parameter and Computational Cost

Each gate introduces:
- additional parameters
- additional nonlinearities
- higher memory and compute requirements

This limits scalability in resource-constrained settings.

---

### 2. Gate Saturation Still Exists

Although gradients through \(c_t\) are stable, gate activations:

$$
\sigma(\cdot)
$$

can still saturate, leading to:
- slow learning
- brittle optimization under poor initialization

---

### 3. Sequential Computation Bottleneck

Like all recurrent models, LSTMs assume:
- strict temporal ordering
- no parallelism across time

This fundamentally limits throughput.

---

## Why Even LSTMs Were Not the End

Despite solving memory preservation, LSTMs still assume:

- sequential computation is acceptable
- memory must be propagated step-by-step
- long-range dependency modeling requires recurrence

As sequence lengths and datasets grew, these assumptions became untenable.

The next step required:
- removing recurrence
- enabling direct access to all context
- trading temporal compression for content-based retrieval

---

## Summary

LSTMs introduce the **most important correction** in recurrent modeling:

> memory must be explicit, additive, and protected.

They fix:
- uncontrolled memory decay
- gradient instability
- memory–output interference

However, they do so at the cost of:
- increased complexity
- sequential computation
- limited parallelism

These trade-offs motivated a shift away from recurrence entirely.

---

## References to Derivations

### Understanding Core Limitations in GRUs (Motivation):

![GRU Limitations](../../math/lstm/discussing_core_limitations_in_gru_01.png)  
![GRU Limitations](../../math/lstm/discussing_core_limitations_in_gru_02.png)

---

### LSTM Gate Equations and State Update:

![LSTM Equations](../../math/lstm/lstm_equations.png)  
![LSTM Derivation](../../math/lstm/lstm_derivation.png)

---

### Gradient Flow Analysis in LSTM Cell:

![Gradient Flow 1](../../math/lstm/gradient_analysis/gradient_flow_analysis_lstm_01.png)  
![Gradient Flow 2](../../math/lstm/gradient_analysis/gradient_flow_analysis_lstm_02.png)  
![Gradient Flow 3](../../math/lstm/gradient_analysis/gradient_flow_analysis_lstm_03.png)

---

### Core Memory Problems and Architectural Fixes:

![Core Problem](../../math/lstm/understanding_core_problem.png)  
![Proposed Fix](../../math/lstm/proposing_core_fixes_in_gru.png)
