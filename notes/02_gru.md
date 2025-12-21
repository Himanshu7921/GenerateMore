# 02: Gated Recurrent Units (GRU)

## Context

Gated Recurrent Units (GRUs) were introduced as a **minimal corrective response** to the structural failures of vanilla RNNs.

Rather than abandoning recurrence, GRUs ask a precise question:

> *Can we preserve the recursive structure of RNNs while giving the model explicit control over what information to keep and what to overwrite?*

GRUs do **not** aim to solve all long-range dependency issues.  
They aim to fix the **most damaging failure mode** of RNNs: *uncontrolled memory decay*.

This file documents:
- the assumptions GRUs introduce,
- what problems they fix (and what they don’t),
- and why even GRUs eventually become insufficient.

---

## Core Assumptions

### 1. Memory Preservation Must Be Explicitly Controlled

GRUs assume that **memory retention cannot be left to implicit dynamics**.

Instead of a single hidden update, GRUs introduce gating:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

This assumes:
- Memory should persist by default
- Updates should occur **only when useful**
- Forgetting must be a learned decision

This is a direct rejection of the RNN compression assumption.

---

### 2. Multiplicative Gates Can Stabilize Gradient Flow

GRUs assume that **additive state updates**, modulated by gates, reduce gradient decay:

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx (1 - z_t)
$$

instead of repeated multiplication by \( W_{hh} \).

This assumes:
- Gates can learn near-identity mappings
- Gradient flow can be preserved by design, not luck

---

### 3. A Single State Vector Is Still Sufficient

GRUs **do not introduce a separate memory cell**.

They assume:
- One state vector can serve as both memory and output
- Selective gating compensates for lack of structural separation

This assumption simplifies the architecture — but limits expressiveness.

---

## What Failed (in RNNs, Fixed by GRUs)

### 1. Uncontrolled Memory Overwrite

RNN update rule:

$$
h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1})
$$

forces **every timestep** to overwrite memory.

GRUs fix this via the **update gate**:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$

allowing:
- near-zero updates when memory should persist
- near-complete overwrite when new information matters

This turns memory decay from an *emergent side effect* into a *learned decision*.

---

### 2. Gradient Collapse Through Time

In RNNs, gradients depend on:

$$
\prod_t W_{hh}
$$

In GRUs, gradients depend primarily on:

$$
\prod_t (1 - z_t)
$$

Since \( z_t \in [0,1] \), the model can learn regimes where:

$$
z_t \approx 0 \Rightarrow h_t \approx h_{t-1}
$$

creating **stable gradient highways** across long spans.

---

### 3. Short-Term Bias in Optimization

RNNs tend to optimize for **immediate loss reduction**, sacrificing long-term signals.

GRUs mitigate this by allowing:
- delayed updates
- selective integration of inputs

This improves retention of rare but important events.

---

## What GRUs Still Fail To Solve

GRUs are **not a final solution**.

They introduce new assumptions — and new limits.

---

### 1. Memory and Output Are Still Entangled

GRUs store memory in the same vector used for output:

$$
h_t \rightarrow \text{used directly for prediction}
$$

This means:
- Internal memory is constantly exposed
- Memory can be corrupted by output pressure
- No protected long-term storage exists

This becomes problematic at scale.

---

### 2. Gates Are Still Learned via Multiplicative Dynamics

While gates stabilize gradients, they rely on:

$$
\sigma(\cdot)
$$

which:
- can saturate
- can become insensitive under large activations
- introduces its own optimization fragility

GRUs reduce gradient issues — they do not eliminate them.

---

### 3. Single Memory Stream Limits Expressiveness

GRUs assume:
- One memory stream is enough
- All temporal dependencies share the same representation

This limits:
- hierarchical memory
- multi-timescale reasoning
- long-term abstraction

---

## Why the Next Model Was Needed

The limitations above motivate **further structural separation**.

Any successor to GRUs must:

---

### 1. Separate Memory From Exposure

Memory must persist **even when not directly influencing output**.

This requires:
- a dedicated memory path
- controlled read/write access

---

### 2. Preserve Gradients Without Relying on Gates Alone

Instead of hoping gates learn identity behavior, the architecture should **guarantee additive flow**:

$$
c_t = c_{t-1} + \text{controlled write}
$$

This is stronger than gate-based mitigation.

---

### 3. Support Longer and More Structured Dependencies

As model depth and scale increase:
- memory must be protected
- interference must be minimized
- optimization pressure must be decoupled from storage

These requirements lead directly to **explicit cell-based memory architectures**.

---

## Summary

GRUs fix the *most catastrophic* failure of RNNs:

> uncontrolled memory decay.

They do so by introducing **learned gating**, which:
- stabilizes gradients
- enables selective memory updates
- significantly improves long-range modeling

However, GRUs still assume:
1. memory and output can share the same representation
2. a single state vector is sufficient
3. gates alone can manage all temporal complexity

These assumptions hold **partially**, but break at scale.

To move further, memory itself must become a **first-class architectural component**.

That step leads to **explicit cell-based recurrent models**.

---

## References to Derivations

### GRU Gate Formulation and Update Dynamics:

![GRU Gate Derivation](../../math/gru/gru_gate_derivation_01.png)  
![GRU Gate Derivation](../../math/gru/gru_gate_derivation_02.png)