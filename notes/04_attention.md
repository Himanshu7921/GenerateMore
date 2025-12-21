# 04: Attention Mechanism

## Context

Attention marks a **structural break** from recurrent modeling.

While LSTMs solved the problem of *memory preservation*, they retained a deeper assumption that eventually became the bottleneck:

> *Information must be processed sequentially and compressed through time.*

Attention rejects this assumption entirely.

Instead of **propagating memory**, attention performs **direct information retrieval** over the entire context.

This file documents:
- the assumptions attention introduces,
- what recurrence fundamentally fails to scale,
- and why abandoning recurrence became necessary.

---

## Core Assumptions

### 1. Sequence Modeling Can Be Reframed as Retrieval

Attention assumes that modeling a sequence does **not** require step-by-step state propagation.

Instead, it assumes:

> Relevant information can be retrieved directly from past representations.

This reframes sequence modeling from:
- *temporal compression* → *content-based lookup*

Formally, each token representation attends to all others:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

This assumes:
- Dependencies are **not inherently sequential**
- Order can be injected separately
- Global access is more important than temporal continuity

---

### 2. Similarity Is a Sufficient Proxy for Relevance

Attention assumes that relevance can be approximated via **vector similarity**.

Dot-product attention relies on:

$$
QK^\top
$$

as a measure of alignment.

This assumes:
- Learned embedding space encodes semantic relationships
- Similar vectors correspond to relevant dependencies

This is a **strong inductive bias**.

---

### 3. Memory Does Not Need to Be Persistent

Unlike recurrent models, attention assumes:
- Memory does not need to persist across steps
- Representations can be recomputed at each layer

There is **no state carried forward**.

All context is re-evaluated at every layer.

---

## What Recurrent Models Failed to Solve

### 1. Sequential Computation Bottleneck

RNNs, GRUs, and LSTMs all assume:

$$
h_t \text{ must be computed before } h_{t+1}
$$

This enforces:
- strict temporal dependency
- no parallelism across time

As datasets and sequence lengths grew, this became **computationally prohibitive**.

---

### 2. Compression Inevitability

Even with explicit memory cells, recurrent models must **compress history**:

$$
\text{history} \rightarrow c_t \rightarrow h_t
$$

This introduces:
- information loss
- interference between unrelated dependencies
- pressure to prioritize short-term signals

Scaling does not remove this bottleneck.

---

### 3. Long-Range Dependency Degradation

Recurrent models rely on:
- repeated state updates
- gate behavior
- optimization stability

Even LSTMs degrade when:
- sequences become very long
- dependencies are sparse
- memory must persist over hundreds or thousands of steps

---

## Architectural Shift Introduced by Attention

Attention replaces **memory propagation** with **representation interaction**.

---

### 1. Linear Projections as Role Separation

Input representations are projected into:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

This assumes:
- A single representation can serve multiple roles
- Relevance (Q,K) and content (V) should be decoupled

This replaces gated control with **learned geometric alignment**.

---

### 2. Normalized Global Interaction

Attention scores are normalized via softmax:

$$
\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}
{\sum_j \exp(q_i \cdot k_j / \sqrt{d_k})}
$$

This enforces:
- competition between tokens
- sparse focus
- bounded contribution

---

### 3. Representation Update via Weighted Aggregation

Each token representation is updated as:

$$
h_i' = \sum_j \alpha_{ij} v_j
$$

This assumes:
- New representations can be formed via **mixtures**
- No explicit state update is required

---

## What Attention Still Fails To Solve

Attention removes recurrence — but introduces new limits.

---

### 1. No Intrinsic Notion of Order

Attention is **permutation-invariant**.

Without external injection:

$$
\text{Attention}(X) = \text{Attention}(\text{permute}(X))
$$

This requires **explicit positional information** to recover sequence order.

---

### 2. Quadratic Complexity

Global interaction requires:

$$
O(n^2)
$$

memory and computation with sequence length.

This becomes prohibitive for long contexts.

---

### 3. Attention Is Not Computation

Attention performs **routing**, not algorithmic processing.

It cannot:
- count
- compare magnitudes
- execute step-by-step logic

It only **reweights existing representations**.

---

## Why Attention Was Necessary Despite These Limits

Despite its weaknesses, attention solved **three unsolved problems**:

1. Removed sequential computation
2. Eliminated memory compression
3. Enabled direct long-range interaction

This trade-off favored **scalability and parallelism** over explicit memory.

---

## Summary

Attention represents a **paradigm shift**:

- from memory propagation → information retrieval
- from recurrence → global interaction
- from temporal compression → representational mixing

It abandons the idea that:
> *history must be summarized*

and replaces it with:
> *history can be queried directly*

This shift enables scale — but sacrifices algorithmic structure.

---

## References to Derivations

### Core Self-Attention Mechanics:

![Self-Attention](../../math/attention/self_attention_mechanism.png)  
![Linear Projections](../../math/attention/linear_projection_in_self_attention.png)  
![Attention Scores](../../math/attention/attention_scores.png)  
![Representation Update](../../math/attention/updation_of_representation.png)

---

### Multi-Head Attention and Output Composition:

![Multi-Head Attention](../../math/attention/MHA_mechanism.png)  
![Multi-Head Derivation](../../math/attention/multi_head_attention_derivation.png)  
![Output Projection](../../math/attention/output_projections_in_MHA.png)

---

### Positional Information and Normalization:

![Positional Encoding](../../math/attention/positional_information_injection.png)  
![Normalization](../../math/attention/normalization.png)

---

### Architectural Overview:

![Core Architecture](../../math/attention/core_architecture.png)
