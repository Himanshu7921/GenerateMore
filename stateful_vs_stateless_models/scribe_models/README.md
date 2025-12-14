# **Scribe Series:  Architecture & Design**

## **Overview**

The **Scribe** series represents the subword-level counterpart to the Astra character-level models.
While Astra models operate at the character granularity, Scribe models are trained on **SentencePiece-based subword tokens**, enabling higher semantic density per timestep and stronger long-range coherence.

All Scribe models are **GRU language models trained entirely from scratch**, implemented manually using PyTorch primitives (without relying on `nn.GRU`), with full control over gating logic, recurrence, and state propagation. PyTorch is used only for tensor operations and optimization, avoiding manual Python loops where possible for efficiency.

The architectural goal of the Scribe family is to study **how recurrent memory behaves when each timestep represents a semantic unit rather than a single character**.

---

## **Recurrent Formulation (Shared Across All Scribe Models)**

$$
\text{Each Scribe model uses the same gated recurrent update rule. Given a subword embedding }
x_t \in \mathbb{R}^{d_{\text{emb}}} \\
\text{ and previous hidden state }
h_{t-1} \in \mathbb{R}^{d_h},
\text{ the GRU cell computes:}
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
\tilde{h}*t = \tanh!\big(W_h x_t + U_h (r_t \odot h*{t-1})\big)
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

where:

* ($r_{t}$) controls **selective memory reset**,
* ( $z_{t}$ ) controls **information overwrite vs retention**,
* ( $\tilde{h}_{t}$ ) represents a **candidate semantic update**,
* ( $h_{t}$ ) is the updated recurrent state.

Because ( $ x_{t} $ ) is a **subword token**, each update corresponds to a **meaningful semantic transition**, not a single character transition.

---

## **Tokenization and Representation**

$$
\text{Let the subword vocabulary size be } |V| \text{, obtained via SentencePiece.}
\\
\text{Each token index } s_t \in \{1, \dots, |V|\} \text{ is mapped to an embedding:}
$$


$$
x_t = E[s_t], \quad E \in \mathbb{R}^{|V| \times d_{\text{emb}}}
$$

Compared to character-level models:

$$
d_{\text{emb}}^{\text{Scribe}} \gg d_{\text{emb}}^{\text{Astra}}
$$

* Each timestep encodes **morphological or semantic units**
* Fewer timesteps are required to represent long text spans

This increases the **information-per-step**, placing stronger demands on the recurrent memory.

---

## **Model Architecture**

All Scribe models follow the same high-level structure:

$$
\text{Input Tokens}
;\rightarrow;
\text{Embedding Layer}
;\rightarrow;
\text{Stacked GRU Layers}
;\rightarrow;
\text{Linear Projection}
;\rightarrow;
\text{Vocabulary Logits}
$$

Formally, for a stack of ( L ) GRU layers:

$$
h_t^{(0)} = x_t
$$

$$
h_t^{(\ell)} = \text{GRU}^{(\ell)}!\left(h_t^{(\ell-1)}, h_{t-1}^{(\ell)}\right),
\quad \ell = 1,\dots,L
$$

$$
\text{logits}_t = W_o h_t^{(L)} + b_o
$$

Dropout is applied **between recurrent layers** to regularize deep recurrence.

---

## **Scribe Model Variants**

### **Architectural Summary**

| Model        | Layers | Hidden Dim | Embedding Dim | Parameters | Size (MB) | Params (M) | Expected Behavior                               |
| ------------ | ------ | ---------- | ------------- | ---------: | --------: | ---------: | ----------------------------------------------- |
| **Scribe-α** | 2      | 256        | 128           |  1,075,688 |      4.10 |       1.08 | Baseline subword LM; local & mid-range patterns |
| **Scribe-β** | 3      | 512        | 256           |  5,102,056 |     19.46 |       5.10 | Strong phrase-level coherence                   |
| **Scribe-γ** | 4      | 768        | 384           | 14,439,400 |     55.08 |      14.44 | High-quality fluent generation                  |

---

## **Training Configuration**

All Scribe models are trained using the same optimization strategy, with scale-dependent hyperparameters.

### **Optimization Objective**

Given a token sequence $( (s_1, \dots, s_{T}) )$, training minimizes:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(s_{t+1} \mid s_1, \dots, s_t)
$$


where:

$$
P(s_{t+1} \mid s_1, \dots, s_t)
=
\operatorname{Softmax}(\text{logits}_t)
$$


### **Training & Optimization Settings**

| Model    | Optimizer | Learning Rate | Epochs | Scheduler         | Dropout |
| -------- | --------- | ------------- | ------ | ----------------- | ------- |
| Scribe-α | AdamW     | 2e-3          | 10     | CosineAnnealingLR | 0.10    |
| Scribe-β | AdamW     | 1.5e-3        | 15     | CosineAnnealingLR | 0.10    |
| Scribe-γ | AdamW     | 1e-3          | 20     | CosineAnnealingLR | 0.15    |

Weight decay is fixed at **0.01** across all models.

---

## **Design Rationale**

### **Why Subwords + GRU?**

* Subwords reduce sequence length while increasing semantic density.
* GRU gating learns **when to preserve vs overwrite semantic context**.
* Stateful recurrence enables modeling of dependencies spanning **dozens of tokens**, not characters.

Formally, stateful inference allows approximation of:

$$
P(s_t \mid s_1, \dots, s_{t-1})
$$

while stateless inference collapses to:

$$
P(s_t \mid s_{t-1})
$$

making memory preservation **strictly necessary** for meaningful subword modeling.

---

## **Summary**

The Scribe series demonstrates how **recurrent memory scales with semantic granularity**.
As model depth and hidden dimensionality increase, the GRU transitions from capturing local morphology (Scribe-α) to maintaining coherent phrase- and discourse-level structure (Scribe-γ).

These models serve as controlled experimental evidence that **stateful recurrence is not optional** when modeling language at the subword level.