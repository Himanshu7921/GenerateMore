# GenerateMore: Language Modeling from First Principles

## Abstract

This repository documents my independent, research-oriented study of **language modeling architectures**, reconstructed **from first principles**.

The work traces the architectural evolution from:

> **statistical models → MLPs → RNNs → GRUs → LSTMs → Attention → Transformers**

with every transition motivated by **explicit assumptions, failure cases, and mathematical necessity** rather than framework convenience.

All core models are implemented **from scratch** using PyTorch only for tensor operations and automatic differentiation.  
No high-level recurrent or attention abstractions are used.

---

## Motivation

Modern deep-learning frameworks hide critical architectural assumptions behind high-level APIs.  
While this accelerates development, it obscures **why models work**, **where they fail**, and **what correctness actually means**.

While implementing sequence models manually, I observed a recurring pattern:

> A model can train successfully, achieve low loss, and still be **structurally incorrect** at inference time.

In particular, recurrent models can silently fail when **hidden-state propagation assumptions are violated**, even if training appears stable.

This repository exists to document those observations **rigorously and reproducibly**, using math, code, and controlled experiments.

---

## Scope of Work

This project covers **three tightly coupled dimensions**:

---

### A. From-Scratch Architectural Implementations

I implemented the following model families directly from their original formulations:

- Statistical n-gram language models  
- Embedding-based MLP language models (Bengio et al., 2003)  
- Vanilla RNNs with explicit BPTT  
- Gated Recurrent Units (GRUs)  
- Long Short-Term Memory networks (LSTMs)  
- Self-Attention and Multi-Head Attention  
- Transformer encoder architectures  

Each implementation exposes:

- state update mechanics  
- memory assumptions  
- gradient flow behavior  
- representational limits  

All finalized implementations live in `src/`.

---

### B. Architectural Reasoning and Mathematical Validation

For every model family, I explicitly documented:

- modeling assumptions  
- structural failure modes  
- why the next architecture was necessary  

This reasoning is captured in:

- `notes/` — architectural analysis and evolution  
- `math/` — handwritten derivations, gradient analysis, and diagrams  

The notes are intended to be read **sequentially**, forming a coherent research narrative.

---

### C. Controlled Experimental Studies

Using the from-scratch implementations, I conducted focused experiments on:

- training vs inference mismatch in recurrent models  
- stateful vs stateless inference behavior  
- scaling depth vs preserving correctness  
- character-level vs subword-level modeling  

Experiments are isolated in `experiments/` and designed to test **one assumption at a time**.

---

## Repository Structure (How to Navigate)
- notes/ → architectural assumptions, failures, motivations
- math/ → derivations and gradient analysis referenced by notes
- notebooks/ → exploratory implementations with full reasoning and storytelling
- src/ → clean, professional .py implementations (final artifacts)
- experiments/ → controlled empirical studies
- artifacts/ → trained model checkpoints (excluded from version control)


**Recommended reading order:**

1. `notes/`
2. `math/`
3. `src/`
4. `experiments/`
5. `notebooks/` (for historical exploration)

---

## Key Model Families

### Astra-GRU (Character-Level Language Models)

Purpose:
- analyze recurrence mechanics
- study memory scaling with depth
- expose inference-time failure modes

Models:
- Astra-α (1-layer GRU)
- Astra-β (2-layer GRU)
- Astra-γ (3-layer GRU)

---

### Scribe-GRU (Subword-Level Language Models)

Purpose:
- analyze semantic coherence
- study long-range dependency modeling
- amplify inference-time errors

Subword modeling makes **correct state propagation essential**, not optional.

---

### Leviathan-LSTM (Large-Scale Recurrent Model)

Purpose:
- compare LSTM vs GRU gating at scale
- study explicit cell-state memory retention
- observe over-capacity behavior on small corpora

This model serves as a **stress test** for recurrent architectures.

---

## Key Experimental Findings

1. **Training loss alone cannot validate recurrent models**
2. **A GRU without carried state is functionally non-recurrent**
3. **Correct inference matters more than architectural depth**
4. **Subword modeling amplifies state-handling errors**
5. **Implementation correctness precedes model scaling**

---

## Implementation Principles

- No use of `nn.RNN`, `nn.GRU`, or `nn.LSTM`
- Explicit hidden-state and cell-state management
- No in-place mutation of recurrent state
- Separation of:
  - model definition
  - training logic
  - experimentation
  - visualization

The implementations in `src/` are **consolidated results**, not exploratory code.

---

## References

- Bengio et al., 2003 — *A Neural Probabilistic Language Model*  
- Cho et al., 2014 — *Learning Phrase Representations using RNN Encoder–Decoder*  
- Hochreiter & Schmidhuber, 1997 — *Long Short-Term Memory*  
- Vaswani et al., 2017 — *Attention Is All You Need*  
- Karpathy, 2022 — *makemore*

---

## Conclusion

> In sequence modeling, **architectural correctness matters more than scale**.

This repository demonstrates — through derivation, implementation, and controlled experimentation — that violating modeling assumptions can silently invalidate results, even when training appears successful.

Understanding **why architectures evolved** is essential to using them correctly.