# Notes: Architectural Reasoning Log

This directory contains my **conceptual and mathematical reasoning notes** while studying language modeling architectures from first principles.

These notes are **not tutorials** and **not summaries of papers**.  
They document *why* each architectural transition was necessary, focusing on:

- modeling assumptions
- structural limitations
- failure cases
- and the motivation for the next model

The goal is to make the **evolution of architectures inevitable**, not historical.

---

## How to Read These Notes

Each file follows the same structure:

1. **Context**  
   Why this model exists and what problem it attempts to solve.

2. **Core Assumptions**  
   Explicit assumptions about:
   - memory
   - optimization
   - data
   - computation

3. **What Failed**  
   Structural or mathematical limitations, not implementation bugs.

4. **Why the Next Model Was Needed**  
   The precise architectural gap that motivated the next design.

5. **Mathematical References**  
   Linked derivations and diagrams stored in the `math/` directory.

These notes should be read **in order**.

---

## Reading Order

### 01 — Recurrent Neural Networks (RNN)
`01_rnn.md`

- Introduces recurrence
- Shows why fixed-size hidden states fail
- Analyzes gradient instability via BPTT
- Establishes the memory compression problem

---

### 02 — Gated Recurrent Units (GRU)
`02_gru.md`

- Introduces learned gating
- Explains how GRUs mitigate memory decay
- Analyzes gradient flow improvement
- Shows why gating alone is insufficient at scale

---

### 03 — Long Short-Term Memory (LSTM)
`03_lstm.md`

- Introduces explicit memory cells
- Separates memory storage from output exposure
- Explains additive gradient flow
- Analyzes scaling behavior and remaining bottlenecks

---

### 04 — Attention Mechanism
`04_attention.md`

- Abandons recurrence entirely
- Reframes sequence modeling as information retrieval
- Explains global interaction and parallelism
- Identifies new assumptions and limitations

---

### 05 — Transformer (Attention + Computation)
`05_transformer.md`

- Explains why attention alone is insufficient
- Introduces depth, nonlinearity, and residual structure
- Shows how computation is layered on top of retrieval
- Establishes the modern sequence modeling paradigm

---

## Relationship to the Rest of the Repository

- **`math/`**  
  Contains handwritten derivations, gradient analyses, and architectural diagrams referenced in these notes.

- **`src/`**  
  Contains from-scratch implementations corresponding to the concepts discussed here.

- **`experiments/`**  
  Contains controlled experiments validating specific claims made in these notes (e.g. stateful vs stateless inference).

- **`notebooks/`**  
  Used only for visualization and exploratory analysis, not for core implementations.

---

## Design Philosophy

These notes follow one guiding principle:

> *Architectures are responses to failure, not inventions.*

Every model discussed here exists because the previous one made assumptions that eventually broke.

Understanding those assumptions is more important than memorizing equations.

---

## Intended Audience

These notes are written for:
- myself (as a long-term reference)
- researchers reviewing my work
- engineers who care about **why models work**, not just how to run them

They assume familiarity with:
- linear algebra
- gradient-based optimization
- basic neural network concepts

They intentionally avoid:
- API-level explanations
- framework-specific details
- training recipes