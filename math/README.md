# Mathematical Derivations and Architectural Analysis

This directory contains the **mathematical backbone** of the repository.

All files here are **hand-constructed derivations, diagrams, and gradient analyses** created while studying language modeling architectures from first principles.

These are **not copied figures from papers** and **not decorative diagrams**.  
They exist to make **assumptions, failure modes, and architectural fixes explicit** at the mathematical level.

---

## Purpose of This Directory

The goal of this directory is to answer one question:

> *Why does this architecture work — and where does it mathematically break?*

Each subdirectory corresponds to a specific model family and contains:
- state update derivations
- gradient flow analysis
- architectural failure illustrations
- structural fixes introduced by the next model

These artifacts are referenced directly from the `notes/` directory.

---

## How to Read These Files

- Files are intended to be read **alongside the markdown notes**, not in isolation.
- The recommended flow is:
  
  **notes → math → code**

- Mathematical symbols may differ slightly from standard papers, as they reflect my own derivation process rather than canonical notation.

---

## Directory Structure and Contents

### `rnn/` — Recurrent Neural Networks

Contains derivations related to:
- recurrent state update equations
- Backpropagation Through Time (BPTT)
- gradient instability through repeated Jacobians

Key focus:
- why fixed-size hidden states fail
- why long-range gradients vanish or explode

Referenced primarily in:
- `notes/01_rnn.md`

---

### `gru/` — Gated Recurrent Units

Contains gate-level derivations illustrating:
- update gate behavior
- reset gate interaction
- additive vs multiplicative gradient paths

Key focus:
- how gating mitigates uncontrolled memory decay
- why gradient stability improves compared to RNNs

Referenced primarily in:
- `notes/02_gru.md`

---

### `lstm/` — Long Short-Term Memory Networks

Contains detailed analysis of:
- explicit memory cell formulation
- separation of memory and output
- additive gradient flow through the cell state
- architectural limitations of GRUs that motivated LSTMs

Subdirectories:
- `core_problems/` — conceptual memory failures in earlier models
- `gradient_analysis/` — gradient flow behavior through LSTM cells

Key focus:
- why explicit memory is necessary
- how LSTMs structurally preserve long-range dependencies

Referenced primarily in:
- `notes/03_lstm.md`

---

### `attention/` — Attention Mechanism

Contains diagrams and derivations illustrating:
- self-attention mechanics
- query–key–value projections
- attention score normalization
- representation updates
- multi-head attention decomposition
- positional information injection

Key focus:
- abandoning recurrence
- reframing sequence modeling as retrieval
- global interaction vs temporal compression

Referenced primarily in:
- `notes/04_attention.md`

---

### `mlp/` — Feedforward Language Models

Contains foundational derivations for:
- embedding-based MLP language models
- fixed-window context modeling
- limitations of non-recurrent architectures

Used as historical grounding for sequence modeling.

---

## Design Philosophy

These derivations follow one guiding principle:

> *Architectural changes are mathematical necessities, not design preferences.*

Every diagram here exists because:
- an assumption failed
- a gradient collapsed
- memory was overwritten
- or a structural bottleneck emerged

Understanding these failures is more important than memorizing final equations.

---

## Relationship to the Rest of the Repository

- **`notes/`**  
  Conceptual and architectural reasoning that references these derivations.

- **`src/`**  
  From-scratch implementations whose correctness depends on the math shown here.

- **`experiments/`**  
  Empirical validation of claims supported by these derivations.

---

## Status

This directory is treated as **stable**.

New derivations will be added only when:
- a new architecture is studied
- a new failure mode is identified
- or an existing assumption is re-examined

Existing files are not drafts; they represent finalized reasoning at the time of writing.