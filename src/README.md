# Source Code: From-Scratch Implementations

This directory contains **clean, minimal, and professional `.py` implementations** of the models discussed throughout the repository.

All implementations here are derived **manually from mathematical formulations**, not copied from high-level framework abstractions.

---

## Purpose of This Directory

The role of `src/` is **not exploration** and **not storytelling**.

It exists to provide:

- readable reference implementations  
- explicit architectural structure  
- reproducible model definitions  
- code suitable for inspection and extension  

The **thinking, derivations, and experimental reasoning** that led to these implementations are intentionally **not duplicated here**.

They live in:

- `notes/` — architectural assumptions, failures, and motivations  
- `math/` — handwritten derivations and gradient analysis  
- `notebooks/` — exploratory implementations with full thought process, diagrams, and narrative  

This directory is the **consolidated outcome** of that process.

---

## Design Philosophy

Implementations in `src/` follow these principles:

- **From first principles**  
  No use of high-level recurrent abstractions (e.g. `nn.RNN`, `nn.GRU`, `nn.LSTM`).

- **Separation of concerns**  
  Model definitions are decoupled from:
  - training loops
  - data loading
  - visualization
  - experimentation logic

- **Explicitness over convenience**  
  Tensor shapes, state transitions, and recurrence mechanics are written explicitly.

- **Stability over cleverness**  
  Code is written to be understandable and verifiable, not compact or optimized prematurely.

---

## Relationship to Other Directories

- **`notes/`**  
  Explains *why* each model exists and *what failed* in previous architectures.

- **`math/`**  
  Provides the mathematical justification for every state update and gating mechanism used here.

- **`notebooks/`**  
  Contains earlier exploratory implementations with full reasoning, diagrams, and step-by-step experimentation.

- **`experiments/`**  
  Uses the implementations here to validate specific architectural claims (e.g. stateful vs stateless inference).

---

## Status

At present, this directory is being populated **incrementally**.

Priority is given to:
1. conceptual correctness  
2. mathematical validation  
3. architectural clarity  

before implementation completeness.

This is intentional.

---

## Intended Audience

This code is written for:
- researchers reviewing implementation correctness
- engineers interested in architectural mechanics
- future self-reference when extending to new domains (e.g. vision, diffusion)

It assumes familiarity with:
- PyTorch tensors and autograd
- linear algebra
- sequence modeling fundamentals

It intentionally avoids:
- training recipes
- hyperparameter tuning
- end-to-end pipelines

Those concerns belong elsewhere.