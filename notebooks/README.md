# Notebooks: Exploratory Implementations and Architectural Reasoning

This directory contains **exploratory, from-scratch Jupyter notebooks** used to *develop*, *reason about*, and *validate* the architectures studied in this repository.

These notebooks are **not polished code artifacts**.  
They are **thinking spaces** where mathematical derivations, diagrams, commentary, and implementation evolve together.

They intentionally preserve the **full reasoning process**.

---

## Purpose of This Directory

The notebooks serve four primary roles:

1. **Architectural Exploration**  
   Implementing models step-by-step to understand their internal mechanics.

2. **Mathematical Validation**  
   Translating equations and derivations directly into executable code.

3. **Storytelling and Commentary**  
   Preserving the *why* behind every design decision, not just the final result.

4. **Failure Analysis**  
   Observing where assumptions break during training and inference.

This directory represents the **developmental stage** of understanding — before consolidation into clean implementations.

---

## Relationship to Other Directories

- **`notes/`**  
  Contains the *finalized architectural reasoning* distilled from these notebooks.

- **`math/`**  
  Holds the formal derivations and diagrams referenced and tested here.

- **`src/`**  
  Contains the clean, professional `.py` implementations extracted from these notebooks after validation.

- **`experiments/`**  
  Uses the finalized implementations to run controlled studies.

In short:

> **Notebooks = thinking**  
> **Notes = reasoning**  
> **Math = evidence**  
> **Src = truth**

---

## Reading Order (Recommended)

The notebooks are roughly ordered to reflect **conceptual progression**:

1. Statistical and MLP-based language models  
2. Vanilla RNNs and BPTT  
3. GRU architectures (from scratch and framework comparison)  
4. LSTM architectures and limitations  
5. Architectural motivation beyond recurrence  
6. Transformer implementation

They are not meant to be read end-to-end, but as **contextual companions** to the notes.

---

## Design Philosophy

These notebooks deliberately prioritize:

- clarity over conciseness  
- explicit math over abstraction  
- commentary over optimization  
- understanding over performance  

They may:
- repeat equations
- contain intermediate variables
- include explanatory text blocks
- show multiple failed attempts

This is intentional.

---

## What These Notebooks Are Not

- Production-ready pipelines  
- Reusable libraries  
- Hyperparameter-optimized benchmarks  
- Minimal implementations  

Those concerns are addressed elsewhere.

---

## Intended Audience

These notebooks are written primarily for:
- myself (to preserve reasoning over time)
- researchers reviewing architectural understanding
- engineers interested in *how* models are derived, not just used

They assume:
- familiarity with PyTorch
- comfort with linear algebra and gradients
- patience to read reasoning, not just results

---

## Status

This directory is considered **historical and stable**.

While notebooks may be added during exploration of new architectures, existing notebooks are not refactored into clean code here.

Instead, validated ideas are promoted to `src/`.