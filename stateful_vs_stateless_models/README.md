# **Astra & Scribe: Stateful vs Stateless GRU Experiments**

## **Overview**

This repository documents a focused experimental study on **recurrent language models**, centered around two internally developed GRU-based model families:

* **Astra Models** — character-level GRU language models
* **Scribe Models** — subword-level (SentencePiece) GRU language models

The primary goal of this project is **not** to introduce new architectures, but to **empirically demonstrate the critical role of stateful inference in recurrent neural networks**.

All models were:

* implemented as **custom GRU architectures**
* trained end-to-end using **PyTorch**
* designed from scratch (no `nn.GRU` usage)
* optimized for performance using PyTorch tensor operations (avoiding manual Python loops)

---

## **Core Experiment Motivation**

Both Astra and Scribe models were trained **twice** under identical training conditions, differing only in **how inference (sampling) was performed**.

### Key Question

> What happens if a GRU is trained correctly, but its hidden state is *not* carried forward during inference?

This repository answers that question concretely.

---

## **Inference Strategies Compared**

### **Method 1 — Stateless Sampling**

* The hidden state (`h_prev`) is **reset to `None`** at every generation step.
* Each forward pass is conditionally independent.
* Despite correct training, the model behaves like a **short-context conditional model** during sampling.

This method intentionally violates the recurrent assumption to observe failure modes.

---

### **Method 2 — Stateful Sampling (Correct GRU Usage)**

* The hidden state is initialized once:

  ```
  h_prev = None
  ```
* Then **explicitly carried forward** across generation steps:

  ```
  logits, h_prev = model(input_ids[:, -1:], h_prev)
  ```
* This allows the model to accumulate and preserve temporal context (memory).

The updated and correct implementation can be found here:

📄 **`7_shakespeare_generator.ipynb`** (root directory)

---

## **Results & Comparisons**

A **side-by-side qualitative comparison** of stateless vs stateful sampling is provided for all model variants.

### Astra Models

* `astra_models/astra_alpha`
* `astra_models/astra_beta`
* `astra_models/astra_gamma`

### Scribe Models

* `scribe_models/scribe_alpha`
* `scribe_models/scribe_beta`
* `scribe_models/scribe_gamma`

Each directory contains:

* identical trained weights
* identical training setup
* **only inference differs**

The differences in coherence, structure, and long-range consistency are substantial and clearly visible.

---

## **Model Families**

### **Astra Models — Character-Level GRU**

* Input granularity: **characters**
* Dataset: Shakespeare corpus
* Objective: next-character prediction
* Designed to study **sequence modeling mechanics**

All Astra models share:

* custom GRU cell implementation
* stacked GRU layers
* CrossEntropy loss
* AdamW optimizer
* Cosine Annealing learning rate schedule

#### **Astra Architecture & Training Summary**

| Model   | Embedding Dim | Hidden Dim | Layers | Epochs | Learning Rate | Dropout | Params    |
| ------- | ------------- | ---------- | ------ | ------ | ------------- | ------- | --------- |
| Astra-α | 64            | 128        | 1      | 10     | 3e-3          | 0.1     | 86,913    |
| Astra-β | 128           | 256        | 2      | 15     | 2e-3          | 0.1     | 715,713   |
| Astra-γ | 256           | 512        | 3      | 20     | 1e-3          | 0.1     | 4,383,041 |

Common settings:

* Batch size: 64
* Sequence length: 128
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Weight decay: 0.01

---

### **Scribe Models — Subword-Level GRU (SentencePiece)**

* Input granularity: **subword tokens**
* Vocabulary: SentencePiece tokenizer
* Objective: next-token prediction
* Designed to study **semantic and lexical coherence**

Compared to Astra, Scribe models:

* use larger embeddings
* use deeper GRU stacks
* model longer semantic dependencies

#### **Scribe Architecture & Training Summary**

| Model    | Embedding Dim | Hidden Dim | Layers | Epochs | Learning Rate | Dropout | Params     |
| -------- | ------------- | ---------- | ------ | ------ | ------------- | ------- | ---------- |
| Scribe-α | 128           | 256        | 2      | 10     | 2e-3          | 0.1     | 1,075,688  |
| Scribe-β | 256           | 512        | 3      | 15     | 1.5e-3        | 0.1     | 5,102,056  |
| Scribe-γ | 384           | 768        | 4      | 20     | 1e-3          | 0.15    | 14,439,400 |

Common settings:

* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Weight decay: 0.01
* Loss: CrossEntropyLoss

---

## **Key Takeaway**

> A GRU trained with Backpropagation Through Time **must** carry its hidden state during inference.
> Otherwise, the model collapses into a stateless conditional generator and discards learned temporal dependencies.

This repository demonstrates that:

* training can appear successful even with incorrect inference
* loss curves alone are insufficient to validate recurrent models
* **stateful inference is not optional — it is fundamental**

---

## **Why This Repository Exists**

This project exists to:

* document a subtle but critical RNN failure mode
* provide concrete evidence through controlled experiments
* serve as a reference for anyone implementing recurrent models from scratch
* reinforce the distinction between **training dynamics** and **inference behavior**

---

## **Lessons Learned**

### **1. Training success does not guarantee correct inference**

The most important lesson from this work is that **a recurrent model can be trained correctly and still be used incorrectly**.

All Astra and Scribe models were trained using Backpropagation Through Time on full sequences, where hidden states flowed across time as intended. Training loss decreased smoothly, and validation performance appeared reasonable.

However, during early experiments, the hidden state was unintentionally **reset at every inference step**, causing a mismatch between training and generation behavior.

This demonstrates that:

* low loss alone is not proof of correct model usage
* inference-time behavior must align with training-time assumptions

---

### **2. A GRU without carried state is not a GRU language model**

Resetting the hidden state during sampling effectively reduces a GRU to a **stateless conditional model**.

In this configuration:

* long-term dependencies cannot accumulate
* the model behaves similarly to an n-gram or shallow feedforward model
* additional training epochs provide diminishing returns

The architecture itself remains recurrent, but its **functional behavior is no longer recurrent**.

---

### **3. Stateful inference can outperform deeper training**

One of the most striking observations in this study is that:

> A single epoch of stateful inference produced higher-quality text than dozens of epochs under stateless inference.

This improvement occurred **without**:

* adding parameters
* changing architecture
* modifying optimization strategy

The gains came solely from **correctly preserving memory during generation**, highlighting how critical inference design is for recurrent models.

---

### **4. Recurrent models must be evaluated qualitatively, not just quantitatively**

Loss curves alone failed to reveal the inference-time issue.

Only through:

* qualitative text inspection
* long-range coherence analysis
* dialogue structure observation

did the problem become evident.

This reinforces the importance of **task-appropriate evaluation** for generative sequence models.

---

### **5. Framework abstractions hide essential mechanics**

High-level APIs often abstract away state handling, making it easy to overlook how memory flows through a model.

By implementing GRU models from scratch, this project exposed:

* the explicit role of hidden state
* the separation between training dynamics and inference usage
* how subtle implementation choices drastically affect behavior

This understanding would be difficult to obtain when relying exclusively on framework-provided recurrent modules.

---

### **6. Correctness precedes scale**

Before increasing:

* model depth
* hidden dimensionality
* training duration
* dataset size

it is essential to ensure that **core modeling assumptions are respected**.

This project shows that:

* correctness of recurrence is more impactful than additional capacity
* scaling a flawed setup only amplifies inefficiency

---

## **Implementation Notes**

* All GRU models are **implemented from scratch**
* PyTorch is used only for:

  * tensor operations
  * automatic differentiation
  * optimization
* No high-level recurrent modules (`nn.GRU`, `nn.LSTM`) are used
* Manual Python loops were avoided where possible for efficiency

---

> This repository serves as a reminder that in sequence modeling, *how* a model is used can matter more than *how large* it is.