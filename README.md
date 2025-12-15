# **GenerateMore: Language Modeling from First Principles**

## **Abstract**

This repository documents my independent, research-oriented exploration of **language modeling**, progressing from **statistical methods** to **deep recurrent neural networks**, all implemented **from scratch**.

The project has two complementary goals:

1. To **reconstruct and understand classical language modeling architectures** (MLP, RNN, GRU, LSTM) directly from their original formulations and research papers.
2. To conduct a **focused experimental study on training–inference mismatch in GRU-based language models**, demonstrating that **stateful inference is a correctness requirement**, not an implementation detail.

All models are implemented manually using PyTorch **only for tensor operations and autograd**, with no reliance on high-level recurrent abstractions.

---

## **1. Motivation**

Modern deep learning frameworks abstract away many details of recurrence and state handling. While this accelerates development, it also obscures **critical modeling assumptions**, particularly the role of hidden-state propagation during inference.

Through implementing language models from first principles, I observed that:

> A recurrent model can be trained correctly, achieve low loss, and still fail catastrophically if its inference-time state handling violates the recurrence assumption.

This repository exists to document that observation rigorously and reproducibly.

---

## **2. Scope of Work**

This project covers **two intertwined research directions**:

### **A. From-Scratch Language Model Implementations**

I implemented and trained multiple families of language models inspired directly by foundational research:

* Statistical n-gram models
* Embedding-based MLP language models (Bengio et al., 2003)
* Vanilla RNNs
* Gated Recurrent Units (GRUs)
* Long Short-Term Memory networks (LSTMs)

Each model family was implemented explicitly to expose:

* recurrence mechanics
* parameterization
* optimization behavior
* representational limits

---

### **B. Stateful vs Stateless Inference in GRUs**

Using my custom GRU implementations, I conducted controlled experiments comparing:

* **Stateless inference** (hidden state reset each step)
* **Stateful inference** (hidden state preserved across steps)

The models, weights, and training procedures are identical; **only inference behavior differs**.

---

## **3. Experimental Design**

All recurrent models were:

* implemented from scratch (no `nn.RNN`, `nn.GRU`, or `nn.LSTM`)
* trained with Backpropagation Through Time
* optimized using AdamW
* evaluated qualitatively via long-form generation

### **Inference Modes**

**Stateless Inference**

```
h_prev = None
logits = model(x_t, h_prev)
```

**Stateful Inference (Correct Usage)**

```
h_prev = None
logits, h_prev = model(x_t, h_prev)
```

This distinction is the **central experimental variable** in the GRU studies.

---

## **4. Model Families**

## **4.1 Astra-GRU (Character-Level Language Models)**

* Dataset: Tiny Shakespeare
* Tokenization: Characters
* Objective: Next-character prediction
* Purpose:

  * study recurrence mechanics
  * analyze memory scaling with depth
  * expose inference-time failure modes

---

### **Astra-α Configuration**

| Component            | Specification   |
| -------------------- | --------------- |
| Model                | Astra-α (GRU)   |
| Embedding Dim        | 64              |
| Hidden Dim           | 128             |
| Number of Layers     | 1               |
| Dropout              | 0.1             |
| Vocabulary           | Character-level |
| Trainable Parameters | **86,913**      |
| Model Size           | **0.33 MB**     |

**Training Configuration**

| Setting         | Value             |
| --------------- | ----------------- |
| Optimizer       | AdamW             |
| Learning Rate   | 3e-3              |
| Weight Decay    | 0.01              |
| Scheduler       | CosineAnnealingLR |
| Batch Size      | 64                |
| Sequence Length | 128               |
| Epochs          | 10                |
| Device          | CUDA              |

**Architecture**

```
Embedding → GRULayer → Linear
```

---

### **Astra-β Configuration**

| Component            | Value                      |
| -------------------- | -------------------------- |
| Model Name           | **Astra-β**                |
| Architecture         | 2-layer GRU (from scratch) |
| Embedding Dim        | 128                        |
| Hidden Dim           | 256                        |
| Dropout              | 0.1                        |
| Epochs               | 15                         |
| Optimizer            | AdamW                      |
| Learning Rate        | 2e-3                       |
| Scheduler            | CosineAnnealingLR          |
| Batch Size           | 64                         |
| Sequence Length      | 128                        |
| Device               | CUDA                       |
| Trainable Parameters | **715,713**                |
| Model Size           | **2.73 MB**                |

**Architecture**

```
Embedding → GRULayer → GRULayer → Linear
```

---

### **Astra-γ Configuration**

| Component            | Value                      |
| -------------------- | -------------------------- |
| Model Name           | **Astra-γ**                |
| Architecture         | 3-layer GRU (from scratch) |
| Embedding Dim        | 256                        |
| Hidden Dim           | 512                        |
| Dropout              | 0.1                        |
| Epochs               | 20                         |
| Optimizer            | AdamW                      |
| Learning Rate        | 1e-3                       |
| Scheduler            | CosineAnnealingLR          |
| Batch Size           | 64                         |
| Sequence Length      | 128                        |
| Device               | CUDA                       |
| Trainable Parameters | **4,383,041**              |
| Model Size           | **16.72 MB**               |

**Architecture**

```
Embedding → GRULayer → GRULayer → GRULayer → Linear
```

---

## **4.2 Scribe-GRU (Subword-Level Language Models)**

* Tokenization: SentencePiece (subword units)
* Objective: Next-token prediction
* Purpose:

  * study semantic coherence
  * analyze long-range dependency modeling
  * amplify inference-time failure modes

Subword modeling makes **state propagation substantially more critical** than character-level modeling.

---

### **Scribe-α Configuration**

| Component            | Value         |
| -------------------- | ------------- |
| Model Name           | **Scribe-α**  |
| Architecture         | 2-layer GRU   |
| Embedding Dim        | 128           |
| Hidden Dim           | 256           |
| Dropout              | 0.1           |
| Epochs               | 10            |
| Learning Rate        | 2e-3          |
| Trainable Parameters | **1,075,688** |
| Model Size           | **4.1 MB**    |

---

### **Scribe-β Configuration**

| Component            | Value         |
| -------------------- | ------------- |
| Model Name           | **Scribe-β**  |
| Architecture         | 3-layer GRU   |
| Embedding Dim        | 256           |
| Hidden Dim           | 512           |
| Dropout              | 0.1           |
| Epochs               | 15            |
| Learning Rate        | 1.5e-3        |
| Trainable Parameters | **5,102,056** |
| Model Size           | **19.5 MB**   |

---

### **Scribe-γ Configuration**

| Component            | Value          |
| -------------------- | -------------- |
| Model Name           | **Scribe-γ**   |
| Architecture         | 4-layer GRU    |
| Embedding Dim        | 384            |
| Hidden Dim           | 768            |
| Dropout              | 0.15           |
| Epochs               | 20             |
| Learning Rate        | 1e-3           |
| Trainable Parameters | **14,439,400** |
| Model Size           | **55.1 MB**    |

---

## **4.3 Leviathan-LSTM (Large-Scale Recurrent Language Model)**

* Dataset: Tiny Shakespeare
* Tokenization: Characters
* Objective: Next-character prediction
* Purpose:

  * study **LSTM gating vs GRU gating** at scale
  * analyze **long-range memory retention** under explicit cell state
  * evaluate **over-capacity recurrent models** on small corpora
  * compare convergence and overfitting behavior against deep GRUs

Leviathan-LSTM represents a **capacity-stress test** for recurrent architectures, pushing LSTM recurrence to ~24M parameters while maintaining correct stateful inference and layer-wise memory separation.

---

### **Leviathan-LSTM Configuration**

| Component            | Value                       |
| -------------------- | --------------------------- |
| Model Name           | **Leviathan-LSTM**          |
| Architecture         | 3-layer LSTM (from scratch) |
| Embedding Dim        | 768                         |
| Hidden Dim           | 1024                        |
| Dropout              | 0.2                         |
| Epochs               | 6                           |
| Optimizer            | AdamW                       |
| Learning Rate        | 1e-3                        |
| Weight Decay         | 0.01                        |
| Scheduler            | CosineAnnealingLR           |
| Batch Size           | 64                          |
| Sequence Length      | 128                         |
| Device               | CUDA                        |
| Trainable Parameters | **24,248,129**              |
| Model Size           | **92.50 MB**                |

---

### **Architecture**

```
Embedding
   ↓
LSTMLayer (Layer 1)
   ↓
LSTMLayer (Layer 2)
   ↓
LSTMLayer (Layer 3)
   ↓
LayerNorm
   ↓
Linear Projection
```

Each LSTM layer maintains **independent hidden and cell states**
$((h_t^{(l)}, c_t^{(l)}))$, ensuring proper hierarchical memory flow.

---

### **Implementation Notes**

* LSTM cell implemented **from scratch** (no `nn.LSTM`)
* Single bias per gate (input-side), differing from PyTorch’s dual-bias design
* Explicit management of:

  * hidden state $(h_{t})$
  * cell state $(c_{t})$
* **No in-place state mutation**, ensuring autograd correctness
* Stateful inference with manual propagation of (h, c) across time

---

### **Training Behavior**

Leviathan-LSTM exhibits **rapid convergence** due to its large memory capacity:

* validation loss reaches minimum within **3 epochs**
* overfitting begins early, consistent with over-parameterized recurrent models
* training was intentionally limited to **6 epochs** to preserve generalization

This behavior contrasts with GRU-γ, which requires more epochs to reach comparable expressiveness.

---

### **Comparative Context**

| Model          | Params (M) | Layers | Memory Type   | Overfitting Onset |
| -------------- | ---------- | ------ | ------------- | ----------------- |
| Astra-γ GRU    | 4.38       | 3      | Hidden only   | Late              |
| Scribe-γ GRU   | 14.44      | 4      | Hidden only   | Medium            |
| Leviathan-LSTM | 24.25      | 3      | Hidden + Cell | Early             |

Leviathan-LSTM confirms that **capacity amplifies both learning speed and memorization risk**, reinforcing the importance of early stopping and inference-time correctness.

---

### **Key Takeaway**

> LSTM cell state enables stronger long-range dependency modeling than GRUs, but large LSTM models on small corpora converge extremely fast and must be trained conservatively.

Leviathan-LSTM serves as a **reference point** for understanding how explicit memory gates scale in recurrent architectures.


## **5. Key Experimental Findings**

1. **Training loss alone cannot validate recurrent models**
2. **A GRU without carried state is functionally non-recurrent**
3. **Correct inference yields larger gains than increased depth or epochs**
4. **Subword models amplify inference-time errors**
5. **Implementation correctness precedes model scaling**

---

## **6. Implementation Notes**

* All models are implemented **from scratch**
* No use of:

  * `nn.RNN`
  * `nn.GRU`
  * `nn.LSTM`
* PyTorch is used only for:

  * tensor operations
  * automatic differentiation
  * optimization
* Hidden-state propagation is **explicit and manually controlled**


## See [Appendix A — Formal Recurrent Model Equations](docs/appendix_equations.md) for the full mathematical formulation.


## **7. References**

* Bengio et al., 2003 — *A Neural Probabilistic Language Model*
* Cho et al., 2014 — *Learning Phrase Representations using RNN Encoder–Decoder*
* Hochreiter & Schmidhuber, 1997 — *Long Short-Term Memory*
* Karpathy, 2022 — *makemore*

---

## **Conclusion**

> In sequence modeling, **correctness of recurrence matters more than architectural scale**.

This repository demonstrates—through direct implementation and controlled experimentation—that **stateful inference is fundamental to recurrent language models**, and that violating this assumption silently invalidates model behavior.
