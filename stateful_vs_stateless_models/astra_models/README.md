# **Astra Models: Architecture and Design**

The **Astra family** consists of three GRU-based language models — **Astra-α**, **Astra-β**, and **Astra-γ** — designed to study how **capacity, depth, and recurrence interact with memory preservation** in character-level language modeling.

All Astra models share the **same core design philosophy** and differ only in **scale** (embedding size, hidden size, and number of recurrent layers).

Each model is implemented as a **custom GRU from scratch using PyTorch**, avoiding framework-level recurrent abstractions to retain full control over:

* gate equations,
* state propagation,
* and inference-time memory handling.

---

## **Common Design Principles (Across All Astra Models)**

### **Recurrent Core**

All Astra variants use a **stacked GRU architecture**, where each GRU layer consists of a manually implemented `GRUCell` with:

* update gate ( z_t )
* reset gate ( r_t )
* candidate state ( \tilde{h}_t )

The recurrence follows the standard GRU update:

$
h_t = (1 - z_t), h_{t-1} + z_t, \tilde{h}_t
$

This formulation allows **partial memory overwrite**, enabling controlled retention and update of historical context.

---

### **Layered Recurrence**

Each Astra model stacks multiple `GRULayer`s:

* the **first layer** consumes token embeddings,
* higher layers consume the hidden states of the layer below.

This creates a **hierarchical temporal representation**:

* lower layers focus on short-range character transitions,
* higher layers aggregate longer-range contextual patterns.

---

### **Embedding and Output Projection**

* Input tokens are mapped via a learnable **embedding layer**
* Final hidden states are projected into vocabulary space using a **linear decoder**

$
\text{logits}_t = W h_t + b
$

This projection is applied **at every time step**, enabling autoregressive next-character prediction.

---

### **Training Setup (Shared)**

All Astra models are trained under identical procedural assumptions:

* Optimizer: **AdamW**
* Scheduler: **CosineAnnealingLR**
* Loss: **Cross-Entropy**
* Batch Size: **64**
* Sequence Length: **128**
* Dropout applied **between recurrent layers**

This ensures that observed differences arise from **architecture scale**, not training discrepancies.

---

## **Astra-α — Baseline Recurrent Model**

### **Design Goal**

Astra-α serves as the **minimal recurrent baseline**, intended to:

* validate correctness of GRU equations,
* establish a reference for memory behavior,
* highlight failure modes under stateless inference.

### **Architecture**

| Component            | Value  |
| -------------------- | ------ |
| Embedding Dimension  | 64     |
| Hidden Dimension     | 128    |
| GRU Layers           | 1      |
| Dropout              | 0.1    |
| Trainable Parameters | 86,913 |

### **Design Characteristics**

* Single-layer recurrence
* Limited representational capacity
* Highly sensitive to inference-time state handling
* Ideal for isolating **conceptual correctness**

---

## **Astra-β — Intermediate Capacity Model**

### **Design Goal**

Astra-β is designed to capture **mid-range dependencies** and richer stylistic patterns by increasing both **depth and width**.

### **Architecture**

| Component            | Value   |
| -------------------- | ------- |
| Embedding Dimension  | 128     |
| Hidden Dimension     | 256     |
| GRU Layers           | 2       |
| Dropout              | 0.1     |
| Trainable Parameters | 715,713 |

### **Design Characteristics**

* Two stacked recurrent layers
* Improved abstraction hierarchy
* Stronger separation between local and contextual representations
* Memory handling becomes increasingly critical at this scale

---

## **Astra-γ — High-Capacity Recurrent Model**

### **Design Goal**

Astra-γ is explicitly designed to test **long-range memory utilization** and the limits of recurrent modeling at character level.

### **Architecture**

| Component            | Value     |
| -------------------- | --------- |
| Embedding Dimension  | 256       |
| Hidden Dimension     | 512       |
| GRU Layers           | 3         |
| Dropout              | 0.1       |
| Trainable Parameters | 4,383,041 |

### **Design Characteristics**

* Deep recurrent stack
* Large hidden state enables persistent context
* Capable of modeling dialogue structure and multi-line continuity
* Most sensitive to stateless inference failures
* Fully exposes the cost of discarding memory

---

## **Scaling Behavior Across Astra Models**

| Model   | Depth   | Hidden Size | Capacity | Memory Sensitivity |
| ------- | ------- | ----------- | -------- | ------------------ |
| Astra-α | Shallow | Small       | Low      | Moderate           |
| Astra-β | Medium  | Medium      | Moderate | High               |
| Astra-γ | Deep    | Large       | High     | Extreme            |

As model capacity increases, **correct state propagation becomes exponentially more important**.
Astra-γ demonstrates that scale cannot compensate for broken recurrence.

---

## **Summary**

The Astra family is intentionally constructed as a **controlled scaling study**:

* same equations,
* same training setup,
* same dataset,
* different capacity.

This design makes Astra an ideal testbed for understanding:

* recurrence vs depth,
* memory preservation,
* inference-time failure modes,
* and why **stateful inference is not optional for recurrent models**.