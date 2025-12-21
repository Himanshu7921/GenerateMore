# **Astra-α: Stateless vs Stateful GRU Inference**

## **Overview**

Astra-α is the smallest model in the **Astra-GRU family**, trained on Shakespeare at the **character level** using a GRU implemented **from scratch in PyTorch** (manual cell equations, no `nn.GRU`), with PyTorch used only for tensor operations and optimization.

This experiment isolates **one variable only**:

> **How inference behavior changes when the GRU hidden state is *not* preserved vs when it *is preserved***.

The **architecture, weights, training procedure, and dataset are identical** in both cases.
Only the **sampling strategy differs**.

---

## **Model Architecture & Training Setup**

### **Architecture Summary**

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

### **Training Configuration**

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

---

## **Inference Modes**

### **1. Stateless Sampling (Incorrect for RNNs)**

During sampling, the hidden state is **discarded at every step**:

```python
h_prev = None
logits = model(x_t, h_prev)
```

This forces the GRU to behave as if **no prior context exists beyond the current token**.

---

### **2. Stateful Sampling (Correct)**

During sampling, the hidden state is **explicitly carried forward**:

```python
h_prev = None
logits, h_prev = model(x_t, h_prev)
```

This allows the GRU to accumulate information across time steps.

---

## **Qualitative Results**

### **Greedy Sampling**

#### Stateless

```
ROMEO: the the the the the the the the the the the the the the the the the the ...
```

#### Stateful

```
ROMEO: the son, the son, the son, the son, the son, the son, the son, ...
```

**Observation:**
Both collapse under greedy decoding (expected), but the **stateful version preserves phrase-level consistency**, while the stateless version collapses immediately.

---

### **Temperature Sampling (T = 0.8)**

#### Stateless

```
KING: lind se alivearalatare' chorineeres ary, withing hande lar wligisas...
```

Characteristics:

* Severe phonetic noise
* No sentence-level coherence
* Abrupt token transitions

#### Stateful

```
KING: deed, thou, or to him of thy brother who letwear the torness is the mocting...
```

Characteristics:

* Recognizable syntactic structure
* Clause-level continuity
* Shakespeare-like rhythm emerging

---

### **Top-K Sampling (k = 30)**

#### Stateless

```
FIRST CITIZEN: wiserakend mplanenttham fre?
I: I'MBlentoor...
```

* High randomness
* Weak dialogue structure
* Poor long-range consistency

#### Stateful

```
FIRST CITIZEN: when 'I deed, if not pray of the panciendou' go's;
The king up me fear...
```

* Dialogue formatting preserved
* Multi-line coherence
* Dramatically improved readability

---

## **Mathematical Explanation**

### **What a GRU Is Supposed to Compute**

A GRU defines a **state transition function**:

$$
h_t = f(x_t, h_{t-1})
$$

The output distribution is:

$$
P(x_{t+1} \mid h_t)
$$

---

### **What Happened in Stateless Inference**

By resetting the hidden state at every step:

$$
h_{t-1} = 0 \quad \forall t
$$

the model effectively computes:

$$
h_t = f(x_t, 0)
$$

which implies:

$$
P(x_{t+1} \mid x_t)
$$

This is **mathematically equivalent to a first-order Markov (bigram-like) model**, regardless of how the model was trained.

> The GRU *learned* long-range dependencies, but inference **collapsed them**.

---

### **What Happens in Stateful Inference**

When the hidden state is preserved:

$$
h_t = f(x_t, h_{t-1})
$$

the model represents:

$$
h_t = f(x_t, x_{t-1}, x_{t-2}, \dots)
$$

leading to:

$$
P(x_{t+1} \mid x_1, x_2, \dots, x_t)
$$

This is the **intended behavior of a recurrent language model**.

---

## **Why Training Still “Worked” Before**

During training, the model was unrolled over full sequences using **Backpropagation Through Time (BPTT)**:

$$
\mathcal{L} = \sum_{t=1}^{T} \ell(x_{t+1}, P(x_{t+1} \mid h_t))
$$

Hidden states were **correctly propagated during training**, so the GRU **did learn memory dynamics**.

The failure occurred **only at inference**, where the learned state was silently discarded.

---

## **Key Takeaways**

* A GRU **without preserved hidden state is not a recurrent model**
* Stateless sampling reduces a GRU to a conditional n-gram model
* Passing `h_prev` during inference unlocks the full expressive power **without changing parameters**
* One line of code restored **orders of magnitude** more effective context modeling

---

## **Conclusion**

Astra-α demonstrates that **memory is not optional** in recurrent models.

The dramatic quality jump between stateless and stateful inference is not due to:

* more parameters
* longer training
* better optimization

It is purely the result of **respecting the mathematical definition of recurrence**.
