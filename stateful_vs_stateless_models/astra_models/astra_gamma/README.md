# **Astra-γ:  Stateful vs Stateless GRU Generation**

## **Overview**

Astra-γ is the **largest and most expressive model** in the Astra-GRU family.
With **three recurrent layers** and a **hidden size of 512**, it is explicitly designed to model **long-range character dependencies, speaker continuity, and multi-line dramatic structure** in Shakespearean text.

This README presents a controlled comparison between:

* **Stateless Sampling** — hidden state reset at every generation step
* **Stateful Sampling** — hidden state preserved and propagated across steps

Both experiments use **identical trained weights**.
Only the inference procedure differs.

---

## **Model Configuration**

| Component            | Value                      |
| -------------------- | -------------------------- |
| Model Name           | **Astra-γ**                |
| Architecture         | 3-layer GRU (from scratch) |
| Embedding Dim        | 256                        |
| Hidden Dim           | 512                        |
| Number of Layers     | 3                          |
| Optimizer            | AdamW                      |
| Learning Rate        | 1e-3                       |
| Scheduler            | CosineAnnealingLR          |
| Dropout              | 0.1                        |
| Batch Size           | 64                         |
| Sequence Length      | 128                        |
| Epochs               | 20                         |
| Device               | CUDA                       |
| Trainable Parameters | **4,383,041**              |
| Model Size           | **16.72 MB**               |

---

## **Architecture Summary**

```
embedding → GRULayer → GRULayer → GRULayer → Linear
```

Key properties:

* Custom GRUCell implementation
* Explicit state propagation
* Manual recurrence (no PyTorch RNN black boxes)
* Designed to test **memory scaling behavior**

---

## **Stateless Sampling Results**

> Hidden state is **discarded at every time step**
> Despite training on long sequences, inference collapses memory

---

### **Greedy Sampling**

```
ROMEO: the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the 
```

**Observation**

* Severe repetition
* No accumulation of meaning
* Model behaves like a high-confidence loop generator

---

### **Temperature Sampling (T = 0.8)**

```
KING: theroure
O, f sthe all tid, teef send he tr
Toupourarerrey hean s t shonge stofin; I me s.
Thisar in, kissthamy e and ote, all's dls ceend, stendeajuth n f thirn.
O:

BULBEThoun,
```

**Observation**

* Phonetic structure exists
* Lines drift without narrative anchoring
* Context resets implicitly every step

---

### **Top-K Sampling (k = 30)**

```
FIRST CITIZEN: wis s myoucer fung tllind sthot g.
SUClod.

Gh geay annchano lmerd inrr,
Hvere thad boure t, bleat tobe t y hereras, abut'd pany theny, thanod se, I
RGourgor nthyour,
```

**Observation**

* Vocabulary richness wasted
* Deeper layers provide no benefit without memory
* Output resembles noisy character soup

---

## **Stateful Sampling Results**

> Hidden state is **explicitly preserved across time**

---

### **Greedy Sampling**

```
ROMEO: the sea, and the state of men,
When they shall have a strange one another sense,
Which we have seen the sea to see the crown,
And therefore I will not be a prince's death,
I will not be a present deat
```

**Observation**

* Multi-line coherence
* Repetition replaced by thematic continuation
* Memory persists across sentences

---

### **Temperature Sampling (T = 0.8)**

```
KING: tremble of the people.

CORIOLANUS:
This not an open land
A man of care, let us kill this man,
Dreaming of thy life as the rest be right;
But I can be a ravisher, and revenge it out
On me.

BRUTUS:
Sh
```

**Observation**

* Speaker roles preserved
* Dramatic conflict emerges
* Clause-level logic maintained across lines

---

### **Top-K Sampling (k = 30)**

```
FIRST CITIZEN: that, for a deal of peace!

RICHARD:
What straight else, come. Assemble! my Larcius,
Revolted voices in his wit;
And that the tables there no figures speedy he may give both ye.

KING RICHARD II:
Wher
```

**Observation**

* Scene-like structure
* Natural dialogue transitions
* Strong stylistic fidelity to Shakespeare

---

## **Why Stateless Inference Fails (Mathematical Explanation)**

A GRU is defined as a recurrent state transition:

$$
h_t = f(x_t, h_{t-1})
$$

---

### **Stateless Inference**

In stateless sampling, the hidden state is reset:

$$
h_{t-1} = 0 \quad \forall t
$$

Thus the model computes:

$$
P(x_t \mid x_{t-1})
$$

Regardless of depth or parameter count, this **collapses the GRU into a conditional n-gram model**.

---

### **Stateful Inference**

With memory preserved:

$$
h_t = f(x_t, h_{t-1}), \quad h_0 \neq 0
$$

The model approximates:

$$
P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

This enables:

* Long-range dependency tracking
* Speaker identity persistence
* Narrative continuity
* Dramatic rhythm and structure

---

## **Why Astra-γ Shows the Largest Gap**

With **three recurrent layers**:

* Lower layer → character transitions
* Middle layer → phrase structure
* Upper layer → discourse-level context

Stateless inference **destroys all three layers simultaneously**.

Stateful inference allows Astra-γ to fully exploit its depth and width.

---

## **Key Takeaways**

* Astra-γ without state is **massively underutilized**
* Memory preservation unlocks exponential qualitative gains
* Training already learned the memory rules — inference must respect them
* Bigger RNNs amplify both correctness and mistakes

---

## **Conclusion**

Astra-γ proves that:

> **Scale does not compensate for broken recurrence.**

When used correctly, Astra-γ transitions from incoherent noise to **structured theatrical dialogue** — not by retraining, not by new parameters, but by **preserving the hidden state**.

This experiment conclusively demonstrates that:

> **A recurrent model is only recurrent if its state survives inference.**