# **Astra-β: Stateful vs Stateless GRU Generation**

## **Overview**

Astra-β is the **medium-capacity model** in the Astra-GRU family.
It extends Astra-α by introducing a **two-layer recurrent stack** with significantly larger hidden representations, enabling stronger modeling of **mid-range and long-range character dependencies**.

This README documents a **critical experimental comparison**:

* **Stateless Sampling** (hidden state reset at every step)
* **Stateful Sampling** (hidden state carried across generation steps)

Both experiments use the **same trained weights**.
The only difference is whether the recurrent memory is preserved during inference.

---

## **Model Configuration**

| Component            | Value                      |
| -------------------- | -------------------------- |
| Model Name           | **Astra-β**                |
| Architecture         | 2-layer GRU (from scratch) |
| Embedding Dim        | 128                        |
| Hidden Dim           | 256                        |
| Number of Layers     | 2                          |
| Optimizer            | AdamW                      |
| Learning Rate        | 2e-3                       |
| Scheduler            | CosineAnnealingLR          |
| Dropout              | 0.1                        |
| Batch Size           | 64                         |
| Sequence Length      | 128                        |
| Epochs               | 15                         |
| Device               | CUDA                       |
| Trainable Parameters | **715,713**                |
| Model Size           | **2.73 MB**                |

---

## **Architecture Summary**

```
embedding → GRULayer → GRULayer → Linear
```

* Custom GRUCell implementation
* Manual recurrence and hidden-state control
* No PyTorch RNN black boxes (only Linear ops for speed)

---

## **Stateless Sampling Results**

> Hidden state **discarded at every step**
> Equivalent to resetting memory during inference

### **Greedy Sampling**

```
ROMEO: the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the 
```

**Observation**
Complete collapse into high-probability loops — a hallmark of **memoryless generation**.

---

### **Temperature Sampling (T = 0.8)**

```
KING: t wane wint wosome athe wry. s, we wer
NGONTh
And o bute t t at th erwh ad
Thansis womanen, ocore coomelelling fove y,
Wee were thes ather.
CAngoumin t falvethe ourthesaroucid youthar,
```

**Observation**

* Some phonetic structure
* Frequent incoherence
* No sustained narrative or speaker consistency

---

### **Top-K Sampling (k = 30)**

```
FIRST CITIZEN: t, CHat.
DUCONI man, irmearant carulangmirs din m t ane my, m.
I ndomice hie ing, alath s wange n;
G?
Jul: ts; we whechis hevished, beve meang a werthe fotindurs his ish CHis:
Ak m.
Tond wheshale atha
```

**Observation**

* Formatting partially preserved
* Still locally noisy
* Context does not accumulate meaningfully

---

## **Stateful Sampling Results**

> Hidden state **explicitly preserved across time**

### **Greedy Sampling**

```
ROMEO: the state, and the state,
And the state of the state of the state,
And the state of the state of the state,
And the state of the state of the state,
And the state of the state of the state,
And the st
```

**Observation**

* Repetition still present (expected with greedy decoding)
* Phrase-level coherence persists across lines

---

### **Temperature Sampling (T = 0.8)**

```
KING: the fatle of this:
But my lord, you have send for his son to Dare you well me so sir, that the princess to be.

ANGELO:
How!
In him as they say, the city of the measure as honours of justice
By some c
```

**Observation**

* Clear sentence structure
* Speaker transitions respected
* Dramatic rhythm emerges

---

### **Top-K Sampling (k = 30)**

```
FIRST CITIZEN: possibatary, but if the mother revenge,
That my arms open laugh with wretched me dull bosom; and where your eyes
Why he was doned above, fair dance will have me to bid thee Dermanacle, inman: yet, and
```

**Observation**

* Strong theatrical cadence
* Context maintained across multiple clauses
* Invented words follow Shakespearean morphology

---

## **Why Stateless Inference Fails (Mathematical View)**

A GRU is defined by the recurrence:

$$
h_t = f(x_t, h_{t-1})
$$

### **Stateless Sampling**

During stateless inference:

$$
h_{t-1} = 0 \quad \forall t
$$

So generation collapses to:

$$
P(x_t \mid x_{t-1})
$$

This is effectively a **conditional n-gram model**, regardless of how powerful the GRU is.

---

### **Stateful Sampling**

When hidden state is preserved:

$$
h_t = f(x_t, h_{t-1}), \quad h_0 \neq 0
$$

The model approximates:

$$
P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

This enables:

* accumulation of context
* speaker consistency
* narrative structure
* stylistic continuity

---

## **Why Astra-β Shows a Larger Gap Than Astra-α**

With **two recurrent layers**:

* Lower layer captures local character transitions
* Upper layer integrates longer-range abstractions

When memory is discarded, **both layers are neutralized**.

When memory is preserved, Astra-β’s depth translates directly into **qualitative gains**.

---

## **Key Takeaways**

* Astra-β is **not stronger by architecture alone**
* Its real power appears **only under stateful inference**
* Stateless sampling silently reduces a GRU to a Markov process
* Memory preservation is **not optional** for recurrent models

---

## **Conclusion**

Astra-β clearly demonstrates that:

> **A deeper GRU without carried state is computationally expensive noise.**

Stateful inference unlocks the representational capacity already learned during training — no new parameters, no retraining, just **correct usage of recurrence**.
