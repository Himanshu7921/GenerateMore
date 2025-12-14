# **Scribe-α: Stateless vs Stateful Inference**

Scribe-α is a **subword-level GRU language model** trained using SentencePiece tokenization.
Unlike Astra (character-level), Scribe operates on **semantically meaningful units**, which makes **hidden-state propagation during inference critical**.

This section compares **two inference regimes** applied to the *same trained weights*.

---

## **1. Stateless Inference**

### **Inference Setup**

At every decoding step, the hidden state is **reset**:

```python
h_prev = None
logits, h_prev = model(x_t, h_prev)  # h_prev discarded next step
```

---

### **Observed Generations (Stateless)**

**Greedy Sampling**

```
ROMEO: ... the sounds, and the sounds, and the sounds ...
```

**Temperature Sampling**

* Locally fluent
* Grammatically plausible phrases
* Rapid semantic drift
* Speaker identity weakly preserved

**Top-k Sampling**

* Long outputs
* Good surface-level grammar
* Poor global coherence
* Sentences do not build on earlier meaning

---

### **Why This Happens (Mathematically)**

Although Scribe-α is *trained* as a recurrent model, stateless inference collapses it into:

$$
P(s_{t+1} \mid s_t)
$$

instead of the intended:

$$
P(s_{t+1} \mid s_1, \dots, s_t)
$$

The GRU update:

$$
h_t = f(x_t, h_{t-1})
$$

is effectively evaluated as:

$$
h_t = f(x_t, 0)
$$

at every timestep.

**Result:**
The model behaves like a **conditional subword classifier**, not a sequence model.

---

## **2. Stateful Inference (Correct Usage)**

### **Inference Setup**

Hidden state is explicitly carried forward:

```python
h_prev = None
for t in range(T):
    logits, h_prev = model(x_t, h_prev)
```

---

### **Observed Generations (Stateful)**

**Greedy Sampling**

```
ROMEO: I have I have I have I have I have ...
```

* Repetition now reflects **learned discourse patterns**, not random loops

**Temperature Sampling**

* Stronger dialogue structure
* Speakers respond to prior context
* Meaning accumulates across turns
* Less semantic reset

**Top-k Sampling**

* Multi-character interaction
* Scene-like continuity
* Far stronger long-range dependencies
* Output resembles *actual dramatic text*, not fragments

---

### **Why This Works (Mathematically)**

With state preserved:

$$
h_t = h_{t-1} + z_t(\tilde{h_{t}} - h_{t-1})
$$

* When ( $z_t \approx 0$ ): memory is retained
* When ( $z_t \approx 1$ ): new semantic content is injected

This enables the correct factorization:

$$
P(s_{t+1} \mid s_1, \dots, s_t)
$$

The model finally behaves as a **true autoregressive language model**.

---

## **3. Why the Difference Is Especially Large for Scribe**

Subword tokens encode **semantic units**, not characters.

Therefore:

* Stateless Scribe ≈ *semantic Markov model*
* Stateful Scribe ≈ *contextual language model*

Losing memory at the subword level is far more destructive than at the character level, which explains the **dramatic qualitative jump** after enabling stateful inference.

---

## **Key Insight**

> **Training a GRU teaches it *how* to use memory.
> Inference decides *whether* that memory is actually used.**

Scribe-α proves that **hidden-state propagation is not optional** — it is the defining property of recurrent language models.
