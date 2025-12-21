# **Scribe-β: Stateless vs Stateful Inference**

Scribe-β is a **larger-capacity subword-level GRU language model** trained using SentencePiece tokenization.
Compared to Scribe-α, β has **higher representational power**, making it *even more sensitive* to how hidden state is handled during inference.

This document compares **stateless vs stateful generation** using the **same trained Scribe-β weights**, isolating the effect of hidden-state propagation.

---

## **1. Stateless Inference**

### **Inference Setup**

At every decoding step, the recurrent state is **discarded**:

```python
h_prev = None
logits, h_prev = model(x_t, h_prev)  # h_prev reset each step
```

The model is never allowed to accumulate context beyond the current token.

---

### **Observed Generations (Stateless)**

**Greedy Sampling**

```
ROMEO: I amongs, and the curse, and the curse, and the curse, and the curse, ...
```

* Strong repetition
* Phrase-level collapse
* No discourse progression

---

**Temperature Sampling**

```
KING: I'ded call me, itself, But mished, this, he is my words. From words to the goes...
```

* Rich vocabulary fragments
* Local syntactic structure
* Severe semantic drift
* Speaker identity unstable
* No long-range consistency

---

**Top-k Sampling**

```
FIRST CITIZEN: And fit upon mineself! Pars: Sh, Accord? Bolent a tastly, by your love...
```

* Long outputs
* High lexical diversity
* Scene structure fails to persist
* Sentences do not reference earlier meaning

---

### **Why This Happens (Mathematically)**

Stateless inference reduces the GRU to:

$$
P(s_{t+1} \mid s_t)
$$

instead of the trained objective:

$$
P(s_{t+1} \mid s_1, \dots, s_t)
$$

The recurrence:

$$
h_t = f(x_t, h_{t-1})
$$

is effectively evaluated as:

$$
h_t = f(x_t, 0)
$$

at every timestep.

**Result:**
Scribe-β degenerates into a **powerful subword classifier**, not a sequence model.
Extra capacity amplifies *local fluency* but cannot recover *global coherence*.

---

## **2. Stateful Inference (Correct Usage)**

### **Inference Setup**

Hidden state is **explicitly preserved** across decoding steps:

```python
h_prev = None
for t in range(T):
    logits, h_prev = model(x_t, h_prev)
```

The GRU’s memory is now allowed to function as trained.

---

### **Observed Generations (Stateful)**

**Greedy Sampling**

```
ROMEO: I have done, and free, and free, and free, and free, ...
```

* Repetition reflects **learned rhythmic patterns**
* Stable speaker voice
* Clear persistence of intent

---

**Temperature Sampling**

```
KING: Cans of my poor eyes Wheretinge amen's that-axford...
CORIOLANUS: They thus ga! Abever, That times, Avant overs never...
```

* Multi-speaker interaction
* Contextual responses
* Meaning accumulates across turns
* Dramatically improved continuity

---

**Top-k Sampling**

```
FIRST CITIZEN: For my law, Against, To the ready, and force...
QUEEN ELIZABETH: Why, and his brothers from this cast onceard...
```

* Scene-like progression
* Cross-line semantic dependence
* Dialogue coherence emerges
* Output resembles structured dramatic text

---

### **Why This Works (Mathematically)**

With state preserved, the GRU update:

$$
h_t = h_{t-1} + z_t(\tilde{h_t} - h_{t-1})
$$

functions as intended:

* If ( z_t \approx 0 ) → memory retained
* If ( z_t \approx 1 ) → new information injected

This enables the correct factorization:

$$
P(s_{t+1} \mid s_1, \dots, s_t)
$$

Scribe-β finally operates as a **true autoregressive language model**, not a conditional predictor.

---

## **3. Why the Difference Is Even Larger for Scribe-β**

Scribe-β operates on **semantic subword units** *and* has higher capacity.

Therefore:

* Stateless Scribe-β ≈ **high-capacity semantic Markov model**
* Stateful Scribe-β ≈ **context-aware generative language model**

Larger models **suffer more** when memory is disabled, because they are explicitly trained to exploit long-range dependencies.

---

## **Key Insight**

> **Scaling a recurrent model increases its dependence on memory.
> Disabling state during inference negates that advantage entirely.**

Scribe-β demonstrates that **stateful inference is not an optimization** —
it is a **correctness requirement** for recurrent language models.
