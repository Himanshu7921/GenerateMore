# Shared Memory vs Layer-Wise Memory in Stacked GRU Models

## Overview

This repository documents a controlled experiment analyzing the effect of **shared hidden state (memory)** versus **layer-wise independent hidden states** in **stacked GRU architectures**.

Three GRU models of increasing capacity were trained and evaluated on a **character-level name generation task**.
The experiment reveals a **fundamental architectural property of deep recurrent networks**:
**stacked RNNs require independent memory per layer to form hierarchical representations.**

---

## Models Studied

| Model              | Layers | Embedding Dim | Hidden Dim | Trainable Params |
| ------------------ | ------ | ------------- | ---------- | ---------------- |
| **GRU-α (Small)**  | 1      | 64            | 128        | 79,772           |
| **GRU-β (Medium)** | 2      | 96            | 256        | 675,996          |
| **GRU-γ (Large)**  | 3      | 128           | 512        | 4,155,420        |

All models were implemented **from scratch in PyTorch**, without using `nn.GRU`.

---

## Training Configuration

### Common Settings

* **Task**: Character-level language modeling (name generation)
* **Loss**: CrossEntropyLoss
* **Batch size**: 64
* **Device**: CUDA (if available)
* **Optimizer**: Adam / AdamW
* **Sampling**: Autoregressive generation

---

### GRU-α (Small)

| Setting       | Value  |
| ------------- | ------ |
| Epochs        | 35     |
| Learning Rate | 1e-3   |
| Dropout       | 0.0    |
| Optimizer     | Adam   |
| Parameters    | 79,772 |

**Validation Snapshot**

```
Epoch 10  | Val Loss: 1.8285
Epoch 20  | Val Loss: 1.8177
Epoch 30  | Val Loss: 1.8371
```

---

### GRU-β (Medium)

| Setting       | Value   |
| ------------- | ------- |
| Epochs        | 40      |
| Learning Rate | 5e-4    |
| Dropout       | 0.1     |
| Optimizer     | Adam    |
| Parameters    | 675,996 |

**Validation Snapshot**

```
Epoch 10  | Val Loss: 1.7802
Epoch 20  | Val Loss: 1.7466
Epoch 30  | Val Loss: 1.7566
Epoch 40  | Val Loss: 1.7695
```

---

### GRU-γ (Large)

| Setting       | Value     |
| ------------- | --------- |
| Epochs        | 60        |
| Learning Rate | 3e-4      |
| Dropout       | 0.2       |
| Optimizer     | AdamW     |
| LayerNorm     | Yes       |
| Parameters    | 4,155,420 |

**Validation Snapshot**

```
Epoch 10  | Val Loss: 1.7953
Epoch 20  | Val Loss: 1.7780
Epoch 30  | Val Loss: 1.8047
Epoch 40  | Val Loss: 1.8487
Epoch 50  | Val Loss: 1.8888
Epoch 60  | Val Loss: 1.9268
```

---

## Experimental Setups

### Setup-1: **Shared Hidden Memory Across Layers**

In this setup, **multiple GRU layers reuse the same hidden state**, effectively forcing all layers to read from and write to a shared memory vector.

* GRU-α: unaffected (single layer)
* GRU-β: both layers share `h_prev`
* GRU-γ: all three layers share `h_prev`

#### Example (GRU-β, shared memory)

```python
h1, h_prev_1 = self.gru_layer_1(x, h_prev_1)
h2, h_prev_2 = self.gru_layer_2(h1, h_prev_1)  # shared memory
```

#### Example (GRU-γ, shared memory)

```python
h1, h_prev_1 = self.gru_layer_1(x, h_prev_1)
h2, h_prev_2 = self.gru_layer_2(h1, h_prev_1)
h3, h_prev_3 = self.gru_layer_3(h2, h_prev_2)
```

---

### Setup-2: **Independent Memory Per Layer (Correct Design)**

Each GRU layer maintains **its own hidden state**, allowing hierarchical temporal abstraction.

#### Example (GRU-β, decoupled memory)

```python
h1, h_prev_1 = self.gru_layer_1(x, h_prev_1)
h2, h_prev_2 = self.gru_layer_2(h1, h_prev_2)
```

#### Example (GRU-γ, decoupled memory)

```python
h1, h_prev_1 = self.gru_layer_1(x, h_prev_1)
h2, h_prev_2 = self.gru_layer_2(h1, h_prev_2)
h3, h_prev_3 = self.gru_layer_3(h2, h_prev_3)
```

The updated implementation is available in
`6_GRU_Pytorch_version.ipynb`.

---

## Results

### Setup-1: Shared Memory (Failure Mode)

**GRU-γ Output Characteristics**

* Severe repetition
* Fixed-point attractors
* Low diversity
* Phonetic loops

Example outputs:

```
Laneronelanarisyutyu
Lanikyunlikyunlejanl
Lanlikyunelanaranlik
```

**GRU-β Output Characteristics**

* Syllabic repetition
* Weak structure
* Mode collapse

Example outputs:

```
Yanidandindediddiddi
Manitanitanitanitani
Lintinzitinititielit
```

---

### Setup-2: Layer-Wise Memory (Correct Behavior)

**GRU-γ Output Characteristics**

* Realistic names
* Structural diversity
* Stable long-range dependencies

Example outputs:

```
Rishvi
Radley
Hanan
Riona
Herschel
Lindsey
```

**GRU-β Output Characteristics**

* Improved phonetic realism
* Variable length
* Reduced repetition

Example outputs:

```
Zara
Kayla
Annalize
Shaylyn
Dannie
Wood
```

---

## Key Insight

> **Depth in recurrent networks comes from hierarchical memory, not repeated computation.**

Sharing hidden state across layers:

* collapses depth
* causes destructive interference
* produces repetition and attractors

Independent memory per layer:

* enables abstraction
* stabilizes training
* produces coherent generation

---

## Conclusion

This experiment empirically demonstrates that:

* **Stacked GRUs must maintain independent hidden states per layer**
* Shared memory destroys hierarchical learning
* Proper memory separation is essential for deep recurrent models

This insight directly applies to **GRU, LSTM, and RNN architectures**, and explains why framework implementations enforce layer-wise state separation.

---

## Reproducibility

* Notebook: `6_GRU_Pytorch_version.ipynb`
* Framework: PyTorch
* Device: CUDA
* Sampling: Autoregressive

---

## Author Notes

This experiment was discovered **accidentally**, but highlights a **core theoretical principle** of deep recurrent networks that is often obscured by high-level APIs.

> Depth in RNNs is not “more computation per timestep”;
it is “more independent state trajectories over time.”
