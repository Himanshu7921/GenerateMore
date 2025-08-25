# **Neural & Statistical Language Modeling Project**

This repository documents my research-oriented journey from **statistical models** to **deep neural networks** for **character-level language modeling** (e.g., generating names).

The project evolves progressively:

1. From **co-occurrence counts** and **probability distributions** (Statistical Approach).
2. To **trainable shallow neural networks**.
3. To **embedding-based MLPs** (inspired by Bengio et al., 2003).
4. Finally, to **deep multi-layer perceptrons with BatchNorm**, implemented almost entirely **from scratch**.

---

## **Project Structure**

```
├── 1_stastical_approach.ipynb      # Count-based bigram model (Part 1)
├── 2_neural_network_approach.ipynb # Intro to NN language modeling (Part 2)
├── 3_MLP_v1.ipynb                  # 2-layer MLP with embeddings (Bengio-style)
├── 3_MLP_v2.ipynb                  # Improved MLP, optimization tweaks
├── 3_MLP_v3.ipynb                  # Deeper MLP exploration
├── 3_MLP_v4.ipynb                  # Deep MLPs from scratch (Linear, BatchNorm, Tanh)
├── LICENSE
├── README.md
├── names.txt                       # Dataset of names for training
```

---

## **Progressive Research Timeline**

| Version          | Notebook                           | Description                 | Key Innovations                                                                               | Limitations                                 |
| ---------------- | ---------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Statistical**  | `1_stastical_approach.ipynb`       | Bigram probability model    | Simple counts, probability distributions                                                      | No generalization beyond seen pairs         |
| **Neural v1–v2** | `2_neural_network_approach.ipynb`  | First NN for LM             | Trainable parameters + backpropagation                                                        | Shallow, limited representation             |
| **MLP v1**       | `3_MLP_v1.ipynb`                   | 2-layer MLP with embeddings | Learns distributed word representations (Bengio et al., 2003)                                 | Small-scale, no normalization               |
| **MLP v2–v3**    | `3_MLP_v2.ipynb`, `3_MLP_v3.ipynb` | Scaling MLPs deeper         | Better optimization, nonlinearities                                                           | Still unstable without normalization        |
| **MLP v4**       | `3_MLP_v4.ipynb`                   | Deep custom MLP framework   | From-scratch **Linear, BatchNorm, Tanh**, training loop. Comparison of Plain vs BatchNorm MLP | Training stability sensitive to hyperparams |

---

## **Core Concepts**

### Experiment 1: Statistical Approach (Part 1)

* Constructed **bigram probability distributions** from dataset.
* No learnable parameters → purely count-based.
* Limitation: fails on unseen contexts.

---

### Experiment 2: Neural Network Approach (Part 2)

* Transitioned to **trainable parameters** via **backpropagation**.
* Introduced **softmax** for normalized predictions.
* Demonstrated improvement over raw statistics.

---

### Experiment 3: MLP Approaches (Part 3)

#### v1: Embedding-based MLP

* Inspired by **Bengio et al. (2003)**.
* Learned **word embeddings** instead of sparse one-hot vectors.
* Demonstrated ability to generalize and capture semantic structure.

#### v2–v3: Scaling Deeper

* Extended depth of MLPs, explored nonlinearities.
* Identified optimization bottlenecks (slow convergence, gradient issues).

#### v4: Deep MLP with BatchNorm (from scratch)

* Implemented **custom deep learning framework**:

  * `Linear` layers with Xavier initialization.
  * `BatchNorm1d` with running statistics + learnable γ, β.
  * `Tanh` activation function.
  * `MLPModel` class with training loop + name generation.
* Compared **Plain MLP** vs **BatchNorm MLP**:

  * BatchNorm allowed **faster training, higher learning rates, and stability**.
  * Generated **higher-quality names** vs earlier MLPs.

---

## **Key Learnings Across Versions**

1. **From Counts → Parameters**: Bigram counts → trainable neural models.
2. **From One-hot → Embeddings**: Low-dimensional continuous spaces capture semantic/syntactic similarity.
3. **From Shallow → Deep**: Multi-layer architectures can model richer contexts.
4. **From Vanilla → BatchNorm**: Stability and convergence drastically improved with normalization.

---

## **References**

* Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model*. JMLR, 3, 1137–1155. [PDF](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* Karpathy, A. (2022). *The makemore series*. [Link](https://karpathy.ai)
