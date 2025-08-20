# Generate More: Character-Level Language Modeling
## Abstract

This repository investigates the problem of character-level language modeling, beginning from purely statistical baselines and extending toward neural network architectures. The study explores bigram models, bag-of-words, multilayer perceptrons (MLPs), recurrent neural networks (RNNs), gated recurrent units (GRUs), and transformers. By contrasting count-based statistical methods with neural approaches, this work emphasizes the trade-offs between interpretability, simplicity, and expressive capacity in sequence modeling.

---

## 1. Introduction

Language modeling is a fundamental task in natural language processing (NLP), aimed at estimating the probability distribution of a sequence of tokens. At the character level, the model learns dependencies between individual characters rather than entire words.

The motivation of this project, named **Generate More**, is to construct models that produce additional sequences resembling the style of the training data. Beginning with statistical bigram modeling, this repository demonstrates the limitations of purely frequency-based approaches and motivates the use of parameterized neural networks capable of generalization.

---

## 2. Related Work

* **Statistical N-gram Models** have historically provided simple baselines by capturing local token dependencies.
* **Neural Language Models** (e.g., Bengio et al., 2003) introduced distributed representations and enabled capturing longer-range dependencies.
* **Recurrent Architectures** such as RNNs and GRUs advanced sequence modeling through temporal recurrence.
* **Transformers** (Vaswani et al., 2017) represent the current state-of-the-art, leveraging attention mechanisms for efficient context modeling.

This repository replicates this progression to illustrate the conceptual evolution from counts to learned representations.

---

## 3. Methodology

### 3.1 Statistical Bigram Model

**Bigram Count Construction**

```python
# Construct bigram frequency counts from character sequences
biagram_count = dict()
for w in words:
    chars = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chars, chars[1:]):
        bigram = (ch1, ch2)
        biagram_count[bigram] = biagram_count.get(bigram, 0) + 1
```

**Probability Normalization & Sampling**

```python
# Normalize counts into conditional probabilities
P = N.float()
P /= P.sum(dim=1, keepdim=True)

# Sequential sampling procedure for text generation
generator = torch.Generator().manual_seed((2147483647))
for i in range(5):
    idx = 0
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, generator=generator, replacement=True).item()
        print(''.join(itos[idx]), end = '')
        if idx == 0:
            break
    print()
```

**Loss Calculation: Negative Log-Likelihood**

```python
# Compute the average negative log-likelihood of the dataset
log_likelihood = 0.0
n = 0
for w in words:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        bigram = (ch1, ch2) 
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        prob = P[idx1, idx2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        negative_log_likelihood = -log_likelihood
        n += 1

print(f"log likelihood = {log_likelihood}")
print(f"negative log likelihood (Loss Function) = {(negative_log_likelihood):.4f}")
avg_statical_loss = negative_log_likelihood/n
print(f"Average negative log likelihood = {avg_statical_loss:.4f}")
```

---

### 3.2 Neural Approaches

**Training Data Preparation**

```python
# Construct (input, output) training pairs for neural network models
x, y = [], []
for w in words[:5]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]

        print(f"{bigram} = {idx1}, {idx2}")
        print(f"If input is: '{bigram[0]}' ({idx1}) | output will be: '{bigram[1]}' ({idx2})")
        x.append(idx1)
        y.append(idx2)

print()
print("Constructed training pairs: ", end = " ")
print(f"x = {x}, y = {y}")
```

---

## 4. Experiments

### 4.3 Neural Models

**Loss Evaluation Example**

```python
# Training our Neural Network
import torch.nn.functional as F
xenc = F.one_hot(x, num_classes = 27).float()

for i in range(500):
    # Forward pass
    logits = xenc @ weight
    count = torch.exp(logits)
    probs = count / count.sum(dim = 1, keepdims = True) # Probabilities for next character assigned by the Model
    loss = -probs[torch.arange(len(y)), y].log().mean()

    if i % 100 == 0:
        print(f"Loss at epoch {i}: {loss}")

    # Backward pass
    weight.grad = None
    loss.backward()

    # Update the Model's Parameter
    lr = 50
    weight.data -= lr * weight.grad

# For storing it into a dict
neural_network_loss = loss
```

---

## 5. Discussion

The experiments highlight the following observations:

1. **Statistical Models** provide interpretability and efficiency but are limited to memorized patterns.
2. **Neural Networks** extend modeling capacity by learning representations that capture structure beyond observed counts.
3. **Loss Functions** such as negative log-likelihood serve as a unifying principle, applicable to both statistical and neural approaches.

---

## 6. Conclusion

This repository presents an incremental journey through character-level language modeling, beginning with statistical bigram models and advancing toward neural networks. The results confirm that while statistical approaches offer an accessible baseline, neural models significantly improve performance by capturing longer dependencies and learning generalizable representations.

Future directions include scaling experiments to larger corpora, extending architectures with attention mechanisms, and benchmarking against state-of-the-art transformer-based language models.

---

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3, 1137–1155.
2. Mikolov, T., Karafiát, M., Burget, L., Cernocký, J., & Khudanpur, S. (2010). Recurrent neural network based language model. *Interspeech*.
3. Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP*.
4. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Karpathy, A. (2022). *The makemore series*. Retrieved from [https://karpathy.ai](https://karpathy.ai)
