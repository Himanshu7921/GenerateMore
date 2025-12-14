# **Appendix A — Formal Recurrent Model Equations**

This appendix documents the **exact mathematical formulations** of the recurrent architectures implemented in this repository.
All equations correspond directly to the **from-scratch implementations** used in the Astra, Scribe, and Zeta model families.

---

## **A.1 Vanilla Recurrent Neural Network (RNN)**

Given an input sequence $( {x_t}_{t=1}^T )$, the vanilla RNN updates its hidden state as:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

The output logits are computed as:

$$
o_t = W_{ho} h_t + b_o
$$

and converted to probabilities via softmax:

$$
p(y_t \mid x_{\le t}) = \text{softmax}(o_t)
$$

### Limitations

* Susceptible to vanishing gradients
* Short effective memory horizon
* Motivates gated architectures (GRU, LSTM)

---

## **A.2 Gated Recurrent Unit (GRU)**

The GRU introduces gating mechanisms to regulate information flow.

### **Update Gate**

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

Controls how much of the previous hidden state is retained.

---

### **Reset Gate**

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

Controls how much past information is used to compute the candidate state.

---

### **Candidate Hidden State**

$$
\tilde{h}*t = \tanh(W_h x_t + U_h (r_t \odot h*{t-1}) + b_h)
$$

---

### **Hidden State Update**

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

This convex interpolation allows the model to **preserve long-term memory** or **overwrite it with new information**.

---

### **Output Layer**

$$
o_t = W_o h_t + b_o
$$

$$
p(y_t \mid x_{\le t}) = \text{softmax}(o_t)
$$

---

### **Inference-Time Assumption**

The GRU formulation **assumes**:

$$
h_t = f(x_t, h_{t-1})
$$

Resetting $( h_{t-1} )$ during inference violates this assumption and collapses the model into a stateless conditional predictor.

---

## **A.3 Long Short-Term Memory (LSTM)**

The LSTM introduces an explicit **cell state** $( c_t )$ to improve long-range gradient flow.

---

### **Input Gate**

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

---

### **Forget Gate**

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

---

### **Output Gate**

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

---

### **Candidate Cell State**

$$
{g_{t}} = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

---

### **Cell State Update**

$$
c_t = f_t \odot c_{t-1} + i_t \odot {g}_t
$$

---

### **Hidden State Update**

$$
h_t = o_t \odot \tanh(c_t)
$$

---

### **Output Distribution**

$$
p(y_t \mid x_{\le t}) = \text{softmax}(W_y h_t + b_y)
$$

---

## **A.4 Backpropagation Through Time (BPTT)**

All recurrent models in this repository are trained using **Backpropagation Through Time** over unrolled sequences of length ( T ).

The total loss over a sequence is:

$$
\mathcal{L} = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

Gradients are computed by recursively applying the chain rule backward through time:

$$
\frac{\partial \mathcal{L}}{\partial \theta}
=
\sum_{t=1}^{T}
\frac{\partial \mathcal{L}}{\partial h_t}
\frac{\partial h_t}{\partial \theta}
$$



where $( \theta )$ denotes all trainable parameters.

---

## **A.5 Relation to Implementations in This Repository**

* `5_RNN_vanilla.ipynb` implements **A.1**
* `6_GRU.ipynb` and `6_GRU_Pytorch_version.ipynb` implement **A.2**
* `7_shakespeare_generator*.ipynb` demonstrate **inference-time implications** of A.2
* `8_LSTM.ipynb` implements **A.3**

All equations above correspond **one-to-one** with the code in this repository.