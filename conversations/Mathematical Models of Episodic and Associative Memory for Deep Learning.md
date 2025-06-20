# Mathematical Models of Episodic and Associative Memory for Deep Learning

This report presents a comprehensive overview of mathematical models of episodic and associative memories that are structured around key-value pairs and are particularly suited for implementation within deep learning frameworks. The models discussed range from classical neural network architectures to more recent developments in memory-augmented and attention-based networks.

## Introduction

Episodic memory refers to the recollection of past personal experiences, which are often indexed by spatio-temporal context. Associative memory, a more general concept, involves the ability to store and retrieve patterns based on their relationships. In the context of deep learning, these memory systems are often modeled using key-value structures, where a **key** is used to address or query the memory, and a **value** represents the stored information. This paradigm has proven powerful for tasks requiring context-dependent processing, reasoning, and one-shot learning.

This document details the mathematical foundations of several prominent key-value memory models. For each model, we will define the key and value representations, and the mathematical operations for writing to and reading from the memory.

---

## 1. Key-Value Memory Networks

Key-Value Memory Networks are a class of models explicitly designed to leverage an external memory component organized as a key-value store. They have been successfully applied to tasks like question answering and language modeling.

### Mathematical Formulation

Let the memory be a set of slots $(\mathbf{k}_1, \mathbf{v}_1), (\mathbf{k}_2, \mathbf{v}_2), \dots, (\mathbf{k}_N, \mathbf{v}_N)$, where $\mathbf{k}_i \in \mathbb{R}^{d_k}$ is the *i*-th key vector and $\mathbf{v}_i \in \mathbb{R}^{d_v}$ is the corresponding value vector. Given a query vector (or "read key") $\mathbf{q} \in \mathbb{R}^{d_k}$, the retrieval process involves two main steps: addressing and value retrieval.

**1. Addressing:** A relevance score between the query $\mathbf{q}$ and each key $\mathbf{k}_i$ is computed. A common choice for the scoring function is the dot product, followed by a softmax normalization to obtain a set of weights:

$$w_i = \frac{\exp(\mathbf{q} \cdot \mathbf{k}_i)}{\sum_{j=1}^{N} \exp(\mathbf{q} \cdot \mathbf{k}_j)}$$

where $w_i$ represents the weight of the *i*-th memory slot.

**2. Value Retrieval:** The final output vector, or "read vector" $\mathbf{r}$, is a weighted sum of the value vectors:

$$\mathbf{r} = \sum_{i=1}^{N} w_i \mathbf{v}_i$$

**Writing to Memory:** In its simplest form, writing to a Key-Value Memory Network involves appending a new key-value pair to the memory. More complex models allow for the modification of existing memory slots, often using mechanisms similar to those in Neural Turing Machines.

---

## 2. Hopfield Networks

Classical and modern Hopfield networks can be understood as associative memories. When adapted to a key-value framework, the stored patterns can be considered values, and a partial or noisy version of a pattern can act as the key for retrieval. Modern Hopfield networks, in particular, have a strong connection to the attention mechanism in Transformers.

### Mathematical Formulation

#### Classical Hopfield Network

In a classical Hopfield network, the memory is stored in the weights of the network. To store a set of *N* bipolar patterns $\{\mathbf{v}_i\}_{i=1}^N$, where $\mathbf{v}_i \in \{-1, 1\}^d$, the weight matrix $\mathbf{W}$ is defined by the Hebbian learning rule:

$$\mathbf{W} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{v}_i \mathbf{v}_i^T$$

**Retrieval:** Given a key (a probe pattern) $\mathbf{q} \in \{-1, 1\}^d$, the network state is iteratively updated until it converges to one of the stored patterns. The update rule for a neuron *j* at time *t*+1 is:

$$s_j(t+1) = \text{sgn}\left(\sum_{k=1}^{d} W_{jk} s_k(t)\right)$$

where $\mathbf{s}(0) = \mathbf{q}$. The converged state is the retrieved value.

#### Modern Hopfield Network (Dense Associative Memory)

Modern Hopfield networks introduce a continuous energy function that allows for a much larger storage capacity. The retrieval process can be formulated as a single-step update, which aligns well with deep learning layers.

Given a set of stored patterns (values) $\{\mathbf{v}_i\}_{i=1}^N$, which are the columns of a matrix $\mathbf{V}$, and a key (state pattern) $\mathbf{q}$, the update rule to retrieve the associated pattern is:

$$\mathbf{q}_{\text{new}} = \text{softmax}(\beta \mathbf{V}^T \mathbf{q}) \mathbf{V}$$

Here, $\beta$ is an inverse temperature parameter. This formulation is strikingly similar to the self-attention mechanism, where the queries, keys, and values are all derived from the same set of input vectors.

---

## 3. Sparse Distributed Memory (SDM)

Sparse Distributed Memory is a mathematical model of human long-term memory that uses a very large, high-dimensional binary address space. It is a content-addressable memory that can retrieve information from noisy or incomplete cues.

### Mathematical Formulation

Let the address space be $\{0, 1\}^n$, where *n* is large (e.g., 1000). The memory consists of a set of *M* physical storage locations, each with an address $\mathbf{a}_j \in \{0, 1\}^n$ and a content vector $\mathbf{c}_j \in \mathbb{Z}^d$.

**Writing to Memory:** To write a data vector (value) $\mathbf{v} \in \{-1, 1\}^d$ at a specified address (key) $\mathbf{k} \in \{0, 1\}^n$:
1.  **Addressing:** A subset of physical memory locations is selected based on the Hamming distance between their addresses $\mathbf{a}_j$ and the write key $\mathbf{k}$. A location *j* is selected if $d_H(\mathbf{k}, \mathbf{a}_j) \le r$, where *r* is a predefined radius.
2.  **Writing:** The data vector $\mathbf{v}$ is added to the content vectors of all selected locations:
    $$
    \mathbf{c}_j \leftarrow \mathbf{c}_j + \mathbf{v} \quad \text{for all } j \text{ such that } d_H(\mathbf{k}, \mathbf{a}_j) \le r
    $$

**Reading from Memory:** To read the data associated with a read key $\mathbf{q} \in \{0, 1\}^n$:
1.  **Addressing:** The same Hamming distance-based selection is performed to find all physical locations *j* such that $d_H(\mathbf{q}, \mathbf{a}_j) \le r$.
2.  **Retrieval:** The content vectors of the selected locations are summed to produce an aggregate vector $\mathbf{s} = \sum_{j} \mathbf{c}_j$.
3.  **Decoding:** The final output vector $\mathbf{r} \in \{-1, 1\}^d$ is obtained by thresholding the aggregate vector:
    $$
    r_i = \text{sgn}(s_i)
    $$

---

## 4. Neural Turing Machines (NTM) and Differentiable Neural Computers (DNC)

NTMs and their successors, DNCs, are memory-augmented neural networks that couple a neural network controller with an external memory matrix. These models can learn to perform algorithmic tasks by reading from and writing to the memory.

### Mathematical Formulation

Let the memory be a matrix $\mathbf{M} \in \mathbb{R}^{N \times W}$, where *N* is the number of memory locations and *W* is the width of each memory vector.

**Addressing Mechanisms:** Both NTMs and DNCs use a combination of content-based and location-based addressing to produce a weighting over the memory locations.

* **Content-based addressing:** A key vector $\mathbf{k}_t$ emitted by the controller at time *t* is compared to each memory location $\mathbf{M}_t(i)$ using a similarity measure $K(\cdot, \cdot)$, typically cosine similarity. This produces a content-based weighting $\mathbf{w}_t^c$:
    $$
    w_t^c(i) = \frac{\exp(\beta_t K(\mathbf{k}_t, \mathbf{M}_t(i)))}{\sum_{j=1}^N \exp(\beta_t K(\mathbf{k}_t, \mathbf{M}_t(j)))}
    $$
    where $\beta_t$ is a key strength also emitted by the controller.

* **Location-based addressing:** This allows for iterative access to memory locations. The final weighting $\mathbf{w}_t$ is a combination of the content-based weighting and the weighting from the previous timestep, modulated by gating mechanisms.

**Reading from Memory:** The read vector $\mathbf{r}_t$ is a weighted sum of the memory vectors:

$$\mathbf{r}_t = \sum_{i=1}^N w_t(i) \mathbf{M}_t(i)$$

**Writing to Memory:** Writing is a combination of erasing and adding new information. The controller emits an erase vector $\mathbf{e}_t$ and an add vector $\mathbf{a}_t$. The memory is updated as follows:

$$\mathbf{M}_{t+1}(i) = \mathbf{M}_t(i) \odot (\mathbf{1} - w_t(i) \mathbf{e}_t) + w_t(i) \mathbf{a}_t$$

where $\odot$ denotes element-wise multiplication and $\mathbf{1}$ is a vector of ones.

**DNC Enhancements:** DNCs introduce more sophisticated mechanisms, including a usage vector to track free memory and a temporal link matrix to preserve the order of writes, enabling dynamic memory allocation and retrieval of sequential information.

---

## 5. Attention Mechanism and Transformers

The attention mechanism, particularly the scaled dot-product attention used in Transformers, can be interpreted as a form of soft, content-addressable memory.

### Mathematical Formulation

In the context of self-attention, the input sequence of vectors is transformed into queries ($\mathbf{Q}$), keys ($\mathbf{K}$), and values ($\mathbf{V}$).

Let the input be a set of *n* vectors, packed into matrices $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, $\mathbf{K} \in \mathbb{R}^{n \times d_k}$, and $\mathbf{V} \in \mathbb{R}^{n \times d_v}$.

**Addressing and Retrieval:** The attention scores are computed by taking the dot product of each query with all keys. These scores are then scaled and passed through a softmax function to obtain the weights. The output is a weighted sum of the value vectors. This is all performed in a single matrix multiplication:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

Here, the rows of $\mathbf{K}$ act as the keys and the rows of $\mathbf{V}$ as the values. The query vectors in $\mathbf{Q}$ are used to retrieve a context-dependent representation from the values. This mechanism allows the model to dynamically focus on different parts of the input sequence, effectively treating the entire input as a temporary, differentiable key-value memory.

## Conclusion

The mathematical models presented here provide a foundational toolkit for incorporating episodic and associative memory into deep learning systems. From the explicit key-value stores of Memory Networks to the implicit, dynamic memory of the Transformer's attention mechanism, these formalisms offer diverse and powerful ways to imbue neural networks with the ability to store and retrieve information, a crucial component for tackling complex cognitive tasks. The convergence of these models, particularly the relationship between modern Hopfield networks and attention, highlights a promising direction for future research in building more capable and general artificial intelligence.
