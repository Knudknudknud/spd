# Dataset Proposals: Where GNN Over SPD Wins

## Notation

Throughout, SPD decomposes each weight matrix $W^l$ into rank-1 subcomponents:

$$W^l \approx \sum_{c=1}^{C} \vec{U}^l_c \vec{V}^{l\top}_c$$

and learns **independent** causal importance functions:

$$g^l_c(x) = \sigma_H\!\bigl(\gamma^l_c(h^l_c(x))\bigr), \quad h^l_c(x) = \sum_j V^l_{c,j}\, a^l_j(x)$$

Each $\gamma^l_c$ is a small MLP that sees **only its own scalar** $h^l_c(x)$. This is the weakness we exploit.

---

## Option 1: Modular Addition (Grokking)

### Task

Given a prime $p$, learn the map:

$$f: \mathbb{Z}_p \times \mathbb{Z}_p \to \mathbb{Z}_p, \quad f(a,b) = (a + b) \bmod p$$

Inputs $a, b$ are one-hot encoded, so $\mathbf{e}_a, \mathbf{e}_b \in \{0,1\}^p$.

### What the model learns

After grokking, the model represents each input $a$ using $K$ **Fourier components** at frequencies $k \in \{1, \dots, (p-1)/2\}$. For each frequency $k$, the embedding of input $a$ is:

$$\phi_k(a) = \begin{pmatrix} \cos\!\bigl(\tfrac{2\pi k a}{p}\bigr) \\[4pt] \sin\!\bigl(\tfrac{2\pi k a}{p}\bigr) \end{pmatrix} \in \mathbb{R}^2$$

This maps $a$ to a point on the unit circle $S^1$. Each frequency $k$ defines an **irreducible rank-2 representation** of $\mathbb{Z}_p$.

### How the model computes addition

For a given frequency $k$, the model computes $\phi_k(a+b)$ from $\phi_k(a)$ and $\phi_k(b)$ using the angle-addition identities:

$$\cos\!\bigl(\tfrac{2\pi k(a+b)}{p}\bigr) = \cos\!\bigl(\tfrac{2\pi k a}{p}\bigr)\cos\!\bigl(\tfrac{2\pi k b}{p}\bigr) - \sin\!\bigl(\tfrac{2\pi k a}{p}\bigr)\sin\!\bigl(\tfrac{2\pi k b}{p}\bigr)$$

$$\sin\!\bigl(\tfrac{2\pi k(a+b)}{p}\bigr) = \sin\!\bigl(\tfrac{2\pi k a}{p}\bigr)\cos\!\bigl(\tfrac{2\pi k b}{p}\bigr) + \cos\!\bigl(\tfrac{2\pi k a}{p}\bigr)\sin\!\bigl(\tfrac{2\pi k b}{p}\bigr)$$

In matrix form, this is a **rotation**:

$$\phi_k(a+b) = R_k(b)\,\phi_k(a), \quad R_k(b) = \begin{pmatrix} \cos\theta_{k,b} & -\sin\theta_{k,b} \\ \sin\theta_{k,b} & \cos\theta_{k,b} \end{pmatrix}$$

where $\theta_{k,b} = \tfrac{2\pi k b}{p}$.

The rotation matrix $R_k(b)$ is **rank 2**. It cannot be factored into a single rank-1 matrix — both the $\cos$ and $\sin$ rows are needed simultaneously.

### The mechanism structure in the weights

For each frequency $k$, the relevant weight matrices contain a rank-2 block that implements $R_k$. In the attention/MLP weights, this manifests as two directions $\vec{u}^{(k)}_{\cos}, \vec{u}^{(k)}_{\sin}$ spanning a 2D plane.

Define the subspace for frequency $k$:

$$\mathcal{S}_k = \text{span}\!\bigl(\vec{u}^{(k)}_{\cos},\, \vec{u}^{(k)}_{\sin}\bigr)$$

The mechanism for frequency $k$ is irreducibly rank-2: projecting onto either $\vec{u}^{(k)}_{\cos}$ or $\vec{u}^{(k)}_{\sin}$ alone destroys the rotation.

### Why SPD's independent gates fail

SPD will decompose the weight into rank-1 subcomponents. Two of these, call them $c_1$ and $c_2$, will correspond to the $\cos$ and $\sin$ directions of frequency $k$:

$$c_1 \leftrightarrow \vec{u}^{(k)}_{\cos}\vec{v}^{(k)\top}_{\cos}, \qquad c_2 \leftrightarrow \vec{u}^{(k)}_{\sin}\vec{v}^{(k)\top}_{\sin}$$

Their importance functions are:

$$g_{c_1}(x) = \sigma_H\!\bigl(\gamma_{c_1}(h_{c_1}(x))\bigr), \qquad g_{c_2}(x) = \sigma_H\!\bigl(\gamma_{c_2}(h_{c_2}(x))\bigr)$$

The problem: the correct importance values satisfy the **joint constraint**

$$g_{c_1}(x) \approx 1 \iff g_{c_2}(x) \approx 1$$

Both must be active together or both inactive. But $\gamma_{c_1}$ only sees $h_{c_1}(x)$ and $\gamma_{c_2}$ only sees $h_{c_2}(x)$. Under superposition with other frequencies, these scalars contain interference:

$$h_{c_1}(x) = \underbrace{\vec{v}^{(k)\top}_{\cos}\, \vec{a}(x)}_{\text{signal from freq } k} + \underbrace{\sum_{k' \neq k} \langle \vec{v}^{(k)}_{\cos},\, \vec{v}^{(k')}_{\cos/\sin} \rangle\, \alpha_{k'}(x)}_{\text{interference from other frequencies}}$$

Without knowing $h_{c_2}(x)$ (the $\sin$ partner), $\gamma_{c_1}$ cannot reliably determine whether its signal comes from frequency $k$ being active or from interference. Similarly for $\gamma_{c_2}$.

### Why a GNN succeeds

A GNN over the subcomponent graph can propagate messages between $c_1$ and $c_2$. After message passing, node $c_1$ has access to an aggregated representation that includes $h_{c_2}(x)$:

$$\tilde{h}_{c_1}(x) = \text{AGG}\!\bigl(h_{c_1}(x),\, \{h_{c'}(x)\}_{c' \in \mathcal{N}(c_1)}\bigr)$$

This allows the GNN to learn the joint decision:

$$g_{c_1}(x) = \sigma_H\!\bigl(\gamma_{c_1}(\tilde{h}_{c_1}(x))\bigr)$$

Now $\gamma_{c_1}$ can condition on whether its $\sin$ partner $c_2$ is also receiving signal, which disambiguates true activation of frequency $k$ from interference.

---

## Option 2: Rank-$r$ Toy Model of Superposition

### Construction

Generalise the Toy Model of Superposition (Elhage et al., 2022) from scalar features to $r$-dimensional features.

**Standard TMS (rank-1 features):** We have $n$ scalar features in $m < n$ dimensions.

$$\hat{\mathbf{x}} = \text{ReLU}(W^\top W\,\mathbf{x} + \mathbf{b}), \quad W \in \mathbb{R}^{m \times n}$$

Each mechanism is a single column $\vec{w}_i \in \mathbb{R}^m$, which is rank-1.

**Rank-$r$ generalisation:** We have $n$ features, each $r$-dimensional. Total input dimension is $d = n \cdot r$. Hidden (bottleneck) dimension is $m$ where $m < n \cdot r$, forcing superposition.

$$\hat{\mathbf{x}} = \sigma\!\bigl(W^\top W\,\mathbf{x} + \mathbf{b}\bigr), \quad W \in \mathbb{R}^{m \times (n \cdot r)}$$

Group the columns of $W$ into $n$ blocks:

$$W = \bigl[\underbrace{W_1}_{r \text{ cols}} \mid \underbrace{W_2}_{r \text{ cols}} \mid \cdots \mid \underbrace{W_n}_{r \text{ cols}}\bigr]$$

where $W_i \in \mathbb{R}^{m \times r}$ is the embedding for feature $i$. Each $W_i$ is a **rank-$r$ mechanism**: when feature $i$ is active, all $r$ columns of $W_i$ are jointly needed to encode and decode the $r$-dimensional input.

### Data distribution

Input $\mathbf{x} \in \mathbb{R}^{n \cdot r}$ is generated blockwise. For each feature $i \in \{1, \dots, n\}$:

$$\mathbf{x}^{(i)} \sim \begin{cases} \text{Unif}(S^{r-1}) & \text{with probability } s \\ \mathbf{0} \in \mathbb{R}^r & \text{with probability } 1 - s \end{cases}$$

where $s \ll 1$ is the sparsity parameter and $S^{r-1}$ is the unit $(r{-}1)$-sphere (sampling on the sphere ensures all $r$ dimensions are needed — you can't ignore any).

The full input is $\mathbf{x} = (\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)})^\top$.

### Training objective

Same as standard TMS — autoencoding with importance weighting:

$$\mathcal{L} = \sum_{i=1}^{n} \lambda_i \, \|\hat{\mathbf{x}}^{(i)} - \mathbf{x}^{(i)}\|^2$$

where $\lambda_i$ are importance weights (typically decaying) and $\hat{\mathbf{x}}^{(i)}$ is the reconstruction of the $i$-th block.

### Ground truth mechanisms

The ground truth decomposition has exactly $n$ parameter components, one per feature. Component $i$ is the rank-$r$ matrix:

$$M_i = W_i W_i^\top \in \mathbb{R}^{m \times m} \quad (\text{rank } r)$$

or equivalently the $r$ rank-1 sub-matrices $\{\vec{w}_{i,1}\vec{w}_{i,1}^\top, \dots, \vec{w}_{i,r}\vec{w}_{i,r}^\top\}$ that must be grouped together.

Component $i$ should be **active** (importance $\approx 1$) when $\mathbf{x}^{(i)} \neq \mathbf{0}$ and **inactive** (importance $\approx 0$) otherwise.

### Why SPD's independent gates fail

SPD decomposes $W$ into $C$ rank-1 subcomponents. For feature $i$, it should recover $r$ subcomponents $\{c_{i,1}, \dots, c_{i,r}\}$ corresponding to the $r$ columns of $W_i$.

The causal importance function for subcomponent $c_{i,j}$ only sees:

$$h_{c_{i,j}}(x) = \vec{v}_{c_{i,j}}^\top \vec{a}(x)$$

Under superposition, $\vec{v}_{c_{i,j}}$ is not orthogonal to the columns of other features, so:

$$h_{c_{i,j}}(x) = \underbrace{\alpha_{i,j}(x)}_{\text{signal from feature } i, \text{ dim } j} \;+\; \underbrace{\sum_{(i',j') \neq (i,j)} \langle \vec{v}_{c_{i,j}}, \vec{v}_{c_{i',j'}} \rangle \, \alpha_{i',j'}(x)}_{\text{interference}}$$

The independent gate must decide: "is feature $i$ active?" But it has two problems:

**(a) Ambiguity from interference.** The scalar $h_{c_{i,j}}$ could be large either because feature $i$ is active or because several other features interfere constructively. Without seeing $h_{c_{i,j'}}$ for $j' \neq j$ (the other dimensions of the same feature), the gate can't disambiguate. If all $r$ dimensions of feature $i$ show correlated signal, that's strong evidence feature $i$ is truly active — but an independent gate can't check this.

**(b) The co-activation constraint.** The correct importance satisfies:

$$g_{c_{i,1}}(x) = g_{c_{i,2}}(x) = \cdots = g_{c_{i,r}}(x)$$

All $r$ subcomponents of a feature must be jointly active or jointly inactive. Each independent gate must independently arrive at the same binary decision, using only its own scalar. As $r$ grows, the probability that all $r$ independent gates agree correctly **decreases**, even if each individual gate has high accuracy:

$$P(\text{all correct}) = P(\text{one correct})^r \to 0 \quad \text{as } r \to \infty$$

### Why a GNN succeeds

A GNN with edges between subcomponents of the same feature can enforce consistency. After message passing over the group $\{c_{i,1}, \dots, c_{i,r}\}$:

$$\tilde{h}_{c_{i,j}}(x) = \text{AGG}\!\bigl(h_{c_{i,j}}(x),\, \{h_{c_{i,j'}}(x)\}_{j' \neq j}\bigr)$$

The aggregated representation allows each node to:

1. **Vote on activation**: if $r-1$ of the $r$ dimensions show signal, the $r$-th can infer it should also be active, even if its own scalar is ambiguous
2. **Reject interference**: if only 1 of $r$ dimensions shows signal (likely interference), all $r$ gates can agree to stay inactive
3. **Enforce joint consistency**: the GNN's output importance values are functions of the shared representation, so $g_{c_{i,1}} \approx g_{c_{i,2}} \approx \cdots \approx g_{c_{i,r}}$ by construction

### The key experiment: sweep over $r$

Fix $n$ (number of features), $m$ (hidden dimension), and $s$ (sparsity). Train target models for $r \in \{1, 2, 3, 4, 8, 16\}$. Decompose each with SPD and with your GNN-SPD.

**Prediction:**
- At $r = 1$: both methods perform equally (no multi-dimensional structure to exploit)
- At $r = 2$: GNN begins to outperform
- At $r \geq 4$: the gap widens significantly, as SPD's independent gates increasingly fail the co-activation constraint

Measure using:
- **MMCS** (mean max cosine similarity) between recovered and ground-truth mechanisms (after clustering SPD subcomponents into rank-$r$ groups)
- **Importance accuracy**: fraction of datapoints where all subcomponents of an active feature have $g > 0.5$ and all subcomponents of inactive features have $g < 0.5$
- **Reconstruction loss** of the decomposed model

---

## Summary

| | Option 1: Modular Addition | Option 2: Rank-$r$ TMS |
|---|---|---|
| Source of rank-$k$ | Fourier frequencies → rank-2 rotations | Explicit rank-$r$ feature embeddings |
| Ground truth known? | Yes (Fourier analysis) | Yes (by construction) |
| Controllable $k$? | Fixed at $k=2$ per frequency | Yes, sweep $r$ |
| Ecological validity | High (studied in real LLMs) | Lower (synthetic) |
| Cleanness of argument | Moderate (transformer internals) | Very clean (direct) |

**Recommended thesis structure:** Option 2 first (clean, controllable, irrefutable), then Option 1 (connects to real models and existing literature).