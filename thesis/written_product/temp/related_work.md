1. The unit-of-decomposition problem
Sets up the central tension: mech interp wants atoms, but what counts as an atom is contested.

Sharkey et al. — Open Problems in Mechanistic Interpretability — https://arxiv.org/abs/2501.16496 (already in your bibtex)
Elhage et al. — A Mathematical Framework for Transformer Circuits — https://transformer-circuits.pub/2021/framework/index.html (could live here as foundation, or in §4)

2. The linear representation hypothesis and its limits
Establishes that dependent / multi-dimensional features are real, motivating why decomposition needs to handle them.

Park, Choe & Veitch — The Linear Representation Hypothesis and the Geometry of Large Language Models — https://arxiv.org/abs/2311.03658
Elhage et al. — Toy Models of Superposition (already in your bibtex as elhage2022toymodelssuperposition)
Mendel — SAE feature geometry is outside the superposition hypothesis — https://www.alignmentforum.org/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis
Csordás, Potts, Manning & Geiger — Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations — https://arxiv.org/abs/2408.10920
Engels et al. — Not All Language Model Features Are One-Dimensionally Linear (already in your bibtex as engels2025languagemodelfeaturesonedimensionally) — brief mention here, formal treatment in methods

3. Activation-space decomposition
Walks through SAEs as the dominant approach, then the failure modes that motivate moving to parameter space. Circuits folded in as a parallel activation-side branch.
SAE lineage:

Cunningham et al. — Sparse Autoencoders Find Highly Interpretable Features (already in your bibtex as sparseautoencodershighlyinterpretable)
Bricken et al. — Towards Monosemanticity — https://transformer-circuits.pub/2023/monosemantic-features/index.html
Templeton et al. — Scaling Monosemanticity — https://transformer-circuits.pub/2024/scaling-monosemanticity/
Gao et al. — Scaling and Evaluating Sparse Autoencoders — https://arxiv.org/abs/2406.04093

Failure modes (the critique):

Chanin et al. — A is for Absorption (already in your bibtex as absorptionstudyingfeaturesplitting)
Leask et al. — Sparse Autoencoders Do Not Find Canonical Units of Analysis (already in your bibtex as sparseautoencoderscanonicalunits)
Bussmann et al. — Showing SAE Latents Are Not Atomic Using Meta-SAEs — https://www.alignmentforum.org/posts/TMAmHh4DdMr4nCSr5/showing-sae-latents-are-not-atomic-using-meta-saes
Bussmann, Nabeshima, Karvonen & Nanda — Learning Multi-Level Features with Matryoshka Sparse Autoencoders — https://arxiv.org/abs/2503.17547

Circuits (parallel branch):

Olah et al. — Zoom In: An Introduction to Circuits — https://distill.pub/2020/circuits/zoom-in/
Marks et al. — Sparse Feature Circuits — https://arxiv.org/abs/2403.19647

Bridge to parameters:

Dunefsky, Chlenski & Nanda — Transcoders Find Interpretable LLM Feature Circuits — https://arxiv.org/abs/2406.11944

4. Parameter-space decomposition
Your direct lineage. Most space goes here.
Foundation (if not in §1):

Elhage et al. — A Mathematical Framework for Transformer Circuits — https://transformer-circuits.pub/2021/framework/index.html

Direct lineage:

Bushnaq et al. — The Local Interaction Basis — https://arxiv.org/abs/2405.10928
Bushnaq et al. — Using Degeneracy in the Loss Landscape for Mechanistic Interpretability (LIB companion paper) — https://arxiv.org/abs/2405.10927
Braun et al. — APD (already in your bibtex as AttributionBasedParameterDecomposition)
Bushnaq et al. — SPD (already in your bibtex as stochasticparameterdecomposition)

MDL / SLT theoretical grounding:

Lau, Furman, Wang, Murfet & Wei — The Local Learning Coefficient: A Singularity-Aware Complexity Measure — https://arxiv.org/abs/2308.12108

5. Causal abstraction (optional, brief)
Contrasting paradigm — engages the same "what's the right unit?" question from a different angle. Skippable if space is tight.

Geiger, Wu, Potts, Icard & Goodman — Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS) — https://arxiv.org/abs/2303.02536

A few cross-cutting notes:
Web articles (Transformer Circuits, Distill, Alignment Forum) need @misc or @online BibTeX entries with URL and access date — they don't have arXiv IDs.
§3's failure-modes subsection is doing the heavy lifting argumentatively — it's where SAEs get critiqued enough that parameter decomposition becomes the natural next move. Don't underweight it.
§4 is where you'd ideally close with a sentence flagging the gap your work fills (existing parameter decomposition assumes feature independence; multi-dim features break this), so the transition into your contribution is clean.
Want to start drafting §1, or jump straight to §4?