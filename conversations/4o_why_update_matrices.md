In the **Temporal Context Model (TCM)**, the learning rule for **contextual retrieval** updates the matrix $M^{FT}$, which maps items to context, to encode how an item retrieves context over time. This matrix plays a crucial role in enabling contiguity and asymmetric recall effects.

### Why Should $M^{FT}$ and $M^{TF}$ Be Updated?

Updating these matrices allows the model to:

1. **Encode temporal associations**: Every time an item $f_i$ is presented, the context vector $t_i$ is updated and then stored in association with $f_i$. This allows the item to later retrieve the context in which it occurred, a prerequisite for contextual drift and associative recall.
2. **Enable asymmetric retrieval**: To model the empirical finding that forward associations are stronger than backward ones (i.e., you're more likely to recall the item that follows a given one than the one that precedes it), the updates to $M^{FT}$ must incorporate both:

   * Preexperimental context $t_i^{\text{IN}}$
   * Newly learned context $t_i$

This is achieved through the learning rule (Eq. 12 in the paper):

$$
M^{FT}_{i+1} = M^{FT}_i \tilde{P}_{f_i} + A_i M^{FT}_i P_{f_i} + B_i t_i f_i^\top
$$

Where:

* $P_{f_i}$ and $\tilde{P}_{f_i} = I - P_{f_i}$ are projection operators that isolate parts of the matrix parallel and orthogonal to $f_i$
* $A_i$ is a decay factor
* $B_i$ scales the new Hebbian association
* This rule ensures that old associations are **decayed** and new associations (between $f_i$ and updated $t_i$) are **added**, maintaining stability and plasticity.

In symmetric models, you could have:

$$
M^{TF} = \sum f_i t_i^\top \quad \text{(for context-to-item recall)}
$$

$$
M^{FT} = \sum t_i f_i^\top \quad \text{(for item-to-context retrieval)}
$$

This symmetry ensures that contextual recall supports bidirectional interactions, but with asymmetries encoded via decay and differential contributions of preexperimental vs newly-learned context.

---

### Summary: Purpose of Updating $M^{FT}$ and $M^{TF}$

* **$M^{FT}$** (item → context) is updated to enable **retrieved context** that guides future contextual drift and explains the **lag-recency** and **asymmetry** in recall.
* **$M^{TF}$** (context → item) is used to cue memory by similarity of current context $t$ to previous contexts $t_j$ where each $f_j$ was encoded.
* Both matrices are critical to **episodic encoding and retrieval**, especially in accounting for the **directional nature of recall** and **temporal clustering**.

These updates reflect a **learning process during study** and are essential for the model to maintain the history of item-context relationships. Without them, the model would be unable to support context-based associative recall or explain forward recall advantages observed in human memory.

