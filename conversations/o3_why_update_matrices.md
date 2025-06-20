### 1  What the two matrices do

* **$M^{TF}$** (context → item) stores “where and when I saw it.”  Each time an item $f_i$ appears, you lay down an outer-product trace $f_i\,t_i^{\!\top}$. At test, multiplying the *current* context cue $t$ through $M^{TF}$ produces an activation pattern over items, so recent or context-matching items pop out .
* **$M^{FT}$** (item → context) lets the item *bring its past back with it*.  Presenting $f_i$ retrieves a composite context vector

$$
t^{\text{IN}}_i = M^{FT}_i\,f_i
$$

which is then integrated into the drifting context state and drives recall chaining .

---

### 2  Learning rule for $M^{FT}$ (contextual-retrieval matrix)

When an item is *repeated* later in the list (or during recall) we want it to cue **both** its long-term, pre-experimental context and the freshly learned list context.  Howard & Kahana derive the constraint

$$
t^{\text{IN}}_r = A_i\,t^{\text{IN}}_i + B_i\,t_i,
$$

with coefficients $A_i/B_i=c$ (a free asymmetry parameter) and chosen so $\lVert t^{\text{IN}}_r\rVert=1$ .
To implement this for *every* item independently, they decompose $M^{FT}$ into components parallel and orthogonal to the item vector $f_i$ and update only the parallel slice:

$$
\boxed{M^{FT}_{i+1}=M^{FT}_{i}\,\tilde P_{f_i}
\;+\;A_i\,M^{FT}_{i}\,P_{f_i}
\;+\;B_i\,t_i\,f_i^{\!\top}}
\tag{12}
$$

where $P_{f_i}=f_i f_i^{\!\top}/\lVert f_i\rVert^{2}$ and $\tilde P_{f_i}=I-P_{f_i}$ .

* The **first term** leaves weights unrelated to $f_i$ intact.
* The **second term** *decays* the slice that already points from $f_i$ to its older contexts by factor $A_i<1$.
* The **third term** writes a fresh Hebbian trace $t_i f_i^{\!\top}$ scaled by $B_i$.

This “item-specific unlearning + relearning” guarantees (i) the retrieved vector stays unit length and (ii) the cue gradually shifts from remote to recent context instead of growing without bound.

---

### 3  Why do we have to keep updating?

| Matrix   | What would break if we **didn’t** update?                                                                                                                                                      | Why update?                                                                                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $M^{TF}$ | The current context could not retrieve today’s items—new experiences would be invisible during recall.                                                                                         | Each study trial forms new item–context links so the cue $t$ can find those items later .                                                                                   |
| $M^{FT}$ | A repeated or recalled item would always drag you back to *its very first* context (or to an over-grown average).  Contiguity gradients and forward-recall asymmetry would vanish or saturate. | Updating lets the item retrieve a **mixture** of old and freshly drifted context, reproducing graded, asymmetric lag-recency effects and preventing runaway weight growth . |

In short, continual Hebbian **writing** and controlled **decay** keep the bidirectional item-context associations *plastic but bounded*.  That plasticity is essential for:

* capturing the statistics of the current episode,
* allowing context drift to be *caused* by retrieval (not just by time), and
* supporting the empirically observed recency and contiguity patterns across seconds to minutes.

These same principles carry over to modern memory-augmented LLMs: keys (context vectors) must update as content is reused so that retrieval cues stay informative without letting any single key dominate, preserving both stability and adaptability.

