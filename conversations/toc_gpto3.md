**Proposed Table of Contents**

1. Renaming `keys`/`values` to `mem_keys`/`mem_values` and brainstorming richer memory architectures.&#x20;
2. Surveying benchmark protocols for evaluating memory-augmented networks and advising which to adopt.&#x20;
3. Weighing rapid, Adam-style optimisation against slower, human-like SGD for episodic memory.&#x20;
4. Verifying the simple episodic-memory code handles batched queries correctly.&#x20;
5. Stepping through a batched `update()` call to confirm gradient flow and shapes.&#x20;
6. Clarifying when to wrap evaluation in `torch.no_grad()` and the role of `zero_grad()`.&#x20;
7. Questioning the value of a plain loss-curve “adaptation-speed” test for episodic memory.&#x20;
8. Adding three systematic tests—adaptation, capacity, forgetting—to `episodic_memory.py`.&#x20;
9. Designing a “TrueEpisodicMemory” class and experiments for temporal interference & sequence recall.&#x20;
10. Fixing a context-dimension mismatch that crashed the new episodic test script.&#x20;
11. Switching to learning-based storage, adding visualisations, and seeing real episodic effects.&#x20;
12. Interpreting plots that now show temporal interference, context dependence, and sequential structure.&#x20;
13. Making results reproducible with fixed random seeds and observing sensitivity to initialisation.&#x20;
14. Proposing better initialisation, diversity regularisation, and consolidation to cut that sensitivity.&#x20;
15. Outlining a hybrid MoE-plus-memory system to slash data and parameter requirements.&#x20;
16. Retracting bold “90 % less data” claims and compiling nuanced, citation-backed distinctions between FFN, attention, and episodic memories.&#x20;

