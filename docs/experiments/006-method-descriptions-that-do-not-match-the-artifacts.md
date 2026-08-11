---
id: 006
kind: manuscript
title: Four method/display descriptions in the draft do not match the artifacts that produced them
status: open
analysis: open_vocab_retrieval
opened: 2026-07-28
closed:
runs:
report:
answer:
---

## Question

The numbers are largely current; the *descriptions* of how they were produced are not. Four
of them, spanning two figures.

**Open-vocabulary retrieval**

- The matched gallery was built **without a concreteness norms file**, so it is
  noun-dominant and frequency-matched — **not** "the 5,000 most common concrete nouns".
- The qualitative projection is **metric MDS on cosine distance**, not t-SNE. The draft uses
  both descriptions in different places.

**Language versus vision**

- The current paper figure and Results use **GloVe/Word2Vec versus DINOv3/MoCo**. Methods
  still mention **DINOv2** and, elsewhere, **ConceptNet plus a six-model family**.

## What was tried

No compute. Each description was checked against its producing module.

## Result

Not a compute question. Every correction is a matter of reading the producing artifact.

## Next

- Standardise each description on the artifact that produced it.
- Keep **nDCG paired with its matched permutation null** — never report a bare ~0.65.
- Decide whether the extra embedding models are supplementary or dropped, then use **one**
  model set consistently across Results and Methods.
- **Preserve the negative result**: peak-bin word-level language-vs-vision contrasts are all
  non-significant, and MoCo is the strongest category competitor. That is a finding, not a
  gap to be tidied away.
