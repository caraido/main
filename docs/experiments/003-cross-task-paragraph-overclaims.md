---
id: 003
kind: manuscript
title: The draft's cross-task paragraph quotes retired numbers and overclaims an amodal code
status: answered
analysis: cross_task_cotrain
opened: 2026-07-28
closed: 2026-08-11
runs:
report:
answer: >
  Moot. Alec retired the cross-task material from the manuscript as a whole on 2026-08-11 —
  auditory naming performance is underwhelming and new analyses are needed first. The
  paragraph is not being corrected; it is being removed.
---

## Question

The active draft's cross-task paragraph is obsolete in four separate ways at once, and
three of them are claim-level rather than wording-level.

It quotes the old **upsampled** values (picture 0.241, auditory 0.251; retention 78%/87%),
treats **VIP** as a live electrode-importance measure, and describes "amodal"
electrodes and representations.

Current boundaries:

- the paper's canonical ROI condition is **`balance=none`**, not auditory upsampling;
- **VIP was removed 2026-07-23** and must not appear in Results or Methods;
- **no ROI survives the BH-corrected group test**;
- naive PN↔AN transfer is **at chance**;
- the supported claim is a **shared, alignable subspace**, recovered through co-training
  plus low-dimensional alignment — **never an "amodal code"**.

## What was tried

No compute. This is the draft disagreeing with analyses that have already run.

## Result

Not a compute question.

## Next

**Decision, Alec 2026-08-11: the cross-task material is retired from the manuscript as a
whole.** Not corrected — removed. The reason is upstream of the wording: auditory naming
performance is underwhelming, and new analyses are needed before any cross-task claim is
worth making. So the four corrections above are moot as manuscript edits.

What this does and does not mean:

- **The manuscript section goes.** Including the amodal wording, the upsampled numbers and
  the VIP material.
- **The code and results stay.** `analysis/cross_task/`, `results/cross_task_*` and
  `figures_for_paper/cross_task/` are untouched by this decision — a retired section is not
  a pruning authorisation, and `cross_task_transfer` is still the negative control behind
  the framing.
- **The scientific boundaries still hold** and must not be quietly relaxed if the material
  returns: no ROI survives the BH-corrected group test, naive PN↔AN transfer is at chance,
  and the supported claim was a shared alignable subspace, never an amodal code.
- **VIP is being removed repo-wide** — see entry
  [009](009-tracked-doc-corrections-not-applied.md).

A successor entry should be opened when the new auditory analyses are defined.
