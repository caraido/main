---
id: 003
kind: manuscript
title: The draft's cross-task paragraph quotes retired numbers and overclaims an amodal code
status: open
analysis: cross_task_cotrain
opened: 2026-07-28
closed:
runs:
report:
answer:
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

- Replace the VIP section with the region knockout / Jacobian / covariance analysis.
- **Do not nominate single electrodes as implant targets** from the retired VIP analysis.
- If Alec wants upsampling to become the primary condition, that is a **prospective method
  decision to make before rerunning**, not a manuscript-only wording change.
- `analysis/README.md` and `README.md:262` still describe VIP as live — see entry 009.
