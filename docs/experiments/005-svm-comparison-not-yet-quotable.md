---
id: 005
kind: manuscript
title: Joon's fixed-class SVM comparison is claimed in Results but has no number behind it
status: open
analysis: semantic_regression
opened: 2026-07-28
closed:
runs:
report:
answer:
---

## Question

The Results contain a literal `classifier XX` placeholder while simultaneously claiming
**"no measurable cost"**, closely matched accuracy, and the same peak bin. Those are three
quantitative claims resting on a number that has not been obtained.

This is the highest-risk open item in the draft: the claim is already written as though
settled, so it can survive into submission without anyone noticing the placeholder.

## What was tried

Nothing. The SVM arm is Joon's, and the value has not been supplied.

## Result

No result. Do not infer one from the regression arm.

## Next

- Obtain the classifier value from Joon.
- Before quoting it, confirm that **cohort, window, split, category definition and the
  statistical comparison are matched** between the two arms. Any of those differing makes
  "no measurable cost" unsupportable regardless of the numbers.
- Until then keep the comparison as an explicit placeholder.
- The Discussion's **"15 individual words"** must be labelled as the *classifier* arm, so it
  is not read as the regression arm's per-participant vocabularies and open gallery.
