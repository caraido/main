# `tests/cotrain_latent_space/` — do both tasks land in one 2-D space?

```bash
python -m tests.cotrain_latent_space.run \
    --patients AA KAW LH PV RB SE WBH \
    --why "one line, required"
```

## The question

The cross-task figure's retired MDS panel compared **two separately trained** decoders and
could not show category clusters that also mapped to the same place. This asks it the other
way round: fit **one co-trained** decoder and look at the space it actually builds, so "both
tasks project into the same space" holds by construction and the only open question is
whether **category** is visible in it.

Three views come off a single word-grouped out-of-fold fit, so they describe the same model
and the choice between them is about presentation, not about which decoder got trained:

| view | what it is |
|---|---|
| `latent` | the co-trained PLS latent space itself — two of its ten components |
| `lda` | category discriminants fitted on **picture** trials, auditory projected in |
| `glove` | metric MDS on cosine distance of the co-trained model's predicted GloVe |

## What it found (7 participants, `tpm`/h10, `balance=downsample`)

**No promotable panel.** Cross-task category-centroid alignment, computed in each view's own
2-D space at the word level:

| view | mean | median | p < 0.05 |
|---|---|---|---|
| `glove` | **0.356** | 0.338 | 1/7 |
| `latent` | 0.175 | 0.282 | 1/7 |
| `lda` | 0.029 | 0.105 | 0/7 |

`glove` is the best of the three and still thin. `lda` — the strongest claim if it had worked
— is at chance, consistent with naive PN→AN transfer being at chance in this data.

Four things worth keeping even though the panel is not:

- **Single-trial clouds overlap almost completely in all three views.** That is the finding,
  not a plotting failure. Alec's original read of the retired panel was right, and switching
  to a co-trained decoder does not fix it.
- **Per-word means made the numbers *worse*, and are the honest ones.** Trial-level alignment
  is optimistic: a word presented many times contributes many near-identical points, which
  tightens the permutation null. (The retired panel's 0.258 was trial-level.)
- **The co-trained latent space barely encodes task at all** — max single-component
  picture-vs-auditory AUC is 0.60–0.81 across participants. The worry that components 1–2
  would be task axes was wrong. Category is instead spread thinly: the two plotted components
  are never 1 and 2, and the best one is component 7 or 10 for some participants.
- **Most per-category cross-task shifts are below resolution.** The 95 % bootstrap ellipses on
  the centroids overlap. Pooling across participants (Procrustes to a common frame) is the
  only obvious way to shrink them, and would need its own control since the alignment would
  then be fitted on the thing being shown.

## Status

Answered and **not promoted**. Record in
[`docs/experiments/018`](../../docs/experiments/018-promoting-roi-analysis-into-the-cross-task-figure.md).
Kept rather than archived because the fit is cheap (~3 s/participant) and the next idea —
pooled Procrustes — reuses all of it.

## Files

| File | What it is |
|---|---|
| `run.py` | CLI → `open_run` → per-participant loop → `run.headline()` |
| `latent.py` | the co-trained OOF fit, word means, component diagnostics, the three views |
| `figures.py` | per-participant view figure + latent-component diagnostics |

Outputs go to `results/cotrain_latent_space/<run_id>/{figures,source_data}/` via
`utils.run_context.open_run` — never a hand-composed path.
