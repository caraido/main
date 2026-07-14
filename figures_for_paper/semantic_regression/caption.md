# Figure caption — Cross-patient semantic-decoding time courses

Cross-patient semantic-decoding time courses (GloVe). Held-out decoding accuracy as a
function of time in two naming tasks — picture naming (N=12, aligned to trial onset); auditory naming (N=6, aligned to auditory stimulus onset) — with kernel-PLS (Nystroem RBF kernel followed by
PLS regression onto GloVe word-embedding targets); each participant in a distinct colour,
kept the same in every panel. Columns = task, rows = metric.

*Picture naming* (**a**, **b**, **c**, **d**; N=12). **a** Category accuracy. **b** Top-1 word-retrieval accuracy. **c** Top-3 word-retrieval accuracy. **d** Top-5 word-retrieval accuracy.

*Auditory naming* (**e**, **f**, **g**, **h**; N=6). **e** Category accuracy. **f** Top-1 word-retrieval accuracy. **g** Top-3 word-retrieval accuracy. **h** Top-5 word-retrieval accuracy.

Within a metric family the y-scale is shared across panels and across tasks (the word top-k rows share one scale; the category row has its own), so accuracies are directly comparable between tasks.

Coloured bars below the chance line are a per-participant significance raster (rows ordered by peak
accuracy, highest at top): time bins after the alignment cue where the observed mean accuracy
exceeds the 99th percentile of the shuffled-null distribution at that bin (per-bin one-sided
permutation test, p < 0.01; bins before the alignment cue are not tested). Dashed
line: mean shuffled chance across participants. Dotted vertical line at 0 s: that task's alignment
cue. Shaded vertical bands: mean cue time across participants ± 1 s.d.; cues identical across
participants (the group-warped auditory stimulus offset) are drawn as a single line without a band.
The alignment cue itself, and cues falling outside a panel's time window, are excluded. x-axis in
seconds. Participants are identified by display ID (NUEx###).
