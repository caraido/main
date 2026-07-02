# -*- coding: utf-8 -*-
"""
tests.open_vocab_retrieval
==========================
Open-vocabulary (zero-shot) word retrieval for the semantic-decoding paper.

Converts the closed-set retrieval result (decoded GloVe embedding scored only
against the stimulus wordset) into an *open-gallery* retrieval result: the
decoded embedding is scored against a large external lexicon so the decoder can
retrieve words never seen in training.

Module map (mirrors the implementation guide, one concern per file):

  gallery      Step 1 — build/lemmatize/filter the open lexicon + per-word meta
  predict_io   Step 2 — CV over per-patient pkls -> predicted embeddings, folds,
                          held-out (zero-shot) flags
  retrieval    Step 3 — cosine retrieval + tie-safe rank computation
  metrics      Step 4-5 — rank_metrics(), ndcg_independent(), category Hit@k/mAP
  relevance    Step 5 — WordNet graders (relevance independent of decode target)
  stats        Step 6 — permutation null, group-level Wilcoxon, frequency control
  sweeps       Step 7 — gallery-size and gallery-variant sweeps
  figures      Step 8 + paper figures
  run          orchestration CLI

The decode target is a LANGUAGE embedding (GloVe 840B, matching the project's
kernel_pls_cosine runs); the gallery is GloVe embeddings of the lexicon words,
so query and gallery live in the same space.  Cosine is kept as the similarity
(consistent with the training objective), and the project's canonical
mean-centre convention (subtract the gallery centroid before cosine) is reused
from ``utils.retrieval``.
"""
