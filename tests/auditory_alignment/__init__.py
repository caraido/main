# -*- coding: utf-8 -*-
"""
tests/auditory_alignment — which behavioral cue triggers the decodable semantic signal?

Auditory-naming decoding normally puts trials on a common timeline by *time-warping*.
Warping is too vague to say *which* event (auditory stimulus, go cue, or speech onset)
actually triggers the semantic information. This pilot instead re-references the ORIGINAL
neural data to each candidate cue one at a time, decodes, and compares which alignment
yields the strongest and most temporally *locked* semantic readout. The cue that
de-smears the informative bin is the trigger.

Pipeline (all reuse — no re-implementation of binning/cleaning):
    align_runner  drive main/semantic_regression.py in-process with --warp none --align {cue}
                  (its align block already slices raw hg_data around the cue, floors the
                  window to whole bins so the cue lands on a bin boundary, trims to the min
                  window across trials, then runs the full clean+fit pipeline). One result
                  set per (cue, patient); compact per-bin metric arrays cached to npz.
    metrics       extract per-bin obs/null arrays from a fitted BasicRegressor; the
                  cue-relative bin-offset time axis (x=0 exactly at the aligned cue).
    aggregate     load all cells; group time-courses on the integer bin-offset grid; peak
                  table (height, latency, width); cross-patient other-cue bands; argmax vote.
    stats         per-bin per-patient permutation p (free from the fit's shuffled null),
                  Fisher-combined across patients (the n=6 Wilcoxon floor is p=1/64), BH-FDR;
                  paired Wilcoxon between alignments on peak values.
    figures       figures_for_paper-style panels: headline peak-height x temporal-locking
                  scatter, per-metric 4-cue time-course grid, peak / latency box+points,
                  within-patient argmax-cue vote, cue x metric heatmap, per-patient detail.
    report        auditory_alignment_report.html (base64 figures + tables + caveats).
    run           CLI orchestrator: python -m tests.auditory_alignment.run

Metrics compared: cosine (descriptive; no stored null), category(indep) accuracy,
word top-1/3/5, plus R2 (fit significance with a real null). Cues: stim_on, stim_off,
go_cue, voice_on. Six auditory patients: AA AZ DR LH RB WBH.

Nothing outside tests/ imports this. Outputs go under results/auditory_alignment/ via
utils.paths.results_dir. Run in the Speech conda env (project pkls need dill).
"""
