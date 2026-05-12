# -*- coding: utf-8 -*-
"""
utils.cli -- shared argparse builders for the pipeline scripts.

Canonical flag naming convention (use dashes, never underscores):

    --patients PATIENT [PATIENT ...]    list of patient IDs (nargs='+')
    --patient PATIENT                   single patient ID
    --task {picture_naming,auditory_naming}
    --warp {none,linear}                time-warping mode
    --align CUE                         alignment event
    --align-back FLOAT, --align-forward FLOAT
    --bin-size INT                      temporal bin size (ms)
    --epochs INT                        number of training epochs
    --model {kernel_pls,pls,krr,linear_ridge}
    --closest {l2,cosine}               retrieval metric
    --embedding NAME [NAME ...]         subset of embeddings
    --pls-components INT
    --pca-components INT
    --n-splits INT                      k-fold splits
    --seed INT
    --shuffles INT                      permutation count
    --run-dir PATH                      a results/<pipeline>/<run_id> path or 'latest'
    --results-dir PATH                  results/<pipeline>/ root
    --out-dir PATH                      output directory for this script
    --fig-dir PATH                      figure output directory
    --data-dir PATH                     data/ root
    --in-dir PATH                       generic input directory
    --smoke                             quick sanity-check mode (one patient, few epochs)
    --quiet                             suppress progress output
    --resume                            resume partial run

Public API:
    common_parser(prog=None, description=None, *, flag_groups=None)
        Build an ArgumentParser pre-populated with common flag groups.
        flag_groups: list of strings naming groups to include, e.g.
            ['patient', 'training', 'paths']
        Defaults to ['patient', 'training', 'paths'].

    add_patient_flags(parser)
    add_training_flags(parser)
    add_path_flags(parser)
    add_smoke_flag(parser)
"""

import argparse


# Underscore <-> dash aliases for back-compat with old commands.
# When migrating a script, replace each `parser.add_argument('--run_dir', ...)`
# with `parser.add_argument('--run-dir', '--run_dir', dest='run_dir', ...)`.
LEGACY_ALIASES = {
    '--run-dir':       ['--run_dir'],
    '--results-dir':   ['--results_dir'],
    '--out-dir':       ['--out_dir', '--out'],   # --out collapsed too
    '--fig-dir':       ['--fig_dir'],
    '--data-dir':      ['--data_dir'],
    '--in-dir':        ['--in_dir'],
    '--epochs':        ['--n_epochs'],
    '--pls-components': ['--pls_components'],
    '--pca-components': ['--pca_dims'],
    '--n-splits':      ['--n_splits'],
    '--shuffles':      ['--n_shuffle'],
    '--lex-embedding': ['--lex_embedding'],
    '--vis-embedding': ['--vis_embedding'],
    '--var-cutoff':    ['--var_cutoff'],
    '--align-method':  ['--align_method'],
    '--align-target':  ['--align_target'],
    '--run-rsa':       ['--run_rsa'],
    '--model-run-dir': ['--model_run_dir'],
    '--vanilla-run-dir': ['--vanilla_run_dir'],
    '--vanilla-fig-dir': ['--vanilla_fig_dir'],
    '--krr-run-dir':   ['--krr_run_dir'],
    '--model-label':   ['--model_label'],
}


def _add(parser, *names, **kwargs):
    """Wrapper around add_argument that also adds any registered legacy aliases."""
    canonical = names[0]
    extra = LEGACY_ALIASES.get(canonical, [])
    return parser.add_argument(*names, *extra, **kwargs)


def add_patient_flags(parser):
    """Patient / cohort selection flags."""
    _add(parser, '--patients', type=str, nargs='+', default=None,
         help='Patient IDs to process (e.g. AA AZ VB). Default: all auto-discovered.')
    _add(parser, '--patient', type=str, default=None,
         help='Single patient ID. Convenience alias for --patients PATIENT.')


def add_training_flags(parser):
    """Training / model selection flags."""
    _add(parser, '--task', type=str, default='picture_naming',
         choices=['picture_naming', 'auditory_naming', 'auditory_repetition'],
         help='Behavioural task. Default: picture_naming.')
    _add(parser, '--model', type=str, default='kernel_pls',
         choices=['kernel_pls', 'pls', 'krr', 'linear_ridge'],
         help='Regression model. Default: kernel_pls.')
    _add(parser, '--closest', type=str, default='cosine',
         choices=['cosine', 'l2'],
         help='Retrieval metric. Default: cosine.')
    _add(parser, '--epochs', type=int, default=50,
         help='Number of training epochs (or repeats for some models).')
    _add(parser, '--pls-components', type=int, default=10,
         help='Number of PLS components (PLS / Kernel PLS models).')
    _add(parser, '--pca-components', type=int, default=None,
         help='Optional PCA dimensions before regression.')
    _add(parser, '--n-splits', type=int, default=5,
         help='k for k-fold cross-validation.')
    _add(parser, '--seed', type=int, default=0,
         help='Random seed.')
    _add(parser, '--shuffles', type=int, default=0,
         help='Number of permutation null shuffles (0 = none).')


def add_path_flags(parser):
    """Path / output flags."""
    _add(parser, '--run-dir', type=str, default=None,
         help='Path to a specific results/<pipeline>/<run_id> directory, or "latest".')
    _add(parser, '--results-dir', type=str, default=None,
         help='Root of results/ tree to scan for runs.')
    _add(parser, '--out-dir', type=str, default=None,
         help='Output directory for this script (defaults to results/<pipeline>/<run_id>/).')
    _add(parser, '--fig-dir', type=str, default=None,
         help='Figure output directory.')
    _add(parser, '--data-dir', type=str, default=None,
         help='Data root directory (defaults to ./data).')


def add_alignment_flags(parser):
    """Temporal alignment flags."""
    _add(parser, '--warp', type=str, default='none',
         choices=['none', 'linear'],
         help='Time-warping mode.')
    _add(parser, '--align', type=str, default='none',
         help='Behavioural event to align trials around.')
    _add(parser, '--align-back', type=float, default=None,
         help='Seconds before alignment cue.')
    _add(parser, '--align-forward', type=float, default=None,
         help='Seconds after alignment cue.')


def add_smoke_flag(parser):
    _add(parser, '--smoke', action='store_true',
         help='Smoke-test mode: one patient, few epochs, quick output.')


def add_quiet_flag(parser):
    _add(parser, '--quiet', action='store_true',
         help='Suppress progress output.')


# Map of group name -> populator function
GROUP_BUILDERS = {
    'patient':   add_patient_flags,
    'training':  add_training_flags,
    'paths':     add_path_flags,
    'alignment': add_alignment_flags,
    'smoke':     add_smoke_flag,
    'quiet':     add_quiet_flag,
}


def common_parser(prog=None, description=None, *, flag_groups=None):
    """Build an ArgumentParser pre-populated with the named flag groups.

    Defaults to ['patient', 'training', 'paths'] which covers most pipeline
    scripts; add 'alignment' / 'smoke' / 'quiet' as needed via flag_groups=...
    """
    if flag_groups is None:
        flag_groups = ['patient', 'training', 'paths']
    parser = argparse.ArgumentParser(prog=prog, description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for group in flag_groups:
        if group not in GROUP_BUILDERS:
            raise KeyError(f"Unknown CLI flag group: {group!r}; expected one of {list(GROUP_BUILDERS)}")
        GROUP_BUILDERS[group](parser)
    return parser
