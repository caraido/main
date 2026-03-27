"""
PATCH_pls_support.py
====================
Instructions for adding PLS support to semantic_regression.py.

Apply these changes manually (or run this script to auto-patch):
    python PATCH_pls_support.py

Changes:
  1. Add PLSRegression import
  2. Add --model CLI argument (krr, linear_ridge, pls, kernel_pls)
  3. Modify _make_regressor_pipeline() to accept model mode
  4. Modify run_regressions() to accept model mode and adjust y_reducer
  5. Update meta.json to capture model mode
"""

import re
import sys
import os

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # ── 1. Add PLSRegression import ───────────────────────────────────────────
    old_import = "from sklearn.linear_model import Ridge"
    new_import = ("from sklearn.linear_model import Ridge\n"
                  "from sklearn.cross_decomposition import PLSRegression")
    if 'PLSRegression' not in content:
        content = content.replace(old_import, new_import)
        print("  [+] Added PLSRegression import")

    # ── 2. Add PLS_COMPONENTS constant ────────────────────────────────────────
    if 'PLS_COMPONENTS' not in content:
        content = content.replace(
            "PARALLEL_WORKERS   = 10",
            "PARALLEL_WORKERS   = 10\nPLS_COMPONENTS     = 10        # n_components for PLS regression"
        )
        print("  [+] Added PLS_COMPONENTS constant")

    # ── 3. Replace _make_regressor_pipeline() ─────────────────────────────────
    old_pipeline = '''def _make_regressor_pipeline():
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('ridge',    Ridge(alpha=KRR_ALPHA)),
    ])'''

    new_pipeline = '''def _make_regressor_pipeline(mode='krr'):
    """
    Build the regression pipeline.

    Parameters
    ----------
    mode : str
        One of:
          'krr'          — Nystroem(RBF) + Ridge (current default, nonlinear + regularized)
          'linear_ridge' — Ridge only (linear + regularized)
          'pls'          — PLSRegression (linear, implicit regularization via n_components)
          'kernel_pls'   — Nystroem(RBF) + PLSRegression (nonlinear + implicit regularization)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if mode == 'krr':
        return Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('ridge',    Ridge(alpha=KRR_ALPHA)),
        ])
    elif mode == 'linear_ridge':
        return Pipeline([
            ('ridge', Ridge(alpha=KRR_ALPHA)),
        ])
    elif mode == 'pls':
        return Pipeline([
            ('pls', PLSRegression(n_components=PLS_COMPONENTS, scale=False)),
        ])
    elif mode == 'kernel_pls':
        return Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('pls',      PLSRegression(n_components=PLS_COMPONENTS, scale=False)),
        ])
    else:
        raise ValueError(f"Unknown model mode: {mode!r}. "
                         f"Choose from: krr, linear_ridge, pls, kernel_pls")'''

    if old_pipeline in content:
        content = content.replace(old_pipeline, new_pipeline)
        print("  [+] Replaced _make_regressor_pipeline()")
    else:
        print("  [!] Could not find _make_regressor_pipeline() to replace — apply manually")

    # ── 4. Update run_regressions() to accept model_mode ─────────────────────
    old_run = "def run_regressions(pdata, embeddings, n_epochs, closest='l2'):"
    new_run = "def run_regressions(pdata, embeddings, n_epochs, closest='l2', model_mode='krr'):"
    if old_run in content:
        content = content.replace(old_run, new_run)
        print("  [+] Updated run_regressions() signature")

    # Update the pipeline + y_reducer construction inside run_regressions
    old_br = "        br = BasicRegressor(_make_regressor_pipeline(), y_reducer=PCA(Y_PCA_COMPONENTS))"
    new_br = ("        # PLS handles dimensionality reduction internally — skip PCA\n"
              "        use_pca = model_mode not in ('pls', 'kernel_pls')\n"
              "        br = BasicRegressor(\n"
              "            _make_regressor_pipeline(mode=model_mode),\n"
              "            y_reducer=PCA(Y_PCA_COMPONENTS) if use_pca else None,\n"
              "        )")
    if old_br in content:
        content = content.replace(old_br, new_br)
        print("  [+] Updated BasicRegressor construction for PLS")

    # Update the call to run_regressions in main()
    old_call = '''            regressors = run_regressions(
                pdata, embeddings,
                n_epochs=args.epochs,
                closest=args.closest,
            )'''
    new_call = '''            regressors = run_regressions(
                pdata, embeddings,
                n_epochs=args.epochs,
                closest=args.closest,
                model_mode=args.model,
            )'''
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("  [+] Updated run_regressions() call in main()")

    # ── 5. Add --model CLI argument ───────────────────────────────────────────
    old_closest_arg = """    parser.add_argument(
        '--closest', choices=['l2', 'cosine'], default='l2',
        help='Retrieval similarity metric (l2 = Euclidean, cosine = cosine similarity)',
    )"""
    new_closest_arg = """    parser.add_argument(
        '--closest', choices=['l2', 'cosine'], default='l2',
        help='Retrieval similarity metric (l2 = Euclidean, cosine = cosine similarity)',
    )
    parser.add_argument(
        '--model', choices=['krr', 'linear_ridge', 'pls', 'kernel_pls'],
        default='krr',
        help='Regression model: krr (Nystroem+Ridge, default), linear_ridge, '
             'pls (Partial Least Squares), kernel_pls (Nystroem+PLS)',
    )"""
    if '--model' not in content:
        if old_closest_arg in content:
            content = content.replace(old_closest_arg, new_closest_arg)
            print("  [+] Added --model CLI argument")

    # ── 6. Update run_id to include model name ────────────────────────────────
    old_run_id = "    run_id = f'{timestamp}_KRR_{args.closest}_{args.epochs}ep'"
    new_run_id = "    run_id = f'{timestamp}_{args.model}_{args.closest}_{args.epochs}ep'"
    if old_run_id in content:
        content = content.replace(old_run_id, new_run_id)
        print("  [+] Updated run_id to include model name")

    # ── 7. Update meta.json to capture model mode ─────────────────────────────
    if "'closest':              args.closest," in content and "'model_mode'" not in content:
        content = content.replace(
            "'closest':              args.closest,",
            "'closest':              args.closest,\n"
            "        'model_mode':           args.model,"
        )
        print("  [+] Added model_mode to meta.json")

    # Update regressor_pipeline string in meta
    old_pipeline_str = "        'regressor_pipeline':   'Nystroem(kernel=\"rbf\") → Ridge(alpha={})'.format(KRR_ALPHA),"
    new_pipeline_str = ("        'regressor_pipeline':   f'{args.model}: ' + {\n"
                        "            'krr': f'Nystroem(rbf) → Ridge(α={KRR_ALPHA})',\n"
                        "            'linear_ridge': f'Ridge(α={KRR_ALPHA})',\n"
                        "            'pls': f'PLSRegression(n={PLS_COMPONENTS})',\n"
                        "            'kernel_pls': f'Nystroem(rbf) → PLSRegression(n={PLS_COMPONENTS})',\n"
                        "        }.get(args.model, '?'),")
    if old_pipeline_str in content:
        content = content.replace(old_pipeline_str, new_pipeline_str)
        print("  [+] Updated regressor_pipeline in meta.json")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nPatch applied to {filepath}")


if __name__ == '__main__':
    target = os.path.join(os.path.dirname(__file__), 'semantic_regression.py')
    print(f"Patching: {target}\n")
    patch_file(target)
    print("\nDone! Now you can run:")
    print("  python semantic_regression.py --model pls --closest cosine")
    print("  python semantic_regression.py --model krr   # (default, unchanged)")
