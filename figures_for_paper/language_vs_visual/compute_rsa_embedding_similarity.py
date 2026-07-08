# -*- coding: utf-8 -*-
"""
Panel a — Procrustes similarity between the four embedding models.

Shows that the language models (GloVe, Word2Vec) and the vision models (DINOv3, MoCo)
encode the stimulus set in informationally distinct ways — the premise for the whole
figure ("families blind to one another by construction"). Stimulus-level, patient-independent.

Method (following main/notebooks/embedding_comparison.ipynb, exec 24 + 26):
  * one vector per stimulus *concept* per model
      - language: GloVe-840B-300d / Word2Vec-google-news-300 vector of the concept word
      - vision:  mean of the concept's exemplar images' pooled embedding
                 (dinov3_pooled 384-d / moco_pooled 512-d)
  * PCA each model's concept matrix to a common k components
  * between models: Procrustes alignment (orthogonal_procrustes) of the PCA spaces, scored by the
    mean per-concept cosine similarity between the rotated source and the target → 4×4 matrix.

Outputs → source_data/
  panel_a_procrustes_matrix.csv   (primary — models × models Procrustes similarity)
  supp_rsa_spearman_matrix.csv    (supplement — second-order Spearman-of-cosine-RDMs)

Run (Speech env; cwd = main/):
  python figures_for_paper/language_vs_visual/compute_rsa_embedding_similarity.py
"""

import os
import re
import sys
import pickle as pk
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial import procrustes as sp_procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

PCA_K = 10   # common PCA dimensionality before Procrustes (n_concepts≈64; top-10 PCs, denoised)

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(HERE))
SRC_DIR = os.path.join(HERE, 'source_data')
EMB_DIR = os.path.join(MAIN_DIR, 'embeddings', 'pictureNaming extended all')
MODELS = ['GloVe', 'Word2Vec', 'DINOv3', 'MoCo']


def _concept(w):
    """'airplane1' -> 'airplane' (strip trailing exemplar index)."""
    return re.sub(r'\d+$', '', str(w)).strip().lower()


def _vision_concept_matrix(pkl, key):
    d = pk.load(open(os.path.join(EMB_DIR, pkl), 'rb'))
    words = [_concept(w) for w in d['words']]
    vecs = np.asarray(d[key], dtype=np.float64)
    by = {}
    for c, v in zip(words, vecs):
        by.setdefault(c, []).append(v)
    return {c: np.mean(vs, axis=0) for c, vs in by.items()}


def _language_lookup():
    os.chdir(MAIN_DIR)                                   # torchtext GloVe reads ./.vector_cache
    from torchtext.vocab import GloVe
    glove = GloVe(dim=300, name='840B')
    import gensim.downloader as gensim_api
    w2v = gensim_api.load('word2vec-google-news-300')

    def glove_vec(c):
        v = glove[c].numpy()
        return v if np.linalg.norm(v) > 0 else None

    def w2v_vec(c):
        for key in (c, c.capitalize(), c.upper()):
            if key in w2v:
                return np.asarray(w2v[key], dtype=np.float64)
        return None
    return glove_vec, w2v_vec


def procrustes_similarity(X, Y):
    """Procrustes similarity = 1 − disparity, where disparity is scipy.spatial.procrustes'
    standardized residual sum-of-squares after optimal translation/scaling/rotation of X onto Y
    (0 = identical geometry). Inputs must share width (we PCA both to a common k first).
    Higher = more similar representational geometry."""
    _, _, disparity = sp_procrustes(X, Y)
    return float(1.0 - disparity)


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    dino = _vision_concept_matrix('dinov3_layerwise_embeddings.pk', 'dinov3_pooled')
    moco = _vision_concept_matrix('moco_ssl_resnet18_layerwise_embeddings.pk', 'moco_pooled')
    glove_vec, w2v_vec = _language_lookup()

    concepts = sorted(set(dino) & set(moco))
    rows = {m: [] for m in MODELS}
    kept = []
    for c in concepts:
        g, w = glove_vec(c), w2v_vec(c)
        if g is None or w is None:
            continue                                     # drop OOV concepts (keep models aligned)
        rows['GloVe'].append(g)
        rows['Word2Vec'].append(w)
        rows['DINOv3'].append(dino[c])
        rows['MoCo'].append(moco[c])
        kept.append(c)
    print(f"[rsa] {len(kept)}/{len(concepts)} concepts with all four models present")

    raw = {m: np.vstack(rows[m]) for m in MODELS}
    n = len(MODELS)

    # ── primary: Procrustes similarity on a common PCA space ──────────────────
    k = min(PCA_K, len(kept) - 1)
    pcs = {m: PCA(n_components=k).fit_transform(raw[m]) for m in MODELS}
    proc = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            proc[i, j] = proc[j, i] = procrustes_similarity(pcs[MODELS[i]], pcs[MODELS[j]])
    pout = pd.DataFrame(proc, index=MODELS, columns=MODELS); pout.insert(0, 'model', MODELS)
    pout.to_csv(os.path.join(SRC_DIR, 'panel_a_procrustes_matrix.csv'), index=False)

    # ── supplement: second-order Spearman of cosine RDMs ─────────────────────
    rdms = {m: pdist(raw[m], metric='cosine') for m in MODELS}
    rsa = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(rdms[MODELS[i]], rdms[MODELS[j]])
            rsa[i, j] = rsa[j, i] = rho
    sout = pd.DataFrame(rsa, index=MODELS, columns=MODELS); sout.insert(0, 'model', MODELS)
    sout.to_csv(os.path.join(SRC_DIR, 'supp_rsa_spearman_matrix.csv'), index=False)

    print(f"[rsa] {len(kept)} concepts, k={k} PCs -> panel_a_procrustes_matrix.csv (+ supp_rsa_spearman_matrix.csv)")
    print("[rsa] Procrustes similarity:")
    with pd.option_context('display.float_format', lambda v: f'{v:.3f}'):
        print(pout.set_index('model'))
    # sanity: within-family should exceed cross-family
    wl = proc[0, 1]; wv = proc[2, 3]
    cross = np.mean([proc[0, 2], proc[0, 3], proc[1, 2], proc[1, 3]])
    print(f"[rsa] within-language={wl:.3f}  within-vision={wv:.3f}  mean-cross={cross:.3f}"
          f"  {'OK (within>cross)' if min(wl, wv) > cross else 'WARNING: cross not lower'}")


if __name__ == '__main__':
    main()
