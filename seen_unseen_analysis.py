import sys, os, pickle, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import PIC_RUN, run_dir   # noqa: E402

RUN_DIR = str(run_dir(PIC_RUN))

with open(os.path.join(RUN_DIR, 'meta.json')) as f:
    meta = json.load(f)

patients = meta['succeeded_patients']
embedding_names = meta['embedding_names']

rows = []
for patient in patients:
    pkl_path = os.path.join(RUN_DIR, patient, 'semantic_regression_results.pkl')
    if not os.path.exists(pkl_path):
        continue
    print(f'Loading {patient}...')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    labels_all = data['clean_answer_labels']
    n_trials = len(labels_all)
    for emb_name in embedding_names:
        if emb_name not in data['regressors']:
            continue
        br = data['regressors'][emb_name]
        if not br.all_retrieval_pairs:
            continue
        word2idx = br.word_to_index
        trial_word_idx = np.array([word2idx[w] for w in labels_all], dtype=np.uint16)
        for pair in br.all_retrieval_pairs:
            bin_idx = pair['bin_index']
            test_idx = pair['test_indices']
            true_widx = pair['true_word_idx']
            pred_widx = pair['pred_word_idx']
            train_idx = np.setdiff1d(np.arange(n_trials), test_idx)
            train_words = set(trial_word_idx[train_idx].tolist())
            seen_mask = np.array([w in train_words for w in true_widx])
            unseen_mask = ~seen_mask
            n_seen = int(seen_mask.sum())
            n_unseen = int(unseen_mask.sum())
            correct = (true_widx == pred_widx)
            acc_seen = float(correct[seen_mask].mean()) if n_seen > 0 else float('nan')
            acc_unseen = float(correct[unseen_mask].mean()) if n_unseen > 0 else float('nan')
            acc_all = float(correct.mean())
            cat_seen = float('nan')
            cat_unseen = float('nan')
            if 'pred_category_idx_indep' in pair and br.word_index_to_category_index is not None:
                true_cat = br.word_index_to_category_index[true_widx]
                pred_cat = pair['pred_category_idx_indep']
                cat_correct = (true_cat == pred_cat)
                cat_seen = float(cat_correct[seen_mask].mean()) if n_seen > 0 else float('nan')
                cat_unseen = float(cat_correct[unseen_mask].mean()) if n_unseen > 0 else float('nan')
            rows.append({'patient': patient, 'embedding': emb_name, 'bin_index': bin_idx, 'n_test': len(test_idx), 'n_seen': n_seen, 'n_unseen': n_unseen, 'frac_unseen': n_unseen/len(test_idx), 'word_acc_seen': acc_seen, 'word_acc_unseen': acc_unseen, 'word_acc_all': acc_all, 'cat_acc_seen': cat_seen, 'cat_acc_unseen': cat_unseen})
    del data
    import gc; gc.collect()

df = pd.DataFrame(rows)
out_path = os.path.join(RUN_DIR, 'seen_unseen_analysis.csv')
df.to_csv(out_path, index=False)
print(f'Saved {len(df)} rows to {out_path}')

for emb in embedding_names:
    sub = df[df['embedding'] == emb]
    if sub.empty: continue
    by_pat_bin = sub.groupby(['patient','bin_index']).agg({'word_acc_seen':'mean','word_acc_unseen':'mean','word_acc_all':'mean','cat_acc_seen':'mean','cat_acc_unseen':'mean','n_unseen':'mean','frac_unseen':'mean'}).reset_index()
    best = by_pat_bin.loc[by_pat_bin.groupby('patient')['word_acc_all'].idxmax()]
    print(f'{emb}: seen={best["word_acc_seen"].mean():.4f}+/-{best["word_acc_seen"].std():.4f} unseen={best["word_acc_unseen"].mean():.4f}+/-{best["word_acc_unseen"].std():.4f} cat_seen={best["cat_acc_seen"].mean():.4f} cat_unseen={best["cat_acc_unseen"].mean():.4f} frac_unseen={best["frac_unseen"].mean():.4f} n_unseen={best["n_unseen"].mean():.1f}')
