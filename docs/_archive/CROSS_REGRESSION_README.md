# Cross-Task Semantic Regression

## Overview

This extension adds **cross-regression** capabilities to the existing cross-decoding paradigm. Instead of predicting discrete semantic categories, the regressor predicts continuous word embeddings from neural activity.

## Key Additions

### 1. Enhanced `BasicRegressor` Class

The `BasicRegressor` class in `model.py` has been extended with **ranked accuracy** metrics:

#### New Attributes
- `all_ranked_accuracy`: Tracks whether the predicted embedding's closest match is the correct word (top-1 accuracy)
- `all_top_k_accuracy`: Dictionary tracking top-k accuracy for different k values (e.g., top-3, top-5, top-10)

#### New Parameters in `fit()` method
```python
def fit(self, n_epochs=50, parallel=None, closest='l2', use_kfold=False, n_splits=5,
        compute_ranked_accuracy=True, top_k_values=[1, 3, 5, 10]):
```

- `compute_ranked_accuracy` (bool): Enable/disable ranked accuracy computation
- `top_k_values` (list): List of k values for computing top-k accuracy

#### New Method: `_compute_ranked_accuracy()`

This method computes ranked accuracy by:
1. Taking the predicted embedding from the regressor
2. Finding the k-nearest embeddings in the full dataset
3. Checking if the true target embedding is among the top-k nearest neighbors

**Ranked Accuracy Interpretation:**
- **Top-1 Accuracy**: The regressor predicts an embedding that is closest to the correct word's embedding
- **Top-k Accuracy**: The true word's embedding is among the k-nearest neighbors to the prediction
- This metric answers: "Can the regressor predict approximately the correct word embedding?"

### 2. New Notebook: `cross_semantic_regression.ipynb`

A comprehensive notebook that mirrors the structure of `cross_semantic_decoding.ipynb` but for regression:

#### Workflow:
1. **Load Data**: Same pipeline as cross-decoding
2. **Load Word Embeddings**: Map words to continuous embeddings (Word2Vec, GloVe, BERT, etc.)
3. **Within-Task Regression**: Train and test regressors within each task
4. **Cross-Task Regression**: Train on one task, test on another
5. **Visualizations**: 
   - R² scores over time
   - Ranked accuracy (top-1) over time
   - Top-k accuracy comparison
   - Summary statistics

## Usage Example

```python
from model import BasicRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np

# Prepare data
neural_data = ...  # Shape: (trials, time_bins, channels)
word_embeddings = ...  # Shape: (trials, embedding_dim)

# Setup regressor
x_reducer = PCA(n_components=50)  # Reduce neural dimensionality
regressor = Ridge(alpha=1.0)

br = BasicRegressor(regressor, x_reducer=x_reducer, y_reducer=None)
br.load_data(neural_data, word_embeddings, n_bins_history=1)

# Fit with ranked accuracy
br.fit(
    n_epochs=20,
    use_kfold=True,
    n_splits=5,
    compute_ranked_accuracy=True,
    top_k_values=[1, 3, 5, 10, 20]
)

# Access results
r2_scores = br.all_test_score.mean(0)  # Mean R² across epochs
ranked_acc = br.all_ranked_accuracy.mean(0)  # Mean top-1 accuracy
top_5_acc = br.all_top_k_accuracy[5].mean(0)  # Mean top-5 accuracy

print(f"Peak R² score: {r2_scores.max():.3f}")
print(f"Peak ranked accuracy (top-1): {ranked_acc.max():.3f}")
print(f"Peak top-5 accuracy: {top_5_acc.max():.3f}")
```

## Cross-Task Regression Example

```python
from sklearn.model_selection import KFold

# Train on Picture Flashing, test on Picture Naming
pf_data = ...  # Picture Flashing neural data
pf_embeddings = ...  # Picture Flashing word embeddings
pn_data = ...  # Picture Naming neural data
pn_embeddings = ...  # Picture Naming word embeddings

# Reformat data
from utils import reformat
pf_X_to_use = reformat(pf_data, n_bins_history=1)
pn_X_to_use = reformat(pn_data, n_bins_history=1)

# Cross-task evaluation
n_bins = len(pf_X_to_use)
cross_task_scores = np.zeros(n_bins)
cross_task_ranked_acc = np.zeros(n_bins)
cross_task_top5_acc = np.zeros(n_bins)

for t in range(n_bins):
    X_pf_t = pf_X_to_use[t]
    X_pn_t = pn_X_to_use[t]
    
    kf = KFold(n_splits=5, shuffle=True)
    fold_scores = []
    fold_ranked = []
    fold_top5 = []
    
    for train_idx, _ in kf.split(X_pf_t):
        # Train on PF
        X_train = X_pf_t[train_idx]
        y_train = pf_embeddings[train_idx]
        
        pca = PCA(n_components=50)
        X_train_low = pca.fit_transform(X_train)
        X_pn_low = pca.transform(X_pn_t)
        
        reg = Ridge(alpha=1.0)
        reg.fit(X_train_low, y_train)
        
        # Test on PN
        y_pred = reg.predict(X_pn_low)
        score = reg.score(X_pn_low, pn_embeddings)
        fold_scores.append(score)
        
        # Compute ranked accuracy
        n_correct_top1 = 0
        n_correct_top5 = 0
        for i, pred in enumerate(y_pred):
            distances = np.sum((pn_embeddings - pred) ** 2, axis=1)
            sorted_idx = np.argsort(distances)
            if sorted_idx[0] == i:
                n_correct_top1 += 1
            if i in sorted_idx[:5]:
                n_correct_top5 += 1
        
        fold_ranked.append(n_correct_top1 / len(y_pred))
        fold_top5.append(n_correct_top5 / len(y_pred))
    
    cross_task_scores[t] = np.mean(fold_scores)
    cross_task_ranked_acc[t] = np.mean(fold_ranked)
    cross_task_top5_acc[t] = np.mean(fold_top5)

print(f"Peak cross-task R²: {cross_task_scores.max():.3f}")
print(f"Peak cross-task ranked accuracy: {cross_task_ranked_acc.max():.3f}")
print(f"Peak cross-task top-5 accuracy: {cross_task_top5_acc.max():.3f}")
```

## Interpretation Guide

### R² Score
- Measures how well the regressor predicts the exact embedding values
- Range: (-∞, 1], where 1 is perfect prediction
- Sensitive to embedding magnitude and dimensionality

### Ranked Accuracy (Top-1)
- **More interpretable** than R² for understanding word-level performance
- Answers: "Does the predicted embedding point to the correct word?"
- Range: [0, 1], where 1 means all predictions are closest to the correct word

### Top-K Accuracy
- More lenient metric: "Is the correct word among the k-nearest neighbors?"
- Useful for understanding approximate prediction quality
- Higher k values should yield higher accuracy

### Example Interpretation
```
Peak R² score: 0.42
Peak ranked accuracy (top-1): 0.65
Peak top-5 accuracy: 0.85
```
This means:
- The regressor explains 42% of variance in embeddings
- 65% of predictions are closest to the correct word
- 85% of predictions have the correct word in the top-5 nearest neighbors

## Advantages Over Classification

1. **Continuous representation**: Captures semantic similarity, not just discrete categories
2. **Transferability**: Embeddings trained on large corpora (Word2Vec, BERT) generalize well
3. **Semantic distance**: Errors can be meaningful (confusing "cat" with "dog" vs "table")
4. **Ranked accuracy**: Provides interpretable metric for regression quality

## Files Modified

1. **`model.py`**:
   - Added `all_ranked_accuracy` and `all_top_k_accuracy` attributes to `BasicRegressor`
   - Extended `fit()` method with ranked accuracy parameters
   - Modified `_fit()` method to compute ranked accuracy
   - Added `_compute_ranked_accuracy()` helper method

2. **`cross_semantic_regression.ipynb`** (NEW):
   - Complete pipeline for cross-task regression
   - Visualizations for R² scores, ranked accuracy, and top-k accuracy
   - Comprehensive summary statistics

## Next Steps

1. **Load actual word embeddings**: Replace dummy embeddings with real ones (Word2Vec, GloVe, BERT)
2. **Run within-task regression**: Establish baseline performance
3. **Run cross-task regression**: Evaluate generalization
4. **Compare with classification**: See if continuous embeddings provide better insights
5. **Analyze semantic confusions**: Use embedding distances to understand prediction errors

## Questions?

For questions or issues, refer to:
- `model.py`: Implementation details
- `cross_semantic_regression.ipynb`: Usage examples
- `cross_semantic_decoding.ipynb`: Original cross-decoding paradigm
