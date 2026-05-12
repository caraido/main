# Cross-Task Semantic Regression: Auditory Task Support

## Overview

The cross-task semantic regression notebook has been enhanced to support **auditory naming** and **auditory repetition** tasks, with flexible time alignment strategies to handle the longer stimulus presentation periods in auditory tasks.

## Key Enhancement: Time Alignment Strategies

### The Challenge

Different tasks have fundamentally different temporal characteristics:

| Task Type | Stimulus Duration | Semantic Processing Timing |
|-----------|------------------|---------------------------|
| Picture Flashing | ~0.3-0.5s | Immediate |
| Picture Naming | ~0.5-0.8s | Early-to-mid trial |
| Auditory Repetition | ~0.8-1.5s | Mid-trial |
| **Auditory Naming** | **~2-3s** | **Late in presentation** |

**Problem:** Auditory naming has much longer stimulus presentation because:
1. The auditory stimulus (spoken word) takes time to play
2. Semantic processing happens during/after hearing the full word
3. Response (voice onset) occurs much later than in visual tasks

**Solution:** Implement flexible time alignment strategies.

## Three Alignment Methods

### 1. `time_warp` (Recommended for Cross-Task Analysis)

**Purpose:** Align comparable cognitive stages across tasks

**How it works:**
```python
alignment_method = 'time_warp'
reference_task = 'picture_naming'  # Use this as timing reference

# For auditory naming:
# Original: [trial_onset] -------- [stimulus] ---------- [voice_onset]
#                0s                    2.5s                  3.0s
#
# Warped:   [trial_onset] -- [stimulus] -- [voice_onset]
#                0s             0.5s           0.8s
#           (compressed to match picture naming timing)
```

**Implementation:**
- Uses interpolation to compress/expand neural activity timecourse
- Aligns key events: go cue → voice onset
- Preserves neural dynamics while standardizing timeline

**When to use:**
- Comparing semantic representations across modalities (auditory ↔ visual)
- Training on one task, testing on another
- Understanding shared cognitive processes

**Advantages:**
- ✅ Enables direct cross-task comparison
- ✅ Aligns comparable cognitive stages
- ✅ Better cross-task generalization expected

**Disadvantages:**
- ⚠️ May distort native temporal dynamics
- ⚠️ Interpolation can introduce artifacts

### 2. `keep_original` (Best for Within-Task Analysis)

**Purpose:** Preserve native task timing

**How it works:**
```python
alignment_method = 'keep_original'

# Each task keeps its original trial length:
# Picture Naming:    30 time bins (0.5s per trial)
# Auditory Naming:   120 time bins (2.0s per trial)
# Picture Flashing:  20 time bins (0.3s per trial)
```

**When to use:**
- Analyzing individual tasks separately
- Understanding task-specific temporal dynamics
- When timing precision is critical (e.g., studying auditory processing stages)

**Advantages:**
- ✅ Preserves native neural dynamics
- ✅ No interpolation artifacts
- ✅ Best for task-specific findings

**Disadvantages:**
- ⚠️ Cannot concatenate tasks directly
- ⚠️ Requires separate analysis per task
- ⚠️ Difficult to compare across tasks

### 3. `truncate` (Not Recommended for Auditory)

**Purpose:** Simple alignment by cutting to minimum length

**How it works:**
```python
alignment_method = 'truncate'

# All tasks truncated to shortest:
# Picture Naming:    20 time bins (truncated from 30)
# Auditory Naming:   20 time bins (truncated from 120)  ❌ Loses critical info!
# Picture Flashing:  20 time bins (unchanged)
```

**When to use:**
- Quick exploratory analysis
- When all tasks have similar lengths

**Advantages:**
- ✅ Simple to implement
- ✅ All tasks same length

**Disadvantages:**
- ❌ **Loses critical information from auditory naming**
- ❌ Semantic processing happens at 2-3s, truncation cuts this off
- ❌ Not recommended for auditory tasks

## New Functions

### `load_task_data()`

Unified data loader for all task types:
```python
task_data = load_task_data(
    task_path='data/RB/auditoryNaming_all_data.mat',
    task_type='auditory_naming',
    bin_size=100,
    bad_channels=[],
    bad_trials=[],
    shank_of_interest=['A', 'B', 'C'],
    word_to_embedding=word_to_embedding_dict
)
```

Returns:
- `clean_data_binned`: Neural activity (trials × channels × time_bins)
- `clean_target_words`: Target words for each trial
- `embeddings`: Word embeddings for each trial
- `clean_trial_onset`, `clean_voice_onset`, etc.: Timing information
- `task_type`: Task identifier

### `time_warp_trials()`

Warps neural activity to match reference timing:
```python
warped_data = time_warp_trials(
    data_binned=auditory_data,
    source_onsets=auditory_go_cue_times,
    source_offsets=auditory_voice_onset_times,
    target_onsets=visual_go_cue_time,  # Reference timing
    target_offsets=visual_voice_onset_time
)
```

**Key features:**
- Preserves pre-onset and post-offset periods
- Compresses/expands middle period (stimulus → response)
- Uses linear interpolation for smooth warping
- Per-trial warping to handle variability

## Usage Examples

### Example 1: Cross-Modality Analysis with Time Warping

```python
# Configuration
alignment_method = 'time_warp'
reference_task = 'picture_naming'

# Load tasks
tasks_to_load = [
    {'file': 'pictureNaming_all_data.mat', 'task_type': 'picture_naming'},
    {'file': 'auditoryNaming_all_data.mat', 'task_type': 'auditory_naming'},
]

# ... load data ...

# Train on auditory naming, test on picture naming
br_an = BasicRegressor(Ridge(alpha=1.0), x_reducer=PCA(n_components=50))
br_an.load_data(auditory_neural_warped, auditory_embeddings, n_bins_history=1)
br_an.fit(n_epochs=20, use_kfold=True, compute_ranked_accuracy=True)

# Test on picture naming
# Cross-task code here...

print(f"Cross-modality ranked accuracy: {cross_modal_acc.max():.3f}")
# Expected: Higher accuracy with warping than without
```

### Example 2: Within-Task Analysis with Original Timing

```python
# Configuration
alignment_method = 'keep_original'

# Analyze auditory naming with native timing
br_an_original = BasicRegressor(Ridge(alpha=1.0), x_reducer=PCA(n_components=50))
br_an_original.load_data(auditory_neural_original, auditory_embeddings, n_bins_history=1)
br_an_original.fit(n_epochs=20, use_kfold=True, compute_ranked_accuracy=True)

# Plot ranked accuracy over time
plt.plot(br_an_original.all_ranked_accuracy.mean(0))
plt.axvline(x=25, label='Stimulus onset', linestyle='--')
plt.axvline(x=90, label='Expected semantic processing', linestyle='--', color='red')
plt.xlabel('Time bin (100ms)')
plt.ylabel('Ranked Accuracy')
plt.legend()
```

### Example 3: Compare Warping vs Original

```python
alignment_method = 'time_warp'

# Load with warping
br_warped = BasicRegressor(...)
br_warped.load_data(auditory_neural_warped, embeddings, ...)
br_warped.fit(...)

# Load original data
br_original = BasicRegressor(...)
br_original.load_data(auditory_neural_original, embeddings, ...)
br_original.fit(...)

# Compare
print(f"Warped peak accuracy: {br_warped.all_ranked_accuracy.mean(0).max():.3f}")
print(f"Original peak accuracy: {br_original.all_ranked_accuracy.mean(0).max():.3f}")
print(f"Warped peak at: {br_warped.all_ranked_accuracy.mean(0).argmax() * 0.1:.2f}s")
print(f"Original peak at: {br_original.all_ranked_accuracy.mean(0).argmax() * 0.1:.2f}s")
```

## Expected Findings

### Within-Task Performance

| Task | Expected Peak Ranked Accuracy | Peak Timing |
|------|------------------------------|-------------|
| Picture Flashing | 0.6-0.8 | 0.3-0.5s |
| Picture Naming | 0.5-0.7 | 0.5-0.8s |
| Auditory Repetition | 0.4-0.6 | 0.8-1.2s |
| Auditory Naming (original) | 0.4-0.6 | 2.0-2.5s |
| Auditory Naming (warped) | 0.4-0.6 | 0.5-0.8s (aligned) |

### Cross-Task Generalization

**With time warping:**
```
Train on auditory naming → test on picture naming
Expected ranked accuracy: 0.3-0.5 (good generalization)

Train on picture naming → test on auditory naming  
Expected ranked accuracy: 0.3-0.5 (good generalization)
```

**Without time warping (keep_original):**
```
Train on auditory naming → test on picture naming
Expected ranked accuracy: 0.1-0.2 (poor generalization)
Reason: Temporal misalignment between tasks
```

## Recommended Analysis Pipeline

### Step 1: Load and Prepare Data

```python
# Set alignment method
alignment_method = 'time_warp'
reference_task = 'picture_naming'

# Load all tasks
tasks_to_load = [
    {'file': 'pictureNaming_all_data.mat', 'task_type': 'picture_naming'},
    {'file': 'pictureFlashing_all_data.mat', 'task_type': 'picture_flashing'},
    {'file': 'auditoryNaming_all_data.mat', 'task_type': 'auditory_naming'},
    {'file': 'auditoryRepetition_all_data.mat', 'task_type': 'auditory_repetition'},
]

# Load and apply alignment
# (see notebook for full code)
```

### Step 2: Within-Task Regression

```python
# Train and test on each task separately
for task_type in ['picture_naming', 'auditory_naming', ...]:
    br = BasicRegressor(Ridge(alpha=1.0), x_reducer=PCA(n_components=50))
    br.load_data(neural_data, embeddings, n_bins_history=1)
    br.fit(n_epochs=20, use_kfold=True, compute_ranked_accuracy=True, 
           top_k_values=[1, 3, 5, 10])
    
    print(f"{task_type}: Peak ranked accuracy = {br.all_ranked_accuracy.mean(0).max():.3f}")
```

### Step 3: Cross-Task Analysis

```python
# Cross-task pairs to test:
pairs = [
    ('picture_flashing', 'picture_naming'),  # Visual-visual
    ('auditory_naming', 'picture_naming'),   # Auditory-visual
    ('auditory_repetition', 'auditory_naming'),  # Auditory-auditory
]

for train_task, test_task in pairs:
    # Train on train_task, test on test_task
    # Measure cross-task ranked accuracy
```

### Step 4: Interpret Results

1. **High within-task accuracy** → Task has decodable semantic information
2. **High cross-task accuracy** → Shared semantic representations
3. **Low cross-task accuracy** → Task-specific representations or timing issues

## Troubleshooting

### Issue: Poor cross-task generalization with auditory tasks

**Solution:**
- Ensure `alignment_method = 'time_warp'` is set
- Check that reference task has appropriate timing
- Verify warping is actually applied (check shape changes)

### Issue: Different numbers of time bins after loading

**Solution:**
- This is expected with `keep_original` method
- Either switch to `time_warp` or analyze tasks separately
- Don't concatenate tasks with different lengths

### Issue: Peak accuracy at unexpected time

**Solution:**
- For auditory naming with original timing, peak should be late (~2-3s)
- For warped auditory naming, peak should align with reference task
- Check alignment_method and verify warping applied correctly

## Key Takeaways

1. **Time warping is essential for comparing auditory and visual tasks**
2. **Auditory naming requires special handling due to long stimulus presentation**
3. **Use `time_warp` for cross-task, `keep_original` for within-task**
4. **Ranked accuracy provides interpretable semantic prediction metric**
5. **Compare warped vs original to validate findings**

## Files Updated

- **`cross_semantic_regression.ipynb`**: Added auditory task support and alignment methods
- **Configuration section**: Added alignment_method parameter
- **Data loading**: Added load_task_data() and time_warp_trials() functions
- **Analysis sections**: Added cross-modality analysis examples
- **Recommendations**: Added detailed guidance for auditory tasks

## Next Steps

1. Load actual word embeddings (Word2Vec, GloVe, BERT)
2. Run within-task regression on all tasks
3. Compare warped vs original timing for auditory tasks
4. Run cross-modality analysis (auditory ↔ visual)
5. Analyze semantic processing timing differences
6. Publish findings!
