# Summary of Changes to utils.py

## Overview
Changed all plotting functions to use fixed time intervals between x-axis ticks instead of a fixed number of ticks. The ticks are now always aligned to round numbers and "zeroed" at the `back` parameter.

## Changes Made

### 1. `plot_on_channel()`
- **Parameter change**: `n_xticks=6` → `tick_interval=1.0`
- **Default**: 1.0 second between ticks
- **Behavior**: Ticks are generated at fixed intervals (e.g., -2, -1, 0, 1, 2, 3, 4) aligned to 0

### 2. `plot_accuracy_plotly()`
- **Parameter change**: `n_xticks=6` → `tick_interval=1.0`
- **Default**: 1.0 second between ticks
- **Behavior**: Same as above

### 3. `interactive_3d_scatter_plot()`
- **Parameter change**: `n_xticks=6` → `tick_interval=1.0`
- **Default**: 1.0 second between ticks
- **Passes parameter to**: `plot_accuracy_plotly()`

### 4. `interactive_channel_importance()`
- **Parameter change**: `n_xticks=20` → `tick_interval=0.5`
- **Default**: 0.5 second between ticks (more granular due to heatmap)
- **Passes parameter to**: `plot_accuracy_plotly()`

### 5. `interactive_confusion_accuracy()`
- **Parameter change**: `n_xticks=20` → `tick_interval=0.5`
- **Default**: 0.5 second between ticks (more granular due to confusion matrix)
- **Passes parameter to**: `plot_accuracy_plotly()`

## Algorithm

The new tick generation algorithm:
```python
# Generate ticks at fixed intervals, aligned to 0
tick_start = -np.floor(back / tick_interval) * tick_interval
tick_end = np.ceil(forward / tick_interval) * tick_interval
x_ticks = np.arange(tick_start, tick_end + tick_interval/2, tick_interval)
```

## Example

If `back=2.1`, `forward=4.2`, and `tick_interval=1`:
- `tick_start = -floor(2.1/1) * 1 = -2`
- `tick_end = ceil(4.2/1) * 1 = 5`
- `x_ticks = [-2, -1, 0, 1, 2, 3, 4, 5]`

This ensures ticks are at nice round numbers regardless of the exact values of `back` and `forward`.

## Usage

You can now easily adjust the tick density by changing the `tick_interval` parameter:
- `tick_interval=0.5` → ticks every 0.5 seconds (finer)
- `tick_interval=1.0` → ticks every 1 second (default)
- `tick_interval=2.0` → ticks every 2 seconds (coarser)
