# -*- coding: utf-8 -*-
"""utils.interactive -- interactive plotly visualisations."""

import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qual

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Color palettes also used here.
DATA_COLORS = ['blue', 'orange', 'grey', 'cyan', 'magenta', 'yellow', 'grey', 'green', 'purple', 'black']
LINE_COLORS = ['black', 'blue', 'green', 'magenta', 'cyan']

def interactive_3d_scatter_plot(
    data_list, label_list, main_data, *extra_data, title_3D,
    data_std=None, back=None, forward=None,
    lines=None, line_labels=None, tick_interval=1.0, title_2D=None,
    data_labels=None, p=None, label_name=None
):
    """
    Synchronized 3D+2D Plotly animation with slider, exportable as HTML.
    """
    n_time = len(data_list)
    assert len(label_list) == n_time

    # Helper for colors
    def get_colors(labels):
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab10')
        color_map = {label: mcolors.to_hex(cmap(i % cmap.N)) for i, label in enumerate(unique_labels)}
        return np.array([color_map[label] for label in labels]), color_map, unique_labels

    # Label mapping
    if label_name is not None:
        unique_labels = np.unique(label_list[0])
        assert len(label_name) == len(unique_labels), "label_name must match number of unique labels"
        label_map = {label: name for label, name in zip(unique_labels, label_name)}
    else:
        label_map = {label: str(label) for label in np.unique(label_list[0])}

    # 2D accuracy plot (get static traces, we'll animate the vertical line)
    fig2d, x_vals = plot_accuracy_plotly(
        main_data, *extra_data, data_std=data_std, back=back, forward=forward,
        lines=lines, line_labels=line_labels, tick_interval=tick_interval, title=title_2D,
        data_labels=data_labels, p=p, truncated=len(main_data)-len(data_list)
    )
    vline_index = len(fig2d.data) - 1
    y_min, y_max = fig2d.layout.yaxis.range

    # --- Create subplots: 3D scatter (scene), 2D accuracy (x/y) ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=(title_3D, title_2D)
    )

    # Add legend traces for each label (static, for legend only)
    labels0 = label_list[0]
    _, color_map, unique_labels = get_colors(labels0)
    for label in unique_labels:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=8, color=color_map[label], opacity=0.8),
            name=label_map[label],
            showlegend=True,
            legendgroup='3d'
        ), row=1, col=1)

    # Add static 2D traces (except vline)
    for i, trace in enumerate(fig2d.data):
        if i != vline_index:
            trace.legendgroup = '2d'
            trace.showlegend = True
            fig.add_trace(trace, row=1, col=2)

    # Add placeholders for animated traces (these will be updated by frames)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        marker=dict(size=5, color=[], opacity=0.8),
        showlegend=False,
        legendgroup='3d'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False,
        legendgroup='2d'
    ), row=1, col=2)

    # --- Animation frames ---
    frames = []
    for t in range(n_time):
        # 3D scatter
        X = data_list[t]
        labels = label_list[t]
        colors, _, _ = get_colors(labels)
        scatter3d_frame = go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[label_map[l] for l in labels],
            showlegend=False,
            legendgroup='3d'
        )
        # 2D vline
        vline_frame = go.Scatter(
            x=[x_vals[t], x_vals[t]],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Current Time',
            showlegend=False,
            legendgroup='2d'
        )
        # Only update the last two traces (the animated ones)
        frames.append(go.Frame(
            data=[scatter3d_frame, vline_frame],
            name=str(t),
            traces=[len(fig.data)-2, len(fig.data)-1]
        ))

    # Set initial data for the animated traces
    fig.data[-2].x = frames[0].data[0].x
    fig.data[-2].y = frames[0].data[0].y
    fig.data[-2].z = frames[0].data[0].z
    fig.data[-2].marker = frames[0].data[0].marker
    fig.data[-2].text = frames[0].data[0].text

    fig.data[-1].x = frames[0].data[1].x
    fig.data[-1].y = frames[0].data[1].y
    fig.data[-1].line = frames[0].data[1].line

    # --- Slider steps ---
    steps = []
    for t in range(n_time):
        steps.append(dict(
            method="animate",
            args=[[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            label=str(t)
        ))

    # --- Layout ---
    fig.update_layout(
        height=600, width=1200,
        sliders=[{
            "active": 0,
            "pad": {"t": 50},
            "steps": steps,
            "currentvalue": {"prefix": "Time: "}
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
            }]
        }],
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        ),
        legend=dict(itemsizing='constant', font=dict(size=12)),
        margin=dict(l=0, r=0, b=30, t=100),
        xaxis2=dict(title="Time (s)"),
        yaxis2=dict(title="Accuracy", range=[y_min, y_max]),
        showlegend=True
    )

    fig.frames = frames

    return fig


def interactive_channel_importance(
    importance_data, decoder_accuracy, chance_accuracy,
    voice_onset, voice_offset, green_screen_onset, trial_offset,
    p_values=None, tick_interval=0.5, title_heatmap=None, title_accuracy=None,
    data_std=None, back=0, forward=None, lines=None, line_labels=None,
    data_labels=None
):
    """
    Interactive visualization of channel importance over time.
    Top: Channel importance heatmap
    Bottom: Decoding accuracy plot with event markers
    """
    n_time = len(importance_data)
    
    # Find the maximum shape and pad all arrays to match
    max_shape = max(arr.shape[0] for arr in importance_data)
    padded_importance_data = []
    for arr in importance_data:
        if arr.shape[0] < max_shape:
            padded = np.pad(arr, ((0, max_shape - arr.shape[0]), (0, 0)), 
                          mode='constant', constant_values=0)
            padded_importance_data.append(padded)
        else:
            padded_importance_data.append(arr)
    
    importance_data = padded_importance_data
    
    # Create subplots: heatmap (top), accuracy plot (bottom)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(title_heatmap, title_accuracy),
        row_heights=[0.6, 0.4],
        specs=[[{"type": "heatmap"}], [{"type": "xy"}]]
    )

    # Get the accuracy plot using plot_accuracy_plotly
    fig2d, x_vals = plot_accuracy_plotly(
        decoder_accuracy, chance_accuracy,
        data_std=data_std,
        back=back, forward=forward or trial_offset.mean(),
        lines=lines or [0, np.nanmean(green_screen_onset),
                       np.nanmean(voice_onset), np.nanmean(voice_offset)],
        line_labels=line_labels or ['trial onset', 'go cue', 'voice on', 'voice off'],
        tick_interval=tick_interval,
        title=title_accuracy,
        data_labels=data_labels or ["test accuracy", "chance", "train accuracy"],
        p=p_values
    )

    # Add all traces from accuracy plot
    for trace in fig2d.data:
        fig.add_trace(trace, row=2, col=1)

    # Add heatmap with smaller colorbar
    fig.add_trace(
        go.Heatmap(
            z=importance_data[0],
            colorscale='Viridis',
            showscale=True,
            zmin=np.min([np.min(arr) for arr in importance_data]),
            zmax=np.max([np.max(arr) for arr in importance_data]),
            colorbar=dict(
                len=0.5,  # Make colorbar 50% of original size
                y=0.8,    # Position it higher
                thickness=15  # Make it thinner
            )
        ), row=1, col=1
    )

    # Create frames for animation
    frames = []
    x_vals = np.linspace(0, trial_offset.mean(), len(importance_data))
    for t in range(len(importance_data)):
        frames.append(go.Frame(
            data=[
                # Keep all static traces unchanged
                *fig2d.data[:-1],  # All traces except the last one (vertical line)
                # Update vertical line
                go.Scatter(
                    x=[x_vals[t], x_vals[t]],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                # Update heatmap
                go.Heatmap(
                    z=importance_data[t],
                    colorscale='Viridis',
                    showscale=True,
                    zmin=np.min([np.min(arr) for arr in importance_data]),
                    zmax=np.max([np.max(arr) for arr in importance_data]),
                    colorbar=dict(
                        len=0.5,
                        y=0.8,
                        thickness=15
                    )
                )
            ],
            name=str(t)
        ))

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        margin=dict(t=100),
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "method": "animate",
                    "args": [[str(k)], {
                        "frame": {"duration": 0, "redraw": True},  # Set duration to 0 for smoother sliding
                        "mode": "immediate",
                        "transition": {"duration": 0}  # Remove transition for smoother updates
                    }],
                    "label": str(k)
                } for k in range(len(frames))
            ],
            "currentvalue": {"prefix": "Time: "}
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 200, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            }]
        }]
    )

    # Update axes
    fig.update_xaxes(title="Channel index", row=1, col=1)
    fig.update_yaxes(title="Time bin", row=1, col=1)

    fig.frames = frames

    return fig


def interactive_confusion_accuracy(
    true_labels_list, pred_labels_list, 
    decoder_accuracy, chance_accuracy,
    voice_onset, voice_offset, green_screen_onset, trial_offset,
    p_values=None, tick_interval=0.5, title_confusion=None, title_accuracy=None,
    data_std=None, back=0, forward=None, lines=None, line_labels=None,
    data_labels=None, normalize=None
):
    """
    Interactive visualization combining confusion matrix and accuracy plot.
    Left: Animated confusion matrix
    Right: Decoding accuracy plot with time indicator
    
    Parameters:
    -----------
    true_labels_list : list of arrays
        List of true labels for each time point
    pred_labels_list : list of arrays
        List of predicted labels for each time point
    normalize : None or {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be normalized.
    """
    assert len(true_labels_list) == len(pred_labels_list)
    n_time = len(true_labels_list)
    
    # Get all unique labels to ensure consistent matrix size
    all_labels = np.unique(np.concatenate([np.unique(np.concatenate(true_labels_list)),
                                         np.unique(np.concatenate(pred_labels_list))]))
    all_labels.sort()  # Ensure consistent order

    if forward is None:
        forward = trial_offset.mean()
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title_confusion, title_accuracy),
        column_widths=[0.5, 0.5],
        specs=[[{"type": "heatmap"}, {"type": "xy"}]]
    )

    # Get the accuracy plot using plot_accuracy_plotly
    fig2d, x_vals = plot_accuracy_plotly(
        decoder_accuracy, chance_accuracy,
        data_std=data_std,
        back=back, forward=forward or trial_offset.mean(),
        lines=lines or [0, np.nanmean(green_screen_onset),
                       np.nanmean(voice_onset), np.nanmean(voice_offset)],
        line_labels=line_labels or ['trial onset', 'go cue', 'voice on', 'voice off'],
        tick_interval=tick_interval,
        title=title_accuracy,
        data_labels=data_labels or ["test accuracy", "chance", "train accuracy"],
        p=p_values
    )


    # Create initial confusion matrix
    cm = confusion_matrix(true_labels_list[0], pred_labels_list[0], 
                         labels=all_labels, normalize=normalize)
    
    # Add confusion matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=all_labels,
            y=all_labels,
            colorscale='Blues',
            showscale=True,
            text=cm.astype(str),  # Show counts in each cell
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(
                len=0.5,
                y=0.8,
                thickness=15
            )
        ),
        row=1, col=1
    )
    for trace in fig2d.data:
        fig.add_trace(trace, row=1, col=2)
    # Create frames
    frames = []
    x_vals = np.linspace(-back, forward, n_time)
    n_static_traces = len(fig2d.data) - 1  # number of static traces from accuracy plot
    
    for t in range(n_time):
        cm = confusion_matrix(true_labels_list[t], pred_labels_list[t], 
                            labels=all_labels, normalize=normalize)
        
        frames.append(go.Frame(
            data=[
                # First data element: Update confusion matrix (trace 0)
                go.Heatmap(
                    z=cm,
                    x=all_labels,
                    y=all_labels,
                    colorscale='Blues',
                    showscale=True,
                    text=cm.astype(str),
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorbar=dict(
                        len=0.5,
                        y=0.8,
                        thickness=15
                    )
                ),
                # Keep all static accuracy traces (traces 1 to n_static_traces)
                *fig2d.data[:-1],
                # Last element: Update vertical line (trace n_static_traces + 1)
                go.Scatter(
                    x=[x_vals[t], x_vals[t]],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                )
            ],
            traces=[0] + list(range(1, n_static_traces + 2)),  # Update all traces in order
            name=str(t)
        ))


    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,
        margin=dict(t=100),
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "method": "animate",
                    "args": [[str(k)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": str(k)
                } for k in range(n_time)
            ],
            "currentvalue": {"prefix": "Time: "}
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 200, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            }]
        }]
    )

    # Update axes
    fig.update_xaxes(title="Predicted label", row=1, col=1)
    fig.update_yaxes(title="True label", row=1, col=1)

    fig.frames = frames

    return fig
