# -*- coding: utf-8 -*-
"""utils.plotting -- static (matplotlib) and simple plotly plot helpers."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qual

DATA_COLORS = ['blue', 'orange', 'grey', 'cyan', 'magenta', 'yellow', 'grey', 'green', 'purple', 'black']
LINE_COLORS = ['black', 'blue', 'green', 'magenta', 'cyan']

def get_channel_colors(channel_names):
    """
    Assigns colors to channels based on their letter prefix.
    Channels with the same letter prefix get the same color.
    
    Args:
        channel_names: List of channel names (e.g., ['A1', 'A2', 'B1', 'B2', 'C1'])
        
    Returns:
        List of colors corresponding to each channel name
    """
    
    # Extract unique letter prefixes
    letter_prefixes = []
    for name in channel_names:
        # Extract letter(s) at the beginning of the channel name
        match = re.match(r'^([A-Za-z]+)', str(name))
        if match:
            letter_prefixes.append(match.group(1).upper())
        else:
            letter_prefixes.append('UNKNOWN')
    
    # Get unique prefixes while preserving order
    unique_prefixes = []
    for prefix in letter_prefixes:
        if prefix not in unique_prefixes:
            unique_prefixes.append(prefix)
    
    # Use plotly's qualitative color palette
    from plotly.colors import qualitative as qual
    colors = qual.Plotly  # ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', ...]
    
    # Create color mapping for each unique prefix
    prefix_to_color = {}
    for i, prefix in enumerate(unique_prefixes):
        prefix_to_color[prefix] = colors[i % len(colors)]
    
    # Assign colors to each channel based on its prefix
    channel_colors = [prefix_to_color[prefix] for prefix in letter_prefixes]
    
    return channel_colors


def to_rgba(color_str, alpha=0.2):
    """Convert a color string (e.g., 'red') to rgba string"""
    r, g, b = np.array(mcolors.to_rgb(color_str)) * 255
    return f'rgba({int(r)}, {int(g)}, {int(b)}, {alpha})'


def plot_on_channel(data, column=None, lines=None, line_labels=None, back=None, forward=None, same_ylim=True,
                    tick_interval=1.0, suptitle=None, CI=None,title=None,channel_colors=None):
    """
    Plot data from multiple channels in a grid layout.

    Args:
        data: list of 2D numpy arrays or a single 2D array (n_channels x n_bins)
        column: number of columns for subplot grid
        lines: list of times (in seconds) for vertical lines
        back: seconds before 0 (optional)
        forward: seconds after 0 (optional)
        same_ylim: bool, whether to fix the y-axis across subplots
        tick_interval: float, time interval between x-ticks in seconds (default 1.0)
    Returns:
        fig, ax: matplotlib figure and axes
    """
    # Handle if single array is passed instead of list
    if isinstance(data, np.ndarray):
        data = [data]

    n_data = len(data)
    n_channels, n_bins = data[0].shape
    data_color = DATA_COLORS * ((n_data // 7) + 1)
    line_color = LINE_COLORS * ((n_data // 5) + 1)

    # Set up the subplot grid
    if column is None:
        column = int(np.ceil(np.sqrt(n_channels)))
    row = n_channels // column + (n_channels % column != 0)
    fig, ax = plt.subplots(row, column, figsize=(18, row * 1.5), squeeze=False)

    # Set up x-axis values and ticks
    if back is not None and forward is not None:
        x_vals = np.linspace(-back, forward, n_bins)
        # Generate ticks at fixed intervals, aligned to 0
        tick_start = -np.floor(back / tick_interval) * tick_interval
        tick_end = np.ceil(forward / tick_interval) * tick_interval
        x_ticks = np.arange(tick_start, tick_end + tick_interval/2, tick_interval)
    else:
        x_vals = np.arange(n_bins)
        x_ticks = np.arange(0, n_bins, max(1, int(n_bins * tick_interval / 10)))

    # Set global y-limits if needed
    if same_ylim:
        global_min = min(np.min(d) for d in data)
        global_max = max(np.max(d) for d in data)
    else:
        global_min = 0.2
        global_max = 0.8
    line_handles = []

    if title is not None:
        assert len(title) == n_channels, "Title list length must match number of channels"
    if channel_colors is not None:
        assert len(channel_colors) == n_channels, "Channel colors length must match number of channels"
    for i in range(n_channels):
        ax_i = ax[i // column][i % column]

        # Plot each dataset
        for d_idx, d in enumerate(data):
            if channel_colors is not None:
                ax_i.plot(x_vals, d[i], color=channel_colors[i], lw=2, )
                if CI is not None:
                    ax_i.fill_between(x_vals, d[i] - CI[d_idx][i], d[i] + CI[d_idx][i], alpha=0.3, color=channel_colors[i],
                                    lw=2)
            else:
                ax_i.plot(x_vals, d[i], color=data_color[d_idx], lw=2, )
                if CI is not None:
                    ax_i.fill_between(x_vals, d[i] - CI[d_idx][i], d[i] + CI[d_idx][i], alpha=0.3, color=data_color[d_idx],
                                  lw=2)

        # Draw vertical lines if provided
        if lines is not None:
            for j, l in enumerate(lines):
                if i == 0:  # Only collect handle once (on the first channel)
                    handle = ax_i.axvline(x=l, color=line_color[j % len(line_color)], linestyle='--', linewidth=1.5,
                                          label=line_labels[j] if line_labels else None)
                    line_handles.append(handle)
                else:
                    ax_i.axvline(x=l, color=line_color[j % len(line_color)], linestyle='--', linewidth=1.5)

        # Draw y=0 line
        ax_i.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax_i.axvline(0, color='grey', linestyle='--', linewidth=1)
        # Set x-ticks
        ax_i.set_xticks(x_ticks)
        if back is not None and forward is not None:
            ax_i.set_xticklabels([f"{tick:.1f}" for tick in x_ticks])
        else:
            ax_i.set_xticklabels([f"{tick:.0f}" for tick in x_ticks])

        # set title
        if title is not None:
            ax_i.set_title(title[i], fontsize=16)

        # Set labels
        if i // column == row - 1:
            ax_i.set_xlabel('Time (s)')
        else:
            ax_i.set_xticklabels([])

        if i % column == 0:
            ax_i.set_ylabel('Amplitude')
        else:
            ax_i.set_yticklabels([])

        # Set consistent y-limits
        if same_ylim:
            ax_i.set_ylim(global_min, global_max)

    # Turn off any empty subplots
    for j in range(n_channels, row * column):
        fig.delaxes(ax[j // column][j % column])
    if suptitle:
        fig.suptitle(suptitle, fontsize=20)
    if line_handles:
        fig.legend(handles=line_handles, loc='lower right', bbox_to_anchor=(0.99, 0.01), frameon=False, fontsize=16)

    fig.tight_layout()
    return fig, ax


def plot_accuracy_plotly(main_data, *extra_data, data_std=None, back=None, forward=None,
                         lines=None, line_labels=None, tick_interval=1.0, title=None, ylabel=None,
                         data_labels=None, p=None, truncated=0,
                         data_colors=None, line_colors=None):
    all_data = (main_data,) + extra_data
    n_data = len(all_data)
    n_bins = main_data.shape[0]

    if data_colors is not None:
        if len(data_colors) != n_data:
            raise ValueError(f"data_colors must have length {n_data}, got {len(data_colors)}")
        data_colors = list(data_colors)
    else:
        data_colors = DATA_COLORS * ((n_data // 10) + 1)

    n_lines = len(lines) if lines else 0
    if line_colors is not None:
        if len(line_colors) != n_lines:
            raise ValueError(f"line_colors must have length {n_lines}, got {len(line_colors)}")
        line_colors = list(line_colors)
    else:
        line_colors = LINE_COLORS * ((n_data // 5) + 1)

    if back is not None and forward is not None:
        x_vals = np.linspace(-back, forward, n_bins)
        # Generate ticks at fixed intervals, aligned to 0
        tick_start = -np.floor(back / tick_interval) * tick_interval
        tick_end = np.ceil(forward / tick_interval) * tick_interval
        x_ticks = np.arange(tick_start, tick_end + tick_interval/2, tick_interval)
    else:
        x_vals = np.arange(n_bins)
        x_ticks = np.arange(0, n_bins, max(1, int(n_bins * tick_interval / 10)))

    traces = []
    if data_std is None:
        data_std = [np.zeros_like(d) for d in all_data]

    for i, (d, s) in enumerate(zip(all_data, data_std)):
        color = data_colors[i]
        label = data_labels[i] if data_labels else f"Data {i+1}"

        # Shaded error band
        traces.append(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([d - s, (d + s)[::-1]]),
            fill='toself',
            fillcolor=to_rgba(color, alpha=0.2),
            line=dict(color='rgba(255,255,255,0.8)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Mean line
        traces.append(go.Scatter(
            x=x_vals, y=d, mode='lines', name=label,
            line=dict(color=color, width=3)
        ))

    # Add vertical lines if any
    if lines:
        for j, l in enumerate(lines):
            traces.append(go.Scatter(
                x=[l, l],
                y=[np.min(main_data), np.max(main_data)],
                mode='lines',
                line=dict(color=line_colors[j % len(line_colors)], dash='dash'),
                name=line_labels[j] if line_labels else None
            ))

    # Create figure
    fig2d = go.Figure(data=traces)

    fig2d.update_layout(
        title=title or "Accuracy",
        xaxis=dict(title="Time (s)", tickvals=x_ticks),
        yaxis=dict(title= ylabel or "Accuracy"),
        margin=dict(l=0, r=0, b=30, t=30),
        legend=dict(x=1.05, y=1, font=dict(size=12)),
        showlegend=True
    )


    y_min = np.min([d - s for d, s in zip(all_data, data_std)])
    y_max = np.max([d + s for d, s in zip(all_data, data_std)])
    fig2d.update_yaxes(range=[y_min, y_max])

    # Add time marker (updated externally)
    x_vals_truncated = x_vals[truncated:]
    vline = go.Scatter(
        x=[x_vals_truncated[0], x_vals_truncated[0]],
        y=[y_min, y_max],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Current Time'
    )
    fig2d.add_trace(vline)

    if p is not None:
        padding = 0.1 * (y_max - y_min) if y_max > y_min else 0.2
        annotation_y = y_min - padding
        annotations = []
        for label, loc in zip(p, x_vals):
            annotations.append(dict(
                x=loc,
                y=annotation_y,
                text=str(label),
                showarrow=False,
                font=dict(size=12),
                xanchor='center'
            ))
        fig2d.update_layout(annotations=annotations)

    return fig2d, x_vals_truncated


def plot_3d_scatter(embeddings, word_category=None, labels=None, title=None):

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=(title)
    )

    if word_category is None:
        fig.add_trace(
            go.Scatter3d(
                x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                mode='markers+text',
                marker=dict(size=4, color='gray'),
            text=labels,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertemplate="word=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>answered</extra>",
            name="answered"
        ),
        row=1, col=1
        )   

    else:
        # colored by category
        uniq_cats = sorted(set(word_category))
        palette = (qual.Plotly + qual.D3 + qual.Set3)
        cat2color = {c: palette[i % len(palette)] for i, c in enumerate(uniq_cats)}
        for c in uniq_cats:
            idx = np.where(np.array(word_category) == c)[0]  # works now because word_category is np.array
            if len(idx) == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings[idx, 0], y=embeddings[idx, 1], z=embeddings[idx, 2],
                    mode='markers+text',
                    marker=dict(size=4, color=cat2color[c]),
                    text=labels[idx],
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    hovertemplate="word=%{text}<br>cat=" + c + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>target</extra>",
                    name=c,
                    legendgroup=c,
                    showlegend=True
                ),
                row=1, col=1
            )

    fig.update_layout(
        height=600, width=800,
        scene=dict(xaxis_title='dim1', yaxis_title='dim2', zaxis_title='dim3'),
        legend=dict(itemsizing='constant')
    )
    return fig
