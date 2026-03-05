import scipy.io as sio
import mat73
import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib import colors as mcolors
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qual
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
nlp = spacy.load('en_core_web_sm')

DATA_COLORS = ['blue', 'orange', 'grey', 'cyan', 'magenta', 'yellow', 'grey', 'green','purple','black']
LINE_COLORS = ['black','blue','green','magenta','cyan']

def reformat_raw(elements, alternative=np.nan):
    """
    Takes nested elements and returns extracted values or NaNs.
    Handles both single arrays and lists of nested elements.
    
    Args:
        elements: Array/list of elements, each potentially nested
        alternative: Value to return for empty/invalid elements
        
    Returns:
        numpy array of extracted values or NaNs, or single value for backward compatibility
    """
    def extract_single(element):
        # Handle single element extraction
        if element is None:
            return alternative
        if isinstance(element, np.ndarray) and element.dtype==float:
            return element
        if isinstance(element, (list, np.ndarray)) and len(element) == 0:
            return alternative
        if not isinstance(element, (list, np.ndarray)):
            return element
        
        # Recursively extract the innermost non-list value
        current = element
        while isinstance(current, (list, np.ndarray)) and len(current) > 0:
            current = current[0]
        
        # Return the extracted value if it's not a list/array, otherwise alternative
        return current if not isinstance(current, (list, np.ndarray)) else alternative
    
    # Check if this is an array/list where each element needs individual processing
    # This happens when we have timing data from MATLAB like trial_onset, voice_onset, etc.
    if isinstance(elements, np.ndarray) and elements.dtype == object:
        # This is an array of nested elements (like from MATLAB data)
        try:
            return np.array([extract_single(element) for element in elements])
        except:
            return np.array([extract_single(element) for element in elements], dtype=object)
    elif isinstance(elements, list) and len(elements) > 0 and isinstance(elements[0], (list, np.ndarray)):
        # This is a list of nested elements
        try:
            return np.array([extract_single(element) for element in elements])
        except:
            return np.array([extract_single(element) for element in elements], dtype=object)
    else:
        # This is a single nested element - extract directly
        return extract_single(elements)

def reformat(data, bins_per_feature):
    reformatted_data = []
    for i in range(data.shape[1]):
        reformatted = data[:, i - np.minimum(i, bins_per_feature - 1):i + 1, :]
        reformatted_data.append(reformatted.reshape(data.shape[0], -1))
    return reformatted_data

def load_all_data(path):
    try:
        data = sio.loadmat(path)['all_data']
    except:
        data = np.array(mat73.loadmat(path)['all_data'], dtype=object)
    return data

def switch_2_number(labels):
    """Convert category labels to numbers.
    
    Args:
        labels: List or array of category labels
        
    Returns:
        Array of numerical labels where each unique category is mapped to an integer
    """
    uniques=np.unique(labels)
    label_number= {}
    for i,u in enumerate(uniques):
        label_number[u]=i
    return np.array([label_number[l] for l in labels])

def switch_2_category(number_labels, original_labels):
    """Convert numerical labels back to category labels.
    
    Args:
        number_labels: Array of numerical labels to convert
        original_labels: Array of original category labels used to establish the mapping
        
    Returns:
        Array of category labels corresponding to the numerical labels
    """
    uniques = np.unique(original_labels)
    number_category = {}
    for i, u in enumerate(uniques):
        number_category[i] = u
    return np.array([number_category[l] for l in number_labels])

def ind_func(x):
    try:
        return x[0][0]
    except:
        return x

def fix_index(array):
    try:
        array = np.array([ind_func(vo) if len(vo) > 0 else np.nan for vo in np.array(array, dtype=float)])
    except:
        array = np.array(array, dtype=float)
    return array

def replace_underscores(text, replacement):
   return re.sub(r'_{2,}', " "+replacement, text)

def add_space_after_comma(text):
    return re.sub(',',', ',text)

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
    

def get_sentence_tense(sentence):
    """
    Determines the tense of a given sentence based on temporal keywords, verb conjugations,
    and sentence structure using spaCy's NLP processing.

    The function first checks for specific temporal keywords that indicate past, present,
    or future tenses. If no keywords are found, it evaluates the tense of the verbs in the
    sentence using their Part-of-Speech (POS) tags. Additionally, it analyzes the sentence
    structure to determine tense in cases like imperative statements or sentences without a
    clear subject.

    :param sentence: The input sentence whose tense needs to be identified.
    :type sentence: str

    :return: The determined tense of the sentence. Possible values are:
             - "past": Indicates the sentence is in past tense.
             - "present": Indicates the sentence is in present tense.
             - "future": Indicates the sentence is in future tense.
             - "NA": Indicates the tense could not be determined (e.g., no verbs or keywords identified).
    :rtype: str
    """
    # Parse the sentence using spaCy
    sentence = sentence.replace('\r', '')
    sentence = sentence.lstrip()
    doc = nlp(sentence.replace(",", ""))

    # Keywords for temporal indicators
    past_keywords = {"yesterday", "last", "ago", "just now", "earlier"}
    present_keywords = {"right now", "usually", "often", "always", "currently", "now", "sometimes", "every", 'please'}
    future_keywords = {"tomorrow", "next", "soon", "in the future", "will"}

    tense = None

    # Identify temporal keywords
    for token in doc:
        if token.text.lower() in past_keywords:
            return "past"
        elif token.text.lower() in present_keywords:
            return "present"
        elif token.text.lower() in future_keywords:
            return "future"

    # Identify tense based on verb conjugation
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            # Check the verb tense
            if token.tag_ in {"VBD", "VBN"}:  # Past tense or past participle
                return "past" if tense is None else tense
            elif token.tag_ in {"VBP", "VBZ"}:  # Present tense
                return "present" if tense is None else tense
            elif token.tag_ == "VB":  # Base form (could be infinitive)
                if "will" in [t.text.lower() for t in doc]:
                    return "future"
                elif tense:
                    return tense

    # 1. Find all root VERBs (should usually be just one)
    root_verbs = [token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]
    if not root_verbs:
        return "NA"

    # We'll assume there's only one root verb in a simple sentence
    root_verb = root_verbs[0]

    # 2. Check for subjects in the sentence
    subjects = [token for token in doc if token.dep_ in ("nsubj", "nsubjpass")]

    # 3. If there's no subject at all, that's usually imperative ("Open the door.")
    if not subjects:
        return "present"

    # 4. If the subject is explicitly "you", it can still be imperative ("You open the door now!")
    #    We check .lemma_ to handle “You” vs. “you”
    if len(subjects) == 1 and subjects[0].lemma_.lower() == "you":
        return "present"

    # If no verbs or temporal indicators found
    return "NA"


def get_sentence_subject_number(sentence):
    """
    Determines the grammatical subject's number and person in a given sentence.

    This function processes a given sentence using spaCy's NLP model and identifies the grammatical
    subject (nsubj) or root of the sentence. Based on the subject's text or grammatical properties,
    it determines whether the subject is first person singular, third person singular, or plural.
    If the subject cannot be identified or categorized, the function returns 'NA'.

    :param sentence: A string representing the sentence to analyze.
    :type sentence: str
    :return: A string specifying the grammatical subject's person and number. Possible values
             include 'first person singular', 'third person singular', 'plural', or 'NA'.
    :rtype: str
    """
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    for token in doc:
        # Look for the subject (nsubj = nominal subject)
        if token.dep_ == "nsubj" or token.dep_ == 'ROOT':
            #
            if token.text.lower() in {"he", "she", "it","i"} or token.tag_ == "NN" or token.tag_ == "NNP":
                return "singular"
            # Plural: 'we', 'they', or plural nouns
            elif token.text.lower() in {"we", "they"} or token.tag_ == "NNS" or token.tag_=="NNPS":
                return "plural"
    return "NA"
def to_rgba(color_str, alpha=0.2):
    """Convert a color string (e.g., 'red') to rgba string"""
    r, g, b = np.array(mcolors.to_rgb(color_str)) * 255
    return f'rgba({int(r)}, {int(g)}, {int(b)}, {alpha})'

def get_sentence_subject_person(sentence):
    """
    Determines the grammatical person of the subject in a given sentence.

    Analyzes the given sentence using spaCy to identify the nominal subject (nsubj)
    and evaluates whether the subject is in the first person, third person,
    or cannot be determined. If the subject includes pronouns such as "I", "we",
    "he", "she", "it", "they", or relevant noun tags, the function categorizes
    them. Returns a string indicating "first" for first-person, "third" for
    third-person, or "NA" if the grammatical person cannot be determined.

    :param sentence:
        The input sentence to analyze for determining the person's subject.
        It is expected as a string.
    :return:
        The grammatical person of the subject found in the sentence. Possible
        return values are:
        - "first": If the subject is in the first person, e.g., "I" or "we".
        - "third": If the subject is in the third person, e.g., "he", "she",
          "it", "they", or singular/plural nouns.
        - "NA": If no valid grammatical person for the subject was determined.
    """
    # Parse the sentence using spaCy
    doc = nlp(sentence)

    for token in doc:
        # Look for the subject (nsubj = nominal subject)
        if token.dep_ == "nsubj" or token.dep_ == 'ROOT':
            # First person singular: 'I'
            if token.text.lower() in {"i","we"}:
                return "first"
            # Third person singular: 'he', 'she', 'it', or a singular noun
            elif token.text.lower() in {"he", "she", "it","they"} or 'NN' in token.tag_:
                return "third"
            # Plural: 'we', 'they', or plural nouns

    return "NA"

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
                         lines=None, line_labels=None, tick_interval=1.0, title=None,ylabel=None,
                         data_labels=None, p=None,truncated=0):
    all_data = (main_data,) + extra_data
    n_data = len(all_data)
    n_bins = main_data.shape[0]
    data_colors = DATA_COLORS * ((n_data // 10) + 1)
    line_colors= LINE_COLORS * ((n_data // 5) + 1)

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

def remove_number(text):
    """
    Remove picture numbers from target labels.
    
    Handles formats like:
    - "word + pic_number" -> "word" (e.g., "bank5" -> "bank")
    - "word + meaning_number + pic_number" -> "word + meaning_number" 
      (e.g., "date15" -> "date1", "fan210" -> "fan2")
    
    Args:
        text (str): Input text with picture numbers
        
    Returns:
        str: Text with only the final picture number removed
    """
    if not isinstance(text, str):
        return text
    
    # Extract the word part (letters) and number part
    word_part = ''.join([char for char in text if char.isalpha()])
    number_part = ''.join([char for char in text if char.isdigit()])
    
    if not number_part:
        # No numbers found, return as is
        return text
    
    # For single digit (1-9): it's just pic_number, remove it entirely
    if len(number_part) == 1:
        return word_part
    
    # For two digits (10, 11-19, 21-29): 
    # - 10: pic_number only, remove entirely -> word
    # - 11-19: meaning_number=1, pic_number=1-9 -> word1
    # - 21-29: meaning_number=2, pic_number=1-9 -> word2
    elif len(number_part) == 2:
        if number_part == '10':
            return word_part  # Just pic_number 10
        elif number_part.startswith('1'):
            return word_part + '1'  # meaning_number 1
        elif number_part.startswith('2'):
            return word_part + '2'  # meaning_number 2
        else:
            return word_part  # Other two-digit numbers are just pic_numbers
    
    # For three digits (110, 210): meaning_number + pic_number=10
    elif len(number_part) == 3:
        if number_part.startswith('1'):
            return word_part + '1'  # meaning_number 1
        elif number_part.startswith('2'):
            return word_part + '2'  # meaning_number 2
        else:
            return word_part  # Fallback
    
    # For other cases, return the word part
    else:
        return word_part