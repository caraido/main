# -*- coding: utf-8 -*-
"""utils.preprocessing -- numerical preprocessing & data-shape helpers."""

import numpy as np

def align_data(data, cue, back, forward, adjusted_fs):
    """Align trial data to a cue timestamp and crop all trials to equal length."""
    data_aligned = []
    for i in range(data.shape[0]):
        trial = data[i]
        truncated = trial[:, int((cue[i] - back) * adjusted_fs):int((cue[i] + forward) * adjusted_fs)]
        data_aligned.append(truncated)
    trial_min_length = np.min([arr.shape[1] for arr in data_aligned])
    data_aligned = np.array([arr[:, :trial_min_length] for arr in data_aligned])
    return data_aligned


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
    """Create lagged feature matrices.

    data: (n_trials, n_bins, n_channels)
    Returns list of (n_trials, n_features) arrays, one per time bin.
    """
    reformatted_data = []
    for i in range(data.shape[1]):
        reformatted = data[:, i - np.minimum(i, bins_per_feature - 1):i + 1, :]
        reformatted_data.append(reformatted.reshape(data.shape[0], -1))
    return reformatted_data


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
