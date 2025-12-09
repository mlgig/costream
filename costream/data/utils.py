"""
Data utility functions for processing signals and labels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd

__all__ = ["get_event_starts", "extract_streaming_data"]


def get_event_starts(labels: np.ndarray) -> Union[int, np.ndarray]:
    """
    Identify the start indices of distinct events from a label array.
    
    Handles contiguous blocks of non-zero labels.
    Example: [0, 0, 1, 1, 1, 0, 2, 2] -> Returns [2, 6]
    
    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels (0=Background, >0=Event).
        
    Returns
    -------
    int or np.ndarray
        -1 if no events found.
        Array of start indices if events exist.
    """
    labels = np.asarray(labels)
    non_zero_indices = np.flatnonzero(labels)
    
    if non_zero_indices.size == 0:
        return -1
        
    # Find gaps > 1 sample to identify distinct event blocks
    gaps = np.diff(non_zero_indices) > 1
    
    # The starts are: the very first non-zero index...
    # ...PLUS the indices immediately following any gap.
    starts = np.insert(non_zero_indices[1:][gaps], 0, non_zero_indices[0])
    
    return starts


def extract_streaming_data(
    subject_map: Dict[str, List[pd.DataFrame]],
    subjects: Sequence[str],
    feature_col: str,
    label_col: str = "label"
) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]]]:
    """
    Extract raw signals and robust event start indices for a list of subjects.
    
    Parameters
    ----------
    subject_map : dict
        Dictionary {subject_id: [DataFrame, ...]}.
    subjects : list
        List of subject IDs to process.
    feature_col : str
        Name of the column containing the signal (e.g., 'mag').
    label_col : str
        Name of the label column.
        
    Returns
    -------
    signals : List[np.ndarray]
        List of 1D float arrays (the streams).
    events : List[Union[int, np.ndarray]]
        List of event start indices (or -1) corresponding to each stream.
    """
    signals = []
    events = []

    for s in subjects:
        if s in subject_map:
            for df in subject_map[s]:
                # 1. Extract Signal
                sig = df[feature_col].to_numpy(dtype=np.float32)
                signals.append(sig)
                
                # 2. Extract Event Starts
                labels = df[label_col].to_numpy()
                starts = get_event_starts(labels)
                events.append(starts)
                
    return signals, events