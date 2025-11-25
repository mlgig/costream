"""
Segmentation logic for creating training datasets from continuous signals.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["create_training_data"]


def _window_view(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    Return a sliding window view of an array.
    """
    arr = np.asarray(arr)
    
    if len(arr) < win:
        if arr.ndim == 1:
            return np.empty((0, win), dtype=arr.dtype)
        else:
            return np.empty((0, win, arr.shape[1]), dtype=arr.dtype)
            
    # Input: (Time, Feats) -> Output: (N_windows, Feats, Win_size)
    if arr.ndim == 1:
        # (N, Win)
        return sliding_window_view(arr, win, axis=0)[::step]
    else:
        # (Time, Feats) -> (N, Feats, Win)
        v = sliding_window_view(arr, win, axis=0)[::step]
        # Transpose to (N, Win, Feats)
        return v.transpose(0, 2, 1)


def create_training_data(
    dfs: Iterable[pd.DataFrame],
    feature_cols: Sequence[str],
    label_col: str = "label",
    *,
    freq: int = 100,
    window_size: float = 7.0,
    step: float = 1.0,
    activity_threshold: float = 1.1,
    drop_below_threshold: bool = True,
    spacing: Union[int, str] = 1,
    event_pos: str = "fixed",
    keep_unsegmented: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) training pairs from a list of signal DataFrames.

    Parameters
    ----------
    dfs : Iterable[pd.DataFrame]
        List of dataframes (one per recording).
    feature_cols : Sequence[str]
        List of columns to use as features.
    label_col : str, default="label"
        Name of the label column. Assumes 0 is background, >0 is event.
    freq : int, default=100
        Sampling rate in Hz.
    window_size : float, default=7.0
        Window size in seconds.
    step : float, default=1.0
        Step size for sliding window in seconds (for negative class).
    activity_threshold : float, default=1.1
        Threshold value for filtering ADL (negative) windows.
    drop_below_threshold : bool, default=True
        - True: Discard windows where max(signal) < threshold (Keep active).
        - False: Discard windows where max(signal) >= threshold (Keep quiet).
    spacing : int, "na" or "multiphase", default=1
        - "na": Center the event in the window.
        - "multiphase": Special mode. Positive offset is fixed at 1s. 
           Negative windows are filtered based on the 2nd second of data.
        - int: Generate multiple positive windows with this spacing.
    event_pos : str, default="fixed"
        "fixed" or "random" offsets for positive window generation.
    keep_unsegmented : bool, default=False
        If True, returns the full raw signals (ragged array) and event index.

    Returns
    -------
    X : np.ndarray
        Shape (N_samples, Window_Len_Samples, N_features) or (N, Win).
    y : np.ndarray
        Shape (N_samples,). 0 for ADL, 1 for Event.
    """
    
    win_len_samples = int(window_size * freq)
    step_samples = int(step * freq)

    # --- 1. Setup Positive Window Offsets ---
    if spacing == "na":
        # Center the fall in the window
        pre_offsets = [int(window_size // 2)]
        
    elif spacing == "multiphase":
        # Special Mode: Exactly 1 second before the fall
        # Adapted from original script: pre_offsets = [1]
        pre_offsets = [1]
        
    else:
        # Standard spacing logic
        s = int(spacing)
        margin = 2 if window_size >= 4.0 else 0
        start_range = margin
        end_range = int(window_size - margin)

        if event_pos == "random":
            rng = np.random.default_rng(42)
            low = max(start_range, 0)
            high = max(end_range, low + 1)
            n_offsets = max(1, int(window_size // s))
            pre_offsets = rng.integers(low, high, n_offsets).tolist()
        else:
            if end_range <= start_range:
                pre_offsets = [int(window_size // 2)]
            else:
                pre_offsets = list(range(start_range, end_range, s))
                if not pre_offsets:
                    pre_offsets = [int(window_size // 2)]

    X_rows, y_rows = [], []

    for df in dfs:
        # Extract signal
        signal = df[feature_cols].to_numpy(dtype=np.float32)
        if signal.shape[1] == 1:
            signal = signal.flatten()

        labels = df[label_col].to_numpy()
        event_indices = np.flatnonzero(labels)
        event_index = event_indices[0] if event_indices.size > 0 else -1

        # -- Unsegmented Mode --
        if keep_unsegmented:
            X_rows.append(signal)
            y_rows.append(event_index)
            continue

        win_list = []
        y_list = []

        # --- 2. Positive Windows ---
        if event_index >= 0:
            for pre in pre_offsets:
                offset_samples = int(pre * freq)
                start = max(0, event_index - offset_samples)
                end = start + win_len_samples
                
                if end <= len(signal):
                    event_win = signal[start:end]
                    win_list.append(event_win)
                    y_list.append(1)
            
            # Negatives come from before the event
            max_offset = max(pre_offsets) if pre_offsets else 0
            max_offset_samples = int(max_offset * freq)
            cut_left = max(0, event_index - max_offset_samples)
            signal_neg = signal[:cut_left]
        else:
            signal_neg = signal

        # --- 3. Negative Windows ---
        if len(signal_neg) >= win_len_samples:
            view = _window_view(signal_neg, win_len_samples, step_samples)
            
            if view.size > 0:
                # --- Filtering Logic ---
                # Check max value. If "multiphase", check only the 2nd second (freq to 2*freq).
                
                target_view = view
                
                if spacing == "multiphase":
                    # Check if window is long enough (at least 2 seconds)
                    if view.shape[1] >= 2 * freq:
                        if view.ndim == 2:
                             # (N, Win) -> Slice columns
                            target_view = view[:, freq : 2 * freq]
                        else:
                            # (N, Win, Feats) -> Slice time dimension (axis 1)
                            target_view = view[:, freq : 2 * freq, :]
                
                # Calculate Max
                if target_view.ndim == 2:
                    max_vals = np.max(np.abs(target_view), axis=1)
                else:
                    max_vals = np.max(np.abs(target_view), axis=(1, 2))
                
                # Apply Threshold
                if drop_below_threshold:
                    # Original logic: mask = main.max() >= thresh
                    mask = max_vals >= activity_threshold
                else:
                    mask = max_vals < activity_threshold
                
                clean_negatives = view[mask]
                
                if clean_negatives.shape[0] > 0:
                    win_list.append(clean_negatives)
                    y_list.extend([0] * clean_negatives.shape[0])

        # -- Aggregate --
        if win_list:
            batch_list = []
            for w in win_list:
                if w.ndim == signal.ndim: 
                    batch_list.append(np.expand_dims(w, axis=0))
                else:
                    batch_list.append(w)
            
            X_rows.append(np.vstack(batch_list))
            y_rows.append(np.array(y_list, dtype=np.uint8))

    # -- Final Return --
    if keep_unsegmented:
        return np.array(X_rows, dtype=object), np.array(y_rows, dtype=int)

    if not X_rows:
        if len(feature_cols) == 1:
             return np.empty((0, win_len_samples), np.float32), np.empty((0,), np.uint8)
        else:
             return np.empty((0, win_len_samples, len(feature_cols)), np.float32), np.empty((0,), np.uint8)

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    
    return X, y