"""
Streaming segmentation and inference utilities.

This module provides the core logic for:
1. Slicing a continuous time-series into sliding windows.
2. Running inference on those windows.
3. Aggregating window-level predictions back into a continuous confidence signal.
"""

from __future__ import annotations

import time
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["generate_sliding_windows", "compute_confidence_map", "sliding_window_inference"]


def generate_sliding_windows(
    ts: np.ndarray,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 0.0,
    pad: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, int]:
    
    ts = np.asarray(ts, dtype=np.float32)
    n = len(ts)
    step_samples = int(step * freq)
    window_samples = int(window_size * freq)

    # Handle empty/short series 
    if n < window_samples:
        empty_shape = (0, window_samples) if ts.ndim == 1 else (0, window_samples, ts.shape[1])
        return np.empty(empty_shape, dtype=ts.dtype),[], np.array([], dtype=bool), 0

    # Padding
    pad_size = 0
    if pad:
        remainder = (n - window_samples) % step_samples
        if remainder != 0:
            pad_size = step_samples - remainder
            pad_width = (0, pad_size) if ts.ndim == 1 else ((0, pad_size), (0, 0))
            ts = np.pad(ts, pad_width, mode='constant', constant_values=0)
            n = len(ts)

    # Sliding Window View
    if ts.ndim == 1:
        all_windows = sliding_window_view(ts, window_samples)[::step_samples]
    else:
        # Slide ONLY along the time axis (axis=0)
        all_windows = sliding_window_view(ts, window_samples, axis=0)[::step_samples]
        # Swap axes to match standard (N_windows, Window_Length, Channels)
        all_windows = all_windows.transpose(0, 2, 1)
    
    n_wins = all_windows.shape[0]
    starts = np.arange(0, n_wins * step_samples, step_samples)
    ends = starts + window_samples
    indices = list(zip(starts, ends))

    # 4. Vectorized Threshold Check (Now supports 3D max pooling)
    if all_windows.ndim == 1:
        max_vals = all_windows
    elif all_windows.ndim == 2:
        max_vals = all_windows.max(axis=1)
    else:
        # For 3D: Get the max value across the whole window and all channels
        max_vals = all_windows.max(axis=(1, 2))
        
    valid_mask = max_vals >= signal_thresh

    return all_windows, indices, valid_mask, pad_size


def compute_confidence_map(
    ts_len: int,
    indices: List[Tuple[int, int]],
    valid_mask: np.ndarray,
    confidence_scores: np.ndarray,
    method: Literal['max', 'mean'] = 'max',
    pad_size: int = 0
) -> np.ndarray:
    """
    Map window-level confidence scores back to the continuous time axis.

    Parameters
    ----------
    ts_len : int
        Length of the original time series (before padding).
    indices : list of tuple
        List of (start, end) tuples for each window.
    valid_mask : np.ndarray
        Boolean array indicating which windows have predictions.
    confidence_scores : np.ndarray
        Predicted scores for the valid windows.
    method : {'max', 'mean'}, default='max'
        Aggregation method for overlapping windows.
    pad_size : int, default=0
        Amount of padding that was added to the signal (to align internal buffers).

    Returns
    -------
    conf_map : np.ndarray
        Confidence score (0.0 to 1.0) for each timepoint.
    """
    total_len = ts_len + pad_size
    
    if method == 'mean':
        conf_map = np.zeros(total_len, dtype=np.float32)
        count_map = np.zeros(total_len, dtype=np.uint16)
    else:
        # Initialize with 0.0 (or -inf if we strictly wanted to distinguish no-pred)
        # Using 0.0 is safer for plotting/thresholding later
        conf_map = np.zeros(total_len, dtype=np.float32)

    score_idx = 0
    for i, (start, end) in enumerate(indices):
        if valid_mask[i]:
            score = confidence_scores[score_idx]
            score_idx += 1

            if method == 'max':
                # Update current max
                current_slice = conf_map[start:end]
                np.maximum(current_slice, score, out=current_slice)
            elif method == 'mean':
                conf_map[start:end] += score
                count_map[start:end] += 1

    if method == 'mean':
        nonzero = count_map > 0
        conf_map[nonzero] /= count_map[nonzero]

    # Crop padding
    if pad_size > 0:
        conf_map = conf_map[:ts_len]

    return conf_map


def sliding_window_inference(
    ts: np.ndarray,
    model,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 0.0,
    method: Literal['max', 'mean'] = 'max',
    pad: bool = False,
    const_confidence: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Run the full streaming inference pipeline on a time series.

    1. Segment signal into windows.
    2. Filter out low-activity windows (optimization).
    3. Predict probabilities using the model.
    4. Reconstruct continuous confidence signal.

    Parameters
    ----------
    ts : np.ndarray
        Input time series (magnitude).
    model : object
        Trained classifier implementing `predict_proba`.
    window_size : float, default=7.0
    step : float, default=1.0
    freq : int, default=100
    signal_thresh : float, default=1.04
    method : {'max', 'mean'}, default='max'
    pad : bool, default=False
    const_confidence : float, optional
        If provided, skips inference and returns a constant confidence map.
        Useful for Dummy/Baseline comparisons.

    Returns
    -------
    conf_map : np.ndarray
        Continuous confidence scores matching len(ts).
    avg_inference_time : float
        Average inference time per sample (in microseconds).
    """
    ts = np.array(ts)
    start_time = time.time()

    # Case: Constant dummy prediction
    if const_confidence is not None:
        conf_map = np.full(len(ts), const_confidence, dtype=np.float32)
        # Mock timing
        total_time_us = 0.0
        return conf_map, total_time_us

    # 1. Generate Windows
    windows, indices, valid_mask, pad_size = generate_sliding_windows(
        ts,
        window_size=window_size,
        step=step,
        freq=freq,
        signal_thresh=signal_thresh,
        pad=pad
    )

    # 2. Extract Valid Windows
    valid_windows = windows[valid_mask]

    # 3. Predict
    if len(valid_windows) > 0:
        # Check if model expects 3D or 2D input.
        # Our segmenter returns (N, Win) for 1D input.
        # If the model expects (N, Win, 1), we might need to expand.
        # But CostClassifierCV usually expects features extracted from these windows.
        # **Crucial Note**: If your model expects raw features (FeatureExtraction), 
        # that usually happens inside the model pipeline. 
        # If your model expects raw signal, this is fine.
        
        # We assume 'model' can handle the shape of valid_windows.
        # valid_windows shape: (N_valid, Window_Size_Samples)
        
        probs = model.predict_proba(valid_windows)
        
        # Handle binary classification [P(0), P(1)]
        if probs.shape[1] == 2:
            confidence_scores = probs[:, 1]
        else:
            confidence_scores = probs.ravel()
    else:
        confidence_scores = np.array([])

    # 4. Map back to signal
    conf_map = compute_confidence_map(
        len(ts),
        indices,
        valid_mask,
        confidence_scores,
        method=method,
        pad_size=pad_size
    )

    stop_time = time.time()
    # Time per sample in the final output (microseconds)
    total_time_us = 1e6 * (stop_time - start_time) / max(1, len(conf_map))

    return conf_map, total_time_us