"""
Event detection logic for streaming evaluation.

This module interprets continuous confidence maps to detect events
and compares them against ground truth to calculate:
- True Positives (TP): Detected event within tolerance.
- False Positives (FP): Detections in background (ADL) or outside tolerance.
- False Negatives (FN): Missed events.
- True Negatives (TN): Estimates of non-event time blocks.
- Delay: Time from actual event to detection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["evaluate_recording", "get_high_confidence_regions", "iou"]


def iou(range_a: range, range_b: range) -> float:
    """Compute Intersection over Union (IoU) of two ranges."""
    set_a = set(range_a)
    set_b = set(range_b)
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0


def get_high_confidence_regions(
    confidence_signal: np.ndarray,
    threshold: float = 0.5,
    min_interval_secs: float = 60.0,
    freq: int = 100
) -> Optional[np.ndarray]:
    """
    Find indices where confidence exceeds threshold.
    
    Applies 'de-bouncing': if multiple peaks occur within `min_interval_secs`,
    only the first one is counted to avoid multiple alarms for the same event.

    Parameters
    ----------
    confidence_signal : np.ndarray
        Continuous confidence scores.
    threshold : float
        Detection threshold.
    min_interval_secs : float, default=60.0
        Minimum time between distinct detections.
    freq : int, default=100
        Sampling rate.

    Returns
    -------
    np.ndarray or None
        Array of start indices of detected events.
    """
    # Find all points above threshold
    high_conf_indices = np.where(confidence_signal >= threshold)[0]
    
    if len(high_conf_indices) == 0:
        return None

    # Calculate difference between consecutive indices
    min_interval_samples = int(min_interval_secs * freq)
    
    # 1. Always keep the first index found
    distinct_detections = [high_conf_indices[0]]
    
    # 2. Iterate and keep only if distance from *last kept* is sufficient
    for idx in high_conf_indices[1:]:
        if idx - distinct_detections[-1] >= min_interval_samples:
            distinct_detections.append(idx)
            
    return np.array(distinct_detections)


def evaluate_recording(
    ts_len: int,
    event_point: int,
    confidence_signal: np.ndarray,
    confidence_thresh: float = 0.5,
    window_size: float = 7.0,
    tolerance: float = 2.0,
    freq: int = 100,
    step: float = 1.0,
    debounce_secs: float = 60.0
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Evaluate a single recording (TimeSeries + ConfidenceMap) against Ground Truth.

    Parameters
    ----------
    ts_len : int
        Length of the time series (samples).
    event_point : int
        Index of the event (-1 if background/ADL).
    confidence_signal : np.ndarray
        Continuous confidence scores.
    confidence_thresh : float
        Threshold for declaring a detection.
    window_size : float
        Window size in seconds.
    tolerance : float
        Allowed margin (seconds) around the event for a correct detection.
    freq : int
        Sampling frequency.
    step : float
        Step size used for windowing (used to estimate 'n_samples' blocks).
    debounce_secs : float
        Minimum time between distinct alarms.

    Returns
    -------
    cm : np.ndarray
        Confusion Matrix [[TN, FP], [FN, TP]] for this recording.
    high_conf : np.ndarray or None
        Indices of detected alarms.
    delay : float
        Detection delay in seconds (0 if missed or ADL).
    """
    
    # 1. Get detections
    high_conf = get_high_confidence_regions(
        confidence_signal, 
        threshold=confidence_thresh, 
        min_interval_secs=debounce_secs, 
        freq=freq
    )

    # 2. Define "blocks" or "samples" for TN calculation
    # (Approximation of total decision points in the file)
    step_samples = int(step * freq)
    window_samples = int(window_size * freq)
    n_decision_blocks = max(1, (ts_len - window_samples) // step_samples)

    # 3. Define Ground Truth Region
    if event_point == -1:
        # Background File: Event range is outside the signal
        event_range = range(ts_len, ts_len + 1)
    else:
        # Event File: Define tolerance window
        # We accept detection if the window overlaps with the event region
        tol_samples = int(tolerance * freq)
        
        left_bound = (event_point - freq) - int((window_size + tolerance) * freq)
        right_bound = (event_point + int((window_size - 1) * freq)) + tol_samples
        
        event_range = range(int(left_bound), int(right_bound))

    # 4. Calculate Metrics
    TP, FP, TN, FN = 0, 0, 0, 0
    delay = 0.0

    if high_conf is None:
        # No alarms raised at all
        FP = 0
        TP = 0
        FN = 1  # Potential miss
        TN = n_decision_blocks - 1 
    else:
        # Check each alarm
        for alarm_idx in high_conf:
            # Define the "detection window" for IoU check
            detection_range = range(alarm_idx, alarm_idx + int((window_size + tolerance) * freq))
            
            overlap = iou(detection_range, event_range)
            
            if overlap > 0:
                TP = 1
                # Calculate delay: Alarm Time - Actual Event Time
                d = (alarm_idx - event_point) / freq
                delay = d
            else:
                FP += 1

        FN = 1 if TP == 0 else 0
        TN = max(0, n_decision_blocks - TP - FP - FN)

    # 5. Fix for Background Files (event_point == -1)
    if event_point == -1:
        TP = 0
        FN = 0 # Cannot miss an event that doesn't exist
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    return cm, high_conf, delay