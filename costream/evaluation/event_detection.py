"""
Event detection logic for streaming evaluation (Multi-Event Support).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, Sequence

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
    """Find start indices of alarms with debouncing."""
    high_conf_indices = np.where(confidence_signal >= threshold)[0]
    
    if len(high_conf_indices) == 0:
        return None

    min_interval_samples = int(min_interval_secs * freq)
    distinct_detections = [high_conf_indices[0]]
    
    for idx in high_conf_indices[1:]:
        if idx - distinct_detections[-1] >= min_interval_samples:
            distinct_detections.append(idx)
            
    return np.array(distinct_detections)


def evaluate_recording(
    ts_len: int,
    event_points: Union[int, Sequence[int]],
    confidence_signal: np.ndarray,
    confidence_thresh: float = 0.5,
    window_size: float = 7.0,
    tolerance: float = 2.0,
    freq: int = 100,
    step: float = 1.0,
    debounce_secs: float = 60.0
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Evaluate a recording against ONE OR MORE ground truth events.

    Parameters
    ----------
    event_points : int or List[int]
        Indices of the events. Use -1 or empty list if no events.

    Returns
    -------
    cm : np.ndarray
        [[TN, FP], [FN, TP]]
    high_conf : np.ndarray
        Detected alarms.
    delay : float
        Average delay of True Positives (0 if none).
    """
    
    # Normalize input to list
    if isinstance(event_points, int):
        ground_truth_events = [] if event_points == -1 else [event_points]
    else:
        # Filter out -1s if mixed in list
        ground_truth_events = [e for e in event_points if e != -1]

    # 1. Get Alarms
    high_conf = get_high_confidence_regions(
        confidence_signal, 
        threshold=confidence_thresh, 
        min_interval_secs=debounce_secs, 
        freq=freq
    )

    # 2. Define Decision Blocks (for TN estimation)
    step_samples = int(step * freq)
    window_samples = int(window_size * freq)
    n_decision_blocks = max(1, (ts_len - window_samples) // step_samples)

    # 3. Setup Ground Truth Ranges
    tol_samples = int(tolerance * freq)
    gt_ranges = []
    
    for ep in ground_truth_events:
        # Define the window of time where a detection counts as a "Hit" for this event
        left_bound = (ep - freq) - int((window_size + tolerance) * freq)
        right_bound = (ep + int((window_size - 1) * freq)) + tol_samples
        gt_ranges.append(range(int(left_bound), int(right_bound)))

    # 4. Matching Logic
    matched_gt_indices = set()
    total_delays = []
    fp_count = 0

    if high_conf is None:
        # No alarms
        tp_count = 0
        fp_count = 0
    else:
        for alarm_idx in high_conf:
            detection_range = range(alarm_idx, alarm_idx + int((window_size + tolerance) * freq))
            
            # Check if this alarm matches ANY ground truth event
            hit_any = False
            for i, gt_range in enumerate(gt_ranges):
                if iou(detection_range, gt_range) > 0:
                    hit_any = True
                    matched_gt_indices.add(i)
                    
                    # Calculate delay (Alarm - Event)
                    # We map this alarm to the specific event 'ep'
                    ep = ground_truth_events[i]
                    d = (alarm_idx - ep) / freq
                    total_delays.append(d)
                    
            if not hit_any:
                fp_count += 1

    tp_count = len(matched_gt_indices)
    # FN = Total Events that were NEVER matched
    fn_count = len(ground_truth_events) - tp_count

    # TN = Remainder
    tn_count = max(0, n_decision_blocks - tp_count - fp_count - fn_count)

    cm = np.array([[tn_count, fp_count], [fn_count, tp_count]])
    avg_delay = float(np.mean(total_delays)) if total_delays else 0.0
    
    return cm, high_conf, avg_delay