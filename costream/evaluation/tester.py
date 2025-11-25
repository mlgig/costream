"""
Experiment runner for training, streaming inference, and evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from ..model.metrics import compute_metrics_from_cm
from ..segmentation.streaming_segmenter import sliding_window_inference
from .event_detection import evaluate_recording

__all__ = ["ModelSpec", "run_experiment"]


@dataclass
class ModelSpec:
    """
    Specification for a model to be trained and evaluated.
    
    Parameters
    ----------
    name : str
        Unique name for the model (e.g., "LR_Cost_Sensitive").
    estimator : BaseEstimator
        The sklearn-compatible estimator instance.
    param_grid : dict, optional
        (Not yet used, but reserved for hyperparameter tuning integration).
    """
    name: str
    estimator: BaseEstimator
    param_grid: Optional[Dict[str, Any]] = None
    
    def clone(self):
        return ModelSpec(self.name, clone(self.estimator), self.param_grid)


def _train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    specs: List[ModelSpec],
    verbose: bool = True
) -> Dict[str, BaseEstimator]:
    """Fit all provided models on the training data."""
    trained_models = {}
    
    if verbose:
        print(f"TRAINING {len(specs)} models on {len(y_train)} samples...")
        
    for spec in specs:
        start = time.time()
        model = clone(spec.estimator)
        
        # Fit
        model.fit(X_train, y_train)
        trained_models[spec.name] = model
        
        if verbose:
            elapsed = time.time() - start
            print(f"  ✓ {spec.name} trained in {elapsed:.2f}s")
            
    return trained_models


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_signals: Sequence[np.ndarray],
    test_event_points: Sequence[int],
    model_specs: List[ModelSpec],
    *,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 1.04,
    tolerance: float = 2.0,
    debounce_secs: float = 60.0,
    ensemble_all: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run a full training and streaming evaluation pipeline.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data (segmented windows).
    test_signals : Sequence[np.ndarray]
        List of continuous raw test signals (e.g. magnitude).
    test_event_points : Sequence[int]
        List of ground truth event indices for test signals (-1 if None).
    model_specs : List[ModelSpec]
        List of models to train and test.
    window_size : float
        Window size in seconds.
    step : float
        Step size in seconds.
    freq : int
        Sampling frequency.
    signal_thresh : float
        Activity threshold for skipping silent windows.
    tolerance : float
        Acceptable delay/margin for True Positives (seconds).
    debounce_secs : float
        Minimum time between distinct alarms.
    ensemble_all : bool
        If True, adds an "Ensemble-All" model that averages confidence maps 
        of all provided models.

    Returns
    -------
    pd.DataFrame
        Table of metrics for each model.
    """
    
    # 1. Train Models
    trained_models = _train_models(X_train, y_train, model_specs, verbose=verbose)
    
    metrics_rows = []
    
    # Store confidence maps for ensemble (Test_Idx -> {ModelName -> Map})
    ensemble_maps: Dict[int, Dict[str, np.ndarray]] = {i: {} for i in range(len(test_signals))}
    
    if verbose:
        print(f"\nTESTING on {len(test_signals)} recordings (Window={window_size}s)...")

    # 2. Evaluate Individual Models
    for name, model in trained_models.items():
        if verbose:
            print(f"  Evaluating {name}...", end=" ", flush=True)
            
        # Determine Threshold
        # CostClassifierCV stores its optimized threshold in .threshold_
        # Standard sklearn models use 0.5
        thresh = getattr(model, "threshold_", 0.5)
        
        # Accumulators
        total_CM = np.zeros((2, 2), dtype=int)
        delays = []
        total_signal_time = 0
        total_runtime = 0.0
        
        for i, (sig, event_point) in enumerate(zip(test_signals, test_event_points)):
            total_signal_time += len(sig)
            
            # A. Streaming Inference
            conf_map, runtime_us = sliding_window_inference(
                sig, model,
                window_size=window_size,
                step=step,
                freq=freq,
                signal_thresh=signal_thresh
            )
            
            # Store for ensemble
            if ensemble_all:
                ensemble_maps[i][name] = conf_map
            
            # B. Event Detection
            cm, _, delay = evaluate_recording(
                ts_len=len(sig),
                event_point=event_point,
                confidence_signal=conf_map,
                confidence_thresh=thresh,
                window_size=window_size,
                tolerance=tolerance,
                freq=freq,
                step=step,
                debounce_secs=debounce_secs
            )
            
            total_CM += cm
            delays.append(delay)
            total_runtime += runtime_us

        # C. Compute Final Metrics for this Model
        # Convert total runtime to average per sample (approx)
        avg_runtime_us = total_runtime / max(1, len(test_signals))
        avg_delay = np.mean(delays) if delays else 0.0
        
        # signal_time in ms for rate metrics (1000 ms/s)
        # We need total signal time in ms
        total_signal_time_ms = (total_signal_time / freq) * 1000
        
        row = compute_metrics_from_cm(
            total_CM,
            signal_time=total_signal_time_ms,
            runtime=avg_runtime_us,
            delay=avg_delay,
            alpha=getattr(model, "alpha", 2.0) # Use model's alpha if available
        )
        row["model"] = name
        row["thresh"] = thresh
        metrics_rows.append(row)
        
        if verbose:
            print("Done.")

    # 3. Evaluate Ensemble (Optional)
    if ensemble_all and len(trained_models) > 1:
        if verbose:
            print("  Evaluating Ensemble-All...", end=" ", flush=True)
            
        total_CM = np.zeros((2, 2), dtype=int)
        delays = []
        total_signal_time = 0
        
        for i, (sig, event_point) in enumerate(zip(test_signals, test_event_points)):
            total_signal_time += len(sig)
            
            # Average the maps
            maps = list(ensemble_maps[i].values())
            # Stack and mean
            avg_conf_map = np.mean(np.vstack(maps), axis=0)
            
            # Detect
            # Ensemble threshold is usually 0.5 for averaged probabilities
            cm, _, delay = evaluate_recording(
                len(sig), event_point, avg_conf_map,
                confidence_thresh=0.5,
                window_size=window_size,
                tolerance=tolerance,
                freq=freq,
                step=step,
                debounce_secs=debounce_secs
            )
            total_CM += cm
            delays.append(delay)
            
        total_signal_time_ms = (total_signal_time / freq) * 1000
        row = compute_metrics_from_cm(
            total_CM,
            signal_time=total_signal_time_ms,
            runtime=0.0, # Ensemble runtime is N/A or sum of parts
            delay=np.mean(delays) if delays else 0.0
        )
        row["model"] = "Ensemble-All"
        row["thresh"] = 0.5
        metrics_rows.append(row)
        
        if verbose:
            print("Done.")

    # 4. Finalize
    df = pd.DataFrame(metrics_rows)
    # Reorder columns
    cols = ["model"] + [c for c in df.columns if c != "model"]
    return df[cols]