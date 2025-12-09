"""
Subject-wise Cross-Validation utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Sequence, Dict, Optional, Union
from sklearn.model_selection import GroupKFold

# Internal package imports
from ..segmentation.training_segmenter import create_training_data
from ..data.utils import extract_streaming_data
from .tester import run_experiment, ModelSpec

__all__ = ["run_subject_cv"]


def run_subject_cv(
    subject_map: Dict[str, List[pd.DataFrame]],
    model_specs: List[ModelSpec],
    feature_cols: Sequence[str],
    label_col: str = "label",
    cv: int = 5,
    random_state: int = 42,
    # Segmentation Params
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    activity_threshold: float = 1.4,
    drop_below_threshold: bool = True,
    spacing: Union[int, str] = 1,
    # Streaming/Eval Params
    tolerance: float = 20,
    debounce_secs: float = 60.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate a Subject-Wise Cross-Validation experiment.

    1. Splits subjects into K folds.
    2. For each fold:
       - Segments the training subjects into (X_train, y_train).
       - Keeps test subjects as continuous streams.
       - Runs training and streaming evaluation.
    3. Aggregates results into a single DataFrame with a 'fold' column.

    Parameters
    ----------
    subject_map : dict
        Mapping of {subject_id: [list of dataframes]}.
        (Usually output of costream.data.loader.load_subject with grouping).
    model_specs : list[ModelSpec]
        Models to evaluate.
    feature_cols : list[str]
        Columns to use as features.
    label_col : str
        Name of label column.
    cv : int
        Number of folds.

    Returns
    -------
    pd.DataFrame
        Combined results suitable for statistical analysis.
    """

    subjects = np.array(list(subject_map.keys()))

    # Handle case where fewer subjects than folds
    if len(subjects) < cv:
        raise ValueError(
            f"Cannot perform {cv}-fold CV with only {len(subjects)} subjects."
        )

    gkf = GroupKFold(n_splits=cv)

    # We use subjects as both X and groups for the splitter
    # (The actual data isn't split here, just the IDs)
    split_gen = gkf.split(subjects, groups=subjects)

    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_gen, start=1):
        train_subjs = subjects[train_idx]
        test_subjs = subjects[test_idx]

        if verbose:
            print(f"\n=== Fold {fold_idx}/{cv} ===")
            print(
                f"  Train Subjects: {len(train_subjs)} | Test Subjects: {len(test_subjs)}"
            )

        # 1. Prepare Training Data (Segmented)
        # Collect all DFs for training subjects
        train_dfs = []
        for s in train_subjs:
            train_dfs.extend(subject_map[s])

        X_train, y_train = create_training_data(
            train_dfs,
            feature_cols=feature_cols,
            label_col=label_col,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=activity_threshold,
            drop_below_threshold=drop_below_threshold,
            spacing=spacing,
        )

        if verbose:
            print(f"  Segmented Train Data: {X_train.shape}")

        # 2. Prepare Test Data (Continuous Streams)
        test_signals, test_events = extract_streaming_data(
            subject_map=subject_map,
            subjects=test_subjs,
            feature_col=feature_cols[0],
            label_col=label_col,
        )

        # 3. Run Experiment for this Fold
        fold_results = run_experiment(
            X_train,
            y_train,
            test_signals=test_signals,
            test_event_points=test_events,
            model_specs=model_specs,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=activity_threshold,
            tolerance=tolerance,
            debounce_secs=debounce_secs,
            verbose=False,
        )

        # Tag with fold index
        fold_results["fold"] = fold_idx
        all_results.append(fold_results)

        if verbose:
            # Print quick summary of this fold
            print("  Fold Results (Mean F1):")
            print(fold_results.groupby("model")["f1-score"].mean())

    # 4. Aggregate
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df
