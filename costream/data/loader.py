"""
Data loading utilities for the Costream package.

This module provides tools to:
1. Group signal files by subject ID.
2. Create customizable CSV loaders that standardize feature columns and generate
   derived features (like vector magnitude) on the fly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

__all__ = ["group_files_by_subject", "make_csv_loader"]


def group_files_by_subject(
    data_dir: Union[str, Path], 
    id_extractor: Optional[Callable[[str], str]] = None,
    file_pattern: str = "*.csv"
) -> Dict[str, List[Path]]:
    """
    Group files in a directory into a mapping of {subject_id: [file_paths]}.

    Args:
        data_dir: The directory containing the signal files.
        id_extractor: A function that extracts a subject ID from a filename (stem).
            If None, defaults to taking the prefix before the first '_' or '-'.
        file_pattern: Glob pattern to match files (default "*.csv").

    Returns:
        A dictionary mapping subject IDs to a sorted list of their file paths.

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    base_dir = Path(data_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Default extractor: "subject01_trial1" -> "subject01"
    if id_extractor is None:
        def _default_extractor(p: Path) -> str:
            name = p.stem
            # Match start of string until the first separator
            m = re.match(r"^([^_-]+)", name)
            return m.group(1) if m else name
        id_extractor = _default_extractor

    mapping = defaultdict(list)
    files = list(base_dir.glob(file_pattern))
    
    if not files:
        print(f"Warning: No files matching '{file_pattern}' found in {base_dir}")

    for path in files:
        sid = id_extractor(path)
        mapping[sid].append(path)

    # Sort files to ensure deterministic order (e.g., trial 1 before trial 2)
    for sid in mapping:
        mapping[sid].sort()
        
    return dict(mapping)


def make_csv_loader(
    feature_cols: Optional[Sequence[str]] = None,
    label_col: str = "label",
    timestamp_col: Optional[str] = None,
    parse_dates: bool = False,
    new_features: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    **read_csv_kwargs
) -> Callable[[Sequence[Union[str, Path]]], List[pd.DataFrame]]:
    """
    Creates a factory function to load CSV files into standardized DataFrames.

    Args:
        feature_cols: List of columns to keep as input features. 
            If None, keeps all columns except timestamp and label.
        label_col: Name of the target column (0=normal, 1=event).
        timestamp_col: Optional name of the timestamp column (for sorting/parsing).
        parse_dates: If True, parses the timestamp column as datetime.
        new_features: Dictionary of {new_col_name: function(df) -> series}.
            Useful for computing magnitude or filtering on load.
        **read_csv_kwargs: Additional arguments passed to pd.read_csv 
            (e.g., sep=';', skiprows=1).

    Returns:
        A function `load_signals(files)` that returns a list of DataFrames.
    """
    
    def load_signals(files: Sequence[Union[str, Path]]) -> List[pd.DataFrame]:
        dfs: List[pd.DataFrame] = []
        
        for f in files:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Prepare date parsing args
            parse_arg = read_csv_kwargs.pop('parse_dates', None)
            if parse_dates and timestamp_col:
                parse_arg = [timestamp_col]

            df = pd.read_csv(path, parse_dates=parse_arg, **read_csv_kwargs)

            # Check label column
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in {path.name}. Columns: {df.columns.tolist()}")
            
            # 1. Generate derived features (e.g., Magnitude)
            if new_features:
                for new_col_name, func in new_features.items():
                    df[new_col_name] = func(df)

            # 2. Determine which columns to keep
            if feature_cols is None:
                # Keep everything that isn't metadata
                exclude = {c for c in (timestamp_col, label_col) if c}
                selected_features = [c for c in df.columns if c not in exclude]
            else:
                selected_features = list(feature_cols)
                # Verify existence
                missing = [c for c in selected_features if c not in df.columns]
                if missing:
                    raise KeyError(f"Missing feature columns in {path.name}: {missing}")
            
            # 3. Filter DataFrame to required structure
            cols_to_keep = selected_features + [label_col]
            if timestamp_col and timestamp_col in df.columns:
                cols_to_keep.append(timestamp_col)
                
            df = df[cols_to_keep].copy()
            
            # Ensure proper types
            df[selected_features] = df[selected_features].astype("float32")
            df[label_col] = df[label_col].astype("int8")
            
            dfs.append(df)
            
        return dfs

    return load_signals