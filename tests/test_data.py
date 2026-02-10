from costream.data.loader import group_files_by_subject, make_csv_loader
import pandas as pd
import numpy as np

def test_group_files(dummy_data_dir):
    mapping = group_files_by_subject(dummy_data_dir)
    
    assert "subj1" in mapping
    assert "subj2" in mapping
    assert len(mapping["subj1"]) == 2
    assert len(mapping["subj2"]) == 1

def test_loader_features(dummy_data_dir):
    # Test on-the-fly feature creation (magnitude)
    loader = make_csv_loader(
        feature_cols=['mag'],
        label_col='label',
        new_features={'mag': lambda df: np.sqrt(df.x**2 + df.y**2)}
    )
    
    files = [dummy_data_dir / "subj1_trial1.csv"]
    dfs = loader(files)
    
    assert len(dfs) == 1
    assert "mag" in dfs[0].columns
    assert "x" not in dfs[0].columns # Should only keep requested features
    assert "label" in dfs[0].columns