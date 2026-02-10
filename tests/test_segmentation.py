import numpy as np
import pandas as pd
from costream.segmentation.training_segmenter import create_training_data
from costream.segmentation.streaming_segmenter import generate_sliding_windows, compute_confidence_map

def test_training_segmentation_univariate(synthetic_dfs):
    # Test Univariate (1 feature) -> Should be 2D (N, Win)
    X, y = create_training_data(
        synthetic_dfs,
        feature_cols=['mag'],
        window_size=2.0,
        freq=100,
        spacing="na",
        signal_thresh=0.0  # <--- UPDATED KEYWORD
    )
    
    # Window = 2s * 100Hz = 200 samples
    assert X.ndim == 2
    assert X.shape[1] == 200
    assert len(X) == len(y)
    assert np.sum(y == 1) > 0
    assert np.sum(y == 0) > 0

def test_training_segmentation_multivariate(synthetic_dfs):
    # Test Multivariate (create dummy 3-axis data)
    for df in synthetic_dfs:
        df['x'] = df['mag']
        df['y'] = df['mag']
        df['z'] = df['mag']

    # 3 features -> Should be 3D (N, Win, 3)
    X, y = create_training_data(
        synthetic_dfs,
        feature_cols=['x', 'y', 'z'],
        window_size=2.0,
        freq=100,
        spacing="na",
        signal_thresh=0.0 # <--- UPDATED KEYWORD
    )
    
    assert X.ndim == 3
    assert X.shape[1] == 200
    assert X.shape[2] == 3

def test_multi_event_extraction(synthetic_dfs):
    # DF 2 has TWO events. We expect TWO positive windows generated.
    multi_event_df = [synthetic_dfs[2]]
    
    X, y = create_training_data(
        multi_event_df,
        feature_cols=['mag'],
        window_size=2.0,
        freq=100,
        spacing="na", # 1 window per event
        signal_thresh=0.0 # <--- UPDATED KEYWORD
    )
    
    positives = y[y==1]
    # Should find exactly 2 events
    assert len(positives) == 2 

def test_streaming_segmentation():
    # 5 second signal (500 samples)
    ts = np.random.rand(500)
    
    windows, indices, mask, pad = generate_sliding_windows(
        ts, window_size=1.0, step=0.5, freq=100, pad=True, signal_thresh=0.0
    )
    
    # 1s window = 100 samples.
    assert windows.shape[1] == 100
    assert len(windows) == len(indices)
    assert len(mask) == len(windows)
    
    # Test Confidence Map Reconstruction
    preds = np.full(len(windows), 0.9)
    mask[:] = True
    
    conf_map = compute_confidence_map(len(ts), indices, mask, preds, method='max', pad_size=pad)
    
    assert len(conf_map) == 500
    assert np.allclose(conf_map, 0.9)