import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Import our new package components
from costream.data.loader import make_csv_loader
from costream.segmentation.training_segmenter import create_training_data
from costream.model.cost_classifier_cv import CostClassifierCV
from costream.model.metrics import cost_score
from costream.segmentation.streaming_segmenter import sliding_window_inference

def test_full_pipeline():
    print("🚀 Starting Component Test...\n")

    # --- 1. SETUP DUMMY DATA ---
    print("[1/4] Generating Dummy Data...")
    os.makedirs("temp_test_data", exist_ok=True)
    
    # Create 2 synthetic recordings (100 Hz, 60 seconds)
    # Subj 1 has a fall at index 2000 (20s)
    # Subj 2 has a fall at index 4000 (40s)
    cols = ["acc_x", "acc_y", "acc_z", "label"]
    
    df1 = pd.DataFrame(np.random.randn(6000, 3), columns=cols[:3])
    df1["label"] = 0
    df1.loc[2000, "label"] = 1 # Fall event
    df1.to_csv("temp_test_data/subj1.csv", index=False)

    df2 = pd.DataFrame(np.random.randn(6000, 3), columns=cols[:3])
    df2["label"] = 0
    df2.loc[4000, "label"] = 1 # Fall event
    df2.to_csv("temp_test_data/subj2.csv", index=False)
    
    print("   -> Created temp_test_data/subj1.csv, subj2.csv")

    # --- 2. TEST DATA LOADER ---
    print("\n[2/4] Testing Data Loader...")
    
    # Define a magnitude function to test feature generation
    def calc_mag(df):
        return np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    
    loader = make_csv_loader(
        feature_cols=["acc_x", "acc_y", "acc_z", "mag"], # We ask for 'mag'
        label_col="label",
        new_features={"mag": calc_mag} # We provide logic for 'mag'
    )
    
    files = ["temp_test_data/subj1.csv", "temp_test_data/subj2.csv"]
    dfs = loader(files)
    
    assert len(dfs) == 2
    assert "mag" in dfs[0].columns
    print(f"   -> Loaded {len(dfs)} files. Shape: {dfs[0].shape}. Columns: {list(dfs[0].columns)}")

    # --- 3. TEST TRAINING SEGMENTATION ---
    print("\n[3/4] Testing Training Segmenter...")
    
    # Create windows (Window=2s, Step=0.5s)
    X, y = create_training_data(
        dfs,
        feature_cols=["mag"], # Use only magnitude for 1D test
        window_size=2.0,
        step=0.5,
        freq=100,
        activity_threshold=0.0, # Keep everything for this test
        spacing=1 # multiple positive windows
    )
    
    print(f"   -> Generated Training Data. X shape: {X.shape}, y shape: {y.shape}")
    print(f"   -> Class distribution: {np.bincount(y)}")
    
    if X.ndim == 3 and X.shape[2] == 1:
        # Flatten if it came out 3D (N, Win, 1) -> (N, Win) for sklearn
        X = X[:, :, 0]

    # --- 4. TEST MODEL & STREAMING ---
    print("\n[4/4] Testing CostClassifierCV & Streaming Inference...")
    
    # Initialize our custom model
    base = LogisticRegression(solver='liblinear')
    model = CostClassifierCV(
        base_estimators=[base],
        n_dirichlet=10, # Keep it fast
        n_thresholds=10,
        cv=2,
        method="dirichlet",
        calibration=None
    )
    
    # Fit
    model.fit(X, y)
    print(f"   -> Model Fitted. Best Threshold: {model.threshold_:.3f}")
    
    # Test Streaming on Subj1 raw data
    raw_signal = dfs[0]["mag"].values
    conf_map, runtime = sliding_window_inference(
        raw_signal,
        model,
        window_size=2.0,
        step=0.5,
        freq=100,
        signal_thresh=0.0 # process all windows
    )
    
    assert len(conf_map) == len(raw_signal)
    print(f"   -> Streaming Inference Complete.")
    print(f"   -> Input len: {len(raw_signal)}, Output len: {len(conf_map)}")
    print(f"   -> Max Confidence: {conf_map.max():.4f}")

    # Clean up
    import shutil
    shutil.rmtree("temp_test_data")
    print("\n✅ ALL SYSTEMS GO! The package structure is working.")

if __name__ == "__main__":
    test_full_pipeline()