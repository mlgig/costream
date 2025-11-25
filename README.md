# Costream: Cost-Sensitive Streaming Event Detection

A Python package for real-time event detection in time-series sensor data using cost-sensitive machine learning. Designed for applications where **False Negatives are more costly than False Positives** (e.g., fall detection, anomaly monitoring).

---

## 🎯 Key Features

- **Cost-Sensitive Classification**: Optimize for custom cost ratios between False Positives and False Negatives
- **Streaming Inference**: Real-time sliding window prediction on continuous signals
- **Ensemble Methods**: Combine multiple base classifiers with automatic weight optimization
- **Flexible Data Loading**: Built-in utilities for multi-subject sensor data with custom feature engineering
- **Comprehensive Evaluation**: Event detection metrics including delay, precision, recall, and cost-based gains
- **Production-Ready**: Efficient sliding window implementation with minimal latency

---

## 📦 Installation

### From Source

```bash
git clone <repository-url>
cd costream_pkg
pip install -e .
```

### Dependencies

- Python >= 3.8
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- openpyxl (for Excel data loading)

---

## 🚀 Quick Start

### 1. Load Data

```python
from costream.data.loader import make_csv_loader, group_files_by_subject
import numpy as np

# Define custom feature extraction
def calculate_magnitude(df):
    return np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2) / 9.81

# Create loader
load_subject = make_csv_loader(
    feature_cols=['mag'],
    label_col='label',
    new_features={'mag': calculate_magnitude}
)

# Group files by subject
subject_map = group_files_by_subject('data/falls')
train_dfs = load_subject(train_files)
```

### 2. Create Training Data

```python
from costream.segmentation.training_segmenter import create_training_data

X_train, y_train = create_training_data(
    train_dfs,
    feature_cols=['mag'],
    window_size=10,        # 10-second windows
    step=1.0,              # 1-second stride
    freq=100,              # 100 Hz sampling rate
    activity_threshold=1.4,
    spacing="multiphase"
)
```

### 3. Train Cost-Sensitive Model

```python
from costream.model.cost_classifier_cv import CostClassifierCV
from sklearn.linear_model import LogisticRegression

model = CostClassifierCV(
    base_estimators=[
        LogisticRegression(solver='liblinear'),
    ],
    alpha=2.0,           # FN cost is 2x FP cost
    method="stacking",   # Ensemble fusion method
    random_state=42
)

model.fit(X_train, y_train)
```

### 4. Run Streaming Evaluation

```python
from costream.evaluation.tester import run_experiment, ModelSpec

results_df = run_experiment(
    X_train, y_train,
    test_signals=test_signals,
    test_event_points=test_events,
    model_specs=[ModelSpec("Cost_Model", model)],
    window_size=10,
    freq=100,
    tolerance=20,  # 20-second detection window
    verbose=True
)

print(results_df)
```

### 5. Visualize Results

```python
from costream.evaluation.visualization import plot_confidence, metric_box

# Plot confidence map for a single recording
plot_confidence(
    signal, conf_map, event_point,
    tp, fp, tn, fn,
    model_name="Cost_Sensitive_Ensemble",
    freq=100
)

# Compare models
metric_box(results_df, "f1-score", title="F1 Score Comparison")
```

---

## 📁 Package Structure

```
costream/
├── data/
│   └── loader.py              # CSV loading and subject grouping utilities
├── segmentation/
│   ├── training_segmenter.py  # Window extraction for training
│   └── streaming_segmenter.py # Real-time sliding window inference
├── model/
│   ├── cost_classifier_cv.py  # Cost-sensitive ensemble classifier
│   └── metrics.py             # Custom cost-based metrics
└── evaluation/
    ├── tester.py              # End-to-end experiment runner
    ├── event_detection.py     # Event detection and evaluation logic
    └── visualization.py       # Plotting utilities
```

---

## 🔬 Core Components

### 1. **Data Loader** (`costream.data.loader`)

Flexible CSV loading with on-the-fly feature engineering:

```python
load_subject = make_csv_loader(
    feature_cols=['mag', 'gyro_z'],     # Select features
    label_col='label',                  # Ground truth column
    new_features={                       # Derived features
        'mag': lambda df: np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    }
)
```

### 2. **Training Segmentation** (`costream.segmentation.training_segmenter`)

Convert continuous signals into labeled windows:

- **Spacing Modes**:
  - `"multiphase"`: Fixed 1-second pre-event offset (specialized for falls)
  - `"na"`: Center event in window
  - `int`: Multiple windows with custom spacing
  
- **Filtering**: Remove windows below activity threshold to reduce noise

### 3. **Cost-Sensitive Classifier** (`costream.model.cost_classifier_cv`)

Meta-estimator that optimizes for custom cost functions:

- **Ensemble Fusion**:
  - `"stacking"`: Logistic regression meta-learner (recommended)
  - `"dirichlet"`: Random search over weight vectors
  
- **Threshold Optimization**: Automatically finds optimal decision boundary
- **Calibration**: Optional probability calibration (sigmoid/isotonic)

### 4. **Streaming Inference** (`costream.segmentation.streaming_segmenter`)

Efficient real-time prediction:

```python
from costream.segmentation.streaming_segmenter import sliding_window_inference

conf_map, runtime = sliding_window_inference(
    signal, model,
    window_size=10,
    step=1.0,
    freq=100
)
```

### 5. **Event Detection** (`costream.evaluation.event_detection`)

Convert continuous confidence scores to discrete events:

- Debouncing to prevent multiple alarms
- Delay calculation for True Positives
- Tolerance windows for matching ground truth

---

## 📊 Metrics

The package computes comprehensive evaluation metrics:

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision & recall |
| **Gain** | -(FP + α × FN) - Custom cost function |
| **FPR** | False Positives per hour of signal |
| **Delay** | Average detection delay (seconds) |
| **Runtime** | Inference time per window (μs) |

---

## 🎨 Visualization

Built-in plotting functions for analysis:

```python
from costream.evaluation.visualization import (
    plot_confidence,       # Signal + confidence map overlay
    metric_box,            # Model comparison boxplots
    plot_grouped_stacked,  # Stacked bar charts
    set_style              # Consistent styling
)
```

---

## 🔍 Example Use Case: Fall Detection

The package was designed for accelerometer-based fall detection:

1. **Load** multi-subject accelerometer data (L5 sensor, 100Hz)
2. **Extract** vector magnitude features
3. **Segment** into 10-second windows with activity filtering
4. **Train** cost-sensitive ensemble (FN cost = 2× FP cost)
5. **Evaluate** on held-out subjects with 20-second tolerance
6. **Visualize** confidence maps and detection delays

See `demo_pipeline.ipynb` for a complete walkthrough.

---

## ⚙️ Configuration

### Key Hyperparameters

```python
# Segmentation
window_size = 10.0        # Window length (seconds)
step = 1.0                # Sliding window stride (seconds)
activity_threshold = 1.4  # Minimum signal magnitude for valid windows

# Model
alpha = 2.0               # Cost ratio (FN/FP)
method = "stacking"       # Ensemble fusion method
calibration = "sigmoid"   # Probability calibration

# Evaluation
tolerance = 20.0          # Detection tolerance (seconds)
debounce_secs = 60.0      # Minimum time between alarms (seconds)
```

---

## 🐛 Troubleshooting

### Error: "Only one class in training data"

**Cause**: No positive (event) samples in `y_train`.

**Solution**: Verify your data segmentation captures event windows:

```python
print(f"Positive samples: {np.sum(y_train == 1)}")
print(f"Negative samples: {np.sum(y_train == 0)}")

# Ensure at least 10+ positive samples
assert np.sum(y_train == 1) >= 10, "Need more event samples!"
```

### Low Recall Despite High Threshold

**Cause**: Cost-sensitive optimization may prioritize reducing FPs over catching all events.

**Solution**: Adjust `alpha` (increase FN penalty) or set `recall_floor`:

```python
model = CostClassifierCV(
    alpha=3.0,           # Increase FN cost
    recall_floor=0.95,   # Require 95% minimum recall
    ...
)
```

---

## 📄 Citation

If you use this package in your research, please cite:

```
[coming soon]
```

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [timiderinola@gmail.com].

---
