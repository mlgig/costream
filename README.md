# Costream: Cost-Sensitive Streaming Event Detection

**Costream** is a Python package for real-time event detection in
time-series sensor data. It is designed for streaming applications where
**False Negatives are more costly than False Positives** (e.g., fall
detection, medical monitoring, anomaly detection).

It provides a full research pipeline including streaming segmentation,
cost-sensitive training, and realistic replay-based evaluation.

------------------------------------------------------------------------

## ⚡ Key Features

-   **Cost-Sensitive Optimization**: Automatic threshold and ensemble
    tuning to minimize custom error costs (e.g., missed falls).
-   **Streaming Inference**: Efficient sliding-window engine that
    simulates real-time deployment.
-   **Smart Ensembling**: Stacking and Dirichlet-based ensemble
    strategies.
-   **Subject-Aware Loading**: Tools for multi-subject datasets
    (Farseeing, SisFall, etc.).
-   **Evaluation Suite**: Streaming metrics including Detection Delay,
    False Alarm Rate, and Event-based F1.

------------------------------------------------------------------------

## 📦 Installation

Clone and install from source:

``` bash
pip install pip install git+https://github.com/mlgig/costream.git
```

**Requirements**\
Python 3.8+, numpy, pandas, scikit-learn, matplotlib\
(Optional) `aeon` for critical difference diagrams

------------------------------------------------------------------------

## 🚀 Quick Start

This example simulates a fall detection pipeline.\
See `examples/demo_notebook.ipynb` for a full walkthrough.

``` python
from costream.segmentation.training_segmenter import create_training_data
from costream.model.cost_classifier_cv import CostClassifierCV
from costream.evaluation.tester import run_experiment, ModelSpec
from sklearn.linear_model import LogisticRegression

# 1. Segment Data for Training
X_train, y_train = create_training_data(
    train_dfs,
    feature_cols=['mag'],
    window_size=4.0,
    freq=100
)

# 2. Train Cost-Sensitive Model
model = CostClassifierCV(
    base_estimators=[LogisticRegression()],
    alpha=3.0,          # Missed events are 3x more costly
    method="stacking"
)
model.fit(X_train, y_train)

# 3. Streaming Evaluation
results = run_experiment(
    X_train, y_train,
    test_signals=test_signals,
    test_event_points=test_events,
    model_specs=[ModelSpec("Cost_Ensemble", model)],
    window_size=4.0,
    tolerance=2.0       # 2s margin for correct detection
)

print(results[['model', 'f1-score', 'gain', 'delay', 'false alarm rate']])
```

------------------------------------------------------------------------

## 📁 Modules

| Module | Description |
| :--- | :--- |
| **`costream.data`** | Loaders for CSV/Feather data and subject grouping (`loader.py`). |
| **`costream.segmentation`** | `training_segmenter.py`: Extracts labeled windows (X, y) for training.<br>`streaming_segmenter.py`: Sliding window engine for real-time inference. |
| **`costream.model`** | `cost_classifier_cv.py`: The core cost-sensitive ensemble estimator.<br>`metrics.py`: Custom cost functions and gain calculations. |
| **`costream.evaluation`** | `tester.py`: End-to-end experiment runner.<br>`event_detection.py`: Converts continuous probabilities into discrete events.<br>`visualization.py`: Plots confidence maps and performance metrics. |

------------------------------------------------------------------------

## 📊 Visualization

Generate confidence-map visualizations:

``` python
from costream.evaluation.visualization import plot_confidence

plot_confidence(
    ts=raw_signal_data,
    c=confidence_scores,
    y=ground_truth_index,
    tp=1, fp=0, tn=50, fn=0,
    model_name="Cost-Sensitive Ensemble",
    freq=100
)
```

------------------------------------------------------------------------

## 📄 Citation

If you use **Costream** in your research, please cite:

``` bibtex
@article{aderinola2025watch,
  title={Watch Your Step: A Cost-Sensitive Framework for Accelerometer-Based Fall Detection in Real-World Streaming Scenarios},
  author={Aderinola, Timilehin B and Palmerini, Luca and D'Ascanio, Ilaria and Chiari, Lorenzo and Klenk, Jochen and Becker, Clemens and Caulfield, Brian and Ifrim, Georgiana},
  journal={arXiv preprint arXiv:2509.11789},
  year={2025}
}
```

*(Citation will be updated once published.)*

------------------------------------------------------------------------

## 📧 Contact & License

Maintained by **Timilehin Aderinola** (timiderinola@gmail.com).\
Licensed under the **MIT License**. See the `LICENSE` file for details.
