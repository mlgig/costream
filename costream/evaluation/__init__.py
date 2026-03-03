from .tester import run_experiment, ModelSpec
from .event_detection import evaluate_recording
from .cross_validation import run_subject_cv, aggregate_cv_results
from .visualization import (
    plot_confidence, 
    plot_detection, 
    metric_box, 
    metric_grid,
    plot_grouped_stacked,
    critical_difference,
    window_bar,
)

__all__ = [
    "run_experiment", 
    "ModelSpec", 
    "evaluate_recording",
    "run_subject_cv",
    "aggregate_cv_results",
    "plot_confidence",
    "plot_detection",
    "metric_box",
    "metric_grid",
    "plot_grouped_stacked",
    "critical_difference",
    "window_bar",
]