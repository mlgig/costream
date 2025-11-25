from .tester import run_experiment, ModelSpec
from .event_detection import evaluate_recording
from .visualization import (
    plot_confidence, 
    plot_detection, 
    metric_box, 
    metric_grid,
    plot_grouped_stacked
)

__all__ = [
    "run_experiment", 
    "ModelSpec", 
    "evaluate_recording",
    "plot_confidence",
    "plot_detection",
    "metric_box",
    "metric_grid",
    "plot_grouped_stacked"
]