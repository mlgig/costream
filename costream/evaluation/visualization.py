"""
Visualization utilities for model evaluation and streaming confidence maps.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Optional dependency for Critical Difference diagrams
try:
    from aeon.visualisation import plot_critical_difference as cd
    _AEON_AVAILABLE = True
except ImportError:
    _AEON_AVAILABLE = False


__all__ = [
    "set_style",
    "metric_box",
    "metric_grid",
    "plot_confidence",
    "plot_detection",
    "critical_difference",
    "save_confusion_matrix",
    "plot_grouped_stacked"
]


def set_style(style: str = "ticks", font_scale: float = 1.1):
    """Set the seaborn plotting style."""
    sns.set_theme(style=style, font_scale=font_scale)


# Set default style on import
set_style()


def metric_box(
    df: pd.DataFrame,
    metric: str,
    *,
    x: str = "model",
    hue: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    grid_ax: Optional[str] = None,
    palette: str = 'tab10',
    type: str = "box"
) -> Axes:
    """
    Draw a box-plot of a specific metric by model.
    """
    ax = ax or plt.gca()
    if type == "box":
        sns.boxplot(
            data=df, x=x, y=metric, order=order, hue=x, 
            ax=ax, width=0.5, palette=palette, showfliers=False
        )
    else: # bar
        sns.barplot(
            data=df, x=x, y=metric, order=order, hue=x,
            ax=ax, palette=palette, errorbar='sd'
        )
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    
    if grid_ax is not None:
        ax.grid(axis=grid_ax, alpha=0.3)
        
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()
    return ax


def metric_grid(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    order: Optional[List[str]] = None,
    n_cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Display multiple metrics in a grid of boxplots.
    """
    n = len(metrics)
    n_cols = max(1, n_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (4 * n_cols, 4 * n_rows),
        sharex=True,
        layout="constrained",
    )
    axs_flat = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    for ax, metric in zip(axs_flat, metrics):
        metric_box(df, metric, order=order, ax=ax, **kwargs)
        # ax.set_title(metric) # metric_box already sets title

    # Hide unused axes
    for ax in axs_flat[len(metrics):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)
        
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    plt.show()


def plot_confidence(
    ts: np.ndarray,
    c: np.ndarray,
    y: Union[int, Sequence[int]],
    tp: int, fp: int, tn: int, fn: int,
    *,
    high_conf: Optional[Sequence[int]] = None,
    ave_time: float = 0.0,
    model_name: str = "",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    thresh_line: Optional[float] = None,
    freq: int = 100
):
    """
    Plot signal (Left Axis) and confidence map (Right Axis/Color).

    Parameters
    ----------
    ts : np.ndarray
        Signal magnitude (1D).
    c : np.ndarray
        Confidence scores [0, 1].
    y : int
        Ground truth event index (-1 if none).
    tp, fp, tn, fn : int
        Confusion matrix counts for this recording.
    high_conf : sequence, optional
        List of start indices for detected alarms.
    ave_time : float
        Average inference time (us or ms) to display in title.
    model_name : str
        Name of model for title.
    thresh_line : float, optional
        Draw a marker on the confidence colorbar at this threshold.
    freq : int
        Sampling rate (for x-axis time conversion).
    """
    x = np.arange(len(ts))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    # --- Left axis: Acceleration ---
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    
    # Dynamic ylim
    min_y, max_y = np.min(ts), np.max(ts)
    ax.set_ylim(min(-0.1, min_y * 1.05), max_y + 0.5)
    
    ax.plot(x, ts, lw=0.8, color="0.25", label="Signal", zorder=1)

    # --- Confidence Segments ---
    # Create colored line segments based on confidence value
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # We color the segment between i and i+1 based on confidence at i
    # (Truncate last point for segments)
    colors = cmap(norm(c[:-1]))
    segments = [[(x[i], ts[i]), (x[i + 1], ts[i + 1])] for i in range(len(ts) - 1)]
    
    lc = mcoll.LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.9, zorder=2)
    ax.add_collection(lc)

    # --- Highlights ---
    # Draw gray spans for detected alarms
    if high_conf is not None:
        # Assume an alarm covers ~2 seconds visually or just highlight the onset
        width = 2.0 * freq 
        for h in high_conf:
            ax.axvspan(h, h + width, color="0.8", alpha=0.4, zorder=0)

    # Draw Ground Truth Line
    # Normalize y to list
    if isinstance(y, int):
        events = [] if y == -1 else [y]
    else:
        events = [e for e in y if e != -1]

    # Plot lines
    for i, event_idx in enumerate(events):
        label = "Event Onset" if i == 0 else None # Only label first line for legend
        ax.axvline(x=event_idx, color="red", linestyle="--", linewidth=1.5, label=label, zorder=3)

    # --- Right Axis: Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Confidence")
    
    # Threshold marker on colorbar
    if thresh_line is not None:
        cbar.ax.hlines(thresh_line, 0, 1, colors="black", lw=2.0)
        cbar.ax.plot(-0.15, thresh_line, marker=r'$\triangleright$', color="black",
                    markersize=10, clip_on=False)

    # --- Formatting ---
    # Convert X-axis samples to Seconds
    def seconds_formatter(x_val, pos):
        return f"{int(x_val / freq)}"
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(seconds_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, prune='both', nbins=10))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Add manual proxy for threshold if present
    if thresh_line is not None:
        triangle = Line2D([], [], color="black", marker=r'$\triangleright$',
                          linestyle="None", markersize=10, label=f"Threshold={thresh_line:.2f}")
        handles.append(triangle)
        labels.append(f"τ={thresh_line:.2f}")

    ax.legend(handles=handles, labels=labels, loc="upper left", frameon=True, framealpha=0.9)

    # Title
    stats = f"TP:{tp} FP:{fp} TN:{tn} FN:{fn} | Time: {ave_time:.1f}µs/sample"
    ax.set_title(f"{model_name}\n{stats} {title or ''}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    if show:
        plt.show()
    
    plt.close(fig)


def plot_detection(
    ts: np.ndarray,
    y: int,
    c: np.ndarray,
    cm: np.ndarray,
    high_conf: Optional[np.ndarray],
    ave_time: float,
    **kwargs,
):
    """
    Wrapper for plot_confidence that handles conditional plotting logic.
    
    Kwargs:
    - plot (bool): Plot always.
    - plot_errors (bool): Plot only if FP > 0 or FN > 0.
    """
    tn, fp, fn, tp = cm.ravel()
    
    should_plot = kwargs.get("plot", False)
    plot_errors = kwargs.get("plot_errors", False)
    
    has_error = (fp > 0) or (fn > 0)
    
    if should_plot or (plot_errors and has_error):
        # Forward valid kwargs to plot_confidence
        valid_keys = plot_confidence.__code__.co_varnames
        plot_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        plot_confidence(
            ts, c, y, tp, fp, tn, fn,
            high_conf=high_conf,
            ave_time=ave_time,
            **plot_kwargs
        )


def critical_difference(
    df: pd.DataFrame,
    metric: str = "f1-score",
    pivot_column: str = "model",
    *,
    alpha: float = 0.05,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Draw a Demšar critical-difference diagram.
    Requires 'aeon' package.
    """
    if not _AEON_AVAILABLE:
        warnings.warn("The 'aeon' package is required for critical_difference plots. "
                      "Pip install aeon to use this feature.")
        return

    # Pivot: Index=Fold/Dataset, Columns=Models, Values=Metric
    # We assume 'fold' or 'seed' exists in df to act as the index
    index_col = "seed" if "seed" in df.columns else "fold"
    if index_col not in df.columns:
        # Fallback: create a dummy index if multiple rows per model exist
        df = df.copy()
        df[index_col] = df.groupby(pivot_column).cumcount()

    pivot = df.pivot_table(index=index_col, columns=pivot_column, values=metric, aggfunc="mean")

    methods = pivot.columns.tolist()
    results = pivot.values  # shape (n_datasets, n_methods)

    # AEON expects (n_estimators, n_datasets)
    # So we might need to transpose depending on version, 
    # but typically cd() takes (n_datasets, n_estimators)?
    # Actually aeon.visualisation.plot_critical_difference docs say:
    # scores : np.array of shape (n_classifiers, n_datasets)
    # So we transpose.
    
    plt.figure(figsize=(10, 6))
    cd(results.T, methods, alpha=alpha)
    
    plt.title(title or f"CD diagram – {metric} (α={alpha})")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def save_confusion_matrix(cm: np.ndarray, labels: List[str] = ["Normal", "Event"], save_path: Optional[str] = None):
    """Plot and save a simple heatmap for a Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_grouped_stacked(
    real_df: pd.DataFrame, 
    dummy_df: pd.DataFrame, 
    metrics: List[str],
    figsize=(8, 5), 
    save_path: Optional[str] = None
):
    """
    Plot grouped bar chart showing:
    1. Improvement over Dummy Baseline (Base Improvement).
    2. Improvement from Tuning (Threshold Tuning Effect).
    """
    records = []
    
    # Ensure thresh column exists (if not, assume all are 0.5)
    if 'thresh' not in real_df.columns:
        real_df['thresh'] = 0.5
    if 'thresh' not in dummy_df.columns:
        dummy_df['thresh'] = 0.5

    for model, g in real_df.groupby("model"):
        # Identify "Base" (untuned) vs "Tuned"
        # We assume untuned is thresh == 0.5 or the first entry if only one exists
        base = g[g["thresh"] == 0.5]
        tuned = g[g["thresh"] != 0.5]
        
        # Get Dummy Baseline (usually averaged)
        dummy_row = dummy_df.mean(numeric_only=True)

        if base.empty and not tuned.empty:
            # Only tuned version exists
            v0 = tuned[metrics].mean().iloc[0] # Fallback
            v1 = v0
        elif not base.empty:
            v0 = base[metrics].mean().iloc[0] # Use first metric as proxy? No, need loop.
        else:
            continue

        for m in metrics:
            val_base = base[m].mean() if not base.empty else tuned[m].mean()
            val_tuned = tuned[m].mean() if not tuned.empty else val_base
            val_dummy = dummy_row[m]
            
            # Prevent div by zero
            if val_dummy == 0: val_dummy = 1e-6

            base_improv = 100 * (val_base - val_dummy) / val_dummy
            tuning_effect = 100 * (val_tuned - val_base) / val_base

            records.append({
                "model": model,
                "metric": m,
                "base_improvement": base_improv,
                "tuning_effect": tuning_effect
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("No data to plot in grouped_stacked.")
        return

    # Plotting Logic
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    
    model_order = df.groupby("model")["base_improvement"].mean().sort_values().index
    metrics_palette = sns.color_palette("tab10", n_colors=len(metrics))
    
    bar_width = 0.8 / len(metrics)
    x = np.arange(len(model_order))
    
    # Background for negative improvement
    ax.axhspan(ax.get_ylim()[0], 0, facecolor="mistyrose", alpha=0.3, zorder=0)

    for i, metric in enumerate(metrics):
        sub = df[df["metric"] == metric].set_index("model").reindex(model_order)
        offset = (i - len(metrics)/2 + 0.5) * bar_width
        
        # Base bars
        ax.bar(x + offset, sub["base_improvement"], width=bar_width, 
               color=metrics_palette[i], label=metric, zorder=2)
        
        # Stacked tuning
        # Only stack if tuning improved things, otherwise it overlaps weirdly 
        # (Simplified visualization for now)
        ax.bar(x + offset, sub["tuning_effect"], width=bar_width,
               bottom=sub["base_improvement"], color=metrics_palette[i], alpha=0.5, hatch='//', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_ylabel("% Improvement vs Dummy")
    ax.axhline(0, color='gray', lw=1)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()