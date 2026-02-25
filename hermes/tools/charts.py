"""Chart generation tools using matplotlib.

Produces static PNG images for embedding in Excel workbooks and Word documents.
All chart functions return the file path to the saved image.  The Agg backend
is used so no display server is required.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from llama_index.core.tools import FunctionTool

from hermes.config import get_config

logger = logging.getLogger(__name__)

# Use a non-interactive backend so charts can be generated on headless servers.
matplotlib.use("Agg")

# Consistent professional styling across all charts.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_chart(fig: matplotlib.figure.Figure, filename: str | None = None) -> str:
    """Save a matplotlib figure to the configured output directory.

    Args:
        fig: The matplotlib figure to save.
        filename: Optional filename (without directory).  A unique name is
            generated if not provided.

    Returns:
        Absolute path to the saved PNG file.
    """
    cfg = get_config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"chart_{uuid.uuid4().hex[:8]}.png"
    if not filename.endswith(".png"):
        filename += ".png"

    filepath = output_dir / filename
    fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart to %s", filepath.resolve())
    return str(filepath.resolve())


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def chart_line(
    title: str,
    x_data: list,
    y_series: dict[str, list],
    x_label: str = "",
    y_label: str = "",
    filename: str | None = None,
) -> str:
    """Create a line chart with one or more data series.

    Args:
        title: Chart title.
        x_data: Values for the x-axis (dates, periods, etc.).
        y_series: Mapping of series name to y-axis values.
            Each list must have the same length as *x_data*.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        filename: Optional output filename.

    Returns:
        Absolute path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, y_values in y_series.items():
        ax.plot(x_data, y_values, label=name, linewidth=2, marker="o", markersize=3)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if len(y_series) > 1:
        ax.legend(loc="best")

    # Rotate x-axis labels if there are many data points.
    if len(x_data) > 10:
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    return _save_chart(fig, filename)


def chart_bar(
    title: str,
    categories: list[str],
    values: dict[str, list[float]],
    x_label: str = "",
    y_label: str = "",
    filename: str | None = None,
) -> str:
    """Create a bar chart with grouped bars for multiple series.

    Args:
        title: Chart title.
        categories: Labels for the x-axis categories.
        values: Mapping of series name to values.  Each list must have
            the same length as *categories*.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        filename: Optional output filename.

    Returns:
        Absolute path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_series = len(values)
    n_categories = len(categories)
    x = np.arange(n_categories)
    bar_width = 0.8 / max(n_series, 1)

    for i, (name, vals) in enumerate(values.items()):
        offset = (i - n_series / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=name)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45 if n_categories > 6 else 0, ha="right")

    if n_series > 1:
        ax.legend(loc="best")

    fig.tight_layout()
    return _save_chart(fig, filename)


def chart_waterfall(
    title: str,
    categories: list[str],
    values: list[float],
    filename: str | None = None,
) -> str:
    """Create a waterfall chart (revenue bridge, EPS bridge, etc.).

    Positive values push the running total up and are shown in green;
    negative values push it down and are shown in red.  The final bar
    shows the net total in blue.

    Args:
        title: Chart title.
        categories: Labels for each bar.
        values: Numerical values for each step.  The last value is
            treated as the total.
        filename: Optional output filename.

    Returns:
        Absolute path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(values)
    cumulative = [0.0] * n
    bottoms = [0.0] * n
    colors = []

    running = 0.0
    for i in range(n - 1):
        bottoms[i] = running if values[i] >= 0 else running + values[i]
        cumulative[i] = abs(values[i])
        colors.append("#2ca02c" if values[i] >= 0 else "#d62728")
        running += values[i]

    # The last bar is the total, drawn from zero.
    bottoms[-1] = 0.0
    cumulative[-1] = running
    colors.append("#1f77b4")

    x = np.arange(n)
    ax.bar(x, cumulative, bottom=bottoms, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on each bar.
    for i in range(n):
        y_pos = bottoms[i] + cumulative[i] / 2
        label = f"{values[i]:+,.0f}" if i < n - 1 else f"{running:,.0f}"
        ax.text(i, y_pos, label, ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.8)

    fig.tight_layout()
    return _save_chart(fig, filename)


def chart_scatter(
    title: str,
    x_data: list[float],
    y_data: list[float],
    labels: list[str] | None = None,
    x_label: str = "",
    y_label: str = "",
    filename: str | None = None,
) -> str:
    """Create a scatter plot.

    Optionally labels each point, useful for peer comparison charts
    (e.g. P/E vs growth rate with company names).

    Args:
        title: Chart title.
        x_data: X-axis values.
        y_data: Y-axis values (must be same length as *x_data*).
        labels: Optional list of point labels.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        filename: Optional output filename.

    Returns:
        Absolute path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x_data, y_data, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    if labels:
        for i, label in enumerate(labels):
            if i < len(x_data) and i < len(y_data):
                ax.annotate(
                    label,
                    (x_data[i], y_data[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.tight_layout()
    return _save_chart(fig, filename)


def chart_heatmap(
    title: str,
    data: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    filename: str | None = None,
) -> str:
    """Create a heatmap for sensitivity tables or correlation matrices.

    Each cell is colour-coded by value and annotated with the numerical
    value.  Blue indicates higher values, red indicates lower.

    Args:
        title: Chart title.
        data: 2D list of numeric values (rows x columns).
        row_labels: Labels for each row.
        col_labels: Labels for each column.
        filename: Optional output filename.

    Returns:
        Absolute path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2), max(6, len(row_labels) * 0.8)))

    arr = np.array(data, dtype=float)
    im = ax.imshow(arr, cmap="RdYlBu", aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Annotate each cell with its value.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if i < arr.shape[0] and j < arr.shape[1]:
                val = arr[i, j]
                # Use white text on dark cells, black on light cells.
                text_color = "white" if abs(val - arr.mean()) > arr.std() else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=9, color=text_color,
                )

    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    return _save_chart(fig, filename)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def create_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for all chart tools."""
    return [
        FunctionTool.from_defaults(
            fn=chart_line,
            name="chart_line",
            description=(
                "Create a line chart with one or more data series. Returns "
                "the file path to the saved PNG image."
            ),
        ),
        FunctionTool.from_defaults(
            fn=chart_bar,
            name="chart_bar",
            description=(
                "Create a bar chart with grouped bars for multiple series. "
                "Returns the file path to the saved PNG image."
            ),
        ),
        FunctionTool.from_defaults(
            fn=chart_waterfall,
            name="chart_waterfall",
            description=(
                "Create a waterfall chart (revenue bridge, EPS bridge). "
                "Positive values are green, negative are red, total is blue. "
                "Returns the file path to the saved PNG image."
            ),
        ),
        FunctionTool.from_defaults(
            fn=chart_scatter,
            name="chart_scatter",
            description=(
                "Create a scatter plot with optional point labels. Useful for "
                "peer comparisons (P/E vs growth). Returns the PNG file path."
            ),
        ),
        FunctionTool.from_defaults(
            fn=chart_heatmap,
            name="chart_heatmap",
            description=(
                "Create a heatmap for sensitivity tables or correlation matrices. "
                "Returns the file path to the saved PNG image."
            ),
        ),
    ]
