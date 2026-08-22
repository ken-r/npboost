from __future__ import annotations

from typing import Any, Optional
import matplotlib.pyplot as plt
import plotnine as pn
from plotnine import (
    theme_minimal,
    theme,
    element_text,
    element_line,
    element_blank,
    element_rect
)

# Model names for plots
NPBOOST_NAME = "NPBoost"
ANPBOOST_NAME = "ANPBoost"
NP_NAME = "NP"
ANP_NAME = "ANP"
GBM_NO_GROUP_NAME = "GBM"
GBM_GROUP_CAT_NAME = "GBM + Group"
GPLINEAR_NAME = "GPLinear"
LME_NAME = "LME"

PLOT_MODEL_NAMES = {
    "npboost": NPBOOST_NAME,
    "anpboost": ANPBOOST_NAME,
    "np": NP_NAME,
    "anp": ANP_NAME,
    "gbm_no_group": GBM_NO_GROUP_NAME,
    "gbm_group_cat": GBM_GROUP_CAT_NAME,
    "gplinear": GPLINEAR_NAME,
    "lme": LME_NAME,
}

CONTEXT_NAME = "Context"
TARGET_NAME = "Target"

# Color palette to reuse across all plots for consistency
PAPER_PALETTE = {
    NPBOOST_NAME: "#4E79A7",
    ANPBOOST_NAME: "#2F5D8C",
    NP_NAME: "#F28E2B",
    ANP_NAME: "#B95F00",
    GBM_NO_GROUP_NAME: "#9C755F",
    GBM_GROUP_CAT_NAME: "#8CD17D",
    GPLINEAR_NAME: "#76B7B2",
    LME_NAME: "#B07AA1",
    "Truth": "#D1495B",
    CONTEXT_NAME: "#D1495B",
    TARGET_NAME: "#2A9D8F",
    "Validation Prediction": "#4E79A7",
}

# Plotnine Academic Theme
PAPER_THEME = (
    theme_minimal(base_size=11, base_family="DejaVu Serif")
    + theme(
        plot_title=element_text(weight="bold", size=14, ha="center"),
        axis_title=element_text(size=11),
        axis_text=element_text(size=10, color="#303030"),
        legend_title=element_text(size=10, weight="bold"),
        legend_text=element_text(size=9),
        legend_position="top",
        panel_grid_major=element_line(color="#D9D9D9", size=0.35),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="#C7C7C7", fill=None, size=0.5),
        plot_background=element_rect(fill="white", color=None),
        panel_background=element_rect(fill="white", color=None),
        figure_size=(8.0, 5.2),
    )
)


# Matplotlib Academic Helper
def apply_academic_style(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    has_legend: bool = False,
) -> None:
    """Apply unified academic theme to a Matplotlib Axes."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif", "Times New Roman"]

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, family="serif")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8, family="serif")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8, family="serif")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C7C7C7")
    ax.spines["bottom"].set_color("#C7C7C7")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.grid(True, which="major", color="#D9D9D9", linestyle="-", linewidth=0.35, alpha=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#303030", labelsize=9)

    if has_legend:
        legend = ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
        if legend:
            legend.get_frame().set_linewidth(0)
