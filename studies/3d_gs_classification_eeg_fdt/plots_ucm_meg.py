import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from src.plotting.roc_curves import plot_roc_curve_on_ax
from src.plotting.confusion_matrix import plot_single_cm


# ---------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.titlesize": 16
})


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# MODE: "compare_scenarios" or "compare_feature_sets"
#   - "compare_scenarios":    fix ONE feature set, compare across scenarios
#   - "compare_feature_sets": fix ONE scenario, compare across feature sets
mode = "compare_feature_sets"

# When mode == "compare_scenarios": set the fixed feature set and list of scenarios
fixed_feature_set = "all_features_fdt_gec"
scenarios_to_compare = ["HC_vs_MCI_nC", "HC_vs_MCI_C", "MCI_nC_vs_MCI_C"]

# When mode == "compare_feature_sets": set the fixed scenario and list of feature sets
fixed_scenario = "HC_vs_MCI_C"
feature_sets_to_compare = [
    "meg_power_conn",
    "meg_thermo",
    "meg_all",
    "gene_meg_all",
    "gmv",
    "gmv_meg_thermo"
]

# Will plot the first n_features_plot + 1 features
n_features_plot = 7

# ---------------------------------------------------------------------
# Display-name mappings
# ---------------------------------------------------------------------
group_comparisons_strs = {
    "HC_vs_AD": r"HC vs AD",
    "HC_vs_MCI_nC": r"HC vs MCI (nC)",
    "HC_vs_MCI_C": r"HC vs MCI (C)",
    "SCD_vs_AD": r"SCD vs AD",
    "SCD_vs_MCI_C": r"SCD vs MCI (C)",
    "SCD_vs_MCI_nC": r"SCD vs MCI (nC)",
    "MCI_nC_vs_MCI_C": r"MCI (nC) vs MCI (C)",
    "MCI_C_vs_AD": r"MCI (C) vs AD",
}

group_comparisons_labels = {
    "HC_vs_AD": [r"HC", r"AD"],
    "HC_vs_MCI_nC": [r"HC", r"MCI (nC)"],
    "HC_vs_MCI_C": [r"HC", r"MCI (C)"],
    "SCD_vs_AD": [r"SCD", r"AD"],
    "SCD_vs_MCI_C": [r"SCD", r"MCI (C)"],
    "SCD_vs_MCI_nC": [r"SCD", r"MCI (nC)"],
    "MCI_nC_vs_MCI_C": [r"MCI (nC)", r"MCI (C)"],
    "MCI_C_vs_AD": [r"MCI (C)", r"AD"],
}

features_strs = {
    "meg_power_conn": "MEG (Power + Conn)",
    "meg_thermo": "MEG (Thermodyn)",
    "meg_all": "MEG (Thermodyn + Power + Conn)",
    "gene_meg_all": "Gene + MEG (T + P + C)",
    "gmv": "GMV",
    "gmv_meg_thermo": "GMV + MEG (Thermodyn)"
}


# ---------------------------------------------------------------------
# Color palette: one distinct color per element being compared
# ---------------------------------------------------------------------
PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]


# ---------------------------------------------------------------------
# Abbreviation function for SHAP x-axis labels
# ---------------------------------------------------------------------
def abbreviate_feature_name(name: str) -> str:
    """
    Compact abbreviation for feature names.

    FDT / EP / Asymmetry features:
        analytical_fdt_int_delta  ->  FDT $\delta$
    GMV features:
        raparc_r_gm_entorhinal   ->  R EC
        laparc_l_gm_parahippocampal -> L parahipp
    """
    # --- GMV features (raparc / laparc prefix) ---
    if name.startswith("raparc") or name.startswith("laparc"):
        hemi = "R" if name.startswith("raparc") else "L"
        # Strip the prefix (raparc_r_gm_ or laparc_l_gm_)
        region = name.split("_gm_", 1)[-1] if "_gm_" in name else name
        gmv_map = {
            "entorhinal": "EC",
            "parahippocampal": "parahipp",
        }
        region = gmv_map.get(region, region)
        return f"{hemi} {region}"

    if name.startswith("aseg"):
        hemi_map = {"l": "L", "r": "R"}
        # e.g. aseg_l_hipp -> L hipp
        parts = name.split("_", 2)  # ['aseg', 'l', 'hipp']
        if len(parts) >= 3:
            hemi = hemi_map.get(parts[1], parts[1])
            return f"{hemi} {parts[2]}"
        return name

    # --- FDT / EP / Asymmetry features (sequential replacements) ---
    label = name
    # Core measure names (apply before band suffixes)
    label = label.replace("analytical_fdt_int", "FDT")
    label = label.replace("entropy_production", "EP")
    label = label.replace("asymmetry", "Asymm")
    # Frequency band suffixes
    label = label.replace("_delta", r" $\delta$")
    label = label.replace("_theta", r" $\Theta$")
    label = label.replace("_alpha", r" $\alpha$")
    label = label.replace("_beta", r" $\beta$")
    label = label.replace("_gamma", r" $\gamma$")

    return label


# ---------------------------------------------------------------------
# Helper: load all data for a given (scenario, feature_set) pair
# ---------------------------------------------------------------------
def load_results(path_results, scenario, feature_set):
    """Return dict with roc_data, avg_cm, feature_importance, pct_selected, imp_sd."""
    path_sce = path_results / "LogReg" / scenario / feature_set

    # ROC data
    roc_data = joblib.load(path_sce / "avg_roc_data_test.joblib")

    # Confusion matrices averaged across folds
    fold_dirs = [
        d for d in os.listdir(path_sce)
        if (path_sce / d).is_dir() and re.fullmatch(r"fold_\d+", d)
    ]
    fold_dirs.sort(key=lambda d: int(d.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under: {path_sce}")

    cms = []
    for d in fold_dirs:
        cm_path = path_sce / d / "test_cm.txt"
        if not cm_path.exists():
            raise FileNotFoundError(f"Expected confusion matrix missing: {cm_path}")
        cms.append(np.loadtxt(cm_path))
    avg_cm = np.mean(np.asarray(cms), axis=0)

    # SHAP feature importance
    df_imp = pd.read_csv(path_sce / "summary_importance.csv")
    feat_imp = {
        feat: val
        for i, (feat, val) in enumerate(zip(df_imp["feature"], df_imp["mean_abs_SHAP"]))
        if i <= n_features_plot
    }
    pct_selected = df_imp["selection_freq"].tolist()[:n_features_plot + 1]
    imp_sd = df_imp["sd"].tolist()[:n_features_plot + 1]

    return {
        "roc_data": roc_data,
        "avg_cm": avg_cm,
        "feature_importance": feat_imp,
        "pct_selected": pct_selected,
        "imp_sd": imp_sd,
    }


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
path_results = path_repo / "Results" / "3d_gs_classification_ucm_meg_fdt_gec"
if mode == "compare_feature_sets":
    path_figures = path_results / mode / fixed_scenario
else:
    path_figures = path_results / mode / fixed_feature_set
path_figures.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Build the list of (key, display_label, scenario, feature_set)
# depending on the chosen mode
# ---------------------------------------------------------------------
if mode == "compare_scenarios":
    items = [
        (sc, group_comparisons_strs[sc], sc, fixed_feature_set)
        for sc in scenarios_to_compare
    ]
    fixed_label = features_strs[fixed_feature_set]
    varying_label = "scenario"
    file_suffix = f"{fixed_feature_set}"
elif mode == "compare_feature_sets":
    items = [
        (fs, features_strs[fs], fixed_scenario, fs)
        for fs in feature_sets_to_compare
    ]
    fixed_label = group_comparisons_strs[fixed_scenario]
    varying_label = "feature_set"
    file_suffix = f"{fixed_scenario.lower()}"
else:
    raise ValueError(f"Unknown mode: {mode}")

# Assign a color to each item
colors = {key: PALETTE[i % len(PALETTE)] for i, (key, *_) in enumerate(items)}

# Load all data
data = {}
for key, display, scenario, feature_set in items:
    data[key] = load_results(path_results, scenario, feature_set)
    data[key]["display"] = display
    data[key]["scenario"] = scenario
    data[key]["feature_set"] = feature_set

n_items = len(items)


# ---------------------------------------------------------------------
# Helper: create a single-hue colormap from white to a given color
# ---------------------------------------------------------------------
def make_single_hue_cmap(color, name="custom"):
    """White-to-*color* colormap for confusion matrices."""
    rgb = mcolors.to_rgb(color)
    return LinearSegmentedColormap.from_list(name, ["white", rgb])


# =====================================================================
# 1) ROC curves (overlaid on a single axes)
# =====================================================================
roc_size = 5 + 0.5 * (n_items - 1)          # grows slightly with more curves
fig_roc, ax_roc = plt.subplots(figsize=(roc_size, roc_size))

for key, display, scenario, feature_set in items:
    rd = data[key]["roc_data"]
    plot_roc_curve_on_ax(
        rd["fpr"],
        rd["tpr"],
        display,
        ax_roc,
        rd["std_tpr"],
        roc_color=colors[key],
    )

ax_roc.legend()
ax_roc.set_box_aspect(1)
ax_roc.set_title(f"ROC Curves ({fixed_label})")

fig_roc.tight_layout()
fig_roc.savefig(path_figures / f"classification_roc_{file_suffix}.pdf", dpi=600)
fig_roc.savefig(path_figures / f"classification_roc_{file_suffix}.svg", dpi=600)


# =====================================================================
# 2) Confusion matrices (side by side, color-matched)
# =====================================================================
cm_single_w = 4.5
fig_cm, axes_cm = plt.subplots(
    1, n_items, figsize=(cm_single_w * n_items + 0.5 * (n_items - 1), cm_single_w)
)
if n_items == 1:
    axes_cm = [axes_cm]

for ax, (key, display, scenario, feature_set) in zip(axes_cm, items):
    cm_labels = group_comparisons_labels[scenario]
    cmap_cm = make_single_hue_cmap(colors[key], name=key)
    plot_single_cm(
        data[key]["avg_cm"],
        cm_labels,
        ax,
        vmin=0,
        vmax=1,
        cmap=cmap_cm,
        fontsize=18,
    )
    ax.set_title(display)
    ax.set_box_aspect(1)

fig_cm.tight_layout()
fig_cm.savefig(path_figures / f"classification_cm_{file_suffix}.pdf", dpi=600)
fig_cm.savefig(path_figures / f"classification_cm_{file_suffix}.svg", dpi=600)


# =====================================================================
# 3) SHAP + Selection Frequency (one subplot per item)
# =====================================================================
shap_single_w = 7
shap_single_h = 4
fig_shap, axes_shap = plt.subplots(
    n_items, 1,
    figsize=(shap_single_w, shap_single_h * n_items + 0.5 * (n_items - 1)),
    squeeze=False,
)
axes_shap = axes_shap[:, 0]  # single col

for idx, (key, display, scenario, feature_set) in enumerate(items):
    ax_bar = axes_shap[idx]
    ax_twin = ax_bar.twinx()

    feat_imp = data[key]["feature_importance"]
    pct_sel = data[key]["pct_selected"]

    features = list(feat_imp.keys())
    x_values = np.arange(len(features))
    abbrev = [abbreviate_feature_name(f) for f in features]

    total_bar_width = 0.8
    single_bar_width = total_bar_width / 2

    color_shap = colors[key]
    # Darken the color for selection frequency bars
    rgb = mcolors.to_rgb(color_shap)
    color_freq = tuple(c * 0.6 for c in rgb)

    ax_bar.bar(
        x_values - single_bar_width / 2,
        list(feat_imp.values()),
        width=single_bar_width,
        label="Mean SHAP value",
        color=color_shap,
    )

    ax_twin.bar(
        x_values + single_bar_width / 2,
        pct_sel,
        width=single_bar_width,
        label="Selection Frequency",
        color=color_freq,
    )

    ax_twin.set(ylim=(0, 1))

    ax_bar.set_xticks(x_values)
    ax_bar.set_xticklabels(abbrev)
    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")

    ax_bar.set_title(display)
    ax_bar.set_ylabel("Mean SHAP value")
    ax_twin.set_ylabel("Selection Frequency")

    # Combined legend
    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax_bar.legend(h1 + h2, l1 + l2, loc="upper right")

    # Color-code spines
    ax_bar.spines["left"].set_color(color_shap)
    ax_bar.tick_params(axis="y", colors=color_shap)
    ax_twin.spines["right"].set_color(color_freq)
    ax_twin.tick_params(axis="y", colors=color_freq)

    ax_twin.set_xlim(
        x_values[0] - 1.25 * single_bar_width,
        x_values[-1] + 1.25 * single_bar_width,
    )

fig_shap.suptitle(f"Feature Importance ({fixed_label})", fontsize=18, y=1.02)
fig_shap.tight_layout()
fig_shap.savefig(
    path_figures / f"classification_shap_{file_suffix}.pdf", dpi=600,
    bbox_inches="tight",
)
fig_shap.savefig(
    path_figures / f"classification_shap_{file_suffix}.svg", dpi=600,
    bbox_inches="tight",
)

plt.show()
