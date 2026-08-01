import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


INPUT_PATH = Path("./avg_results.json")
Q_TYPE = "simple"
BIN_VALUES = (5, 10, 15, 20, 25)
MODELS = ("amber", "redpajama", "olmo")
INDEXES = ("relation_support", "lexical_sro", "lexical_so", "entity_popularity")
METHODS = ("self_consistency", "token_logprob", "verbalized_confidence", "p_true")

MODEL_LABELS = {
    "amber": "AmberChat-7B",
    "redpajama": "RedPajama-7B-Instruct",
    "olmo": "OLMo-7B",
}
INDEX_LABELS = {
    "relation_support": "Relation-Aware Support",
    "lexical_sro": "Lexical S-R-O",
    "lexical_so": "Lexical S-O",
    "entity_popularity": "Entity Popularity",
}
METHOD_LABELS = {
    "self_consistency": "Self-Consistency",
    "token_logprob": "Token Log Prob.",
    "verbalized_confidence": "Verbalized Confidence",
    "p_true": "P(True)",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_map(result_map, expected_parts):
    out = {}
    for key, value in result_map.items():
        parts = key.split("|")
        if len(parts) != expected_parts:
            continue
        out[tuple(parts)] = float(value)
    return out


def build_accuracy_matrix(bin_payload):
    by_cfg = parse_map(bin_payload["accuracy"]["by_config"], expected_parts=3)
    matrix = np.full((len(MODELS), len(INDEXES)), np.nan, dtype=float)
    for i, model in enumerate(MODELS):
        for j, index in enumerate(INDEXES):
            matrix[i, j] = by_cfg.get((model, Q_TYPE, index), np.nan)
    return matrix


def build_method_matrix(bin_payload, task_key, method):
    by_cfg = parse_map(bin_payload[task_key]["by_config"], expected_parts=4)
    matrix = np.full((len(MODELS), len(INDEXES)), np.nan, dtype=float)
    for i, model in enumerate(MODELS):
        for j, index in enumerate(INDEXES):
            matrix[i, j] = by_cfg.get((model, Q_TYPE, method, index), np.nan)
    return matrix


def annotate_matrix(ax, matrix, fmt="{:.2f}", fontsize=8):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", color="black", fontsize=fontsize)


def draw_heat_with_avg_strip(
    ax_main,
    ax_avg,
    matrix,
    x_labels,
    y_labels,
    vmin,
    vmax,
    cmap="YlGn",
    show_y_labels=True,
    annotate_fs_main=8,
    annotate_fs_avg=8,
    main_aspect="auto",
    avg_aspect="auto",
    xtick_fs=8,
    ytick_fs=9,
):
    im = ax_main.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect=main_aspect)
    ax_main.set_xticks(np.arange(len(x_labels)))
    ax_main.set_xticklabels([])
    ax_main.set_yticks(np.arange(len(y_labels)))
    ax_main.set_yticklabels(y_labels if show_y_labels else [], fontsize=ytick_fs)
    annotate_matrix(ax_main, matrix, fontsize=annotate_fs_main)

    col_means = np.nanmean(matrix, axis=0, keepdims=True)
    ax_avg.imshow(col_means, cmap=cmap, vmin=vmin, vmax=vmax, aspect=avg_aspect)
    ax_avg.set_xticks(np.arange(len(x_labels)))
    ax_avg.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=xtick_fs)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg"] if show_y_labels else [], fontsize=ytick_fs)
    annotate_matrix(ax_avg, col_means, fontsize=annotate_fs_avg)
    for spine in ax_avg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    return im


def draw_baselines_heat_simple(payload):
    rows = []
    for b in BIN_VALUES:
        matrix = build_accuracy_matrix(payload["bins"][str(b)])
        rows.append(np.nanmean(matrix, axis=0))
    heat = np.array(rows, dtype=float)

    vmin = 0.6
    vmax = 1.0
    tick_fontsize = 12
    cell_size = 0.72
    # Match absolute figure sizing style of accuracy heatmap (3 main rows + 1 avg row footprint).
    fig = plt.figure(
        figsize=(cell_size * len(INDEXES) + 4.0, cell_size * (len(MODELS) + 1) + 2.2)
    )
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1.8], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_avg = fig.add_subplot(gs[1, 0])

    im = draw_heat_with_avg_strip(
        ax_main=ax,
        ax_avg=ax_avg,
        matrix=heat,
        x_labels=[INDEX_LABELS[idx] for idx in INDEXES],
        y_labels=[str(b) for b in BIN_VALUES],
        vmin=vmin,
        vmax=vmax,
        show_y_labels=True,
        annotate_fs_main=12,
        annotate_fs_avg=12,
        main_aspect="auto",
        avg_aspect="auto",
        xtick_fs=12,
        ytick_fs=12,
    )
    ax.set_ylabel("Bins", fontsize=12)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax_avg.tick_params(axis="x", labelsize=tick_fontsize)
    ax_avg.tick_params(axis="y", labelsize=tick_fontsize)

    fig.canvas.draw()
    main_pos = ax.get_position()
    avg_pos = ax_avg.get_position()
    ax_avg.set_position([main_pos.x0, avg_pos.y0, main_pos.width, avg_pos.height])
    ax_avg.set_xlim(ax.get_xlim())

    cax = fig.add_axes([0.89, 0.17, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spearman ρ")
    fig.subplots_adjust(bottom=0.32, right=0.87, top=0.95, left=0.12)
    fig.savefig("./accuracy_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_method_grid(payload, task_key, out_name, title_prefix, vmin, vmax):
    panels = {}
    all_vals = []
    for method in METHODS:
        rows = []
        for b in BIN_VALUES:
            matrix = build_method_matrix(payload["bins"][str(b)], task_key, method)
            row = np.nanmean(matrix, axis=0)
            rows.append(row)
        heat = np.array(rows, dtype=float)
        panels[method] = heat
        all_vals.extend(heat[~np.isnan(heat)].tolist())
    _ = all_vals
    cmap = "YlGn"
    if task_key == "calibration":
        cmap = LinearSegmentedColormap.from_list("red_to_light", ["#d73027", "#fffde7"])

    tick_fontsize = 10
    fig = plt.figure(figsize=(4.2 * len(METHODS), 4.0))
    gs = fig.add_gridspec(2, len(METHODS), height_ratios=[7.6, 1.4], hspace=0.12, wspace=0.25)
    axes_main = [fig.add_subplot(gs[0, i]) for i in range(len(METHODS))]
    axes_avg = [fig.add_subplot(gs[1, i]) for i in range(len(METHODS))]

    last_im = None
    for i, method in enumerate(METHODS):
        heat = panels[method]

        ax = axes_main[i]
        ax_avg = axes_avg[i]
        last_im = draw_heat_with_avg_strip(
            ax_main=ax,
            ax_avg=ax_avg,
            matrix=heat,
            x_labels=[INDEX_LABELS[idx] for idx in INDEXES],
            y_labels=[str(b) for b in BIN_VALUES],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_y_labels=(i == 0),
            annotate_fs_main=10,
            annotate_fs_avg=10,
            main_aspect="auto",
            avg_aspect="auto",
        )
        ax.set_title(METHOD_LABELS[method], fontsize=14)
        if i == 0:
            ax.set_ylabel("Bins", fontsize=12)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax_avg.tick_params(axis="x", labelsize=tick_fontsize)
        ax_avg.tick_params(axis="y", labelsize=tick_fontsize)

    fig.canvas.draw()
    for ax, ax_avg in zip(axes_main, axes_avg):
        main_pos = ax.get_position()
        avg_pos = ax_avg.get_position()
        ax_avg.set_position([main_pos.x0, avg_pos.y0, main_pos.width, avg_pos.height])
        ax_avg.set_xlim(ax.get_xlim())

    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Spearman ρ")
    fig.subplots_adjust(bottom=0.27, right=0.88, top=0.92)
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    payload = load_json(INPUT_PATH)
    draw_baselines_heat_simple(payload)
    draw_method_grid(
        payload,
        task_key="confidence",
        out_name="./confidence_sensitivity.pdf",
        title_prefix="confidence_methods_heat_by_index_full_simple",
        vmin=0.5,
        vmax=1.0,
    )
    draw_method_grid(
        payload,
        task_key="calibration",
        out_name="./calibration_sensitivity.pdf",
        title_prefix="ece_methods_heat_by_index_simple",
        vmin=-1.0,
        vmax=-0.6,
    )
    print(f"Saved in: {Path('.').resolve()}")


if __name__ == "__main__":
    main()
