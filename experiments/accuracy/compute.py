import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Polygon
from scipy.stats import spearmanr
from tqdm import tqdm

AccuracySeries = Dict[int, float]
ResultTree = Dict[str, Dict[str, Dict[str, AccuracySeries]]]


@dataclass(frozen=True)
class Config:
    """Runtime configuration for result aggregation and chart generation."""

    n_bins: int = 20
    runs_per_setting: int = 20
    dataset_path: str = "../../dataset/exposureQA.json"
    results_path: str = "./results.json"
    models: tuple = ("amber", "redpajama", "olmo", "olmo32")
    question_types: tuple = ("simple", "complex", "template_based")
    indexes: tuple = ("relation_support", "lexical_sro", "lexical_so", "entity_popularity")
    runs_dir: str = "../../dataset/runs"


@dataclass
class ChartData:
    """Container for a single chart panel and its visual metadata."""

    data: List[Dict[int, float]]
    labels: List[str]
    x_label: str
    y_label: str
    title: str


MODEL_LABELS = {
    "amber": "AmberChat-7B",
    "redpajama": "RedPajama-7B-Instruct",
    "olmo": "OLMo-7B",
    "olmo32": "OLMo-32B",
}

QUESTION_TYPE_LABELS = {
    "template_based": "Template",
    "simple": "Simple",
    "complex": "Complex",
}

INDEX_LABELS = {
    "relation_support": "Relation-Aware Support",
    "lexical_sro": "Lexical S-R-O",
    "lexical_so": "Lexical S-O",
    "entity_popularity": "Entity Popularity",
}


def display_model(model: str) -> str:
    """Return display name for a model key."""
    return MODEL_LABELS.get(model, model)


def display_question_type(q_type: str) -> str:
    """Return display name for a question-type key."""
    return QUESTION_TYPE_LABELS.get(q_type, q_type)


def display_index(index: str) -> str:
    """Return display name for an index key."""
    return INDEX_LABELS.get(index, index)


def map_labels(values: Sequence[str], mapping: Dict[str, str]) -> List[str]:
    """Map internal keys to display labels while preserving order."""
    return [mapping.get(v, v) for v in values]


def load_json(path: str) -> Any:
    """Load UTF-8 JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    """Save UTF-8 JSON with deterministic indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_figure(fig: plt.Figure, out_path: str, tight: bool = False) -> None:
    """Save a figure to PDF, with fallback naming on permission errors."""

    save_kwargs = {"dpi": 300}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    try:
        fig.savefig(out_path, **save_kwargs)
    except PermissionError:
        fallback = out_path.replace(".pdf", "_new.pdf")
        fig.savefig(fallback, **save_kwargs)
        print(f"Could not overwrite {out_path}; saved to {fallback} instead.")
    plt.close(fig)


def ensure_required_data(cfg: Config) -> bool:
    """Check required dataset assets and print setup help when missing."""

    missing = []
    if not os.path.isfile(cfg.dataset_path):
        missing.append(cfg.dataset_path)
    if not os.path.isdir(cfg.runs_dir):
        missing.append(cfg.runs_dir)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        print("ERROR: Missing required ExposureQA files/directories:")
        print(missing_list)
        print()
        print("What to do:")
        print("1) Go to the dataset directory of this project.")
        print("2) Run `download.sh`.")
        print("3) Select `ExposureQA-Full` mode to download both the dataset and runs.")
        return False
    return True


def analyze_accuracy_with_binning(
    correct_list: Sequence[int],
    freq_list: Sequence[float],
    n_bins: int,
) -> List[float]:
    """Return mean accuracy in frequency-aligned bins.

    Items with identical frequency stay in the same bin to avoid splitting ties.
    """

    correct = np.asarray(correct_list, dtype=int)
    freq = np.asarray(freq_list, dtype=float)

    mask = freq > 0
    correct = correct[mask]
    freq = freq[mask]
    if len(freq) == 0:
        return []

    order = np.argsort(freq)
    freq = freq[order]
    correct = correct[order]
    target_size = len(freq) / n_bins

    bins = []
    start = 0
    while start < len(freq):
        end = start + 1
        while end < len(freq) and freq[end] == freq[start]:
            end += 1

        group_size = end - start
        if group_size >= target_size * 0.75:
            bins.append((start, end))
            start = end
            continue

        while end < len(freq) and (end - start) < target_size:
            current_freq = freq[end]
            while end < len(freq) and freq[end] == current_freq:
                end += 1
        bins.append((start, end))
        start = end

    return [float(correct[s:e].mean()) for s, e in bins]


def compute_results(
    cfg: Config,
    dataset: Dict[str, Dict[str, Dict[str, float]]],
    recompute: bool = False,
) -> ResultTree:
    """Compute mean binned accuracies over repeated runs for each setting."""

    if not recompute and os.path.exists(cfg.results_path):
        return load_json(cfg.results_path)

    results: ResultTree = {}
    for model in cfg.models:
        results[model] = {}
        for q_type in cfg.question_types:
            results[model][q_type] = {}
            for index in tqdm(cfg.indexes, desc=f"{model}-{q_type}"):
                summed: Optional[np.ndarray] = None
                for run_id in range(1, cfg.runs_per_setting + 1):
                    run_path = os.path.join(cfg.runs_dir, f"{model}_{q_type}_{run_id}.json")
                    run_json = load_json(run_path)
                    correct_list = []
                    freq_list = []
                    for qid, qobj in run_json.items():
                        correct_list.append(qobj["is_correct"])
                        dataset_model = "olmo" if model == "olmo32" else model
                        freq_list.append(dataset[dataset_model][qid][index])
                    binned = np.array(
                        analyze_accuracy_with_binning(correct_list, freq_list, cfg.n_bins)
                    )
                    summed = binned if summed is None else (summed + binned)

                avg = summed / cfg.runs_per_setting
                results[model][q_type][index] = {k: v for k, v in enumerate(avg.tolist(), 1)}

    save_json(cfg.results_path, results)
    return results


def draw_line_charts(
    chart_list: Sequence[ChartData],
    filename: str,
    zoom_range: Optional[Tuple[int, int]] = (1, 8),
    width: float = 4,
) -> None:
    """Draw one line-chart panel per `ChartData` item, optionally with zoom insets."""

    def extract_xy(data_dict: Dict[int, float]) -> Tuple[List[int], List[float]]:
        # Handles both int keys (fresh results) and str keys (JSON-loaded results).
        x_vals = [int(k) for k in data_dict.keys()]
        y_vals = list(data_dict.values())
        return x_vals, y_vals

    def collect_zoom_y(
        series: Sequence[Tuple[List[int], List[float], str]],
        x_min: int,
        x_max: int,
    ) -> List[float]:
        out: List[float] = []
        for x_vals, y_vals, _ in series:
            out.extend([yv for xv, yv in zip(x_vals, y_vals) if x_min <= xv <= x_max])
        return out

    line_styles = ["-", "--", "-.", ":"]
    has_zoom = zoom_range is not None

    if has_zoom:
        fig, axes = plt.subplots(2, len(chart_list), figsize=(width * len(chart_list), 4.5), sharex=False)
        if len(chart_list) == 1:
            axes = np.array(axes).reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, len(chart_list), figsize=(width * len(chart_list), 2.5), sharex=False)
        if len(chart_list) == 1:
            axes = [axes]

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    zoom_links: List[Tuple[plt.Axes, plt.Axes, int, int]] = []

    for idx, chart in enumerate(chart_list):
        ax = axes[0, idx] if has_zoom else axes[idx]
        zoom_ax = axes[1, idx] if has_zoom else None
        plotted_series: List[Tuple[List[int], List[float], str]] = []

        for i, series in enumerate(chart.data):
            x, y = extract_xy(series)
            style = line_styles[i % len(line_styles)]
            label = chart.labels[i] if i < len(chart.labels) else f"Line {i + 1}"
            line, = ax.plot(x, y, linestyle=style, label=label)
            plotted_series.append((x, y, style))
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
        if x:
            ax.set_xticks(range(min(x), max(x) + 1, 3))

        ax.set_xlabel(chart.x_label)
        ax.set_ylabel(chart.y_label)
        ax.set_title(chart.title)

        if has_zoom and plotted_series:
            x_min, x_max = zoom_range
            ax.axvspan(x_min, x_max, color="blue", alpha=0.15, zorder=0)
            for x_vals, y_vals, style in plotted_series:
                zoom_ax.plot(x_vals, y_vals, linestyle=style, linewidth=1.2)
            zoom_ax.set_xlim(x_min, x_max)
            zoom_y = collect_zoom_y(plotted_series, x_min, x_max)
            if zoom_y:
                y_pad = 0.02
                zoom_ax.set_ylim(max(0.0, min(zoom_y) - y_pad), min(1.0, max(zoom_y) + y_pad))
            zoom_ax.set_xticks(range(x_min, x_max + 1, 1))
            zoom_ax.set_title("")
            zoom_ax.set_xlabel("")
            zoom_ax.set_ylabel("")
            zoom_ax.tick_params(axis="both", labelsize=8)
            zoom_ax.grid(True, alpha=0.2)
            for spine in zoom_ax.spines.values():
                spine.set_edgecolor("gray")
                spine.set_linewidth(1.3)
            zoom_links.append((ax, zoom_ax, x_min, x_max))

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), bbox_to_anchor=(0.5, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if has_zoom:
        for _, zoom_ax, _, _ in zoom_links:
            pos = zoom_ax.get_position()
            new_width = pos.width * 0.72
            new_x0 = pos.x0 + (pos.width - new_width) / 2
            zoom_ax.set_position([new_x0, pos.y0, new_width, pos.height])

        for ax, zoom_ax, x_min, x_max in zoom_links:
            con_left = ConnectionPatch(
                xyA=(x_min, 0), coordsA=ax.get_xaxis_transform(),
                xyB=(x_min, 1), coordsB=zoom_ax.get_xaxis_transform(),
                color="gray", linewidth=1.5, alpha=0.95, linestyle="--"
            )
            con_right = ConnectionPatch(
                xyA=(x_max, 0), coordsA=ax.get_xaxis_transform(),
                xyB=(x_max, 1), coordsB=zoom_ax.get_xaxis_transform(),
                color="gray", linewidth=1.5, alpha=0.95, linestyle="--"
            )
            con_left.set_zorder(10)
            con_right.set_zorder(10)
            fig.add_artist(con_left)
            fig.add_artist(con_right)

            top_left_disp = ax.get_xaxis_transform().transform((x_min, 0))
            top_right_disp = ax.get_xaxis_transform().transform((x_max, 0))
            bottom_right_disp = zoom_ax.get_xaxis_transform().transform((x_max, 1))
            bottom_left_disp = zoom_ax.get_xaxis_transform().transform((x_min, 1))
            to_fig = fig.transFigure.inverted().transform
            highlight_poly = Polygon(
                [to_fig(top_left_disp), to_fig(top_right_disp), to_fig(bottom_right_disp), to_fig(bottom_left_disp)],
                closed=True,
                transform=fig.transFigure,
                facecolor="blue",
                edgecolor="none",
                alpha=0.08,
                zorder=2,
            )
            fig.add_artist(highlight_poly)

    save_figure(fig, f"./{filename}.pdf", tight=False)


def draw_heat_charts(chart_list: Sequence[ChartData], filename: str) -> None:
    """Draw Spearman-correlation heatmaps for each chart panel and per-column mean."""

    heatmap_data: List[List[float]] = []
    row_labels: List[str] = []
    col_labels: Optional[List[str]] = None

    for chart in chart_list:
        rho_values = []
        labels = []
        for i, series in enumerate(chart.data):
            # Handles both int keys (fresh results) and str keys (JSON-loaded results).
            x = [int(k) for k in series.keys()]
            y = list(series.values())
            rho, _ = spearmanr(x, y)
            rho_values.append(rho)
            labels.append(chart.labels[i] if i < len(chart.labels) else f"Series {i + 1}")
        heatmap_data.append(rho_values)
        row_labels.append(chart.title)
        if col_labels is None:
            col_labels = labels

    heatmap_data = np.array(heatmap_data)
    col_means = np.nanmean(heatmap_data, axis=0, keepdims=True)

    tick_fontsize = 12
    cell_size = 0.72
    fig = plt.figure(figsize=(cell_size * len(col_labels) + 4.0, cell_size * (len(row_labels) + 1) + 2.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1.8], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_avg = fig.add_subplot(gs[1, 0])

    im = ax.imshow(heatmap_data, cmap="YlGn", vmin=0.6, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=tick_fontsize)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=12)

    ax_avg.imshow(col_means, cmap="YlGn", vmin=0.6, vmax=1, aspect="auto")
    ax_avg.set_xticks(np.arange(len(col_labels)))
    ax_avg.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=tick_fontsize)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg"], fontsize=tick_fontsize)
    for j in range(len(col_labels)):
        ax_avg.text(j, 0, f"{col_means[0, j]:.2f}", ha="center", va="center", color="black", fontsize=12)
    for spine in ax_avg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    fig.subplots_adjust(bottom=0.32, right=0.87, top=0.95, left=0.12)
    fig.canvas.draw()
    main_pos = ax.get_position()
    avg_pos = ax_avg.get_position()
    ax_avg.set_position([main_pos.x0, avg_pos.y0, main_pos.width, avg_pos.height])
    ax_avg.set_xlim(ax.get_xlim())

    cax = fig.add_axes([0.89, 0.17, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spearman ρ")

    save_figure(fig, f"./{filename}.pdf", tight=True)


def build_baseline_charts(
    result_dict: ResultTree,
    q_type: str,
) -> List[ChartData]:
    charts: List[ChartData] = []
    for model in ("amber", "redpajama", "olmo"):
        model_data = result_dict[model][q_type]
        charts.append(
            ChartData(
                data=[model_data[index] for index in Config.indexes],
                labels=map_labels(Config.indexes, INDEX_LABELS),
                x_label="Bins",
                y_label="Accuracy",
                title=display_model(model),
            )
        )
    return charts


def build_difficulty_charts(
    result_dict: ResultTree,
    index: str,
    models: Sequence[str],
) -> List[ChartData]:
    charts: List[ChartData] = []
    for model in models:
        model_data = result_dict[model]
        charts.append(
            ChartData(
                data=[model_data[q_type][index] for q_type in Config.question_types],
                labels=map_labels(Config.question_types, QUESTION_TYPE_LABELS),
                x_label="Bins",
                y_label="Accuracy",
                title=display_model(model),
            )
        )
    return charts


def build_size_charts(
    result_dict: ResultTree,
    index: str,
    question_types: Sequence[str],
) -> List[ChartData]:
    charts: List[ChartData] = []
    for q_type in question_types:
        charts.append(
            ChartData(
                data=[result_dict[model][q_type][index] for model in ("olmo", "olmo32")],
                labels=map_labels(("olmo", "olmo32"), MODEL_LABELS),
                x_label="Bins",
                y_label="Accuracy",
                title=display_question_type(q_type),
            )
        )
    return charts


def main():
    """Compute aggregated results and generate all line and heatmap outputs."""

    cfg = Config()
    if not ensure_required_data(cfg):
        return
    dataset = load_json(cfg.dataset_path)
    result_dict = compute_results(cfg, dataset, recompute=False)

    for q_type in cfg.question_types:
        baseline_charts = build_baseline_charts(result_dict, q_type)
        draw_line_charts(baseline_charts, f"baselines_line_{q_type}")
        draw_heat_charts(baseline_charts, f"baselines_heat_{q_type}")

    for index in cfg.indexes:
        difficulty_charts = build_difficulty_charts(result_dict, index, ["amber", "redpajama", "olmo"])
        draw_line_charts(difficulty_charts, f"difficulty_line_{index}", zoom_range=None, width=2.5)

        size_charts = build_size_charts(result_dict, index, cfg.question_types)
        draw_line_charts(size_charts, f"size_line_{index}", zoom_range=None, width=2.5)


if __name__ == "__main__":
    main()
