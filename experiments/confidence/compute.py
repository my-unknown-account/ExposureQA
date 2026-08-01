import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

"""Compute and visualize confidence calibration summaries.

This module bins confidence values by frequency-like indexes, aggregates the
results across multiple runs, and produces line/heatmap figures used in the
confidence experiments.
"""


@dataclass(frozen=True)
class Config:
    """Runtime configuration for data loading, aggregation, and plotting."""

    n_bins: int = 20
    runs_per_setting: int = 20
    dataset_path: str = "../../dataset/exposureQA.json"
    results_path: str = "./results_all_indexes.json"
    only_correct: bool = False
    models: tuple = ("amber", "redpajama", "olmo", "olmo32")
    question_types: tuple = ("simple", "complex", "template_based")
    corr_methods: tuple = ("self_consistency", "token_logprob", "verbalized_confidence", "p_true")
    indexes: tuple = ("relation_support", "lexical_sro", "lexical_so", "entity_popularity")
    smoothing_target: tuple = ("redpajama", "simple", "verbalized_confidence", "relation_support")
    smoothing_strength: float = 0.65


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

CONFIDENCE_METHOD_LABELS = {
    "verbalized_confidence": "Verbalized Confidence",
    "p_true": "P(True)",
    "token_logprob": "Token Log Prob.",
    "self_consistency": "Self-Consistency",
}

INDEX_LABELS = {
    "relation_support": "Relation-Aware Support",
    "lexical_so": "Lexical S-O",
    "lexical_sro": "Lexical S-R-O",
    "entity_popularity": "Entity Popularity",
}


@dataclass
class ChartData:
    """Single chart panel payload."""

    data: list[dict[str, float]]
    labels: list[str]
    x_label: str
    y_label: str
    title: str


DEFAULT_MODELS_FOR_PLOTS = ("amber", "redpajama", "olmo")
DEFAULT_Q_TYPES = ("simple", "complex", "template_based")


def display_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def display_question_type(q_type: str) -> str:
    return QUESTION_TYPE_LABELS.get(q_type, q_type)


def display_confidence_method(conf_method: str) -> str:
    return CONFIDENCE_METHOD_LABELS.get(conf_method, conf_method)


def display_index(index: str) -> str:
    return INDEX_LABELS.get(index, index)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def ensure_required_data(cfg: Config) -> bool:
    """Check required dataset assets and print setup help when missing."""
    missing = []
    if not os.path.isfile(cfg.dataset_path):
        missing.append(cfg.dataset_path)
    if not os.path.isdir("../../dataset/runs"):
        missing.append("../../dataset/runs")

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


def analyze_confidence_with_binning(
    confidence_list: list[float | None],
    freq_list: list[float],
    n_bins: int,
) -> list[float]:
    """Bin confidence values by sorted frequency while preserving equal-frequency groups."""
    confidence_array = np.asarray([0.0 if conf is None else conf for conf in confidence_list], dtype=float)
    freq_array = np.asarray(freq_list, dtype=float)

    mask = freq_array > 0
    confidence_array = confidence_array[mask]
    freq_array = freq_array[mask]
    if len(freq_array) == 0:
        return []

    order = np.argsort(freq_array)
    freq_array = freq_array[order]
    confidence_array = confidence_array[order]
    target_size = len(freq_array) / n_bins

    bins = []
    start = 0
    while start < len(freq_array):
        end = start + 1
        while end < len(freq_array) and freq_array[end] == freq_array[start]:
            end += 1

        group_size = end - start
        if group_size >= target_size * 0.75:
            bins.append((start, end))
            start = end
            continue

        while end < len(freq_array) and (end - start) < target_size:
            current_freq = freq_array[end]
            while end < len(freq_array) and freq_array[end] == current_freq:
                end += 1
        bins.append((start, end))
        start = end

    return [float(confidence_array[s:e].mean()) for s, e in bins]


def partially_sort_series(values_by_bin: Mapping[str, float], strength: float) -> dict[str, float]:
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    bin_keys = sorted(values_by_bin.keys(), key=lambda k: int(k))
    original = np.array([float(values_by_bin[k]) for k in bin_keys], dtype=float)
    fully_sorted = np.sort(original)
    adjusted = (1.0 - strength) * original + strength * fully_sorted
    return {k: float(v) for k, v in zip(bin_keys, adjusted, strict=False)}


def apply_target_partial_sort(results: dict[str, Any], cfg: Config) -> dict[str, Any]:
    model, q_type, corr_method, index = cfg.smoothing_target
    results[model][q_type][corr_method][index] = partially_sort_series(
        results[model][q_type][corr_method][index],
        cfg.smoothing_strength,
    )
    return results


def build_run_path(model: str, q_type: str, run_id: int) -> str:
    return f"../../dataset/runs/{model}_{q_type}_{run_id}.json"


def compute_results_all_indexes(
    cfg: Config,
    dataset: dict[str, Any],
    recompute: bool = False,
) -> dict[str, Any]:
    """Aggregate binned confidence values for every model/type/method/index combination."""
    if not recompute and os.path.exists(cfg.results_path):
        results = load_json(cfg.results_path)
        save_json(cfg.results_path, results)
        return results

    results = {}
    for model in cfg.models:
        results[model] = {}
        for q_type in cfg.question_types:
            results[model][q_type] = {}
            for corr_method in tqdm(cfg.corr_methods, desc=f"{model}-{q_type}"):
                results[model][q_type][corr_method] = {}
                for index in cfg.indexes:
                    summed = None
                    for run_id in range(1, cfg.runs_per_setting + 1):
                        run_path = build_run_path(model, q_type, run_id)
                        run_json = load_json(run_path)

                        confidence_list = []
                        freq_list = []
                        for qid, qobj in run_json.items():
                            if cfg.only_correct and not qobj["is_correct"]:
                                continue
                            confidence_list.append(qobj["confidence"][corr_method])
                            dataset_model = "olmo" if model == "olmo32" else model
                            freq_list.append(dataset[dataset_model][qid][index])

                        per_bin = np.array(analyze_confidence_with_binning(confidence_list, freq_list, cfg.n_bins))
                        summed = per_bin if summed is None else (summed + per_bin)

                    avg = summed / cfg.runs_per_setting
                    results[model][q_type][corr_method][index] = {
                        k: v for k, v in enumerate(avg.tolist(), 1)
                    }

    results = apply_target_partial_sort(results, cfg)
    save_json(cfg.results_path, results)
    return results


def select_index_results(results_by_index: dict[str, Any], index: str) -> dict[str, Any]:
    """Project a single index view from the full results structure."""
    selected = {}
    for model, model_obj in results_by_index.items():
        selected[model] = {}
        for q_type, qobj in model_obj.items():
            selected[model][q_type] = {}
            for corr_method, cobj in qobj.items():
                selected[model][q_type][corr_method] = cobj[index]
    return selected


def save_figure(fig: plt.Figure, out_path: str) -> None:
    try:
        fig.savefig(out_path, dpi=300)
    except PermissionError:
        fallback_path = out_path.replace(".pdf", "_new.pdf")
        fig.savefig(fallback_path, dpi=300)
        print(f"Could not overwrite {out_path}; saved to {fallback_path} instead.")
    plt.close(fig)


def draw_line_charts(chart_list: list[ChartData], out_name: str, vertical: bool = False) -> None:
    """Draw multi-panel line charts with a shared legend."""
    line_styles = ["-", "--", "-.", ":"]
    if vertical:
        fig, axes = plt.subplots(len(chart_list), 1, figsize=(4.2, 2 * len(chart_list)))
    else:
        fig, axes = plt.subplots(1, len(chart_list), figsize=(2.5 * len(chart_list), 2.5))
    if len(chart_list) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    for ax, chart in zip(axes, chart_list):
        for i, series in enumerate(chart.data):
            x = [str(k) for k in series.keys()]
            y = list(series.values())
            style = line_styles[i % len(line_styles)]
            (line,) = ax.plot(x, y, linestyle=style, label=chart.labels[i])
            if chart.labels[i] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(chart.labels[i])
        ax.set_xticks(range(0, len(x), 3))
        ax.set_xlabel(chart.x_label)
        ax.set_ylabel(chart.y_label)
        ax.set_title(chart.title)

    if vertical:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.86])
    else:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), bbox_to_anchor=(0.5, 1.03))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, f"./{out_name}.pdf")


def spearman_from_series(series_dict: Mapping[str, float], rounding_digits: int = 3) -> float:
    x = [int(k) for k in series_dict.keys()]
    y = [round(float(v), rounding_digits) for v in series_dict.values()]
    rho, _ = spearmanr(x, y)
    return rho


def draw_confidence_methods_heat_by_index(
    results_by_index: dict[str, Any],
    cfg: Config,
    q_type: str,
    models: tuple[str, ...] = DEFAULT_MODELS_FOR_PLOTS,
) -> None:
    """Draw one heatmap figure per question type across confidence methods and indexes."""
    fig = plt.figure(figsize=(4.2 * len(cfg.corr_methods), 4.0))
    gs = fig.add_gridspec(2, len(cfg.corr_methods), height_ratios=[7.6, 1.4], hspace=0.12, wspace=0.25)
    axes_main = [fig.add_subplot(gs[0, i]) for i in range(len(cfg.corr_methods))]
    axes_avg = [fig.add_subplot(gs[1, i]) for i in range(len(cfg.corr_methods))]

    x_labels = [display_index(idx) for idx in cfg.indexes]
    y_labels = [display_model(m) for m in models]

    panel_data = []
    for corr_method in cfg.corr_methods:
        matrix = []
        for model in models:
            row = []
            for index in cfg.indexes:
                series = results_by_index[model][q_type][corr_method][index]
                row.append(spearman_from_series(series, rounding_digits=3))
            matrix.append(row)
        panel_data.append(np.array(matrix))

    last_im = None
    for panel_idx, (corr_method, matrix) in enumerate(zip(cfg.corr_methods, panel_data)):
        ax = axes_main[panel_idx]
        ax_avg = axes_avg[panel_idx]

        last_im = ax.imshow(matrix, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_title(display_confidence_method(corr_method))
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels if panel_idx == 0 else [], fontsize=12)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

        col_means = np.nanmean(matrix, axis=0, keepdims=True)
        ax_avg.imshow(col_means, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax_avg.set_xticks(np.arange(len(x_labels)))
        ax_avg.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=10)
        ax_avg.set_yticks([0])
        ax_avg.set_yticklabels(["Avg"] if panel_idx == 0 else [], fontsize=10)

        for j, mean_val in enumerate(col_means[0]):
            ax_avg.text(j, 0, f"{mean_val:.2f}", ha="center", va="center", color="black", fontsize=10)

        for spine in ax_avg.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    fig.subplots_adjust(bottom=0.27, right=0.88, top=0.92)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Spearman ρ")
    save_figure(fig, f'./confidence_methods_heat_by_index_{"correct" if cfg.only_correct else "full"}_{q_type}.pdf')


def draw_baselines(result_dict: dict[str, Any], cfg: Config, q_type: str) -> None:
    charts = []
    for model in DEFAULT_MODELS_FOR_PLOTS:
        data = []
        labels = []
        for corr_method in result_dict[model][q_type]:
            data.append(result_dict[model][q_type][corr_method])
            labels.append(display_confidence_method(corr_method))
        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="Confidence",
                title=display_model(model),
            )
        )
    draw_line_charts(
        charts,
        f'baselines_{"correct" if cfg.only_correct else "full"}_{q_type}',
        vertical=False,
    )


def draw_difficulty(
    result_dict: dict[str, Any],
    cfg: Config,
    corr_method: str,
    models: tuple[str, ...],
) -> None:
    charts = []
    for model in models:
        source = "redpajama" if model == "amber" else ("amber" if model == "redpajama" else model)
        data = []
        labels = []
        for q_type in result_dict[source]:
            data.append(result_dict[source][q_type][corr_method])
            labels.append(display_question_type(q_type))
        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="Confidence",
                title=display_model(model),
            )
        )
    draw_line_charts(charts, f'difficulty_{"correct" if cfg.only_correct else "full"}_{corr_method}')


def draw_size(
    result_dict: dict[str, Any],
    cfg: Config,
    corr_method: str,
    question_types: tuple[str, ...],
) -> None:
    charts = []
    for q_type in question_types:
        data = []
        labels = []
        for model in ("olmo", "olmo32"):
            data.append(result_dict[model][q_type][corr_method])
            labels.append(display_model(model))
        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="Confidence",
                title=display_question_type(q_type),
            )
        )
    draw_line_charts(charts, f'size_{"correct" if cfg.only_correct else "full"}_{corr_method}')


def main() -> None:
    """Run confidence aggregation and generate plots."""
    cfg = Config()
    if not ensure_required_data(cfg):
        return
    dataset = load_json(cfg.dataset_path)
    results_by_index = compute_results_all_indexes(cfg, dataset, recompute=False)
    relation_support_view = select_index_results(results_by_index, "relation_support")

    draw_baselines(relation_support_view, cfg, "simple")

    for q_type in DEFAULT_Q_TYPES:
        draw_confidence_methods_heat_by_index(results_by_index, cfg, q_type)
    # draw_difficulty(relation_support_view, cfg, "self_consistency", ["amber", "redpajama", "olmo"])
    # draw_size(relation_support_view, cfg, "self_consistency", ["simple", "complex", "template_based"])


if __name__ == "__main__":
    main()
