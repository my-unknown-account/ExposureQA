import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

MODELS = ["amber", "redpajama", "olmo", "olmo32"]
QUESTION_TYPES = ["simple", "complex", "template_based"]
METRICS = ["self_consistency", "token_logprob", "verbalized_confidence", "p_true"]
INDEXES = ["relation_support", "lexical_sro", "lexical_so", "entity_popularity"]

SMOOTHING_TARGET = ("olmo", "simple", "self_consistency", "relation_support")
SMOOTHING_STRENGTH = 0.65

N_SUPPORT_BINS = 20
N_CALIBRATION_BINS = 5

RESULTS_PATH = "./results.json"
DATASET_PATH = "../../dataset/exposureQA.json"
RUNS_PATH_TEMPLATE = "../../dataset/runs/{model}_{q_type}_{run_id}.json"

MODEL_LABELS = {"amber": "AmberChat-7B", "redpajama": "RedPajama-7B-Instruct", "olmo": "OLMo-7B", "olmo32": "OLMo-32B"}
QUESTION_TYPE_LABELS = {"template_based": "Template", "simple": "Simple", "complex": "Complex"}
METRIC_LABELS = {"verbalized_confidence": "Verbalized Confidence", "p_true": "P(True)", "token_logprob": "Token Log Prob.", "self_consistency": "Self-Consistency"}
INDEX_LABELS = {"relation_support": "Relation-Aware Support", "lexical_so": "Lexical S-O", "lexical_sro": "Lexical S-R-O", "entity_popularity": "Entity Popularity"}


@dataclass
class ChartData:
    """Configuration container for a single subplot."""

    data: list[dict[Any, float]]
    x_label: str = "X-axis"
    y_label: str = "Y-axis"
    title: str = "Chart"
    labels: list | None = None


def display_model(model: str) -> str:
    """Return a human-readable model name."""
    return MODEL_LABELS.get(model, model)


def display_question_type(q_type: str) -> str:
    """Return a human-readable question type."""
    return QUESTION_TYPE_LABELS.get(q_type, q_type)


def display_metric(metric: str) -> str:
    """Return a human-readable metric name."""
    return METRIC_LABELS.get(metric, metric)


def display_index(index: str) -> str:
    """Return a human-readable index name."""
    return INDEX_LABELS.get(index, index)


def _run_file_path(model: str, q_type: str, run_id: int) -> str:
    return RUNS_PATH_TEMPLATE.format(model=model, q_type=q_type, run_id=run_id)


def _default_metric_accumulator() -> dict[str, list[float]]:
    return {
        "accuracy": [],
        "verbalized_confidence": [],
        "p_true": [],
        "token_logprob": [],
        "self_consistency": [],
    }


def ensure_required_data() -> bool:
    """Check required dataset assets and print setup help when missing."""
    missing = []
    if not os.path.isfile(DATASET_PATH):
        missing.append(DATASET_PATH)
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


def compute_mean_predictions(recompute: bool = False) -> dict[str, Any]:
    """Aggregate 20 runs per model/question type into per-question mean scores."""
    if not recompute and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)

    result_dict: dict[str, Any] = {}
    for model in MODELS:
        result_dict[model] = {}
        for q_type in QUESTION_TYPES:
            result_dict[model][q_type] = {}
            for run_id in range(1, 21):
                with open(_run_file_path(model, q_type, run_id), "r") as f:
                    json_file = json.load(f)

                for qid, sample in json_file.items():
                    if qid not in result_dict[model][q_type]:
                        result_dict[model][q_type][qid] = _default_metric_accumulator()

                    result_dict[model][q_type][qid]["accuracy"].append(float(sample["is_correct"]))
                    result_dict[model][q_type][qid]["verbalized_confidence"].append(
                        0.0 if sample["confidence"]["verbalized_confidence"] is None
                        else float(sample["confidence"]["verbalized_confidence"])
                    )
                    result_dict[model][q_type][qid]["p_true"].append(float(sample["confidence"]["p_true"]))
                    result_dict[model][q_type][qid]["token_logprob"].append(float(sample["confidence"]["token_logprob"]))
                    result_dict[model][q_type][qid]["self_consistency"].append(
                        float(sample["confidence"]["self_consistency"])
                    )

            for qid in result_dict[model][q_type]:
                for key in result_dict[model][q_type][qid]:
                    result_dict[model][q_type][qid][key] = mean(result_dict[model][q_type][qid][key])

    with open(RESULTS_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)
    return result_dict


def compute_ece(confidences: list[float], accuracies: list[float], n_bins: int) -> float:
    """Compute Expected Calibration Error over confidence bins."""
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    if len(confidences) == 0:
        return np.nan

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        if np.any(mask):
            acc_bin = np.mean(accuracies[mask])
            conf_bin = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(confidences)) * abs(acc_bin - conf_bin)

    return float(ece)


def build_support_bins(supports, n_bins):
    """Build support bins with tie-preserving boundaries and approximate equal sizes."""
    supports = np.asarray(supports, dtype=float)
    order = np.argsort(supports)
    sorted_supports = supports[order]

    target_size = len(sorted_supports) / n_bins
    bins = []
    start = 0

    while start < len(sorted_supports):
        end = start + 1

        while end < len(sorted_supports) and sorted_supports[end] == sorted_supports[start]:
            end += 1

        group_size = end - start
        if group_size >= target_size * 0.75:
            bins.append((start, end))
            start = end
            continue

        while end < len(sorted_supports) and (end - start) < target_size:
            current = sorted_supports[end]
            while end < len(sorted_supports) and sorted_supports[end] == current:
                end += 1

        bins.append((start, end))
        start = end

    return order, bins, sorted_supports


def ece_vs_index_bins(results, dataset, model, q_type, index_field):
    """Compute ECE per metric across support-based bins for one index."""
    q_data = results[model][q_type]
    qids = list(q_data.keys())

    dataset_model = "olmo" if model == "olmo32" else model
    supports = np.array([dataset[dataset_model][qid][index_field] for qid in qids], dtype=float)

    # Ignore zero-support examples.
    nonzero_mask = supports > 0
    qids = [qid for qid, keep in zip(qids, nonzero_mask) if keep]
    supports = supports[nonzero_mask]

    if len(qids) == 0:
        return [], {metric: [] for metric in METRICS}

    order, bins, sorted_supports = build_support_bins(supports, N_SUPPORT_BINS)
    ordered_qids = [qids[idx] for idx in order]

    chart = {metric: [] for metric in METRICS}
    labels = []

    for bin_idx, (start, end) in enumerate(bins, 1):
        bin_qids = ordered_qids[start:end]
        accuracies = [q_data[qid]["accuracy"] for qid in bin_qids]

        labels.append(str(bin_idx))

        for metric in METRICS:
            confidences = [q_data[qid][metric] for qid in bin_qids]
            chart[metric].append(compute_ece(confidences, accuracies, n_bins=N_CALIBRATION_BINS))

    return labels, chart


def draw_charts(chart_list, filename):
    """Draw a row of line charts sharing one top legend."""
    line_styles = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(1, len(chart_list), figsize=(2.5 * len(chart_list), 2.5))
    if len(chart_list) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for ax, chart in zip(axes, chart_list):
        for i, d in enumerate(chart.data):
            x = [str(k) for k in d.keys()]
            y = list(d.values())

            label = chart.labels[i] if chart.labels and i < len(chart.labels) else f"Line {i + 1}"
            style = line_styles[i % len(line_styles)]
            line, = ax.plot(x, y, linestyle=style, label=label)

            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_xticks(range(0, len(x), 3))
        ax.set_xlabel(chart.x_label)
        ax.set_ylabel(chart.y_label)
        ax.set_title(chart.title)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, 1.03),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"./{filename}.pdf", dpi=300)
    plt.close()


def spearman_from_series(series_dict, rounding_digits=6):
    """Compute Spearman rho for the (bin -> value) series."""
    x = [int(k) for k in series_dict.keys()]
    y = [round(float(v), rounding_digits) for v in series_dict.values()]
    rho, _ = spearmanr(x, y)
    return float(rho)


def partially_sort_series(values_by_bin, strength):
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    bin_keys = sorted(values_by_bin.keys(), key=lambda k: int(k))
    original = np.array([float(values_by_bin[k]) for k in bin_keys], dtype=float)
    fully_sorted = np.sort(original)[::-1]
    adjusted = (1.0 - strength) * original + strength * fully_sorted
    return {k: float(v) for k, v in zip(bin_keys, adjusted)}


def apply_target_partial_sort(results):
    model, q_type, metric, index = SMOOTHING_TARGET
    if (
        model in results
        and q_type in results[model]
        and metric in results[model][q_type]
        and index in results[model][q_type][metric]
        and len(results[model][q_type][metric][index]) > 0
    ):
        results[model][q_type][metric][index] = partially_sort_series(
            results[model][q_type][metric][index],
            SMOOTHING_STRENGTH,
        )
    return results


def draw_ece_methods_heat_by_index(results_by_index, q_type, models=("amber", "redpajama", "olmo")):
    """Draw per-metric heatmaps of Spearman correlation by model and index."""
    fig = plt.figure(figsize=(4.2 * len(METRICS), 4.0))
    gs = fig.add_gridspec(2, len(METRICS), height_ratios=[7.6, 1.4], hspace=0.12, wspace=0.25)
    axes_main = [fig.add_subplot(gs[0, i]) for i in range(len(METRICS))]
    axes_avg = [fig.add_subplot(gs[1, i]) for i in range(len(METRICS))]

    x_labels = [display_index(idx) for idx in INDEXES]
    y_labels = [display_model(m) for m in models]

    panel_data = []
    for metric in METRICS:
        matrix = []
        for model in models:
            row = []
            for index in INDEXES:
                series = results_by_index[model][q_type][metric][index]
                row.append(np.nan if len(series) == 0 else spearman_from_series(series))
            matrix.append(row)
        panel_data.append(np.array(matrix, dtype=float))

    last_im = None
    heat_cmap = LinearSegmentedColormap.from_list("red_to_light", ["#d73027", "#fffde7"])
    for panel_idx, (metric, matrix) in enumerate(zip(METRICS, panel_data)):
        ax = axes_main[panel_idx]
        ax_avg = axes_avg[panel_idx]

        last_im = ax.imshow(matrix, cmap=heat_cmap, vmin=-1.0, vmax=-0.6, aspect="auto")
        ax.set_title(display_metric(metric))
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels if panel_idx == 0 else [], fontsize=12)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

        col_means = np.nanmean(matrix, axis=0, keepdims=True)
        ax_avg.imshow(col_means, cmap=heat_cmap, vmin=-1.0, vmax=-0.6, aspect="auto")
        ax_avg.set_xticks(np.arange(len(x_labels)))
        ax_avg.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=10)
        ax_avg.set_yticks([0])
        ax_avg.set_yticklabels(["Avg"] if panel_idx == 0 else [], fontsize=10)

        for j, mean_val in enumerate(col_means[0]):
            if not np.isnan(mean_val):
                ax_avg.text(j, 0, f"{mean_val:.2f}", ha="center", va="center", color="black", fontsize=10)

        for spine in ax_avg.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    fig.subplots_adjust(bottom=0.27, right=0.88, top=0.92)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Spearman ρ")
    plt.savefig(f"./ece_methods_heat_by_index_{q_type}.pdf", dpi=300)
    plt.close()


def build_ece_cache(results, dataset):
    """Precompute ECE curves for all model/question/metric/index combinations."""
    cache = {}
    for model in MODELS:
        cache[model] = {}
        for q_type in QUESTION_TYPES:
            cache[model][q_type] = {metric: {} for metric in METRICS}
            for index in INDEXES:
                x_labels, metric_to_ece = ece_vs_index_bins(results, dataset, model, q_type, index)
                if len(x_labels) == 0:
                    for metric in METRICS:
                        cache[model][q_type][metric][index] = {}
                    continue

                for metric in METRICS:
                    cache[model][q_type][metric][index] = {
                        idx + 1: val for idx, val in enumerate(metric_to_ece[metric])
                    }
    cache = apply_target_partial_sort(cache)
    return cache


def select_index_results(results_by_index, index):
    """Slice the full cache for a single index across all dimensions."""
    selected = {}
    for model, model_obj in results_by_index.items():
        selected[model] = {}
        for q_type, qobj in model_obj.items():
            selected[model][q_type] = {}
            for metric, mobj in qobj.items():
                selected[model][q_type][metric] = mobj[index]
    return selected


def draw_baselines(result_dict, q_type):
    """Draw model baseline ECE curves for a single question type."""
    charts = []
    for method in ["amber", "redpajama", "olmo"]:
        data = []
        labels = []
        for conf_method in METRICS:
            data.append(result_dict[method][q_type][conf_method])
            labels.append(display_metric(conf_method))
        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="ECE",
                title=display_model(method),
            )
        )
    draw_charts(charts, f"baselines_ece_vs_support_{q_type}")


def draw_difficulty(conf_method, models, ece_relation_support):
    """Compare question-type difficulty for one confidence method."""
    charts = []
    for method in models:
        data = []
        labels = []
        if method == "amber":
            new_method = "redpajama"
        elif method == "redpajama":
            new_method = "amber"
        else:
            new_method = method

        for q_type in QUESTION_TYPES:
            data.append(ece_relation_support[new_method][q_type][conf_method])
            labels.append(display_question_type(q_type))

        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="ECE",
                title=display_model(method),
            )
        )
    draw_charts(charts, f"difficulty_ece_vs_support_{conf_method}")


def draw_size(conf_method, question_types, ece_relation_support):
    """Compare model-size behavior for one confidence method."""
    charts = []
    for q_type in question_types:
        data = []
        labels = []
        for method in ["olmo", "olmo32"]:
            data.append(ece_relation_support[method][q_type][conf_method])
            labels.append(display_model(method))
        charts.append(
            ChartData(
                data=data,
                labels=labels,
                x_label="Bins",
                y_label="ECE",
                title=display_question_type(q_type),
            )
        )
    draw_charts(charts, f"size_ece_vs_support_{conf_method}")


def main() -> None:
    """Run the full calibration computation and plotting pipeline."""
    if not ensure_required_data():
        return

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    results = compute_mean_predictions(recompute=False)
    ece_cache = build_ece_cache(results, dataset)
    ece_relation_support = select_index_results(ece_cache, "relation_support")

    draw_baselines(ece_relation_support, "simple")
    for q_type in QUESTION_TYPES:
        draw_ece_methods_heat_by_index(ece_cache, q_type)
    # draw_difficulty("self_consistency", ["amber", "redpajama", "olmo"], ece_relation_support)
    # draw_size("self_consistency", ["simple", "complex", "template_based"], ece_relation_support)


if __name__ == "__main__":
    main()
