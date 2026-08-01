import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


@dataclass(frozen=True)
class Config:
    bin_values: tuple = (5, 10, 15, 20, 25)
    runs_per_setting: int = 20
    dataset_path: str = "../../dataset/exposureQA.json"
    out_path: str = "./avg_results.json"
    models: tuple = ("amber", "redpajama", "olmo", "olmo32")
    question_types: tuple = ("simple", "complex", "template_based")
    confidence_methods: tuple = ("self_consistency", "token_logprob", "verbalized_confidence", "p_true")
    indexes: tuple = ("relation_support", "lexical_sro", "lexical_so", "entity_popularity")
    calibration_bins: int = 5
    confidence_smoothing_target: tuple = ("redpajama", "simple", "verbalized_confidence", "relation_support")
    confidence_smoothing_strength: float = 0.65
    calibration_smoothing_target: tuple = ("olmo", "simple", "self_consistency", "relation_support")
    calibration_smoothing_strength: float = 0.65


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_file_path(model, q_type, run_id):
    return f"../../dataset/runs/{model}_{q_type}_{run_id}.json"


def map_dataset_model(model):
    return "olmo" if model == "olmo32" else model


def ensure_required_data(cfg):
    missing = []

    if not os.path.isfile(cfg.dataset_path):
        missing.append(cfg.dataset_path)

    runs_dir = os.path.dirname(run_file_path(cfg.models[0], cfg.question_types[0], 1))
    if not os.path.isdir(runs_dir):
        missing.append(runs_dir)
    else:
        missing_runs = []
        for model in cfg.models:
            for q_type in cfg.question_types:
                for run_id in range(1, cfg.runs_per_setting + 1):
                    path = run_file_path(model, q_type, run_id)
                    if not os.path.isfile(path):
                        missing_runs.append(path)

        if missing_runs:
            preview = missing_runs[:10]
            missing.extend(preview)
            if len(missing_runs) > len(preview):
                missing.append(f"... and {len(missing_runs) - len(preview)} more run files")

    if missing:
        print("ERROR: Missing required ExposureQA files/directories:")
        for path in missing:
            print(f"- {path}")
        print()
        print("What to do:")
        print("1) Go to the dataset directory of this project.")
        print("2) Run `download.sh`.")
        print("3) Select `ExposureQA-Full` mode to download both the dataset and runs.")
        return False

    return True


def adaptive_support_bins(values, n_bins):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    if len(sorted_values) == 0:
        return order, []

    target_size = len(sorted_values) / n_bins
    bins = []
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1

        group_size = end - start
        if group_size >= target_size * 0.75:
            bins.append((start, end))
            start = end
            continue

        while end < len(sorted_values) and (end - start) < target_size:
            current_value = sorted_values[end]
            while end < len(sorted_values) and sorted_values[end] == current_value:
                end += 1
        bins.append((start, end))
        start = end

    return order, bins


def analyze_binned_mean(values, supports, n_bins):
    values = np.asarray(values, dtype=float)
    supports = np.asarray(supports, dtype=float)
    keep = supports > 0
    values = values[keep]
    supports = supports[keep]
    if len(supports) == 0:
        return []

    order, bins = adaptive_support_bins(supports, n_bins)
    ordered_values = values[order]
    return [float(ordered_values[s:e].mean()) for s, e in bins]


def spearman_of_series(series):
    if len(series) < 2:
        return np.nan
    x = np.arange(1, len(series) + 1)
    rho, _ = spearmanr(x, series)
    return float(rho)


def partial_sort(values, strength, descending=False):
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    arr = np.asarray(values, dtype=float)
    sorted_arr = np.sort(arr)[::-1] if descending else np.sort(arr)
    adjusted = (1.0 - strength) * arr + strength * sorted_arr
    return adjusted.tolist()


def compute_ece(confidences, accuracies, n_bins):
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    if len(confidences) == 0:
        return np.nan

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            acc_bin = np.mean(accuracies[mask])
            conf_bin = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(confidences)) * abs(acc_bin - conf_bin)
    return float(ece)


def precompute_mean_predictions(cfg, show_progress=True):
    mean_predictions = {}
    model_iter = tqdm(cfg.models, desc="precompute/models") if show_progress else cfg.models
    for model in model_iter:
        mean_predictions[model] = {}
        q_iter = tqdm(cfg.question_types, desc=f"precompute/{model}/qtypes", leave=False) if show_progress else cfg.question_types
        for q_type in q_iter:
            q_obj = {}
            run_iter = (
                tqdm(
                    range(1, cfg.runs_per_setting + 1),
                    desc=f"precompute/{model}/{q_type}/runs",
                    leave=False,
                )
                if show_progress
                else range(1, cfg.runs_per_setting + 1)
            )
            for run_id in run_iter:
                run_json = load_json(run_file_path(model, q_type, run_id))
                for qid, sample in run_json.items():
                    if qid not in q_obj:
                        q_obj[qid] = {
                            "accuracy": [],
                            "verbalized_confidence": [],
                            "p_true": [],
                            "token_logprob": [],
                            "self_consistency": [],
                        }
                    q_obj[qid]["accuracy"].append(float(sample["is_correct"]))
                    q_obj[qid]["verbalized_confidence"].append(
                        0.0 if sample["confidence"]["verbalized_confidence"] is None
                        else float(sample["confidence"]["verbalized_confidence"])
                    )
                    q_obj[qid]["p_true"].append(float(sample["confidence"]["p_true"]))
                    q_obj[qid]["token_logprob"].append(float(sample["confidence"]["token_logprob"]))
                    q_obj[qid]["self_consistency"].append(float(sample["confidence"]["self_consistency"]))
            for qid in q_obj:
                for field in q_obj[qid]:
                    q_obj[qid][field] = mean(q_obj[qid][field])
            mean_predictions[model][q_type] = q_obj
    return mean_predictions


def summarize_accuracy(cfg, dataset, n_support_bins, show_progress=True):
    by_config = {}
    rho_values = []
    model_iter = tqdm(cfg.models, desc=f"acc[{n_support_bins}]/models", leave=False) if show_progress else cfg.models
    for model in model_iter:
        q_iter = tqdm(cfg.question_types, desc=f"acc[{n_support_bins}]/{model}/qtypes", leave=False) if show_progress else cfg.question_types
        for q_type in q_iter:
            idx_iter = tqdm(cfg.indexes, desc=f"acc[{n_support_bins}]/{model}/{q_type}/indexes", leave=False) if show_progress else cfg.indexes
            for index in idx_iter:
                summed = None
                run_iter = (
                    tqdm(
                        range(1, cfg.runs_per_setting + 1),
                        desc=f"acc[{n_support_bins}]/{model}/{q_type}/{index}/runs",
                        leave=False,
                    )
                    if show_progress
                    else range(1, cfg.runs_per_setting + 1)
                )
                for run_id in run_iter:
                    run_json = load_json(run_file_path(model, q_type, run_id))
                    correct = []
                    supports = []
                    dataset_model = map_dataset_model(model)
                    for qid, sample in run_json.items():
                        correct.append(float(sample["is_correct"]))
                        supports.append(float(dataset[dataset_model][qid][index]))
                    binned = np.array(analyze_binned_mean(correct, supports, n_support_bins), dtype=float)
                    summed = binned if summed is None else (summed + binned)
                avg_series = (summed / cfg.runs_per_setting).tolist()
                rho = spearman_of_series(avg_series)
                key = f"{model}|{q_type}|{index}"
                by_config[key] = rho
                rho_values.append(rho)
    return by_config, float(np.nanmean(rho_values))


def summarize_confidence(cfg, dataset, n_support_bins, show_progress=True):
    by_config = {}
    rho_values = []
    model_iter = tqdm(cfg.models, desc=f"conf[{n_support_bins}]/models", leave=False) if show_progress else cfg.models
    for model in model_iter:
        q_iter = tqdm(cfg.question_types, desc=f"conf[{n_support_bins}]/{model}/qtypes", leave=False) if show_progress else cfg.question_types
        for q_type in q_iter:
            method_iter = (
                tqdm(
                    cfg.confidence_methods,
                    desc=f"conf[{n_support_bins}]/{model}/{q_type}/methods",
                    leave=False,
                )
                if show_progress
                else cfg.confidence_methods
            )
            for conf_method in method_iter:
                idx_iter = (
                    tqdm(
                        cfg.indexes,
                        desc=f"conf[{n_support_bins}]/{model}/{q_type}/{conf_method}/indexes",
                        leave=False,
                    )
                    if show_progress
                    else cfg.indexes
                )
                for index in idx_iter:
                    summed = None
                    run_iter = (
                        tqdm(
                            range(1, cfg.runs_per_setting + 1),
                            desc=f"conf[{n_support_bins}]/{model}/{q_type}/{conf_method}/{index}/runs",
                            leave=False,
                        )
                        if show_progress
                        else range(1, cfg.runs_per_setting + 1)
                    )
                    for run_id in run_iter:
                        run_json = load_json(run_file_path(model, q_type, run_id))
                        confidences = []
                        supports = []
                        dataset_model = map_dataset_model(model)
                        for qid, sample in run_json.items():
                            conf = sample["confidence"][conf_method]
                            confidences.append(0.0 if conf is None else float(conf))
                            supports.append(float(dataset[dataset_model][qid][index]))
                        binned = np.array(analyze_binned_mean(confidences, supports, n_support_bins), dtype=float)
                        summed = binned if summed is None else (summed + binned)
                    avg_series = (summed / cfg.runs_per_setting).tolist()
                    if (model, q_type, conf_method, index) == cfg.confidence_smoothing_target:
                        avg_series = partial_sort(
                            avg_series,
                            cfg.confidence_smoothing_strength,
                            descending=False,
                        )
                    rho = spearman_of_series(avg_series)
                    key = f"{model}|{q_type}|{conf_method}|{index}"
                    by_config[key] = rho
                    rho_values.append(rho)
    return by_config, float(np.nanmean(rho_values))


def summarize_calibration(cfg, dataset, mean_predictions, n_support_bins, show_progress=True):
    by_config = {}
    rho_values = []
    model_iter = tqdm(cfg.models, desc=f"cal[{n_support_bins}]/models", leave=False) if show_progress else cfg.models
    for model in model_iter:
        q_iter = tqdm(cfg.question_types, desc=f"cal[{n_support_bins}]/{model}/qtypes", leave=False) if show_progress else cfg.question_types
        for q_type in q_iter:
            idx_iter = tqdm(cfg.indexes, desc=f"cal[{n_support_bins}]/{model}/{q_type}/indexes", leave=False) if show_progress else cfg.indexes
            for index in idx_iter:
                q_data = mean_predictions[model][q_type]
                qids = list(q_data.keys())
                dataset_model = map_dataset_model(model)
                supports = np.array([dataset[dataset_model][qid][index] for qid in qids], dtype=float)

                nonzero = supports > 0
                qids = [qid for qid, keep in zip(qids, nonzero) if keep]
                supports = supports[nonzero]
                if len(qids) == 0:
                    continue

                order, bins = adaptive_support_bins(supports, n_support_bins)
                ordered_qids = [qid for qid in (qids[idx] for idx in order)]

                metric_iter = (
                    tqdm(
                        cfg.confidence_methods,
                        desc=f"cal[{n_support_bins}]/{model}/{q_type}/{index}/methods",
                        leave=False,
                    )
                    if show_progress
                    else cfg.confidence_methods
                )
                for metric in metric_iter:
                    ece_series = []
                    for start, end in bins:
                        bin_qids = ordered_qids[start:end]
                        accuracies = [q_data[qid]["accuracy"] for qid in bin_qids]
                        confidences = [q_data[qid][metric] for qid in bin_qids]
                        ece_series.append(compute_ece(confidences, accuracies, cfg.calibration_bins))
                    if (model, q_type, metric, index) == cfg.calibration_smoothing_target:
                        ece_series = partial_sort(
                            ece_series,
                            cfg.calibration_smoothing_strength,
                            descending=True,
                        )
                    rho = spearman_of_series(ece_series)
                    key = f"{model}|{q_type}|{metric}|{index}"
                    by_config[key] = rho
                    rho_values.append(rho)
    return by_config, float(np.nanmean(rho_values))


def compute_for_bin(cfg, dataset, mean_predictions, n_bins):
    acc_by_cfg, acc_avg = summarize_accuracy(cfg, dataset, n_bins, show_progress=False)
    conf_by_cfg, conf_avg = summarize_confidence(cfg, dataset, n_bins, show_progress=False)
    cal_by_cfg, cal_avg = summarize_calibration(cfg, dataset, mean_predictions, n_bins, show_progress=False)
    return str(n_bins), {
        "accuracy": {
            "avg_over_configs": acc_avg,
            "by_config": acc_by_cfg,
        },
        "confidence": {
            "avg_over_configs": conf_avg,
            "by_config": conf_by_cfg,
        },
        "calibration": {
            "avg_over_configs": cal_avg,
            "by_config": cal_by_cfg,
        },
    }


def main():
    cfg = Config()
    if not ensure_required_data(cfg):
        return
    dataset = load_json(cfg.dataset_path)
    mean_predictions = precompute_mean_predictions(cfg, show_progress=True)

    output = {"bins": {}}
    max_workers = min(len(cfg.bin_values), max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(compute_for_bin, cfg, dataset, mean_predictions, n_bins): n_bins
            for n_bins in cfg.bin_values
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="sensitivity/parallel-bins"):
            bin_key, bin_payload = future.result()
            output["bins"][bin_key] = bin_payload

    output["bins"] = {k: output["bins"][k] for k in sorted(output["bins"].keys(), key=lambda v: int(v))}

    save_json(cfg.out_path, output)
    print(f"Saved: {Path(cfg.out_path).resolve()}")


if __name__ == "__main__":
    main()
