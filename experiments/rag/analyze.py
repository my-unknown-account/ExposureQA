from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SETTINGS = [
    "no_context",
    "lexical_so",
    "lexical_sro",
    "support",
    "irrelevant",
]
PLOT_ORDER = ["no_context", "irrelevant", "lexical_so", "lexical_sro", "support"]
PLOT_LABELS = {
    "no_context": "No Context",
    "irrelevant": "Irrelevant",
    "lexical_so": "Lexical S-O",
    "lexical_sro": "Lexical S-R-O",
    "support": "Support",
}
MODEL_LABELS = {"amber": "Amber", "redpajama": "RedPajama", "olmo": "OLMo"}
MODE_COLORS = {"simple": "#1b9e77", "complex": "#d95f02", "template_based": "#7570b3"}


def to_binary_accuracy(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def detect_mode_from_filename(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("_simple"):
        return "simple"
    if stem.endswith("_complex"):
        return "complex"
    if stem.endswith("_template_based"):
        return "template_based"
    return None


def get_correctness_value(setting_payload: dict[str, Any], mode: str | None) -> Any:
    if mode is not None:
        key = f"is_correct_{mode}"
        if key in setting_payload:
            return setting_payload.get(key)

    generic_keys = [k for k in setting_payload if k.startswith("is_correct_")]
    if generic_keys:
        return setting_payload.get(generic_keys[0])

    return setting_payload.get("is_correct_simple", False)


def extract_model(path: Path) -> str | None:
    stem = path.stem
    for suffix in ("_simple", "_complex", "_template_based"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def analyze_file(path: Path, settings: list[str]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = detect_mode_from_filename(path)

    totals = {setting: 0.0 for setting in settings}
    counts = {setting: 0 for setting in settings}
    num_qids = 0

    for qid_payload in data.values():
        if not isinstance(qid_payload, dict):
            continue
        num_qids += 1
        for setting in settings:
            setting_payload = qid_payload.get(setting)
            if not isinstance(setting_payload, dict):
                continue
            correctness_value = get_correctness_value(setting_payload, mode)
            totals[setting] += to_binary_accuracy(correctness_value)
            counts[setting] += 1

    accuracy_by_setting: dict[str, Any] = {}
    valid_scores: list[float] = []
    for setting in settings:
        if counts[setting] == 0:
            accuracy_by_setting[setting] = None
            continue
        score = round(totals[setting] / counts[setting], 4)
        accuracy_by_setting[setting] = score
        valid_scores.append(score)

    file_accuracy_avg = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    return {
        "file": path.name,
        "model": extract_model(path),
        "mode": mode,
        "num_qids": num_qids,
        "accuracy_by_setting": accuracy_by_setting,
        "file_accuracy_avg": file_accuracy_avg,
    }


def plot_grouped_bars(results: list[dict[str, Any]], out_dir: Path) -> None:
    modes = ["simple", "complex", "template_based"]
    models = ["amber", "redpajama", "olmo"]
    result_map = {(r["model"], r["mode"]): r for r in results}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.arange(len(PLOT_ORDER) * len(modes))
    xticklabels = [PLOT_LABELS[s] for _ in modes for s in PLOT_ORDER]

    for i, model in enumerate(models):
        ax = axes[i]
        heights: list[float] = []
        colors: list[str] = []

        for mode in modes:
            entry = result_map.get((model, mode))
            for setting in PLOT_ORDER:
                value = 0.0
                if entry is not None:
                    raw = entry["accuracy_by_setting"].get(setting)
                    value = 0.0 if raw is None else float(raw)
                heights.append(value)
                colors.append(MODE_COLORS[mode])

        ax.bar(x, heights, color=colors, width=0.8)
        ax.set_title(MODEL_LABELS.get(model, model), pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=65, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel("Accuracy")

        for split in (4.5, 9.5):
            ax.axvline(split, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS["simple"]),
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS["complex"]),
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS["template_based"]),
    ]
    fig.legend(
        legend_handles,
        ["Simple", "Complex", "Template-based"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))

    out_path = out_dir / "rag_accuracy.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results_rag"
    settings = DEFAULT_SETTINGS
    output = Path(__file__).resolve().parent / "accuracy.json"

    files = [
        path
        for path in sorted(results_dir.glob("*.json"))
        if not path.name.startswith("accuracy_")
    ]
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {results_dir}")

    results = [analyze_file(path, settings) for path in files]

    for result in results:
        print(f"\n=== {result['file']} ({result['num_qids']} qids) ===")
        print(json.dumps(result["accuracy_by_setting"], ensure_ascii=True))
        print(f"file_accuracy_avg: {result['file_accuracy_avg']}")

    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved output to: {output}")
    plot_grouped_bars(results, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
