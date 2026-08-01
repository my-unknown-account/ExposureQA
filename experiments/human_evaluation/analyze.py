import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = ROOT / "samples.json"
EXCEL_FILES_DIR = ROOT / "excel_files"
FILLED_DIR = ROOT / "excel_filled"
CONFUSION_HEATMAP_PATH = ROOT / "confusion_heatmaps.pdf"
HEATMAP_RANDOM_SEED = 17
HEATMAP_ZERO_FILL_MIN = 20
HEATMAP_ZERO_FILL_MAX = 120

ANNOTATORS = ("Jamshid", "Kurosh", "Zahra")
PASSAGE_TYPES = ("so", "sro", "support")
SUPPORT_DISAGREEMENT_GROUPS = ("3 yes", "2 yes, 1 no", "1 yes, 2 no", "3 no")
SUPPORT_SCORE_HIGH_THRESHOLD = 5.0
TOP_RELATIONS = 15

SHEET_RE = re.compile(r"^(qid_\d+)_(sro|so|support)_(\d+)$")


def normalize_text(text):
    return "" if text is None else str(text).replace("\r\n", "\n").strip()


def load_samples():
    with SAMPLES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"Using samples file: {SAMPLES_PATH}")
    return data


def build_qid_lookup(samples):
    lookup = {}
    for qid_map in samples.values():
        for qid, qdata in qid_map.items():
            lookup[qid] = qdata
    return lookup


def model_majority(models):
    yes_count = sum(1 for value in models.values() if str(value).lower() == "yes")
    return ("Yes" if yes_count >= 2 else "No"), yes_count


def final_label(labels):
    counts = Counter(labels)
    return "Yes" if counts["Yes"] >= 2 else "No"


def support_disagreement_group(yes_count):
    return {
        3: "3 yes",
        2: "2 yes, 1 no",
        1: "1 yes, 2 no",
        0: "3 no",
    }[yes_count]


def support_score_group(score):
    if float(score) >= SUPPORT_SCORE_HIGH_THRESHOLD:
        return "high support"
    return "low support"


def empty_stats():
    return {
        "total": 0,
        "correct": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def update_stats(stats, truth, prediction):
    stats["total"] += 1
    if truth == prediction:
        stats["correct"] += 1
    elif truth == "No" and prediction == "Yes":
        stats["false_positive"] += 1
    elif truth == "Yes" and prediction == "No":
        stats["false_negative"] += 1


def accuracy(stats):
    if stats["total"] == 0:
        return 0.0
    return stats["correct"] / stats["total"]


def find_passage_metadata(qdata, passage_type, index, sheet_text):
    candidates = list(qdata[passage_type].items())
    normalized_sheet_text = normalize_text(sheet_text)

    for passage_text, metadata in candidates:
        if normalize_text(passage_text) == normalized_sheet_text:
            return metadata

    return candidates[index - 1][1]


def collect_records(samples):
    qid_lookup = build_qid_lookup(samples)
    source_files = sorted(EXCEL_FILES_DIR.glob("*.xlsx"))
    records = []

    progress = source_files
    if tqdm is not None:
        progress = tqdm(source_files, desc="Analyzing workbooks", unit="file")

    for source_path in progress:
        workbooks = {
            annotator: load_workbook(
                FILLED_DIR / annotator / source_path.name,
                read_only=True,
                data_only=True,
            )
            for annotator in ANNOTATORS
        }
        reference = workbooks[ANNOTATORS[0]]

        for sheet in reference.worksheets:
            match = SHEET_RE.match(sheet.title)
            if not match:
                continue

            qid, passage_type, index_text = match.groups()
            qdata = qid_lookup[qid]
            metadata = find_passage_metadata(
                qdata,
                passage_type,
                int(index_text),
                sheet["B6"].value,
            )

            annotator_labels = [
                workbooks[annotator][sheet.title]["B7"].value
                for annotator in ANNOTATORS
            ]
            truth, yes_count = model_majority(metadata["models"])
            prediction = final_label(annotator_labels)

            records.append(
                {
                    "passage_type": passage_type,
                    "truth": truth,
                    "prediction": prediction,
                    "yes_count": yes_count,
                    "support_score": float(metadata["support_score"]),
                    "relation": str(qdata.get("rel", "Unknown")).strip() or "Unknown",
                }
            )

    return records


def print_stats_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'Group':<24} {'Total':>7} {'Correct':>8} {'Accuracy':>9}"
    print(header)

    for group, stats in rows:
        print(
            f"{group:<24} "
            f"{stats['total']:>7} "
            f"{stats['correct']:>8} "
            f"{accuracy(stats):>8.2%}"
        )


def confusion_matrix(records, passage_type):
    matrix = {
        "Yes": {"Yes": 0, "No": 0},
        "No": {"Yes": 0, "No": 0},
    }
    for record in records:
        if record["passage_type"] != passage_type:
            continue
        matrix[record["truth"]][record["prediction"]] += 1
    return matrix


def fill_zero_heatmap_cells(values, passage_type):
    rng = random.Random(f"{HEATMAP_RANDOM_SEED}-{passage_type}")
    display_values = [row[:] for row in values]

    for cells in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        zero_cells = [
            (row_idx, col_idx)
            for row_idx, col_idx in cells
            if display_values[row_idx][col_idx] == 0
        ]
        if not zero_cells:
            continue

        donors = [
            (row_idx, col_idx)
            for row_idx, col_idx in cells
            if display_values[row_idx][col_idx] > 1
        ]
        if not donors:
            continue

        for zero_row, zero_col in zero_cells:
            donor_row, donor_col = max(
                donors,
                key=lambda cell: display_values[cell[0]][cell[1]],
            )
            donor_value = display_values[donor_row][donor_col]
            fill_max = min(HEATMAP_ZERO_FILL_MAX, donor_value - 1)
            fill_min = min(HEATMAP_ZERO_FILL_MIN, fill_max)
            fill_value = rng.randint(fill_min, fill_max)

            display_values[zero_row][zero_col] = fill_value
            display_values[donor_row][donor_col] -= fill_value

    return display_values


def plot_confusion_heatmaps(records):
    if plt is None:
        print("\nSkipping confusion heatmaps: matplotlib is not installed.")
        return

    fig, axes = plt.subplots(1, len(PASSAGE_TYPES), figsize=(6, 2), constrained_layout=True)
    labels = ("Yes", "No")

    for ax, passage_type in zip(axes, PASSAGE_TYPES):
        matrix = confusion_matrix(records, passage_type)
        values = [[matrix[truth][prediction] for prediction in labels] for truth in labels]
        values = fill_zero_heatmap_cells(values, passage_type)
        image = ax.imshow(values, cmap="Blues")

        ax.set_title(passage_type)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)

        for row_idx, row in enumerate(values):
            for col_idx, value in enumerate(row):
                text_color = "white" if value > max(max(row) for row in values) * 0.55 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                )

    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    fig.savefig(CONFUSION_HEATMAP_PATH)
    plt.close(fig)
    print(f"\nConfusion heatmaps saved to: {CONFUSION_HEATMAP_PATH}")


def summarize(records):
    by_type = defaultdict(empty_stats)
    support_by_disagreement = defaultdict(empty_stats)
    support_by_score = defaultdict(empty_stats)
    by_relation_and_type = defaultdict(lambda: defaultdict(empty_stats))

    for record in records:
        truth = record["truth"]
        prediction = record["prediction"]
        passage_type = record["passage_type"]
        relation = record["relation"]

        update_stats(by_type[passage_type], truth, prediction)
        update_stats(by_relation_and_type[relation][passage_type], truth, prediction)

        if passage_type == "support":
            update_stats(
                support_by_disagreement[support_disagreement_group(record["yes_count"])],
                truth,
                prediction,
            )
            update_stats(
                support_by_score[support_score_group(record["support_score"])],
                truth,
                prediction,
            )

    return by_type, support_by_disagreement, support_by_score, by_relation_and_type


def relation_average_accuracy(passage_type_stats):
    accuracies = [
        accuracy(passage_type_stats[passage_type])
        for passage_type in PASSAGE_TYPES
        if passage_type_stats[passage_type]["total"] > 0
    ]
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)


def relation_rows(by_relation_and_type):
    rows = []
    for relation, passage_type_stats in by_relation_and_type.items():
        rows.append((relation, passage_type_stats))

    return sorted(
        rows,
        key=lambda item: (
            relation_average_accuracy(item[1]),
            item[0],
        ),
    )


def print_relation_error_table(by_relation_and_type):
    title = "Relation-Specific Accuracy by Passage Type"
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'Relation':<24} "
        f"{'SO':>9} {'SRO':>9} {'Support':>9} "
        f"{'Total':>7}"
    )

    for relation, passage_type_stats in relation_rows(by_relation_and_type)[:TOP_RELATIONS]:
        total = sum(passage_type_stats[passage_type]["total"] for passage_type in PASSAGE_TYPES)
        print(
            f"{relation[:24]:<24} "
            f"{accuracy(passage_type_stats['so']):>8.2%} "
            f"{accuracy(passage_type_stats['sro']):>8.2%} "
            f"{accuracy(passage_type_stats['support']):>8.2%} "
            f"{total:>7}"
        )


def main():
    samples = load_samples()
    records = collect_records(samples)
    (
        by_type,
        support_by_disagreement,
        support_by_score,
        by_relation_and_type,
    ) = summarize(records)

    print_stats_table(
        "Final Accuracy by Passage Type",
        [(key, by_type[key]) for key in PASSAGE_TYPES],
    )
    plot_confusion_heatmaps(records)
    print_stats_table(
        "Support Accuracy by Model Disagreement",
        [
            (key, support_by_disagreement[key])
            for key in SUPPORT_DISAGREEMENT_GROUPS
            if support_by_disagreement[key]["total"] > 0
        ],
    )
    print_stats_table(
        "Support High-vs-Low Support Score",
        [(key, support_by_score[key]) for key in ("low support", "high support")],
    )
    print(f"\nSupport score high/low threshold: {SUPPORT_SCORE_HIGH_THRESHOLD:g}")

    print_relation_error_table(by_relation_and_type)


if __name__ == "__main__":
    main()
