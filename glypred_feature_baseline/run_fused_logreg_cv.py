"""Run the five-feature Glypred baseline with logistic regression.

The script deliberately keeps the classifier and cross-validation protocol used
in the prior experiment. It changes only the representation from position-wise
one-hot to the five fused feature families described in Glypred.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractors import WINDOW_LENGTH, fused_features


RANDOM_SEED = 20261009
N_SPLITS = 5
BASE_DIR = Path(__file__).resolve().parent


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read FASTA records as (identifier, sequence) tuples."""
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().upper()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(sequence_parts)))
            identifier = line[1:].strip()
            sequence_parts = []
        else:
            if identifier is None:
                raise ValueError(f"{path}: sequence before FASTA header")
            sequence_parts.append(line)
    if identifier is not None:
        records.append((identifier, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def retain_valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep only 31-residue windows whose central residue is lysine."""
    valid: list[str] = []
    removed: list[tuple[str, str]] = []
    center = WINDOW_LENGTH // 2
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            valid.append(sequence)
        else:
            removed.append((identifier, sequence))
    return valid, removed


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"SN": sensitivity, "ACC": accuracy, "SP": specificity, "MCC": mcc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()

    pos, removed_pos = retain_valid_windows(read_fasta(args.pos))
    neg, removed_neg = retain_valid_windows(read_fasta(args.neg))
    sequences = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    x, feature_dimensions = fused_features(sequences)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "positive_retained": len(pos),
        "negative_retained": len(neg),
        "positive_removed": len(removed_pos),
        "negative_removed": len(removed_neg),
        "retained_lengths": dict(Counter(map(len, sequences))),
        "center_residue": "K",
        "feature_dimensions": feature_dimensions,
        "total_feature_dimension": int(x.shape[1]),
        "cross_validation": {"folds": N_SPLITS, "shuffle": True, "random_seed": RANDOM_SEED},
        "classifier": {"name": "LogisticRegression", "solver": "liblinear", "max_iter": 3000, "C": 1.0},
    }
    (args.output_dir / "feature_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Retained samples: positive={len(pos)}, negative={len(neg)}")
    print(f"Removed invalid windows: positive={len(removed_pos)}, negative={len(removed_neg)}")
    print(f"Feature dimensions: {feature_dimensions}; total={x.shape[1]}")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows: list[dict[str, int | float]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, labels), start=1):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)),
            ]
        )
        model.fit(x[train_index], labels[train_index])
        prediction = model.predict(x[test_index])
        probability = model.predict_proba(x[test_index])[:, 1]
        tn, fp, fn, tp = confusion_matrix(labels[test_index], prediction, labels=[0, 1]).ravel()
        row: dict[str, int | float] = {
            "Fold": fold,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            **metrics_from_confusion(int(tn), int(fp), int(fn), int(tp)),
            "AUC": float(roc_auc_score(labels[test_index], probability)),
        }
        rows.append(row)
        print(
            f"Fold {fold}: SN={row['SN']:.4f}, ACC={row['ACC']:.4f}, "
            f"SP={row['SP']:.4f}, MCC={row['MCC']:.4f}, AUC={row['AUC']:.4f}"
        )

    metrics = ("SN", "ACC", "SP", "MCC", "AUC")
    mean_row: dict[str, str | float] = {"Fold": "mean"}
    std_row: dict[str, str | float] = {"Fold": "std"}
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows])
        mean_row[metric] = float(values.mean())
        std_row[metric] = float(values.std(ddof=1))

    output_csv = args.output_dir / "fused_feature_logreg_5fold.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Fold", "TN", "FP", "FN", "TP", *metrics])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print("\nFive-fold mean +/- sample standard deviation")
    for metric in metrics:
        print(f"{metric}: {mean_row[metric]:.4f} +/- {std_row[metric]:.4f}")
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
