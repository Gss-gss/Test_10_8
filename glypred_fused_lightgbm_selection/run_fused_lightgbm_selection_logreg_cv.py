"""Five-feature fusion + fold-local LightGBM selection + logistic regression."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractors import WINDOW_LENGTH, fused_features


BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20261009
N_SPLITS = 5
IMPORTANCE_THRESHOLD = 2


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, identifier, chunks = [], None, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().upper()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks)))
            identifier, chunks = line[1:].strip(), []
        else:
            if identifier is None:
                raise ValueError(f"{path}: sequence before FASTA header")
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, "".join(chunks)))
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    center = WINDOW_LENGTH // 2
    kept, removed = [], []
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            kept.append(sequence)
        else:
            removed.append(identifier)
    return kept, removed


def metric_values(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"SN": sn, "ACC": acc, "SP": sp, "MCC": mcc}


def feature_family_counts(selected: np.ndarray, dimensions: dict[str, int]) -> dict[str, int]:
    counts, start = {}, 0
    for name, width in dimensions.items():
        counts[name] = int(selected[start : start + width].sum())
        start += width
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()

    pos, dropped_pos = valid_windows(read_fasta(args.pos))
    neg, dropped_neg = valid_windows(read_fasta(args.neg))
    sequences = pos + neg
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    x, dimensions = fused_features(sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        "positive_retained": len(pos), "negative_retained": len(neg),
        "positive_removed": len(dropped_pos), "negative_removed": len(dropped_neg),
        "retained_lengths": dict(Counter(map(len, sequences))), "center_residue": "K",
        "feature_dimensions": dimensions, "total_feature_dimension": int(x.shape[1]),
        "selector": {
            "model": "LGBMClassifier", "importance": "split", "threshold": f"> {IMPORTANCE_THRESHOLD}",
            "n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 15,
            "selection_scope": "training fold only",
        },
        "classifier": {"model": "LogisticRegression", "solver": "liblinear", "max_iter": 3000},
    }
    (args.output_dir / "feature_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Retained samples: positive={len(pos)}, negative={len(neg)}; total features={x.shape[1]}")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows, selection_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, labels), start=1):
        selector = LGBMClassifier(
            objective="binary", importance_type="split", n_estimators=500, learning_rate=0.05,
            max_depth=5, num_leaves=15, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
        )
        selector.fit(x[train_idx], labels[train_idx])
        selected = selector.feature_importances_ > IMPORTANCE_THRESHOLD
        if not selected.any():
            raise RuntimeError("No features exceeded the LightGBM importance threshold")
        selected_x_train, selected_x_test = x[train_idx][:, selected], x[test_idx][:, selected]
        classifier = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)),
        ])
        classifier.fit(selected_x_train, labels[train_idx])
        prediction = classifier.predict(selected_x_test)
        probability = classifier.predict_proba(selected_x_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(labels[test_idx], prediction, labels=[0, 1]).ravel()
        row = {
            "Fold": fold, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            **metric_values(int(tn), int(fp), int(fn), int(tp)),
            "AUC": float(roc_auc_score(labels[test_idx], probability)),
            "Selected_features": int(selected.sum()),
        }
        rows.append(row)
        selection_rows.append({"Fold": fold, "Selected_features": int(selected.sum()), **feature_family_counts(selected, dimensions)})
        print(f"Fold {fold}: selected={selected.sum()}, ACC={row['ACC']:.4f}, MCC={row['MCC']:.4f}, AUC={row['AUC']:.4f}")

    metrics = ("SN", "ACC", "SP", "MCC", "AUC", "Selected_features")
    mean_row, std_row = {"Fold": "mean"}, {"Fold": "std"}
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows])
        mean_row[metric], std_row[metric] = float(values.mean()), float(values.std(ddof=1))

    with (args.output_dir / "fused_lightgbm_selected_logreg_5fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Fold", "TN", "FP", "FN", "TP", *metrics])
        writer.writeheader(); writer.writerows(rows); writer.writerow(mean_row); writer.writerow(std_row)
    with (args.output_dir / "selected_feature_families_by_fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Fold", "Selected_features", *dimensions.keys()])
        writer.writeheader(); writer.writerows(selection_rows)

    print("\nFive-fold mean +/- sample standard deviation")
    for metric in metrics:
        print(f"{metric}: {mean_row[metric]:.4f} +/- {std_row[metric]:.4f}")


if __name__ == "__main__":
    main()
