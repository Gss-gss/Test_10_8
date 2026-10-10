"""Fold-safe MCC optimisation of fused-feature LightGBM.

Each outer training fold runs three inner folds. Their out-of-fold predictions
choose both model parameters and a fixed decision threshold. The outer test
fold is used once for evaluation. No resampling is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from feature_extractors import WINDOW_LENGTH, fused_features

BASE_DIR = Path(__file__).resolve().parent
SEED = 20261009
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.025), 3)
BASELINE = {"name": "previous_best", "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 30,
            "n_estimators": 400, "max_depth": -1, "reg_lambda": 0.0, "colsample_bytree": 1.0}
CANDIDATES = (
    BASELINE,
    {**BASELINE, "name": "smaller_tree", "num_leaves": 7, "min_child_samples": 50},
    {**BASELINE, "name": "larger_tree", "num_leaves": 31},
    {**BASELINE, "name": "faster_learning", "learning_rate": 0.05},
    {**BASELINE, "name": "slower_learning", "learning_rate": 0.01, "n_estimators": 600},
    {**BASELINE, "name": "regularized", "reg_lambda": 1.0, "colsample_bytree": 0.8},
)
METRICS = ("SN", "ACC", "SP", "MCC", "AUC")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier, pieces = None, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().upper()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(pieces)))
            identifier, pieces = line[1:].strip(), []
        else:
            if identifier is None:
                raise ValueError(f"{path}: sequence before FASTA header")
            pieces.append(line)
    if identifier is not None:
        records.append((identifier, "".join(pieces)))
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    kept, removed = [], []
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[WINDOW_LENGTH // 2] == "K":
            kept.append(sequence)
        else:
            removed.append(identifier)
    return kept, removed


def select_and_scale(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                     seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    selector = LGBMClassifier(objective="binary", importance_type="split", n_estimators=500,
                              learning_rate=0.05, max_depth=5, num_leaves=15,
                              random_state=seed, n_jobs=-1, verbosity=-1)
    selector.fit(x_train, y_train)
    selected = selector.feature_importances_ > 2
    if not np.any(selected):
        raise RuntimeError("No features selected")
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train[:, selected]).astype(np.float32)
    eval_ = scaler.transform(x_eval[:, selected]).astype(np.float32)
    return train, eval_, int(selected.sum())


def model_probability(config: dict, x_train: np.ndarray, y_train: np.ndarray,
                      x_eval: np.ndarray, seed: int) -> np.ndarray:
    params = {key: value for key, value in config.items() if key != "name"}
    model = LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, verbosity=-1, **params)
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    scored = [(float(matthews_corrcoef(y, probabilities >= threshold)), float(threshold)) for threshold in THRESHOLDS]
    best_mcc, threshold = max(scored, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))
    return threshold, best_mcc


def measure(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    tn, fp, fn, tp = map(int, confusion_matrix(y, probabilities >= threshold, labels=[0, 1]).ravel())
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "SN": tp / (tp + fn), "ACC": (tp + tn) / len(y), "SP": tn / (tn + fp),
            "MCC": float(matthews_corrcoef(y, probabilities >= threshold)),
            "AUC": float(roc_auc_score(y, probabilities))}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_summary(rows: list[dict]) -> list[dict]:
    output = list(rows)
    for label, function in (("mean", np.mean), ("std", lambda values: np.std(values, ddof=1))):
        summary = {"Fold": label}
        for metric in METRICS:
            summary[metric] = float(function([row[metric] for row in rows])) if label == "mean" or len(rows) > 1 else 0.0
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos, dropped_pos = valid_windows(read_fasta(args.pos))
    neg, dropped_neg = valid_windows(read_fasta(args.neg))
    x, dimensions = fused_features(pos + neg)
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    audit = {"positive": len(pos), "negative": len(neg), "removed_positive": dropped_pos,
             "removed_negative": dropped_neg, "sequence_length": WINDOW_LENGTH,
             "feature_dimensions": dimensions, "feature_total": x.shape[1], "resampling": "none",
             "validation": "outer 5-fold, inner 3-fold, stratified", "selection": "LightGBM split importance >2, fold-local",
             "threshold_grid": THRESHOLDS.tolist(), "candidate_models": CANDIDATES,
             "model_selection_metric": "pooled inner out-of-fold MCC"}
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Samples: +{len(pos)} / -{len(neg)}; fused features={x.shape[1]}; resampling=none", flush=True)

    optimized_rows, baseline_rows, threshold_only_rows, inner_rows = [], [], [], []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (outer_train, outer_test) in enumerate(outer.split(x, y), start=1):
        print(f"Outer fold {fold}/5", flush=True)
        oof = {candidate["name"]: np.full(len(outer_train), np.nan) for candidate in CANDIDATES}
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        for inner_fold, (relative_train, relative_val) in enumerate(inner.split(x[outer_train], y[outer_train]), start=1):
            train_idx, val_idx = outer_train[relative_train], outer_train[relative_val]
            train_x, val_x, feature_count = select_and_scale(x[train_idx], y[train_idx], x[val_idx], SEED + fold * 100 + inner_fold)
            for candidate in CANDIDATES:
                oof[candidate["name"]][relative_val] = model_probability(
                    candidate, train_x, y[train_idx], val_x, SEED + fold * 1000 + inner_fold)
            print(f"  inner {inner_fold}/3; selected features={feature_count}", flush=True)
        rankings = []
        for candidate in CANDIDATES:
            name, probabilities = candidate["name"], oof[candidate["name"]]
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Missing out-of-fold predictions for {name}")
            threshold, inner_mcc = choose_threshold(y[outer_train], probabilities)
            inner_auc = float(roc_auc_score(y[outer_train], probabilities))
            inner_rows.append({"Fold": fold, "Candidate": name, "Threshold": threshold,
                               "Inner_MCC": inner_mcc, "Inner_AUC": inner_auc,
                               "MCC_at_0.5": float(matthews_corrcoef(y[outer_train], probabilities >= 0.5))})
            rankings.append((inner_mcc, inner_auc, name, threshold, candidate))
        _, _, chosen_name, threshold, chosen = max(rankings, key=lambda entry: (entry[0], entry[1]))
        train_x, test_x, feature_count = select_and_scale(x[outer_train], y[outer_train], x[outer_test], SEED + fold * 10000)
        selected_probability = model_probability(chosen, train_x, y[outer_train], test_x, SEED + fold * 100000)
        optimized_rows.append({"Fold": fold, "Candidate": chosen_name, "Threshold": threshold,
                               "Selected_features": feature_count, **measure(y[outer_test], selected_probability, threshold)})
        if chosen_name == BASELINE["name"]:
            baseline_probability = selected_probability
        else:
            baseline_probability = model_probability(BASELINE, train_x, y[outer_train], test_x, SEED + fold * 100000)
        baseline_rows.append({"Fold": fold, "Candidate": BASELINE["name"], "Threshold": 0.5,
                              "Selected_features": feature_count, **measure(y[outer_test], baseline_probability, 0.5)})
        baseline_threshold = next(row["Threshold"] for row in reversed(inner_rows)
                                  if row["Fold"] == fold and row["Candidate"] == BASELINE["name"])
        threshold_only_rows.append({"Fold": fold, "Candidate": BASELINE["name"], "Threshold": baseline_threshold,
                                    "Selected_features": feature_count,
                                    **measure(y[outer_test], baseline_probability, baseline_threshold)})
        print(f"  selected={chosen_name}, threshold={threshold:.3f}, test MCC={optimized_rows[-1]['MCC']:.4f}, baseline MCC={baseline_rows[-1]['MCC']:.4f}", flush=True)
        result_fields = ["Fold", "Candidate", "Threshold", "Selected_features", "TN", "FP", "FN", "TP", *METRICS]
        write_csv(args.output_dir / "optimized_5fold.csv", add_summary(optimized_rows), result_fields)
        write_csv(args.output_dir / "matched_baseline_5fold.csv", add_summary(baseline_rows), result_fields)
        write_csv(args.output_dir / "threshold_only_5fold.csv", add_summary(threshold_only_rows), result_fields)
        write_csv(args.output_dir / "inner_candidate_scores.csv", inner_rows,
                  ["Fold", "Candidate", "Threshold", "Inner_MCC", "Inner_AUC", "MCC_at_0.5"])
    print("Final mean MCC: optimized={:.4f}, baseline={:.4f}".format(
        np.mean([row["MCC"] for row in optimized_rows]), np.mean([row["MCC"] for row in baseline_rows])), flush=True)


if __name__ == "__main__":
    main()
