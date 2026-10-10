"""Nested-CV comparison of LightGBM, RBF-SVM and GPU MLP on fused Glypred features.

No sample resampling is used. Feature selection, scaling and model selection are
fit only on the appropriate training partition to keep all outer test folds unseen.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_extractors import WINDOW_LENGTH, fused_features

BASE_DIR = Path(__file__).resolve().parent
SEED = 20261009
OUTER_SPLITS, INNER_SPLITS = 5, 3
IMPORTANCE_THRESHOLD = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Candidate:
    name: str
    tag: str
    params: dict


CANDIDATES = (
    Candidate("LightGBM", "lr0.02_leaves15", {"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 30}),
    Candidate("LightGBM", "lr0.05_leaves31", {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20}),
    Candidate("LightGBM", "lr0.10_leaves15", {"learning_rate": 0.10, "num_leaves": 15, "min_child_samples": 20}),
    Candidate("RBF-SVM", "C0.3", {"C": 0.3}),
    Candidate("RBF-SVM", "C1", {"C": 1.0}),
    Candidate("RBF-SVM", "C3", {"C": 3.0}),
    Candidate("MLP", "h128_d0.2_lr1e-3", {"hidden": 128, "dropout": 0.2, "lr": 0.001}),
    Candidate("MLP", "h256_d0.2_lr5e-4", {"hidden": 256, "dropout": 0.2, "lr": 0.0005}),
    Candidate("MLP", "h256_d0.4_lr5e-4", {"hidden": 256, "dropout": 0.4, "lr": 0.0005}),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return records


def valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    kept, removed, center = [], [], WINDOW_LENGTH // 2
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            kept.append(sequence)
        else:
            removed.append(identifier)
    return kept, removed


def select_features(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    selector = LGBMClassifier(objective="binary", importance_type="split", n_estimators=500,
        learning_rate=0.05, max_depth=5, num_leaves=15, random_state=seed, n_jobs=-1, verbosity=-1)
    selector.fit(x_train, y_train)
    mask = selector.feature_importances_ > IMPORTANCE_THRESHOLD
    if not mask.any():
        raise RuntimeError("No feature exceeded the LightGBM split-importance threshold.")
    return mask


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, candidate: Candidate, seed: int) -> np.ndarray:
    set_seed(seed)
    model = MLP(x_train.shape[1], candidate.params["hidden"], candidate.params["dropout"]).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))), batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.params["lr"], weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(30):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_eval).to(DEVICE))).cpu().numpy()


def predict(candidate: Candidate, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> np.ndarray:
    if candidate.name == "LightGBM":
        classifier = LGBMClassifier(objective="binary", n_estimators=400, max_depth=-1, random_state=seed, n_jobs=-1, verbosity=-1, **candidate.params)
        classifier.fit(x_train, y_train)
        return classifier.predict_proba(x_eval)[:, 1]
    if candidate.name == "RBF-SVM":
        classifier = SVC(kernel="rbf", gamma="scale", probability=False, random_state=seed, **candidate.params)
        classifier.fit(x_train, y_train)
        # AUC only needs score ordering.  The sigmoid maps the natural SVM
        # decision threshold of zero to 0.5 without expensive probability calibration.
        decision = np.clip(classifier.decision_function(x_eval), -30, 30)
        return 1.0 / (1.0 + np.exp(-decision))
    return train_mlp(x_train, y_train, x_eval, candidate, seed)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / len(y_true)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "SN": sn, "ACC": acc, "SP": sp,
            "MCC": (tp * tn - fp * fn) / denom if denom else 0.0, "AUC": float(roc_auc_score(y_true, probability))}


def prepare(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    selected = select_features(train_x, train_y, seed)
    scaler = StandardScaler()
    return scaler.fit_transform(train_x[:, selected]).astype(np.float32), scaler.transform(eval_x[:, selected]).astype(np.float32), int(selected.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()
    set_seed(SEED); args.output_dir.mkdir(parents=True, exist_ok=True)
    pos, pos_removed = valid_windows(read_fasta(args.pos)); neg, neg_removed = valid_windows(read_fasta(args.neg))
    sequences = pos + neg; y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    x, dimensions = fused_features(sequences)
    audit = {"device": str(DEVICE), "resampling": "disabled: retained classes are nearly balanced", "positive_retained": len(pos), "negative_retained": len(neg), "positive_removed": len(pos_removed), "negative_removed": len(neg_removed), "retained_lengths": dict(Counter(map(len, sequences))), "features": dimensions, "total_features": int(x.shape[1]), "validation": "nested StratifiedKFold: outer=5, inner=3", "selection": "LightGBM split importance > 2 fit inside each inner/final training partition", "candidates": [{"model": c.name, "setting": c.tag, **c.params} for c in CANDIDATES]}
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Device={DEVICE}; samples pos={len(pos)} neg={len(neg)}; features={x.shape[1]}; resampling=disabled", flush=True)

    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    detail_rows, fold_rows = [], []
    for fold, (outer_train, outer_test) in enumerate(outer.split(x, y), start=1):
        print(f"\nOuter fold {fold}/{OUTER_SPLITS}: inner model selection", flush=True)
        scores: dict[str, list[float]] = {c.tag: [] for c in CANDIDATES}
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold)
        for inner_fold, (relative_train, relative_val) in enumerate(inner.split(x[outer_train], y[outer_train]), start=1):
            train_idx, val_idx = outer_train[relative_train], outer_train[relative_val]
            inner_train_x, inner_val_x, selected_count = prepare(x[train_idx], y[train_idx], x[val_idx], SEED + fold * 100 + inner_fold)
            for candidate in CANDIDATES:
                probability = predict(candidate, inner_train_x, y[train_idx], inner_val_x, SEED + fold * 1000 + inner_fold)
                auc = float(roc_auc_score(y[val_idx], probability)); scores[candidate.tag].append(auc)
                detail_rows.append({"Outer_fold": fold, "Inner_fold": inner_fold, "Model": candidate.name, "Setting": candidate.tag, "Validation_AUC": auc, "Selected_features": selected_count})
        ranked = sorted(CANDIDATES, key=lambda c: (float(np.mean(scores[c.tag])), c.name == "RBF-SVM"), reverse=True)
        best = ranked[0]
        train_x, test_x, selected_count = prepare(x[outer_train], y[outer_train], x[outer_test], SEED + fold * 10000)
        probability = predict(best, train_x, y[outer_train], test_x, SEED + fold * 100000)
        row = {"Fold": fold, "Selected_model": best.name, "Selected_setting": best.tag, "Inner_mean_AUC": float(np.mean(scores[best.tag])), "Selected_features": selected_count, **metrics(y[outer_test], probability)}
        fold_rows.append(row)
        print(f"Fold {fold}: chose {best.name} ({best.tag}), inner AUC={row['Inner_mean_AUC']:.4f}, test AUC={row['AUC']:.4f}, MCC={row['MCC']:.4f}", flush=True)

    metric_names = ("SN", "ACC", "SP", "MCC", "AUC", "Inner_mean_AUC", "Selected_features")
    mean_row, std_row = {"Fold": "mean"}, {"Fold": "std"}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in fold_rows])
        mean_row[name], std_row[name] = float(values.mean()), float(values.std(ddof=1))
    fieldnames = ["Fold", "Selected_model", "Selected_setting", "Inner_mean_AUC", "Selected_features", "TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC"]
    with (args.output_dir / "nested_model_comparison_5fold.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(fold_rows); writer.writerow(mean_row); writer.writerow(std_row)
    with (args.output_dir / "inner_candidate_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Outer_fold", "Inner_fold", "Model", "Setting", "Validation_AUC", "Selected_features"]); writer.writeheader(); writer.writerows(detail_rows)
    print("\nOuter five-fold mean +/- standard deviation", flush=True)
    for name in metric_names:
        print(f"{name}: {mean_row[name]:.4f} +/- {std_row[name]:.4f}", flush=True)


if __name__ == "__main__":
    main()
