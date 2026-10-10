"""Nested comparison of classifiers on 31-residue one-hot sequence features.

The data are nearly balanced after malformed-window filtering, so no resampling
is performed.  Every outer test fold stays completely unseen during selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_INDEX = {residue: index for index, residue in enumerate(ALPHABET)}
WINDOW_LENGTH, SEED = 31, 20261009
OUTER_SPLITS, INNER_SPLITS = 5, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Candidate:
    model: str
    setting: str
    params: tuple[tuple[str, float | int], ...]

    def values(self) -> dict[str, float | int]:
        return dict(self.params)

    @property
    def key(self) -> str:
        return f"{self.model}::{self.setting}"


CANDIDATES = (
    Candidate("LogisticRegression", "C0.1", (("C", 0.1),)),
    Candidate("LogisticRegression", "C1", (("C", 1.0),)),
    Candidate("LogisticRegression", "C10", (("C", 10.0),)),
    Candidate("LightGBM", "lr0.02_leaves15", (("learning_rate", 0.02), ("num_leaves", 15), ("min_child_samples", 30))),
    Candidate("LightGBM", "lr0.05_leaves31", (("learning_rate", 0.05), ("num_leaves", 31), ("min_child_samples", 20))),
    Candidate("LightGBM", "lr0.10_leaves15", (("learning_rate", 0.10), ("num_leaves", 15), ("min_child_samples", 20))),
    Candidate("RBF-SVM", "C0.3", (("C", 0.3),)),
    Candidate("RBF-SVM", "C1", (("C", 1.0),)),
    Candidate("RBF-SVM", "C3", (("C", 3.0),)),
    Candidate("MLP", "h128_d0.2_lr1e-3", (("hidden", 128), ("dropout", 0.2), ("lr", 0.001))),
    Candidate("MLP", "h256_d0.2_lr5e-4", (("hidden", 256), ("dropout", 0.2), ("lr", 0.0005))),
    Candidate("MLP", "h256_d0.4_lr5e-4", (("hidden", 256), ("dropout", 0.4), ("lr", 0.0005))),
)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, identifier, pieces = [], None, []
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
    kept, removed, center = [], [], WINDOW_LENGTH // 2
    for identifier, sequence in records:
        invalid = set(sequence) - set(ALPHABET)
        if invalid:
            raise ValueError(f"{identifier}: unsupported residues {sorted(invalid)}")
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            kept.append(sequence)
        else:
            removed.append(identifier)
    return kept, removed


def one_hot(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), WINDOW_LENGTH, len(ALPHABET)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence):
            out[row, position, AA_INDEX[residue]] = 1.0
    return out.reshape(len(sequences), -1)


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def mlp_probability(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, params: dict[str, float | int], seed: int) -> np.ndarray:
    seed_everything(seed)
    model = MLP(x_train.shape[1], int(params["hidden"]), float(params["dropout"])).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))), batch_size=256, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(30):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y); loss.backward(); optimiser.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_eval).to(DEVICE))).cpu().numpy()


def probability(candidate: Candidate, raw_train: np.ndarray, scaled_train: np.ndarray, y_train: np.ndarray, raw_eval: np.ndarray, scaled_eval: np.ndarray, seed: int) -> np.ndarray:
    params = candidate.values()
    if candidate.model == "LogisticRegression":
        classifier = LogisticRegression(solver="liblinear", max_iter=3000, random_state=seed, **params)
        classifier.fit(scaled_train, y_train); return classifier.predict_proba(scaled_eval)[:, 1]
    if candidate.model == "LightGBM":
        classifier = LGBMClassifier(objective="binary", n_estimators=400, max_depth=-1, random_state=seed, n_jobs=-1, verbosity=-1, **params)
        classifier.fit(raw_train, y_train); return classifier.predict_proba(raw_eval)[:, 1]
    if candidate.model == "RBF-SVM":
        classifier = SVC(kernel="rbf", gamma="scale", probability=False, random_state=seed, **params)
        classifier.fit(scaled_train, y_train)
        score = np.clip(classifier.decision_function(scaled_eval), -30, 30)
        return 1.0 / (1.0 + np.exp(-score))
    return mlp_probability(scaled_train, y_train, scaled_eval, params, seed)


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    prediction = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    sn, sp = tp / (tp + fn), tn / (tn + fp)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "SN": sn, "ACC": (tp + tn) / len(y_true), "SP": sp, "MCC": (tp * tn - fp * fn) / denom if denom else 0.0, "AUC": float(roc_auc_score(y_true, probabilities))}


def split_and_scale(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train_x).astype(np.float32), scaler.transform(eval_x).astype(np.float32)


def summary_rows(rows: list[dict[str, object]], label_key: str) -> list[dict[str, object]]:
    metrics = ("SN", "ACC", "SP", "MCC", "AUC", "Inner_mean_AUC")
    output: list[dict[str, object]] = []
    for label in sorted({str(row[label_key]) for row in rows}):
        subset = [row for row in rows if str(row[label_key]) == label]
        result: dict[str, object] = {label_key: label, "Folds": len(subset)}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in subset])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args(); seed_everything(SEED); args.output_dir.mkdir(parents=True, exist_ok=True)
    pos, dropped_pos = valid_windows(read_fasta(args.pos)); neg, dropped_neg = valid_windows(read_fasta(args.neg))
    sequences = pos + neg; y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8); x = one_hot(sequences)
    audit = {"device": str(DEVICE), "resampling": "disabled: retained classes are nearly balanced", "positive_retained": len(pos), "negative_retained": len(neg), "positive_removed": len(dropped_pos), "negative_removed": len(dropped_neg), "retained_lengths": dict(Counter(map(len, sequences))), "encoding": f"one-hot: {WINDOW_LENGTH} positions x {len(ALPHABET)} residues = {x.shape[1]} features", "validation": "nested StratifiedKFold: outer=5, inner=3", "candidate_count": len(CANDIDATES), "candidates": [{"model": c.model, "setting": c.setting, **c.values()} for c in CANDIDATES]}
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Device={DEVICE}; one-hot shape={x.shape}; pos={len(pos)} neg={len(neg)}; resampling=disabled", flush=True)

    by_model: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in CANDIDATES:
        by_model[candidate.model].append(candidate)
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    inner_rows, outer_model_rows, winner_rows = [], [], []
    for fold, (outer_train, outer_test) in enumerate(outer.split(x, y), start=1):
        print(f"\nOuter fold {fold}/{OUTER_SPLITS}: inner comparison", flush=True)
        candidate_scores: dict[str, list[float]] = {candidate.key: [] for candidate in CANDIDATES}
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold)
        for inner_fold, (rel_train, rel_val) in enumerate(inner.split(x[outer_train], y[outer_train]), start=1):
            train_idx, val_idx = outer_train[rel_train], outer_train[rel_val]
            scaled_train, scaled_val = split_and_scale(x[train_idx], x[val_idx])
            for candidate in CANDIDATES:
                probs = probability(candidate, x[train_idx], scaled_train, y[train_idx], x[val_idx], scaled_val, SEED + fold * 1000 + inner_fold)
                auc = float(roc_auc_score(y[val_idx], probs)); candidate_scores[candidate.key].append(auc)
                inner_rows.append({"Outer_fold": fold, "Inner_fold": inner_fold, "Model": candidate.model, "Setting": candidate.setting, "Validation_AUC": auc})
        scaled_outer_train, scaled_outer_test = split_and_scale(x[outer_train], x[outer_test])
        fold_model_rows = []
        for model, candidates in by_model.items():
            selected = max(candidates, key=lambda item: float(np.mean(candidate_scores[item.key])))
            probs = probability(selected, x[outer_train], scaled_outer_train, y[outer_train], x[outer_test], scaled_outer_test, SEED + fold * 100000)
            row = {"Fold": fold, "Model": model, "Selected_setting": selected.setting, "Inner_mean_AUC": float(np.mean(candidate_scores[selected.key])), **evaluate(y[outer_test], probs)}
            outer_model_rows.append(row); fold_model_rows.append(row)
            print(f"  {model}: {selected.setting}, inner AUC={row['Inner_mean_AUC']:.4f}, test AUC={row['AUC']:.4f}", flush=True)
        winner = max(fold_model_rows, key=lambda item: float(item["Inner_mean_AUC"]))
        winner_rows.append({"Fold": fold, "Selected_model": winner["Model"], "Selected_setting": winner["Selected_setting"], "Inner_mean_AUC": winner["Inner_mean_AUC"], **{key: winner[key] for key in ("TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC")}})
        print(f"Fold {fold} winner: {winner['Model']} ({winner['Selected_setting']})", flush=True)

    metrics = ("SN", "ACC", "SP", "MCC", "AUC", "Inner_mean_AUC")
    summary = summary_rows(outer_model_rows, "Model")
    winner_mean, winner_std = {"Fold": "mean"}, {"Fold": "std"}
    for metric in metrics:
        values = np.array([float(row[metric]) for row in winner_rows]); winner_mean[metric], winner_std[metric] = float(values.mean()), float(values.std(ddof=1))
    with (args.output_dir / "inner_candidate_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Outer_fold", "Inner_fold", "Model", "Setting", "Validation_AUC"]); writer.writeheader(); writer.writerows(inner_rows)
    outer_fields = ["Fold", "Model", "Selected_setting", "Inner_mean_AUC", "TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC"]
    with (args.output_dir / "outer_classifier_comparison_5fold.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=outer_fields); writer.writeheader(); writer.writerows(outer_model_rows)
    summary_fields = ["Model", "Folds", *[f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "std")]]
    with (args.output_dir / "outer_classifier_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary)
    winner_fields = ["Fold", "Selected_model", "Selected_setting", "Inner_mean_AUC", "TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC"]
    with (args.output_dir / "nested_winner_5fold.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=winner_fields); writer.writeheader(); writer.writerows(winner_rows); writer.writerow(winner_mean); writer.writerow(winner_std)
    print("\nNested winner mean +/- standard deviation", flush=True)
    for metric in metrics:
        print(f"{metric}: {winner_mean[metric]:.4f} +/- {winner_std[metric]:.4f}", flush=True)


if __name__ == "__main__":
    main()
