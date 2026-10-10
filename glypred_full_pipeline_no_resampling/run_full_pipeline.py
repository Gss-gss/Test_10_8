"""Paper-aligned Glypred pipeline adapted to the supplied balanced data.

Pipeline: 31-residue K-centered windows -> AAC/PWAA/EBGW/KMER/DR fusion ->
fold-local LightGBM split-importance selection -> BiLSTM with multi-head
attention. Resampling is intentionally disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAPER_ALPHABET = NATURAL_AA + "O"
AA_INDEX = {aa: index for index, aa in enumerate(PAPER_ALPHABET)}
PAIR_INDEX = {pair: index for index, pair in enumerate(product(PAPER_ALPHABET, repeat=2))}
WINDOW = 31
SEED = 20261009
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 8
IMPORTANCE_THRESHOLD = 2


class PaperBiLSTMAttention(nn.Module):
    """BiLSTM, LayerNorm, multi-head attention, GELU, dropout and residual FC."""

    def __init__(self, hidden_size: int = 64, dropout: float = 0.5):
        super().__init__()
        size = hidden_size * 2
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True, bidirectional=True)
        self.norm_lstm = nn.LayerNorm(size)
        self.attention = nn.MultiheadAttention(size, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(size)
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.norm_lstm(x)
        attention, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm_attention(x + attention)
        pooled = x.mean(dim=1)
        hidden = self.fc1(self.dropout(self.gelu(pooled)))
        hidden = self.gelu(hidden + pooled)
        return self.fc2(self.dropout(hidden)).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, identifier, pieces = [], None, []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().upper()
        if not line: continue
        if line.startswith(">"):
            if identifier is not None: records.append((identifier, "".join(pieces)))
            identifier, pieces = line[1:].strip(), []
        else:
            if identifier is None: raise ValueError(f"{path}: sequence before FASTA header")
            pieces.append(line)
    if identifier is not None: records.append((identifier, "".join(pieces)))
    return records


def valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    kept, removed = [], []
    for name, sequence in records:
        if len(sequence) == WINDOW and sequence[15] == "K":
            sequence = sequence.replace("X", "O")
            invalid = set(sequence) - set(PAPER_ALPHABET)
            if invalid: raise ValueError(f"{name}: unsupported residues {sorted(invalid)}")
            kept.append(sequence)
        else: removed.append(name)
    return kept, removed


def feature_blocks(sequences: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    n = len(sequences)
    aac, pwaa, ebgw = np.zeros((n, 20), np.float32), np.zeros((n, 21), np.float32), np.zeros((n, 15), np.float32)
    kmer, dr = np.zeros((n, 441), np.float32), np.zeros((n, 1344), np.float32)
    group_sets = (set("AFGILMPVWCNQSTY"), set("AFGILMPVWHKR"), set("AFGILMPVWDE"))
    for row, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence):
            residue_id = AA_INDEX[residue]
            if residue_id < 20: aac[row, residue_id] += 1
            offset = position - 15
            pwaa[row, residue_id] += offset + abs(offset) / 15
            dr[row, residue_id] += 1
        aac[row] /= 31; pwaa[row] /= 240
        for group, residues in enumerate(group_sets):
            binary = np.array([residue in residues for residue in sequence], dtype=np.float32)
            for part in range(1, 6): ebgw[row, group * 5 + part - 1] = binary[:int(np.floor(31 * part / 5))].mean()
        for start in range(30): kmer[row, PAIR_INDEX[(sequence[start], sequence[start + 1])]] += 1
        kmer[row] /= 30
        for distance in range(1, 4):
            base = 21 + (distance - 1) * 441
            for start in range(31 - distance): dr[row, base + PAIR_INDEX[(sequence[start], sequence[start + distance])]] += 1
    blocks = {"AAC": aac, "PWAA": pwaa, "EBGW": ebgw, "KMER_k2": kmer, "DR_d3": dr}
    return np.hstack(list(blocks.values())).astype(np.float32), {name: matrix.shape[1] for name, matrix in blocks.items()}


def make_loader(x, y, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))), batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def evaluate(model, batches, device, loss_fn):
    model.eval(); total, total_loss, probs, labels = 0, 0.0, [], []
    with torch.no_grad():
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device); logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb); total += len(yb)
            probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(yb.cpu().numpy())
    return total_loss / total, np.concatenate(probs), np.concatenate(labels).astype(np.int8)


def train_and_test(x_train, y_train, x_test, y_test, fold, device):
    """Select features and train only with the outer training fold."""
    selector = LGBMClassifier(objective="binary", importance_type="split", n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=15, random_state=SEED, n_jobs=-1, verbosity=-1)
    selector.fit(x_train, y_train)
    selected = selector.feature_importances_ > IMPORTANCE_THRESHOLD
    if not selected.any(): raise RuntimeError("LightGBM selected no features")
    scaler = StandardScaler().fit(x_train[:, selected])
    x_train = scaler.transform(x_train[:, selected])[..., None].astype(np.float32)
    x_test = scaler.transform(x_test[:, selected])[..., None].astype(np.float32)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + fold)
    fit_idx, valid_idx = next(split.split(x_train, y_train))
    seed_everything(SEED + fold)
    model = PaperBiLSTMAttention().to(device); loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
    fit_batches = make_loader(x_train[fit_idx], y_train[fit_idx], True, SEED + fold)
    valid_batches = make_loader(x_train[valid_idx], y_train[valid_idx], False, SEED + fold)
    best_state, best_loss, best_epoch, waiting = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in fit_batches:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        valid_loss, _, _ = evaluate(model, valid_batches, device, loss_fn)
        if valid_loss < best_loss - 1e-5:
            best_state, best_loss, best_epoch, waiting = deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            waiting += 1
            if waiting >= PATIENCE: break
    model.load_state_dict(best_state)
    _, probability, truth = evaluate(model, make_loader(x_test, y_test, False, SEED + fold), device, loss_fn)
    return selected, best_epoch, best_loss, probability, truth


def metrics(tn, fp, fn, tp):
    sn = tp / (tp + fn) if tp + fn else 0.0; sp = tn / (tn + fp) if tn + fp else 0.0; acc = (tp + tn) / (tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"SN": sn, "ACC": acc, "SP": sp, "MCC": mcc}


def family_counts(selected: np.ndarray, dims: dict[str, int]):
    counts, start = {}, 0
    for name, width in dims.items(): counts[name], start = int(selected[start:start + width].sum()), start + width
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt"); parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt"); parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args(); seed_everything(SEED); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("This full pipeline requires the prepared CUDA environment")
    pos, bad_pos = valid_windows(read_fasta(args.pos)); neg, bad_neg = valid_windows(read_fasta(args.neg)); sequences = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8); x, dims = feature_blocks(sequences); args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"device": str(device), "positive_retained": len(pos), "negative_retained": len(neg), "positive_removed": len(bad_pos), "negative_removed": len(bad_neg), "resampling": "disabled by user request; training data is near-balanced", "feature_dimensions": dims, "total_features": int(x.shape[1]), "lightgbm_selection": {"importance": "split", "threshold": "> 2", "n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 15, "scope": "outer training fold only"}, "classifier": {"architecture": "BiLSTM LayerNorm MultiheadAttention GELU Dropout residual", "learning_rate": 0.01, "weight_decay": 0.005, "batch_size": 64, "max_epochs": 50, "early_stopping_patience": 8, "validation_scope": "10 percent of outer training fold"}}
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Device={device}; samples: positive={len(pos)}, negative={len(neg)}; fused features={x.shape[1]}; resampling=disabled")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); rows = []; selections = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), 1):
        selected, epoch, valid_loss, probability, truth = train_and_test(x[train_idx], y[train_idx], x[test_idx], y[test_idx], fold, device)
        prediction = (probability >= 0.5).astype(np.int8); tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        row = {"Fold": fold, "Selected_features": int(selected.sum()), "Best_epoch": epoch, "Validation_loss": valid_loss, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), **metrics(int(tn), int(fp), int(fn), int(tp)), "AUC": float(roc_auc_score(truth, probability))}; rows.append(row); selections.append({"Fold": fold, "Selected_features": int(selected.sum()), **family_counts(selected, dims)})
        print(f"Fold {fold}: selected={selected.sum()}, epoch={epoch}, ACC={row['ACC']:.4f}, MCC={row['MCC']:.4f}, AUC={row['AUC']:.4f}")
    metric_names = ("SN", "ACC", "SP", "MCC", "AUC", "Selected_features", "Best_epoch", "Validation_loss"); mean, std = {"Fold":"mean"}, {"Fold":"std"}
    for metric in metric_names:
        values = np.array([float(row[metric]) for row in rows]); mean[metric], std[metric] = float(values.mean()), float(values.std(ddof=1))
    with (args.output_dir / "full_pipeline_5fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Fold", "Selected_features", "Best_epoch", "Validation_loss", "TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC"]); writer.writeheader(); writer.writerows(rows); writer.writerow(mean); writer.writerow(std)
    with (args.output_dir / "selected_feature_families.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Fold", "Selected_features", *dims.keys()]); writer.writeheader(); writer.writerows(selections)
    print("\nFive-fold mean +/- sample standard deviation")
    for metric in metric_names: print(f"{metric}: {mean[metric]:.4f} +/- {std[metric]:.4f}")


if __name__ == "__main__": main()
