"""Five-fold evaluation of 31-residue one-hot windows with a Glypred-style classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import GlypredBiLSTMAttention


BASE_DIR = Path(__file__).resolve().parent
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_INDEX = {residue: index for index, residue in enumerate(ALPHABET)}
WINDOW_LENGTH = 31
RANDOM_SEED = 20261009
N_SPLITS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.005
MAX_EPOCHS = 50
PATIENCE = 8
CPU_THREADS = min(4, os.cpu_count() or 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small LSTM batches are substantially slower when every CPU core is used.
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)


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


def keep_valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    center = WINDOW_LENGTH // 2
    kept, dropped = [], []
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            invalid = set(sequence) - set(ALPHABET)
            if invalid:
                raise ValueError(f"{identifier}: unsupported residues {sorted(invalid)}")
            kept.append(sequence)
        else:
            dropped.append(identifier)
    return kept, dropped


def one_hot(sequences: list[str]) -> np.ndarray:
    encoded = np.zeros((len(sequences), WINDOW_LENGTH, len(ALPHABET)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence):
            encoded[row, position, AA_INDEX[residue]] = 1.0
    return encoded


def metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"SN": sn, "ACC": acc, "SP": sp, "MCC": mcc}


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total_count, probabilities, labels = 0.0, 0, [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y_batch.cpu().numpy())
    return total_loss / total_count, np.concatenate(probabilities), np.concatenate(labels).astype(np.int8)


def train_fold(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, fold: int, device: torch.device):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED + fold)
    fit_idx, valid_idx = next(splitter.split(x_train, y_train))
    model = GlypredBiLSTMAttention().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_loader = make_loader(x_train[fit_idx], y_train[fit_idx], BATCH_SIZE, True, RANDOM_SEED + fold)
    valid_loader = make_loader(x_train[valid_idx], y_train[valid_idx], BATCH_SIZE, False, RANDOM_SEED + fold)
    test_loader = make_loader(x_test, y_test, BATCH_SIZE, False, RANDOM_SEED + fold)
    best_state, best_loss, wait, best_epoch = None, float("inf"), 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_batch, y_batch in fit_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        validation_loss, _, _ = evaluate(model, valid_loader, device, criterion)
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, wait, best_epoch = validation_loss, deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    _, probability, truth = evaluate(model, test_loader, device, criterion)
    return probability, truth, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()
    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos, dropped_pos = keep_valid_windows(read_fasta(args.pos))
    neg, dropped_neg = keep_valid_windows(read_fasta(args.neg))
    sequences = pos + neg
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    x = one_hot(sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "positive_retained": len(pos), "negative_retained": len(neg),
        "positive_removed": len(dropped_pos), "negative_removed": len(dropped_neg),
        "retained_lengths": dict(Counter(map(len, sequences))), "center_residue": "K",
        "input_shape": [WINDOW_LENGTH, len(ALPHABET)], "alphabet": ALPHABET,
        "device": str(device),
        "classifier": {
            "architecture": "BiLSTM + LayerNorm + MultiheadAttention + GELU + Dropout + residual FC",
            "hidden_size": 64, "attention_heads": 4, "dropout": 0.5,
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
            "cpu_threads": CPU_THREADS,
            "validation_scope": "10 percent of each training fold",
        },
    }
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Device: {device}; retained positive={len(pos)}, negative={len(neg)}; input={x.shape}")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, labels), start=1):
        probability, truth, epoch, validation_loss = train_fold(x[train_idx], labels[train_idx], x[test_idx], labels[test_idx], fold, device)
        prediction = (probability >= 0.5).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        row = {
            "Fold": fold, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            **metrics(int(tn), int(fp), int(fn), int(tp)), "AUC": float(roc_auc_score(truth, probability)),
            "Best_epoch": epoch, "Validation_loss": validation_loss,
        }
        rows.append(row)
        print(f"Fold {fold}: epoch={epoch}, ACC={row['ACC']:.4f}, MCC={row['MCC']:.4f}, AUC={row['AUC']:.4f}")

    metric_names = ("SN", "ACC", "SP", "MCC", "AUC", "Best_epoch", "Validation_loss")
    mean_row, std_row = {"Fold": "mean"}, {"Fold": "std"}
    for metric in metric_names:
        values = np.array([float(row[metric]) for row in rows])
        mean_row[metric], std_row[metric] = float(values.mean()), float(values.std(ddof=1))
    with (args.output_dir / "onehot_bilstm_attention_5fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Fold", "TN", "FP", "FN", "TP", *metric_names])
        writer.writeheader(); writer.writerows(rows); writer.writerow(mean_row); writer.writerow(std_row)
    print("\nFive-fold mean +/- sample standard deviation")
    for metric in metric_names:
        print(f"{metric}: {mean_row[metric]:.4f} +/- {std_row[metric]:.4f}")


if __name__ == "__main__":
    main()
