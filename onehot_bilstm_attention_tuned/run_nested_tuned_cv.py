"""Outer five-fold One-hot BiLSTM evaluation with fold-local hyperparameter selection."""

from __future__ import annotations

import argparse
import csv
import json
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
AA_INDEX = {letter: index for index, letter in enumerate(ALPHABET)}
WINDOW_LENGTH = 31
RANDOM_SEED = 20261009
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 10
# Candidate choices are evaluated only on the validation subset of each outer training fold.
CANDIDATES = (
    {"name": "A", "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.10, "hidden_size": 64},
    {"name": "B", "learning_rate": 0.001, "weight_decay": 0.0010, "dropout": 0.30, "hidden_size": 64},
    {"name": "C", "learning_rate": 0.003, "weight_decay": 0.0001, "dropout": 0.30, "hidden_size": 64},
    {"name": "D", "learning_rate": 0.0005, "weight_decay": 0.00001, "dropout": 0.10, "hidden_size": 32},
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    return records


def valid_windows(records: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    kept, removed, center = [], [], WINDOW_LENGTH // 2
    for identifier, sequence in records:
        if len(sequence) == WINDOW_LENGTH and sequence[center] == "K":
            invalid = set(sequence) - set(ALPHABET)
            if invalid:
                raise ValueError(f"{identifier}: unsupported residues {sorted(invalid)}")
            kept.append(sequence)
        else:
            removed.append(identifier)
    return kept, removed


def one_hot(sequences: list[str]) -> np.ndarray:
    x = np.zeros((len(sequences), WINDOW_LENGTH, len(ALPHABET)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence):
            x[row, position, AA_INDEX[residue]] = 1.0
    return x


def loader(x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))), batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def evaluate(model: nn.Module, batches: DataLoader, device: torch.device, loss_fn: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval(); total_loss = 0.0; total = 0; all_probabilities = []; all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            total_loss += loss_fn(logits, y_batch).item() * len(y_batch); total += len(y_batch)
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy()); all_labels.append(y_batch.cpu().numpy())
    return total_loss / total, np.concatenate(all_probabilities), np.concatenate(all_labels).astype(np.int8)


def fit_with_validation(x_fit, y_fit, x_valid, y_valid, config, seed, device):
    seed_everything(seed)
    model = GlypredBiLSTMAttention(config["hidden_size"], config["dropout"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    fit_batches = loader(x_fit, y_fit, True, seed)
    valid_batches = loader(x_valid, y_valid, False, seed)
    best_state, best_loss, wait, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_batch, y_batch in fit_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(x_batch), y_batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        validation_loss, _, _ = evaluate(model, valid_batches, device, loss_fn)
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, wait, best_epoch = validation_loss, deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    _, probability, truth = evaluate(model, valid_batches, device, loss_fn)
    return best_epoch, best_loss, float(roc_auc_score(truth, probability))


def refit_and_test(x_train, y_train, x_test, y_test, config, epochs, seed, device):
    seed_everything(seed)
    model = GlypredBiLSTMAttention(config["hidden_size"], config["dropout"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_batches = loader(x_train, y_train, True, seed)
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(x_batch), y_batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return evaluate(model, loader(x_test, y_test, False, seed), device, loss_fn)


def metric_values(tn, fp, fn, tp) -> dict[str, float]:
    sn = tp / (tp + fn) if tp + fn else 0.0; sp = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"SN": sn, "ACC": acc, "SP": sp, "MCC": mcc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=BASE_DIR.parent / "trainPos.txt")
    parser.add_argument("--neg", type=Path, default=BASE_DIR.parent / "trainNeg.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args(); seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this tuned experiment")
    pos, removed_pos = valid_windows(read_fasta(args.pos)); neg, removed_neg = valid_windows(read_fasta(args.neg))
    sequences = pos + neg; labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8); x = one_hot(sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"device": str(device), "positive_retained": len(pos), "negative_retained": len(neg), "positive_removed": len(removed_pos), "negative_removed": len(removed_neg), "retained_lengths": dict(Counter(map(len, sequences))), "input_shape": [31, 21], "outer_cv": "stratified five-fold", "inner_selection": "stratified 15 percent validation split within each outer training fold", "candidates": CANDIDATES, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE}
    (args.output_dir / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Device: {device}; retained samples: positive={len(pos)}, negative={len(neg)}")

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    result_rows, candidate_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, labels), start=1):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED + fold)
        fit_idx, valid_idx = next(inner.split(x[train_idx], labels[train_idx]))
        x_outer, y_outer = x[train_idx], labels[train_idx]
        trials = []
        for candidate_no, config in enumerate(CANDIDATES):
            epoch, val_loss, val_auc = fit_with_validation(x_outer[fit_idx], y_outer[fit_idx], x_outer[valid_idx], y_outer[valid_idx], config, RANDOM_SEED + fold * 100 + candidate_no, device)
            trial = {"Fold": fold, **config, "Best_epoch": epoch, "Validation_loss": val_loss, "Validation_AUC": val_auc}
            trials.append(trial); candidate_rows.append(trial)
        chosen = max(trials, key=lambda trial: (trial["Validation_AUC"], -trial["Validation_loss"]))
        selected_config = next(config for config in CANDIDATES if config["name"] == chosen["name"])
        _, probability, truth = refit_and_test(x_outer, y_outer, x[test_idx], labels[test_idx], selected_config, chosen["Best_epoch"], RANDOM_SEED + fold * 1000, device)
        prediction = (probability >= 0.5).astype(np.int8); tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        row = {"Fold": fold, "Selected_config": chosen["name"], "Selected_epoch": chosen["Best_epoch"], "Selected_validation_AUC": chosen["Validation_AUC"], "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), **metric_values(int(tn), int(fp), int(fn), int(tp)), "AUC": float(roc_auc_score(truth, probability))}
        result_rows.append(row)
        print(f"Fold {fold}: config={chosen['name']}, epoch={chosen['Best_epoch']}, ACC={row['ACC']:.4f}, MCC={row['MCC']:.4f}, AUC={row['AUC']:.4f}")

    metrics = ("SN", "ACC", "SP", "MCC", "AUC", "Selected_epoch", "Selected_validation_AUC")
    mean_row, std_row = {"Fold": "mean"}, {"Fold": "std"}
    for metric in metrics:
        values = np.array([float(row[metric]) for row in result_rows]); mean_row[metric] = float(values.mean()); std_row[metric] = float(values.std(ddof=1))
    with (args.output_dir / "onehot_tuned_bilstm_attention_5fold.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Fold", "Selected_config", "Selected_epoch", "Selected_validation_AUC", "TN", "FP", "FN", "TP", "SN", "ACC", "SP", "MCC", "AUC"])
        writer.writeheader(); writer.writerows(result_rows); writer.writerow(mean_row); writer.writerow(std_row)
    with (args.output_dir / "inner_candidate_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Fold", "name", "learning_rate", "weight_decay", "dropout", "hidden_size", "Best_epoch", "Validation_loss", "Validation_AUC"])
        writer.writeheader(); writer.writerows(candidate_rows)
    print("\nFive-fold mean +/- sample standard deviation")
    for metric in metrics: print(f"{metric}: {mean_row[metric]:.4f} +/- {std_row[metric]:.4f}")


if __name__ == "__main__":
    main()
