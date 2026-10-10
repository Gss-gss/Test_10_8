"""Five-fold evaluation for 31-residue FASTA sequences using one-hot features.

Run from the workspace root:
    python onehot_logistic_regression_cleaned/run_onehot_logistic_regression_cv.py

The script reads trainPos.txt and trainNeg.txt, removes records whose length is
not 31 residues, and evaluates logistic regression using stratified five-fold
cross-validation. Every metric is calculated directly from the fold confusion
matrix.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
RANDOM_SEED = 20261009
N_SPLITS = 5
TARGET_LENGTH = 31
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
OUTPUT_PATH = BASE_DIR / "results" / "onehot_logistic_regression_5fold.csv"


def read_fasta(path: Path) -> list[str]:
    """Read nonempty FASTA records and validate the declared amino-acid alphabet."""
    sequences: list[str] = []
    header: str | None = None
    parts: list[str] = []

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().upper()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences.append("".join(parts))
            header = line[1:].strip()
            parts = []
        else:
            if header is None:
                raise ValueError(f"{path}: sequence found before a FASTA header")
            invalid = set(line) - set(AMINO_ACIDS)
            if invalid:
                raise ValueError(f"{path}: unsupported residues {sorted(invalid)}")
            parts.append(line)

    if header is not None:
        sequences.append("".join(parts))
    if not sequences:
        raise ValueError(f"{path}: no FASTA sequences found")
    return sequences


def retain_target_length(sequences: list[str], target_length: int) -> tuple[list[str], list[str]]:
    """Return the retained sequences and all records excluded by length."""
    kept = [sequence for sequence in sequences if len(sequence) == target_length]
    excluded = [sequence for sequence in sequences if len(sequence) != target_length]
    return kept, excluded


def one_hot_encode(sequences: list[str], width: int) -> np.ndarray:
    """Flatten each 31-position, 21-dimensional one-hot representation."""
    encoded = np.zeros((len(sequences), width, len(AMINO_ACIDS)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        if len(sequence) != width:
            raise ValueError(f"Expected a {width}-residue sequence, received {len(sequence)}")
        for position, residue in enumerate(sequence):
            encoded[row, position, AA_TO_INDEX[residue]] = 1.0
    return encoded.reshape(len(sequences), -1)


def metrics_from_confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    """Compute the requested metrics from TN, FP, FN and TP."""
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {"SN": sensitivity, "ACC": accuracy, "SP": specificity, "MCC": mcc}


def main() -> None:
    positive_raw = read_fasta(DATA_DIR / "trainPos.txt")
    negative_raw = read_fasta(DATA_DIR / "trainNeg.txt")
    positive, excluded_positive = retain_target_length(positive_raw, TARGET_LENGTH)
    negative, excluded_negative = retain_target_length(negative_raw, TARGET_LENGTH)
    sequences = positive + negative
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int8)

    x = one_hot_encode(sequences, TARGET_LENGTH)
    print(f"Positive samples: {len(positive)}")
    print(f"Negative samples: {len(negative)}")
    print(f"Excluded non-{TARGET_LENGTH}-residue samples: "
          f"positive={len(excluded_positive)}, negative={len(excluded_negative)}")
    print(f"Retained sequence lengths: {dict(sorted(Counter(map(len, sequences)).items()))}")
    print(f"One-hot shape: {x.shape} ({TARGET_LENGTH} positions x {len(AMINO_ACIDS)} residues)")
    print(f"Cross-validation: stratified {N_SPLITS}-fold, shuffled, random seed {RANDOM_SEED}")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x, labels), start=1):
        model = LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_SEED)
        model.fit(x[train_index], labels[train_index])
        prediction = model.predict(x[test_index])
        tn, fp, fn, tp = confusion_matrix(labels[test_index], prediction, labels=[0, 1]).ravel()
        row: dict[str, float | int] = {
            "Fold": fold,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            **metrics_from_confusion_matrix(int(tn), int(fp), int(fn), int(tp)),
        }
        fold_rows.append(row)
        print(
            f"Fold {fold}: TN={tn}, FP={fp}, FN={fn}, TP={tp}; "
            f"SN={row['SN']:.4f}, ACC={row['ACC']:.4f}, "
            f"SP={row['SP']:.4f}, MCC={row['MCC']:.4f}"
        )

    metric_names = ("SN", "ACC", "SP", "MCC")
    mean_row: dict[str, float | str] = {"Fold": "mean"}
    std_row: dict[str, float | str] = {"Fold": "std"}
    for name in metric_names:
        values = np.array([float(row[name]) for row in fold_rows])
        mean_row[name] = float(values.mean())
        std_row[name] = float(values.std(ddof=1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["Fold", "TN", "FP", "FN", "TP", *metric_names])
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print("\nFive-fold mean +/- sample standard deviation")
    for name in metric_names:
        print(f"{name}: {mean_row[name]:.4f} +/- {std_row[name]:.4f}")
    print(f"Detailed results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
