"""Five fused sequence feature families used by the Glypred paper."""

from __future__ import annotations

from itertools import product

import numpy as np

NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAPER_ALPHABET = NATURAL_AA + "O"
AA_INDEX = {aa: index for index, aa in enumerate(PAPER_ALPHABET)}
PAIR_INDEX = {pair: index for index, pair in enumerate(product(PAPER_ALPHABET, repeat=2))}
WINDOW_LENGTH = 31


def _normalize(sequence: str) -> str:
    sequence = sequence.strip().upper().replace("X", "O")
    if len(sequence) != WINDOW_LENGTH:
        raise ValueError(f"Expected {WINDOW_LENGTH} residues, got {len(sequence)}")
    invalid = set(sequence) - set(PAPER_ALPHABET)
    if invalid:
        raise ValueError(f"Unsupported residues: {sorted(invalid)}")
    return sequence


def _aac(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), 20), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for residue in sequence:
            if residue in NATURAL_AA:
                out[i, NATURAL_AA.index(residue)] += 1
        out[i] /= WINDOW_LENGTH
    return out


def _pwaa(sequences: list[str]) -> np.ndarray:
    half, out = 15, np.zeros((len(sequences), 21), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence, start=1):
            offset = position - half - 1
            out[i, AA_INDEX[residue]] += offset + abs(offset) / half
        out[i] /= half * (half + 1)
    return out


def _ebgw(sequences: list[str]) -> np.ndarray:
    hydrophobic, polar = set("AFGILMPVW"), set("CNQSTY")
    positive, negative = set("HKR"), set("DE")
    partitions = (hydrophobic | polar, hydrophobic | positive, hydrophobic | negative)
    out = np.zeros((len(sequences), 15), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for group, allowed in enumerate(partitions):
            binary = np.array([residue in allowed for residue in sequence], dtype=np.float32)
            for part in range(1, 6):
                out[i, group * 5 + part - 1] = binary[: int(np.floor(WINDOW_LENGTH * part / 5))].mean()
    return out


def _kmer(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), 441), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for start in range(WINDOW_LENGTH - 1):
            out[i, PAIR_INDEX[(sequence[start], sequence[start + 1])]] += 1
        out[i] /= WINDOW_LENGTH - 1
    return out


def _dr(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), 21 + 3 * 441), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for residue in sequence:
            out[i, AA_INDEX[residue]] += 1
        for distance in range(1, 4):
            offset = 21 + (distance - 1) * 441
            for start in range(WINDOW_LENGTH - distance):
                out[i, offset + PAIR_INDEX[(sequence[start], sequence[start + distance])]] += 1
    return out


def fused_features(sequences: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    sequences = [_normalize(sequence) for sequence in sequences]
    blocks = {"AAC": _aac(sequences), "PWAA": _pwaa(sequences), "EBGW": _ebgw(sequences), "KMER_k2": _kmer(sequences), "DR_d3": _dr(sequences)}
    return np.hstack(list(blocks.values())).astype(np.float32), {name: value.shape[1] for name, value in blocks.items()}
