"""The five sequence feature families used in the earlier fusion experiments."""

from __future__ import annotations

from itertools import product

import numpy as np

NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = NATURAL_AA + "O"
AA_INDEX = {aa: index for index, aa in enumerate(ALPHABET)}
PAIR_INDEX = {pair: index for index, pair in enumerate(product(ALPHABET, repeat=2))}
WINDOW_LENGTH = 31


def fused_features(sequences: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    dimensions = {"AAC": 20, "PWAA": 21, "EBGW": 15, "KMER_k2": 441, "DR_d3": 1344}
    out = np.zeros((len(sequences), sum(dimensions.values())), dtype=np.float32)
    hydrophobic, polar = set("AFGILMPVW"), set("CNQSTY")
    positive, negative = set("HKR"), set("DE")
    groups = (hydrophobic | polar, hydrophobic | positive, hydrophobic | negative)
    for row, original in enumerate(sequences):
        sequence = original.strip().upper().replace("X", "O")
        if len(sequence) != WINDOW_LENGTH or set(sequence) - set(ALPHABET):
            raise ValueError(f"Invalid 31-residue sequence at row {row}")
        for residue in sequence:
            if residue in NATURAL_AA:
                out[row, NATURAL_AA.index(residue)] += 1
            out[row, 497 + AA_INDEX[residue]] += 1
        out[row, :20] /= WINDOW_LENGTH
        for position, residue in enumerate(sequence, start=1):
            offset = position - 16
            out[row, 20 + AA_INDEX[residue]] += offset + abs(offset) / 15
        out[row, 20:41] /= 240
        for group_index, allowed in enumerate(groups):
            binary = np.fromiter((residue in allowed for residue in sequence), dtype=np.float32)
            for part in range(1, 6):
                out[row, 41 + group_index * 5 + part - 1] = binary[: int(np.floor(31 * part / 5))].mean()
        for start in range(30):
            out[row, 56 + PAIR_INDEX[(sequence[start], sequence[start + 1])]] += 1
        out[row, 56:497] /= 30
        for distance in range(1, 4):
            pair_offset = 518 + (distance - 1) * 441
            for start in range(31 - distance):
                out[row, pair_offset + PAIR_INDEX[(sequence[start], sequence[start + distance])]] += 1
    return out, dimensions
