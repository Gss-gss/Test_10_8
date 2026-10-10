"""Paper-aligned sequence feature extractors for 31-residue glycation windows.

The feature order is AAC, PWAA, EBGW, KMER (k=2), and DR (maximum
residue separation 3), matching the five feature families in Glypred.
"""

from __future__ import annotations

from itertools import product

import numpy as np


NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
# Glypred uses O as its artificial terminal residue. The supplied training
# fragments do not contain O or X; X is mapped to O for compatibility.
PAPER_ALPHABET = NATURAL_AA + "O"
AA_INDEX = {aa: index for index, aa in enumerate(PAPER_ALPHABET)}
PAIR_INDEX = {pair: index for index, pair in enumerate(product(PAPER_ALPHABET, repeat=2))}
WINDOW_LENGTH = 31


def normalize_sequence(sequence: str) -> str:
    """Normalize a supplied window to the paper's 21-symbol alphabet."""
    normalized = sequence.strip().upper().replace("X", "O")
    invalid = set(normalized) - set(PAPER_ALPHABET)
    if invalid:
        raise ValueError(f"Unsupported residues: {sorted(invalid)}")
    if len(normalized) != WINDOW_LENGTH:
        raise ValueError(f"Expected a {WINDOW_LENGTH}-residue window, received {len(normalized)}")
    return normalized


def aac(sequences: list[str]) -> np.ndarray:
    """20-dimensional amino-acid composition, excluding the dummy O symbol."""
    output = np.zeros((len(sequences), len(NATURAL_AA)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for residue in sequence:
            if residue in NATURAL_AA:
                output[row, NATURAL_AA.index(residue)] += 1.0
        output[row] /= len(sequence)
    return output


def pwaa(sequences: list[str]) -> np.ndarray:
    """21-dimensional position-weighted amino-acid composition.

    This uses the paper's position weighting with the central residue indexed
    at zero and the 21-symbol alphabet used in the authors' predictor.
    """
    half_window = (WINDOW_LENGTH - 1) // 2
    denominator = half_window * (half_window + 1)
    output = np.zeros((len(sequences), len(PAPER_ALPHABET)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, residue in enumerate(sequence, start=1):
            offset = position - half_window - 1
            weight = offset + abs(offset) / half_window
            output[row, AA_INDEX[residue]] += weight
        output[row] /= denominator
    return output


def ebgw(sequences: list[str], segments: int = 5) -> np.ndarray:
    """15-dimensional encoding based on grouped weight (EBGW), J=5.

    Each of three physicochemical partitions is summarized over five
    increasing prefixes, following the grouping stated in the paper.
    """
    hydrophobic = set("AFGILMPVW")
    polar = set("CNQSTY")
    positive = set("HKR")
    negative = set("DE")
    positive_sets = (
        hydrophobic | polar,
        hydrophobic | positive,
        hydrophobic | negative,
    )
    output = np.zeros((len(sequences), 3 * segments), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for group_index, included in enumerate(positive_sets):
            binary = np.fromiter((1.0 if residue in included else 0.0 for residue in sequence), dtype=np.float32)
            for part in range(1, segments + 1):
                prefix_length = int(np.floor(WINDOW_LENGTH * part / segments))
                output[row, group_index * segments + part - 1] = binary[:prefix_length].mean()
    return output


def kmer(sequences: list[str], k: int = 2) -> np.ndarray:
    """Normalized 2-mer composition over the paper's 21-symbol alphabet."""
    if k != 2:
        raise ValueError("This reproducibility baseline fixes KMER at k=2")
    output = np.zeros((len(sequences), len(PAPER_ALPHABET) ** 2), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for start in range(len(sequence) - 1):
            output[row, PAIR_INDEX[(sequence[start], sequence[start + 1])]] += 1.0
        output[row] /= len(sequence) - 1
    return output


def dr(sequences: list[str], max_distance: int = 3) -> np.ndarray:
    """Distance-residue encoding using composition plus pair counts at d=1..3.

    With the 21-symbol alphabet and ``max_distance=3``, the result has
    21 + 3 * 21^2 = 1344 dimensions, the dimensionality used by the authors'
    public predictor for Protein DR.
    """
    alphabet_size = len(PAPER_ALPHABET)
    output = np.zeros((len(sequences), alphabet_size + max_distance * alphabet_size**2), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for residue in sequence:
            output[row, AA_INDEX[residue]] += 1.0
        cursor = alphabet_size
        for distance in range(1, max_distance + 1):
            for start in range(len(sequence) - distance):
                pair = (sequence[start], sequence[start + distance])
                output[row, cursor + PAIR_INDEX[pair]] += 1.0
            cursor += alphabet_size**2
    return output


def fused_features(sequences: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Extract the five feature families and concatenate them in paper order."""
    normalized = [normalize_sequence(sequence) for sequence in sequences]
    blocks = {
        "AAC": aac(normalized),
        "PWAA": pwaa(normalized),
        "EBGW": ebgw(normalized),
        "KMER_k2": kmer(normalized),
        "DR_d3": dr(normalized),
    }
    dimensions = {name: matrix.shape[1] for name, matrix in blocks.items()}
    return np.hstack(list(blocks.values())).astype(np.float32), dimensions
