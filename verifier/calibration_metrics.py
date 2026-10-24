"""Mathematical Calibration Metrics for Section 3 of the Research Paper.

Implements:
1. Expected Calibration Error (ECE):
       ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
2. Maximum Calibration Error (MCE):
       MCE = max_{m} |acc(B_m) - conf(B_m)|
3. Brier Score:
       BS = (1 / N) * sum_{i=1}^N (p_i - y_i)^2
4. Reliability Diagram Generator for publication-grade curves.
"""
from typing import List, Dict, Any, Tuple
import math


def calculate_ece(
    probabilities: List[float],
    labels: List[int],
    num_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probabilities: Calibrated confidence scores in [0, 1].
        labels: Ground truth binary outcomes (1 for correct/entailed, 0 for incorrect/hallucinated).
        num_bins: Number of equal-width probability bins (default 10).

    Returns:
        (ece, mce, bin_stats)
    """
    if len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have equal length.")
    n = len(probabilities)
    if n == 0:
        return 0.0, 0.0, []

    bin_width = 1.0 / num_bins
    bin_stats: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        # Filter items in this bin [lower, upper) (or <= upper for the last bin)
        if i == num_bins - 1:
            indices = [idx for idx, p in enumerate(probabilities) if bin_lower <= p <= bin_upper]
        else:
            indices = [idx for idx, p in enumerate(probabilities) if bin_lower <= p < bin_upper]

        bin_size = len(indices)
        if bin_size > 0:
            bin_conf = sum(probabilities[idx] for idx in indices) / bin_size
            bin_acc = sum(labels[idx] for idx in indices) / bin_size
            gap = abs(bin_acc - bin_conf)
            weight = bin_size / n
            ece += weight * gap
            mce = max(mce, gap)
        else:
            bin_conf = (bin_lower + bin_upper) / 2.0
            bin_acc = 0.0
            gap = 0.0

        bin_stats.append({
            "bin_index": i + 1,
            "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
            "count": bin_size,
            "confidence": round(bin_conf, 4),
            "accuracy": round(bin_acc, 4),
            "gap": round(gap, 4),
        })

    return round(ece, 4), round(mce, 4), bin_stats


def calculate_brier_score(probabilities: List[float], labels: List[int]) -> float:
    """Computes standard Brier Score (lower is better, 0 is perfect)."""
    if len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have equal length.")
    if not probabilities:
        return 0.0
    n = len(probabilities)
    return round(sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n, 4)


def generate_reliability_diagram(
    probabilities: List[float],
    labels: List[int],
    num_bins: int = 10
) -> Dict[str, Any]:
    """Generates reliability statistics and an ASCII reliability diagram for paper reporting."""
    ece, mce, bin_stats = calculate_ece(probabilities, labels, num_bins)
    brier = calculate_brier_score(probabilities, labels)

    # Render clean ASCII diagram
    lines = [
        "===========================================================",
        f" ACR-RAG Reliability Diagram (ECE: {ece:.4f}, Brier: {brier:.4f}, MCE: {mce:.4f})",
        "===========================================================",
        "Bin Range        | Count | Conf  | Acc   | Calibration Bar (A=Acc, C=Conf)",
        "-----------------+-------+-------+-------+------------------"
    ]
    for b in bin_stats:
        bar_len = 16
        c_pos = int(round(b["confidence"] * bar_len))
        a_pos = int(round(b["accuracy"] * bar_len))
        
        # Build bar string
        bar = list(" " * (bar_len + 1))
        if b["count"] > 0:
            if c_pos == a_pos:
                bar[c_pos] = "★"
            else:
                bar[c_pos] = "C"
                bar[a_pos] = "A"
        bar_str = "".join(bar)

        lines.append(
            f"{b['bin_range']:<16} | {b['count']:<5} | {b['confidence']:.2f}  | {b['accuracy']:.2f}  | |{bar_str}|"
        )
    lines.append("===========================================================")
    ascii_diagram = "\n".join(lines)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "bin_stats": bin_stats,
        "ascii_diagram": ascii_diagram,
    }
