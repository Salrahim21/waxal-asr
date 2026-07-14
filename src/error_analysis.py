"""Error analysis and HTML report generation for WAXAL ASR.

Provides detailed ASR error breakdowns including:
- Highest/lowest WER samples
- Longest/shortest transcripts
- Common substitutions, insertions, and deletions
- Aggregated HTML report
"""

from __future__ import annotations

import html
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-example WER computation
# ---------------------------------------------------------------------------

def compute_per_example_wer(
    references: list[str],
    predictions: list[str],
) -> list[float]:
    """Compute WER for each (reference, prediction) pair.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model predictions.

    Returns:
        List of per-example WER floats.
    """
    wers: list[float] = []
    for ref, pred in zip(references, predictions):
        ref_lower = ref.lower().strip()
        pred_lower = pred.lower().strip()
        if not ref_lower:
            wers.append(0.0 if not pred_lower else 1.0)
        else:
            wers.append(float(jiwer.wer(ref_lower, pred_lower)))
    return wers


# ---------------------------------------------------------------------------
# Edit operation extraction
# ---------------------------------------------------------------------------

def extract_edit_operations(
    references: list[str],
    predictions: list[str],
) -> dict[str, Counter[str]]:
    """Extract substitutions, insertions, and deletions across all examples.

    Uses ``jiwer`` alignment to find word-level edit operations.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model predictions.

    Returns:
        Dict with keys ``"substitutions"``, ``"insertions"``, ``"deletions"``,
        each mapping to a :class:`Counter` of word or word-pair strings.
    """
    substitutions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()

    for ref, pred in zip(references, predictions):
        ref_lower = ref.lower().strip()
        pred_lower = pred.lower().strip()
        if not ref_lower or not pred_lower:
            continue

        try:
            out = jiwer.process_words(ref_lower, pred_lower)
            for alignment in out.alignments:
                for chunk in alignment:
                    ref_words = out.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
                    hyp_words = out.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]

                    if chunk.type == "substitute":
                        for rw, hw in zip(ref_words, hyp_words):
                            substitutions[f"{rw} -> {hw}"] += 1
                    elif chunk.type == "insert":
                        for hw in hyp_words:
                            insertions[hw] += 1
                    elif chunk.type == "delete":
                        for rw in ref_words:
                            deletions[rw] += 1
        except Exception:
            continue

    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


# ---------------------------------------------------------------------------
# Analysis summary
# ---------------------------------------------------------------------------

def build_error_analysis(
    references: list[str],
    predictions: list[str],
    top_n: int = 20,
) -> dict[str, Any]:
    """Build a comprehensive error analysis report.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model predictions.
        top_n: Number of top items for each category.

    Returns:
        Dict containing all analysis results.
    """
    per_wer = compute_per_example_wer(references, predictions)
    indices_sorted = np.argsort(per_wer)

    # Highest WER samples (worst predictions)
    worst_indices = indices_sorted[::-1][:top_n]
    highest_wer = [
        {
            "index": int(i),
            "wer": per_wer[i],
            "reference": references[i],
            "prediction": predictions[i],
        }
        for i in worst_indices
    ]

    # Lowest WER samples (best predictions)
    best_indices = indices_sorted[:top_n]
    lowest_wer = [
        {
            "index": int(i),
            "wer": per_wer[i],
            "reference": references[i],
            "prediction": predictions[i],
        }
        for i in best_indices
    ]

    # By transcript length
    ref_lengths = [len(r.split()) for r in references]
    longest_indices = np.argsort(ref_lengths)[::-1][:top_n]
    shortest_indices = np.argsort(ref_lengths)[:top_n]

    longest_transcripts = [
        {
            "index": int(i),
            "word_count": ref_lengths[i],
            "wer": per_wer[i],
            "reference": references[i],
            "prediction": predictions[i],
        }
        for i in longest_indices
    ]
    shortest_transcripts = [
        {
            "index": int(i),
            "word_count": ref_lengths[i],
            "wer": per_wer[i],
            "reference": references[i],
            "prediction": predictions[i],
        }
        for i in shortest_indices
    ]

    # Edit operations
    edits = extract_edit_operations(references, predictions)

    return {
        "total_examples": len(references),
        "mean_wer": float(np.mean(per_wer)),
        "median_wer": float(np.median(per_wer)),
        "std_wer": float(np.std(per_wer)),
        "per_example_wer": per_wer,
        "highest_wer_samples": highest_wer,
        "lowest_wer_samples": lowest_wer,
        "longest_transcripts": longest_transcripts,
        "shortest_transcripts": shortest_transcripts,
        "common_substitutions": edits["substitutions"].most_common(top_n),
        "common_insertions": edits["insertions"].most_common(top_n),
        "common_deletions": edits["deletions"].most_common(top_n),
    }


def save_error_analysis_json(
    analysis: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save error analysis to JSON (excluding non-serialisable fields).

    Args:
        analysis: Output of :func:`build_error_analysis`.
        output_path: Target file path.

    Returns:
        Path to the saved JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {k: v for k, v in analysis.items() if k != "per_example_wer"}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Error analysis JSON saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text)


def _render_sample_table(samples: list[dict[str, Any]], caption: str) -> str:
    """Render a list of sample dicts as an HTML table."""
    rows = ""
    for s in samples:
        wer_pct = s.get("wer", 0) * 100
        wer_class = "wer-high" if wer_pct > 80 else ("wer-med" if wer_pct > 40 else "wer-low")
        rows += f"""<tr>
            <td>{s.get('index', '')}</td>
            <td class="text-cell">{_esc(s.get('reference', ''))}</td>
            <td class="text-cell">{_esc(s.get('prediction', ''))}</td>
            <td class="{wer_class}">{wer_pct:.1f}%</td>
        </tr>\n"""

    return f"""
    <h3>{_esc(caption)}</h3>
    <table>
        <thead><tr><th>#</th><th>Reference</th><th>Prediction</th><th>WER</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def _render_counter_table(items: list[tuple[str, int]], caption: str, col_header: str) -> str:
    """Render a Counter.most_common() list as an HTML table."""
    rows = ""
    for item, count in items:
        rows += f"<tr><td>{_esc(str(item))}</td><td>{count}</td></tr>\n"
    return f"""
    <h3>{_esc(caption)}</h3>
    <table>
        <thead><tr><th>{_esc(col_header)}</th><th>Count</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def generate_html_report(
    analysis: dict[str, Any],
    output_path: str | Path,
    title: str = "WAXAL ASR — Error Analysis Report",
) -> Path:
    """Generate a standalone HTML error analysis report.

    Args:
        analysis: Output of :func:`build_error_analysis`.
        output_path: Path for the HTML file.
        title: Report title.

    Returns:
        Path to the saved HTML report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    css = """
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px; margin: 0 auto; padding: 20px;
            background: #fafafa; color: #333;
        }
        h1 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #555; margin-top: 25px; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .stat-card {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
        }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #1a73e8; }
        .stat-card .label { font-size: 13px; color: #666; margin-top: 5px; }
        table {
            width: 100%; border-collapse: collapse; margin: 10px 0;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 8px; overflow: hidden;
        }
        th {
            background: #1a73e8; color: white; padding: 10px 12px;
            text-align: left; font-size: 13px;
        }
        td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
        tr:hover { background: #f5f5f5; }
        .text-cell { max-width: 400px; word-wrap: break-word; }
        .wer-high { color: #d32f2f; font-weight: bold; }
        .wer-med  { color: #f57c00; font-weight: bold; }
        .wer-low  { color: #388e3c; font-weight: bold; }
        .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
                   font-size: 12px; color: #999; text-align: center; }
    </style>
    """

    total = analysis["total_examples"]
    mean_wer = analysis["mean_wer"] * 100
    median_wer = analysis["median_wer"] * 100
    std_wer = analysis["std_wer"] * 100

    stats_html = f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="value">{total}</div><div class="label">Total Examples</div></div>
        <div class="stat-card"><div class="value">{mean_wer:.1f}%</div><div class="label">Mean WER</div></div>
        <div class="stat-card"><div class="value">{median_wer:.1f}%</div><div class="label">Median WER</div></div>
        <div class="stat-card"><div class="value">{std_wer:.1f}%</div><div class="label">Std WER</div></div>
    </div>
    """

    body_parts = [
        f"<h1>{_esc(title)}</h1>",
        "<h2>Summary Statistics</h2>",
        stats_html,
        "<h2>Highest WER Samples (Worst Predictions)</h2>",
        _render_sample_table(analysis["highest_wer_samples"], "Samples sorted by WER descending"),
        "<h2>Lowest WER Samples (Best Predictions)</h2>",
        _render_sample_table(analysis["lowest_wer_samples"], "Samples sorted by WER ascending"),
        "<h2>Longest Transcripts</h2>",
        _render_sample_table(analysis["longest_transcripts"], "Longest reference transcripts by word count"),
        "<h2>Shortest Transcripts</h2>",
        _render_sample_table(analysis["shortest_transcripts"], "Shortest reference transcripts by word count"),
        "<h2>Common Edit Operations</h2>",
        _render_counter_table(analysis["common_substitutions"], "Most Common Substitutions", "Substitution (ref -> hyp)"),
        _render_counter_table(analysis["common_insertions"], "Most Common Insertions", "Inserted Word"),
        _render_counter_table(analysis["common_deletions"], "Most Common Deletions", "Deleted Word"),
        '<div class="footer">Generated by WAXAL ASR Error Analysis</div>',
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_esc(title)}</title>
    {css}
</head>
<body>
    {''.join(body_parts)}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    logger.info("HTML error analysis report saved to %s", output_path)
    return output_path
