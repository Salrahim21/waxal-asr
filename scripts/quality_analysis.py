"""Quality analysis for WAXAL ASR submissions.

Generates a markdown report with detailed metrics for a submission CSV.

Usage:
    python scripts/quality_analysis.py submissions/submission_beam_small.csv beam_search_small
"""
from __future__ import annotations

import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GARBLED_SCRIPTS = {
    "KHMER", "DEVANAGARI", "THAI", "TIBETAN", "MYANMAR",
    "BENGALI", "GUJARATI", "KANNADA", "TAMIL", "TELUGU",
    "MALAYALAM", "SINHALA", "LAO", "GEORGIAN", "ARMENIAN",
    "ETHIOPIC", "CHEROKEE", "CANADIAN",
}


def is_garbled(text: str) -> bool:
    if not text or len(text) <= 1:
        return False
    garbled = 0
    for ch in text:
        try:
            script = unicodedata.name(ch, "").split()[0]
            if script in GARBLED_SCRIPTS:
                garbled += 1
        except (ValueError, IndexError):
            pass
    return garbled / len(text) > 0.3


def is_repetitive(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) <= 3:
        return False
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count > len(words) * 0.5


def analyze(submission_path: str, experiment_name: str) -> str:
    path = Path(submission_path)
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    # Per-language stats
    lang_data: dict[str, list[str]] = {}
    for r in rows:
        lang = r["ID"].split("_")[0]
        lang_data.setdefault(lang, []).append(r["Target"].strip())

    # Global metrics
    all_targets = [r["Target"].strip() for r in rows]
    empty_count = sum(1 for t in all_targets if not t)
    single_char = sum(1 for t in all_targets if len(t) == 1)
    punct_only = sum(1 for t in all_targets if t and all(not c.isalnum() for c in t))
    garbled_count = sum(1 for t in all_targets if is_garbled(t))
    repetitive_count = sum(1 for t in all_targets if is_repetitive(t))
    non_empty = [t for t in all_targets if t]
    unique_count = len(set(non_empty))
    dup_count = len(non_empty) - unique_count
    avg_len = sum(len(t) for t in non_empty) / max(len(non_empty), 1)

    lines = [
        f"# Quality Report: {experiment_name}",
        f"",
        f"**Submission:** `{path.name}`",
        f"**Total predictions:** {total}",
        f"",
        f"## Global Metrics",
        f"",
        f"| Metric | Count | % |",
        f"|--------|-------|---|",
        f"| Empty predictions | {empty_count} | {empty_count/total*100:.1f}% |",
        f"| Single character | {single_char} | {single_char/total*100:.1f}% |",
        f"| Punctuation only | {punct_only} | {punct_only/total*100:.1f}% |",
        f"| Garbled script | {garbled_count} | {garbled_count/total*100:.1f}% |",
        f"| Repetitive | {repetitive_count} | {repetitive_count/total*100:.1f}% |",
        f"| Duplicate transcriptions | {dup_count} | {dup_count/total*100:.1f}% |",
        f"| Unique transcriptions | {unique_count} | {unique_count/total*100:.1f}% |",
        f"",
        f"**Average transcript length:** {avg_len:.0f} chars",
        f"",
    ]

    # Per-language breakdown
    lines.append("## Per-Language Breakdown")
    lines.append("")

    # Load ground truth for length comparison
    train_path = PROJECT_ROOT / "Train.csv"
    gt_avg: dict[str, float] = {}
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        for lang in lang_data:
            gt = [r["transcription"] for r in train_rows if r.get("language") == lang]
            gt_avg[lang] = sum(len(t) for t in gt) / max(len(gt), 1) if gt else 0

    for lang in sorted(lang_data):
        preds = lang_data[lang]
        lang_total = len(preds)
        lang_empty = sum(1 for t in preds if not t)
        lang_garbled = sum(1 for t in preds if is_garbled(t))
        lang_rep = sum(1 for t in preds if is_repetitive(t))
        lang_non_empty = [t for t in preds if t]
        lang_avg = sum(len(t) for t in lang_non_empty) / max(len(lang_non_empty), 1)
        lang_unique = len(set(lang_non_empty))
        lang_dup = len(lang_non_empty) - lang_unique
        gt_a = gt_avg.get(lang, 0)
        ratio = lang_avg / gt_a if gt_a else 0

        lines.append(f"### {lang.upper()} ({lang_total} samples)")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Empty | {lang_empty} ({lang_empty/lang_total*100:.1f}%) |")
        lines.append(f"| Garbled | {lang_garbled} ({lang_garbled/lang_total*100:.1f}%) |")
        lines.append(f"| Repetitive | {lang_rep} ({lang_rep/lang_total*100:.1f}%) |")
        lines.append(f"| Duplicates | {lang_dup} |")
        lines.append(f"| Avg pred length | {lang_avg:.0f} chars |")
        if gt_a:
            lines.append(f"| Avg GT length | {gt_a:.0f} chars |")
            lines.append(f"| Length ratio | {ratio:.2f}x |")
        lines.append(f"")

        # Sample predictions
        lines.append(f"**Sample predictions:**")
        lines.append(f"")
        count = 0
        for i, r in enumerate(rows):
            if r["ID"].startswith(lang) and r["Target"].strip() and len(r["Target"].strip()) > 20 and count < 3:
                lines.append(f"- `{r['ID']}`: {r['Target'].strip()[:120]}")
                count += 1
        lines.append(f"")

    # Validation against SampleSubmission
    sample_path = PROJECT_ROOT / "SampleSubmission.csv"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            expected = {r["ID"] for r in csv.DictReader(f)}
        submitted = {r["ID"] for r in rows}
        missing = expected - submitted
        extra = submitted - expected
        lines.append("## Submission Validation")
        lines.append("")
        if missing:
            lines.append(f"**FAIL:** Missing {len(missing)} IDs")
        elif extra:
            lines.append(f"**WARNING:** {len(extra)} extra IDs not in SampleSubmission")
        else:
            lines.append(f"**PASS:** All {len(expected)} IDs present")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/quality_analysis.py <submission.csv> <experiment_name>")
        sys.exit(1)

    submission_path = sys.argv[1]
    experiment_name = sys.argv[2]

    report = analyze(submission_path, experiment_name)

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"quality_report_{experiment_name}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
