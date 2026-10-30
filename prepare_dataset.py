"""Turn a Gmail scrape into a training set and a review file.

Known statuses are applications. Unknown rows are written to data/review.csv so
you can relabel them. A real status means the email is an application. `no`
means it is not. Anything still blank is left out of training.

    python prepare_dataset.py

Personal mail is written under data/, which is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from email_classifier import (
    APPLICATION_STATUSES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    ROOT,
    message_text,
)

DEFAULT_INPUT = ROOT / "output" / "job_applications.json"
DEFAULT_DATA = ROOT / "data"
REVIEW_FIELDS = ["thread_id", "date", "company", "subject", "preview", "suggestion", "label"]
ALLOWED_LABELS = set(APPLICATION_STATUSES) | {NEGATIVE_LABEL}

NO_HINT = re.compile(
    r"\b(?:newsletter|study abroad|survey|still interested|verify your|"
    r"latest .{0,40}jobs|unsubscribe|webinar)\b",
    re.I,
)
APPLIED_HINT = re.compile(
    r"\b(?:received your (?:job )?application|application for|thank you for applying|"
    r"thanks for applying|looking forward to reviewing your application)\b",
    re.I,
)


def suggest_label(record: dict) -> str:
    """Hint from the subject only. Bodies quote older mail and make weak hints."""
    subject = record.get("subject") or ""
    if NO_HINT.search(subject):
        return NEGATIVE_LABEL
    if re.search(r"\bassessment\b", subject, re.I):
        return "online assessment"
    if re.search(r"\binterview\b", subject, re.I):
        return "interview"
    if APPLIED_HINT.search(subject):
        return "applied"
    return ""


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"No scrape file at {path}. Run python scrape_job_applications.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_review_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            thread_id = (row.get("thread_id") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if thread_id:
                labels[thread_id] = label
    return labels


def preview_of(record: dict) -> str:
    text = re.sub(r"\s+", " ", record.get("body") or record.get("snippet") or "").strip()
    if len(text) > 220:
        return text[:217].rstrip() + "..."
    return text


def prepare(source: Path, data_dir: Path) -> None:
    rows = load_rows(source)
    data_dir.mkdir(parents=True, exist_ok=True)
    review_path = data_dir / "review.csv"
    previous = load_review_labels(review_path)

    labeled: list[dict] = []
    review: list[dict] = []
    bad: list[str] = []

    for record in rows:
        status = (record.get("status") or "").strip().lower()
        thread_id = record.get("thread_id") or ""
        base = {
            "thread_id": thread_id,
            "subject": record.get("subject") or "",
            "snippet": record.get("snippet") or "",
            "body": record.get("body") or "",
            "text": message_text(record),
        }
        if status in APPLICATION_STATUSES:
            labeled.append({**base, "label": POSITIVE_LABEL, "status": status, "source": "auto"})
            continue
        if status != "unknown":
            continue
        chosen = previous.get(thread_id, "")
        review.append(
            {
                "thread_id": thread_id,
                "date": record.get("date") or "",
                "company": record.get("company") or "",
                "subject": record.get("subject") or "",
                "preview": preview_of(record),
                "suggestion": suggest_label(record),
                "label": chosen,
            }
        )
        if not chosen:
            continue
        if chosen not in ALLOWED_LABELS:
            bad.append(f"{thread_id}: {chosen}")
            continue
        if chosen == NEGATIVE_LABEL:
            labeled.append({**base, "label": NEGATIVE_LABEL, "status": "", "source": "review"})
        else:
            labeled.append({**base, "label": POSITIVE_LABEL, "status": chosen, "source": "review"})

    labeled_path = data_dir / "labeled.jsonl"
    with labeled_path.open("w", encoding="utf-8") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)

    applications = sum(1 for row in labeled if row["label"] == POSITIVE_LABEL)
    negatives = sum(1 for row in labeled if row["label"] == NEGATIVE_LABEL)
    waiting = sum(1 for row in review if not row["label"])
    print(f"Applications: {applications}")
    print(f"For sure no: {negatives}")
    print(f"Still need a label: {waiting}")
    print(f"Training rows: {labeled_path}")
    print(f"Review file:   {review_path}")
    print("Fill the label column with applied, rejected, interview, online assessment, or no.")
    print("suggestion is only a hint. Training uses label.")
    if bad:
        print("These labels were skipped:")
        for item in bad:
            print(f"  {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local training set from scraped job mail.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Scrape JSON to read.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Folder for the review file and training rows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.input, args.data)
