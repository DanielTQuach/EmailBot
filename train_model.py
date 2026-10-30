"""Train a TF-IDF linear classifier on your own labeled email export.

The model predicts two things: whether the email is an application, and, if it
is, which status it has. Scores are cross-checked against the regex rules.
The saved model is gitignored. PNG charts go in performance/ so the scores can
be committed as a reference. Those charts are not the model, and another
person's inbox will produce different numbers.

    python prepare_dataset.py
    python train_model.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from email_classifier import (
    APPLICATION_STATUSES,
    DEFAULT_MODEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    ROOT,
)
from scrape_job_applications import classify_status, passes_mail_check

DEFAULT_LABELED = ROOT / "data" / "labeled.jsonl"
DEFAULT_CHARTS = ROOT / "performance"
CHART_NOTE = "Reference only. Each person trains a private model on their own mail."


def load_labeled(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"No training rows at {path}. Run python prepare_dataset.py and label data/review.csv."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} is empty. Label data/review.csv, then run prepare_dataset.py again.")
    return rows


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=30000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def cross_validated_predictions(pipeline: Pipeline, texts: list[str], labels: list[str]) -> list[str] | None:
    smallest = min(Counter(labels).values())
    folds = min(5, smallest)
    if folds < 2:
        return None
    return list(cross_val_predict(pipeline, texts, labels, cv=folds))


def print_scores(title: str, labels: list[str], predicted: list[str]) -> None:
    print()
    print(title)
    print(f"  accuracy: {accuracy_score(labels, predicted):.3f}  ({len(labels)} emails)")
    print(classification_report(labels, predicted, zero_division=0))


def save_performance_charts(
    directory: Path,
    application_labels: list[str],
    application_regex: list[str],
    application_model: list[str] | None,
    status_labels: list[str] | None,
    status_regex: list[str] | None,
    status_model: list[str] | None,
) -> list[Path]:
    """Write score PNGs. They contain counts and accuracy, not message text."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("matplotlib is not installed, so no performance PNGs were written.")
        print("  pip install -r requirements.txt")
        return []

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def finish(fig, name: str) -> None:
        fig.text(0.5, 0.01, CHART_NOTE, ha="center", va="bottom", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = directory / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    def confusion(name: str, title: str, labels: list[str], predicted: list[str], order: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        display = ["not an application" if label == NEGATIVE_LABEL else label for label in order]
        ConfusionMatrixDisplay.from_predictions(
            labels,
            predicted,
            labels=order,
            display_labels=display,
            cmap="Blues",
            colorbar=False,
            xticks_rotation=30,
            values_format="d",
            ax=ax,
        )
        accuracy = accuracy_score(labels, predicted)
        ax.set_title(f"{title}\nheld-out accuracy {accuracy:.1%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Labeled")
        finish(fig, name)

    comparison: list[tuple[str, float, float | None]] = [
        (
            "Application or not",
            accuracy_score(application_labels, application_regex),
            None if application_model is None else accuracy_score(application_labels, application_model),
        )
    ]
    if application_model is not None:
        order = [POSITIVE_LABEL, NEGATIVE_LABEL]
        confusion(
            "application_confusion.png",
            "Application or not",
            application_labels,
            application_model,
            order,
        )
    if status_labels and status_regex and status_model:
        comparison.append(
            (
                "Status",
                accuracy_score(status_labels, status_regex),
                accuracy_score(status_labels, status_model),
            )
        )
        present = [status for status in APPLICATION_STATUSES if status in status_labels or status in status_model]
        confusion("status_confusion.png", "Application status", status_labels, status_model, present)
    else:
        stale = directory / "status_confusion.png"
        if stale.is_file():
            stale.unlink()

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels_x = [row[0] for row in comparison]
    x = list(range(len(comparison)))
    width = 0.36
    keyword_bars = ax.bar(
        [item - width / 2 for item in x],
        [row[1] for row in comparison],
        width,
        label="Keyword rules",
        color="#9aa0a6",
    )
    model_heights = [row[2] for row in comparison]
    model_artist = ax.bar(
        [item + width / 2 for item, height in zip(x, model_heights) if height is not None],
        [height for height in model_heights if height is not None],
        width,
        label="TF-IDF model",
        color="#1a73e8",
    )
    for bars in (keyword_bars, model_artist):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, labels_x)
    ax.set_title("Keyword rules vs TF-IDF model")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finish(fig, "accuracy_comparison.png")
    return written


def regex_application_label(row: dict) -> str:
    record = {"subject": row.get("subject") or "", "snippet": row.get("snippet") or "", "body": row.get("body") or ""}
    if passes_mail_check(record):
        return POSITIVE_LABEL
    return NEGATIVE_LABEL


def regex_status(row: dict) -> str:
    return classify_status(row.get("subject") or "", row.get("snippet") or "", (row.get("body") or "")[:4000])


def train(rows: list[dict], model_path: Path, chart_dir: Path = DEFAULT_CHARTS) -> None:
    try:
        import joblib
    except ImportError:
        raise SystemExit("scikit-learn is not installed.\n  pip install -r requirements.txt")

    counts = Counter(row["label"] for row in rows)
    print(f"Training rows: {len(rows)}")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")
    if counts.get(POSITIVE_LABEL, 0) < 2 or counts.get(NEGATIVE_LABEL, 0) < 2:
        raise SystemExit(
            "Need at least 2 applications and 2 for-sure-no emails.\n"
            "Label unknown rows in data/review.csv with a status or no, then run prepare_dataset.py again."
        )

    texts = [row.get("text") or "" for row in rows]
    app_labels = [row["label"] for row in rows]
    regex_labels = [regex_application_label(row) for row in rows]
    print_scores("Regex filter compared with your labels", app_labels, regex_labels)

    application_model = make_pipeline()
    held_out = cross_validated_predictions(application_model, texts, app_labels)
    if held_out is not None:
        print_scores("TF-IDF model, held-out application decision", app_labels, held_out)
    else:
        print()
        print("Too few emails in one class to cross-check the application model.")

    application_rows = [row for row in rows if row["label"] == POSITIVE_LABEL and row.get("status") in APPLICATION_STATUSES]
    status_labels = [row["status"] for row in application_rows]
    status_texts = [row.get("text") or "" for row in application_rows]
    regex_status_labels: list[str] | None = None
    status_held_out: list[str] | None = None
    status_model = None
    if len(set(status_labels)) >= 2 and len(application_rows) >= 4:
        regex_status_labels = [regex_status(row) for row in application_rows]
        print_scores("Regex status compared with your labels", status_labels, regex_status_labels)
        candidate = make_pipeline()
        status_held_out = cross_validated_predictions(candidate, status_texts, status_labels)
        if status_held_out is None:
            print()
            print("Too few emails in one status to cross-check it. Scrapes will keep the keyword status rules.")
        else:
            print_scores("TF-IDF model, held-out status", status_labels, status_held_out)
            model_accuracy = accuracy_score(status_labels, status_held_out)
            regex_accuracy = accuracy_score(status_labels, regex_status_labels)
            if model_accuracy > regex_accuracy:
                candidate.fit(status_texts, status_labels)
                status_model = candidate
                print("Status model beat the keyword rules, so scrapes will use it.")
            else:
                print("Keyword status rules scored higher, so scrapes keep those.")
                print("The model is still used to drop mail that is not an application.")
    else:
        print()
        print("Not enough labeled statuses to train a status model. Application filtering will still be saved.")

    application_model.fit(texts, app_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "application_model": application_model,
            "status_model": status_model,
            "application_statuses": list(APPLICATION_STATUSES),
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
        },
        model_path,
    )
    print(f"Saved model: {model_path}")
    print("Later scrapes use this file instead of the regex filter and status rules.")
    charts = save_performance_charts(
        chart_dir,
        application_labels=app_labels,
        application_regex=regex_labels,
        application_model=held_out,
        status_labels=status_labels if status_held_out else None,
        status_regex=regex_status_labels if status_held_out else None,
        status_model=status_held_out,
    )
    for chart in charts:
        print(f"Chart: {chart}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a personal TF-IDF email classifier.")
    parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED, help="JSONL written by prepare_dataset.py.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Where to write the gitignored model.")
    parser.add_argument(
        "--charts",
        type=Path,
        default=DEFAULT_CHARTS,
        help="Folder for performance PNGs. These scores are safe to commit; the model is not.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_labeled(args.labeled), args.model, args.charts)
