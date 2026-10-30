"""Load a local TF-IDF classifier and decide whether an email is an application.

The trained file is personal and stays out of git. Without it, the scraper keeps
using the regex rules in scrape_job_applications.py.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "email_classifier.joblib"

APPLICATION_STATUSES = ("applied", "rejected", "interview", "online assessment")
POSITIVE_LABEL = "application"
NEGATIVE_LABEL = "no"

_model_path = DEFAULT_MODEL
_model = None
_loaded = False


def set_model_path(path: Path) -> None:
    global _model_path, _model, _loaded
    _model_path = path
    _model = None
    _loaded = False


def message_text(record: dict) -> str:
    return " ".join(
        part.strip()
        for part in (
            record.get("subject") or "",
            record.get("snippet") or "",
            record.get("body") or "",
        )
        if part and part.strip()
    )


def get_model():
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    if not _model_path.is_file():
        _model = None
        return None
    try:
        import joblib
    except ImportError:
        print("scikit-learn is not installed, so the regex rules will be used.")
        print("  pip install -r requirements.txt")
        _model = None
        return None
    _model = joblib.load(_model_path)
    return _model


def predict(model, record: dict) -> dict:
    """Return label application or no, plus a status when the email is kept."""
    text = message_text(record)
    application_model = model["application_model"]
    label = application_model.predict([text])[0]
    result = {"label": label, "status": ""}
    if label != POSITIVE_LABEL:
        return result
    status_model = model.get("status_model")
    if status_model is not None:
        result["status"] = status_model.predict([text])[0]
    return result
