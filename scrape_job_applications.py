"""Scrape job-application emails from Gmail after a manual Google sign-in.

The script never types your password. It opens a real Chrome window, waits until
you finish signing in, and only then opens Gmail and reads matching messages.

Setup:
    pip install -r requirements.txt
    python scrape_job_applications.py

Google usually blocks Playwright's bundled Chromium at the login page, so this
uses installed Google Chrome when it is available. The signed-in session is
stored in .gmail-profile/ and reused on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from email_classifier import get_model, predict, set_model_path

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / ".gmail-profile"
DEFAULT_OUTPUT = ROOT / "output"

LOGIN_URL = (
    "https://accounts.google.com/ServiceLogin?hl=en"
    "&continue=https%3A%2F%2Fmyaccount.google.com%2F"
)

DEFAULT_QUERY = "(thank OR thanks) (application OR applying)"

DATE_FILTER = re.compile(r"\b(?:newer_than|older_than|after|before):", re.I)

ATS_DOMAIN_SUFFIXES = (
    "greenhouse.io",
    "greenhouse-mail.io",
    "lever.co",
    "myworkday.com",
    "smartrecruiters.com",
    "icims.com",
    "ashbyhq.com",
    "workable.com",
    "workablemail.com",
    "jobvite.com",
    "joinhandshake.com",
    "linkedin.com",
    "gmail.com",
    "google.com",
    "googlemail.com",
)

GENERIC_LOCAL_PART = re.compile(
    r"^(no[- ]?reply|noreply|notifications?|jobs?|careers?|talent|"
    r"recruiting|hiring|hr|mailer|info|hello|donotreply)$",
    re.I,
)

GENERIC_NAME = re.compile(
    r"\b(recruiting|talent acquisition|talent|careers|hiring|team|jobs|"
    r"no[- ]?reply|notifications?)\b",
    re.I,
)

SUBJECT_PATTERNS = [
    re.compile(r"^(?P<role>.+?)\s+@\s+(?P<company>.+)$", re.I),
    re.compile(
        r"application sent to (?P<company>.+?)(?:\s+[—–-]\s+here'?s what'?s next)?$",
        re.I,
    ),
    re.compile(
        r"(?:thank you|thanks) from (?P<company>.+?)\s+[-–—]\s+(?P<role>.+)$",
        re.I,
    ),
    re.compile(
        r"(?:thank you|thanks) for applying (?:to|for) (?:the )?"
        r"(?P<role>.+?) (?:role |position |job )?at (?P<company>.+)$",
        re.I,
    ),
    re.compile(
        r"(?:your )?application (?:for|to) (?:the )?"
        r"(?P<role>.+?) (?:role |position )?at (?P<company>.+)$",
        re.I,
    ),
    re.compile(
        r"(?:update on|regarding) your (?:job )?application (?:to|at|with) "
        r"(?P<company>.+)$",
        re.I,
    ),
    re.compile(r"your application (?:to|at|with) (?P<company>.+)$", re.I),
    re.compile(
        r"(?:thank you|thanks) for (?:your )?apply(?:ing)? to (?P<company>.+)$",
        re.I,
    ),
    re.compile(
        r"^(?P<company>.+?)\s*(?::| - | – | — | \|)\s*"
        r"(?:thank you for applying|thanks for applying|"
        r"application (?:received|update)|your application)",
        re.I,
    ),
]

COMPANY_TAIL = re.compile(
    r"\s+[-–—|:]\s+(?:next steps|application|interview|update|"
    r"thank you|thanks for|your application).*$",
    re.I,
)

TRAILING_ATS = re.compile(
    r"\s*(?:\||-|–|—)?\s*(?:via\s+)?(?:"
    r"greenhouse|lever|workday|smartrecruiters|icims|ashby|workable|jobvite|"
    r"linkedin).*$",
    re.I,
)

STATUS_PATTERNS = [
    (
        "rejected",
        [
            r"\bunfortunately\b",
            r"\bnot (?:be )?moving forward\b",
            r"\bother candidates\b",
            r"\bnot selected\b",
        ],
    ),
    (
        "online assessment",
        [
            r"\bassessment\b",
        ],
    ),
    (
        "interview",
        [
            r"\binterview\b",
            r"\bon-?site\b",
        ],
    ),
    (
        "applied",
        [
            r"\bapplication (?:has been |was )?received\b",
            r"\bthank you for applying\b",
            r"\bthanks for applying\b",
            r"\breceived your application\b",
            r"\bthanks for your interest\b",
            r"\bthank you for your interest\b",
        ],
    ),
]

CSV_FIELDS = ["date", "company", "role", "status"]


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" \t-–—|:.")
    value = TRAILING_ATS.sub("", value).strip(" \t-–—|:.")
    return value


def clean_company(value: str) -> str:
    value = COMPANY_TAIL.sub("", clean_text(value)).strip()
    value = re.sub(r"\s+and more$", "", value, flags=re.I).strip(" !")
    if "@" in value:
        value = value.split("@", 1)[0].strip(" -–—|:")
        if value.islower():
            value = value.title()
    if len(value) > 80:
        return ""
    if value.lower() == "ibm":
        return "IBM"
    return value


def clean_role(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"\b(role|position|job|opening)$", "", value, flags=re.I).strip()
    if len(value) > 140:
        return ""
    return value


def strip_reply_prefix(subject: str) -> str:
    text = clean_text(subject)
    while True:
        updated = re.sub(r"^(re|fw|fwd)\s*:\s*", "", text, flags=re.I).strip()
        if updated == text:
            return text
        text = updated


def parse_subject(subject: str) -> dict[str, str]:
    text = strip_reply_prefix(subject)
    for pattern in SUBJECT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groupdict()
        return {
            "company": clean_company(groups.get("company") or ""),
            "role": clean_role(groups.get("role") or ""),
        }
    return {"company": "", "role": ""}


def looks_like_role(role: str) -> bool:
    if not role or not (2 < len(role) <= 140):
        return False
    if re.search(r"\b(thank|thanks|applying|application|interest|received|submit)\b", role, re.I):
        return False
    return bool(
        re.search(
            r"\b(engineer|engineering|developer|intern|internship|co-?op|analyst|scientist|designer|programmer)\b",
            role,
            re.I,
        )
    )


def role_between_company_and_position(company: str, text: str) -> str:
    """Pull the job title sitting between the company name and the word position or role."""
    patterns = []
    if company and len(company.strip()) >= 2:
        company_pattern = re.escape(company.strip())
        patterns.extend(
            [
                rf"{company_pattern}['\u2019]s\s+(?P<role>.+?)\s+(?:position|role)\b",
                rf"(?P<role>.+?)\s+(?:position|role)\s+(?:at|with)\s+{company_pattern}\b",
            ]
        )
    patterns.append(r"(?:for|in|to)\s+the\s+(?P<role>.+?)\s+(?:position|role)\b")
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            role = clean_role(re.sub(r"\s+", " ", match.group("role")))
            if looks_like_role(role):
                return role
    return ""


def company_from_sender(name: str, email: str) -> str:
    domain = email.split("@")[-1].lower() if "@" in (email or "") else ""
    if domain and not any(domain == suffix or domain.endswith("." + suffix) for suffix in ATS_DOMAIN_SUFFIXES):
        parts = domain.split(".")
        org = parts[-2] if len(parts) >= 2 else ""
        if org in {"co", "com", "net", "org", "mail", "email", "careers", "jobs", "recruiting"} and len(parts) >= 3:
            org = parts[-3]
        if org and org not in {"gmail", "google", "yahoo", "outlook", "hotmail", "live"}:
            return clean_company(org.replace("-", " ").title())

    display = GENERIC_NAME.sub("", clean_text(name))
    display = re.sub(r"\s+", " ", display).strip(" \t-–—|:")
    local = email.split("@")[0] if "@" in (email or "") else ""
    if display and not GENERIC_LOCAL_PART.match(display) and display.lower() != local.lower():
        return clean_company(display)
    return ""


def is_handshake_message(record: dict) -> bool:
    email = (record.get("sender_email") or "").lower()
    name = (record.get("sender_name") or "").lower()
    return "joinhandshake.com" in email or "handshake" in name


THANK_YOU_SUBJECT = re.compile(
    r"thank you for applying|thanks for applying|thank you for your application|"
    r"thanks for your application|application sent to",
    re.I,
)


def keep_handshake_message(record: dict) -> bool:
    if not is_handshake_message(record):
        return True
    return bool(THANK_YOU_SUBJECT.search(record.get("subject") or ""))


THANK_WORD = re.compile(r"\bthanks?\b", re.I)
APPLICATION_WORD = re.compile(r"\b(?:application|applying)\b", re.I)


def message_text(record: dict) -> str:
    return " ".join(
        [
            record.get("subject") or "",
            record.get("snippet") or "",
            record.get("body") or "",
        ]
    )


def passes_mail_check(record: dict) -> bool:
    text = message_text(record)
    return bool(THANK_WORD.search(text) and APPLICATION_WORD.search(text))


def assessment_rank(record: dict) -> tuple:
    subject = (record.get("subject") or "").lower()
    mentions_assessment = 0 if "assessment" in subject else 1
    return (mentions_assessment, record.get("date") or "9999-99-99")


def collapse_online_assessments(records: list[dict]) -> list[dict]:
    """Keep one online assessment per company, using the earliest notice."""
    grouped: dict[str, dict] = {}
    order: list[str] = []
    rest: list[dict] = []
    for record in records:
        if record.get("status") != "online assessment":
            rest.append(record)
            continue
        company = re.sub(r"[^a-z0-9]+", "", (record.get("company") or "").lower())
        key = company or record.get("thread_id") or record.get("subject") or ""
        current = grouped.get(key)
        if current is None or assessment_rank(record) < assessment_rank(current):
            grouped[key] = record
            if current is None:
                order.append(key)
    kept = rest + [grouped[key] for key in order]
    kept.sort(key=lambda record: record.get("date") or "", reverse=True)
    return kept


def normalize_date(value: str) -> str:
    text = (value or "").replace("\u202f", " ").replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[A-Za-z]{3},\s+", "", text)
    for fmt in ("%b %d, %Y, %I:%M %p", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


def classify_status(*parts: str) -> str:
    text = "\n".join(part for part in parts if part)
    for status, patterns in STATUS_PATTERNS:
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            return status
    return "unknown"


def build_record(stub: dict, opened: dict | None = None, error: str = "") -> dict:
    opened = opened or {}
    subject = opened.get("subject") or stub.get("subject") or ""
    sender_name = opened.get("sender_name") or stub.get("sender_name") or ""
    sender_email = opened.get("sender_email") or stub.get("sender_email") or ""
    body = (opened.get("body") or "").strip()
    if len(body) > 12000:
        body = body[:12000].rstrip() + "\n\n[truncated]"
    parsed = parse_subject(subject)
    company = parsed["company"] or company_from_sender(sender_name, sender_email)
    role = parsed["role"] or role_between_company_and_position(company, f"{subject}\n{body}")
    snippet = clean_text(stub.get("snippet") or "")
    return {
        "thread_id": stub.get("thread_id") or "",
        "date": normalize_date(opened.get("date") or stub.get("date") or ""),
        "sender_name": clean_text(sender_name),
        "sender_email": (sender_email or "").strip(),
        "subject": clean_text(subject),
        "snippet": snippet,
        "company": company,
        "role": role,
        "status": classify_status(subject, snippet, body[:4000]),
        "body": body,
        "error": error,
    }


def session_is_ready(url: str) -> bool:
    lowered = (url or "").lower()
    pending = ("signin", "servicelogin", "challenge", "identifier", "accountchooser", "webreauth")
    signed_in_host = "myaccount.google.com" in lowered or "mail.google.com/mail" in lowered
    return signed_in_host and not any(token in lowered for token in pending)


def prompt_manual_login(page, timeout_s: int) -> None:
    print()
    print("Sign in to your Google account in the browser window.")
    print("The script does not type your password or 2-step verification codes.")
    print("Finish any prompts Google shows. Gmail opens only after sign-in succeeds.")
    print()

    deadline = time.time() + timeout_s
    next_notice = time.time() + 20
    while time.time() < deadline:
        if page.is_closed():
            raise SystemExit("The browser window was closed before sign-in finished.")
        if session_is_ready(page.url):
            print("Google sign-in detected.")
            return
        if time.time() >= next_notice:
            print("Still waiting for you to finish signing in...")
            next_notice = time.time() + 20
        page.wait_for_timeout(500)

    raise SystemExit(
        "Timed out waiting for Google sign-in. Run the script again and complete login in the browser."
    )


def confirm_existing_session(page) -> None:
    print()
    print("This browser profile is already signed in to Google.")
    print("Press Enter to use it, or switch accounts in the browser window first.")
    if sys.stdin.isatty():
        try:
            input()
        except EOFError:
            pass
    if page.is_closed():
        raise SystemExit("The browser window was closed.")
    if not session_is_ready(page.url):
        print("The open page is no longer a signed-in Google page. Waiting for sign-in.")
        prompt_manual_login(page, timeout_s=600)


def wait_for_mailbox(page, timeout_s: int) -> None:
    print("Opening Gmail.")
    deadline = time.time() + timeout_s
    told_user = False
    while time.time() < deadline:
        if page.is_closed():
            raise SystemExit("The browser window was closed before Gmail finished loading.")
        search = page.locator('input[name="q"], input[aria-label="Search mail"]')
        if search.count() and search.first.is_visible():
            return
        if page.locator("tr.zA").count():
            return
        if not told_user:
            print("If Gmail shows a setup or security dialog, finish it in the browser.")
            told_user = True
        page.wait_for_timeout(500)
    raise SystemExit(
        "Gmail opened, but the message list never appeared. "
        "Set Gmail's display language to English and try again."
    )


def search_has_no_results(page) -> bool:
    return page.locator("text=No messages matched your search").count() > 0


THREAD_LIST_JS = """
function decodeLog(part) {
  try {
    const pad = part + '='.repeat((4 - (part.length % 4)) % 4);
    return atob(pad);
  } catch (error) {
    return '';
  }
}
function threadId(row) {
  const legacy = row.getAttribute('data-legacy-thread-id');
  if (legacy) return legacy;
  const jslog = row.getAttribute('jslog') || '';
  const parts = jslog.match(/[A-Za-z0-9+/]{16,}={0,2}/g) || [];
  for (const part of parts) {
    const decoded = decodeLog(part);
    const legacyThread = decoded.match(/thread-f:\\d+/);
    if (legacyThread) return legacyThread[0];
    const modern = decoded.match(/FMfcg[A-Za-z0-9_-]+/);
    if (modern) return modern[0];
  }
  return '';
}
function visibleRows() {
  return Array.from(document.querySelectorAll('tr.zA')).filter((row) => row.getClientRects().length > 0);
}
"""


def visible_thread_ids(page) -> list[str]:
    return page.evaluate(
        "() => {\n"
        + THREAD_LIST_JS
        + "\nreturn visibleRows().map((row) => threadId(row)).filter(Boolean);\n}"
    )


def run_gmail_search(page, account: int, query: str) -> None:
    page.goto(
        f"https://mail.google.com/mail/u/{account}/#inbox",
        wait_until="domcontentloaded",
    )
    wait_for_mailbox(page, timeout_s=180)
    page.wait_for_selector("tr.zA", timeout=60000)
    before = visible_thread_ids(page)
    search = page.locator('input[aria-label="Search mail"]').first
    search.click()
    page.keyboard.press("Control+A")
    page.keyboard.insert_text(query)
    page.keyboard.press("Enter")

    deadline = time.time() + 30
    while time.time() < deadline:
        if search_has_no_results(page):
            return
        current = visible_thread_ids(page)
        if current and current != before:
            page.wait_for_timeout(800)
            return
        page.wait_for_timeout(400)
    print("Search results did not refresh. Reading the conversations that are on screen.")


def collect_visible_threads(page) -> list[dict]:
    return page.evaluate(
        "() => {\n"
        + THREAD_LIST_JS
        + """
        return visibleRows().map((row) => {
            const senderEl = row.querySelector('span.yP, span.zF, span[email]');
            const subjectEl = row.querySelector('span.bog');
            const snippetEl = row.querySelector('span.y2');
            const dateEl = row.querySelector('td.xW span[title], td.xW span');
            return {
                thread_id: threadId(row),
                sender_name: senderEl ? (senderEl.getAttribute('name') || senderEl.innerText || '') : '',
                sender_email: senderEl ? (senderEl.getAttribute('email') || '') : '',
                subject: subjectEl ? subjectEl.innerText || '' : '',
                snippet: snippetEl ? snippetEl.innerText || '' : '',
                date: dateEl ? (dateEl.getAttribute('title') || dateEl.innerText || '') : '',
            };
        }).filter((row) => row.thread_id || row.subject);
        }"""
    )


def go_to_older_page(page) -> bool:
    previous = page.evaluate(
        """() => {
            const rows = Array.from(document.querySelectorAll('tr.zA'))
                .filter((row) => row.getClientRects().length > 0);
            const subject = rows[0] ? rows[0].querySelector('span.bog') : null;
            return subject ? subject.innerText || '' : '';
        }"""
    )
    if not previous:
        return False
    button = page.locator('[aria-label="Older"]:visible')
    if button.count() == 0:
        return False
    button = button.last
    if (button.get_attribute("aria-disabled") or "").lower() == "true":
        return False
    try:
        button.click(timeout=5000)
    except Exception:
        return False
    try:
        page.wait_for_function(
            """(prev) => {
                const rows = Array.from(document.querySelectorAll('tr.zA'))
                    .filter((row) => row.getClientRects().length > 0);
                const subject = rows[0] ? rows[0].querySelector('span.bog') : null;
                const text = subject ? subject.innerText || '' : '';
                return text && text !== prev;
            }""",
            arg=previous,
            timeout=15000,
        )
    except Exception:
        return False
    return True


def collect_search_results(page, limit: int | None) -> list[dict]:
    deadline = time.time() + 20
    while time.time() < deadline and not collect_visible_threads(page) and not search_has_no_results(page):
        page.wait_for_timeout(400)

    found: list[dict] = []
    seen: set[str] = set()
    while True:
        batch = collect_visible_threads(page)
        added = 0
        for row in batch:
            key = row.get("thread_id") or row.get("subject")
            if not key or key in seen:
                continue
            seen.add(key)
            found.append(row)
            added += 1
            if limit is not None and len(found) >= limit:
                break
        if (limit is not None and len(found) >= limit) or added == 0 or not go_to_older_page(page):
            break
    return found if limit is None else found[:limit]


def read_open_thread(page) -> dict:
    expand = page.locator('[aria-label="Expand all"], [data-tooltip="Expand all"]')
    if expand.count():
        expand.first.click()
        page.wait_for_timeout(400)
    return page.evaluate(
        """() => {
            const senderEl = document.querySelector('span.gD');
            const dateEl = document.querySelector('span.g3');
            const bodies = Array.from(document.querySelectorAll('div.a3s')).map((el) => el.innerText || '');
            return {
                subject: (document.querySelector('h2.hP') || {}).innerText || '',
                sender_name: senderEl ? (senderEl.getAttribute('name') || senderEl.innerText || '') : '',
                sender_email: senderEl ? (senderEl.getAttribute('email') || '') : '',
                date: dateEl ? (dateEl.getAttribute('title') || dateEl.innerText || '') : '',
                body: bodies.join('\\n\\n'),
            };
        }"""
    )


def open_thread(page, account: int, thread_id: str) -> dict:
    page.goto(
        f"https://mail.google.com/mail/u/{account}/#inbox/{thread_id}",
        wait_until="domcontentloaded",
    )
    page.wait_for_selector("h2.hP", timeout=20000)
    return read_open_thread(page)


def scrape_messages(
    page,
    account: int,
    stubs: list[dict],
    pause_s: float,
    list_only: bool,
    records: list[dict],
    on_progress=None,
) -> None:
    total = len(stubs)
    for index, stub in enumerate(stubs, start=1):
        subject = clean_text(stub.get("subject") or "") or "(no subject)"
        if not keep_handshake_message(stub):
            print(f"[{index}/{total}] skipping Handshake mail: {subject}")
            continue
        print(f"[{index}/{total}] {subject}")
        opened = None
        error = ""
        thread_id = stub.get("thread_id") or ""
        if not list_only and thread_id:
            try:
                opened = open_thread(page, account, thread_id)
            except Exception as exc:
                error = str(exc).splitlines()[0]
                print(f"    Could not read this message: {error}")
            if pause_s and not page.is_closed():
                page.wait_for_timeout(int(pause_s * 1000))
        records.append(build_record(stub, opened, error))
        if not accept_record(records[-1]):
            records.pop()
            continue
        if on_progress and index % 10 == 0:
            on_progress(records)


def accept_record(record: dict) -> bool:
    """Keep an email. A trained model replaces the regex filter and status rules."""
    model = get_model()
    if model is None:
        if not passes_mail_check(record):
            print("    skipped, message does not say thank and application or applying")
            return False
        return True
    prediction = predict(model, record)
    if prediction["label"] != "application":
        print("    skipped, trained model says this is not an application")
        return False
    if prediction.get("status"):
        record["status"] = prediction["status"]
    return True


def write_outputs(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "job_applications.json"
    csv_path = output_dir / "job_applications.csv"
    deduped: list[dict] = []
    seen: set[str] = set()
    for record in records:
        if not keep_handshake_message(record):
            continue
        if get_model() is None and not passes_mail_check(record):
            continue
        key = record.get("thread_id") or "|".join(
            [record.get("date") or "", record.get("company") or "", record.get("role") or "", record.get("subject") or ""]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    deduped = collapse_online_assessments(deduped)
    json_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)
    return json_path, csv_path


def launch_browser(playwright, profile_root: Path):
    attempts = [("chrome", "chrome"), ("msedge", "msedge"), ("chromium", None)]
    errors: list[str] = []
    for folder, channel in attempts:
        kwargs = {
            "user_data_dir": str(profile_root / folder),
            "headless": False,
            "viewport": {"width": 1440, "height": 960},
            "locale": "en-US",
            "ignore_default_args": ["--enable-automation"],
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if channel:
            kwargs["channel"] = channel
        try:
            context = playwright.chromium.launch_persistent_context(**kwargs)
            return context, folder
        except Exception as exc:
            errors.append(f"{folder}: {exc}")
    raise SystemExit(
        "Could not open a browser.\n"
        + "\n".join(errors)
        + "\nInstall Google Chrome, or run: python -m playwright install chromium"
    )


def limit_query_to_months(query: str, months: int) -> str:
    if DATE_FILTER.search(query):
        return query
    return f"{query} newer_than:{months}m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually sign in to Google, then scrape job-application emails from Gmail."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Stop after this many emails. Leave it off to read every match in the month window.",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="How many months back to search (default: 3). Example: --months 6",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Gmail search query. Replaces the default job search.")
    parser.add_argument("--account", type=int, default=0, help="Gmail account index in the URL, usually 0.")
    parser.add_argument("--pause", type=float, default=0.4, help="Seconds to wait between opening emails.")
    parser.add_argument("--login-timeout", type=int, default=600, help="Seconds to wait for manual sign-in.")
    parser.add_argument("--list-only", action="store_true", help="Save inbox rows without opening each email.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Directory for the signed-in browser profile.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for JSON and CSV results.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "email_classifier.joblib",
        help="Trained classifier. If this file is missing, regex rules are used.",
    )
    args = parser.parse_args()
    if args.max is not None and args.max < 1:
        parser.error("--max must be at least 1")
    if args.months < 1:
        parser.error("--months must be at least 1")
    args.query = limit_query_to_months(args.query, args.months)
    return args


def main() -> None:
    args = parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright is not installed.\n"
            "  pip install -r requirements.txt\n"
            "  python -m playwright install chromium"
        )

    set_model_path(args.model)
    if get_model() is None:
        print("No trained model found. Using the built-in keyword rules.")
    else:
        print(f"Using trained model: {args.model}")

    records: list[dict] = []
    with sync_playwright() as playwright:
        context, browser_name = launch_browser(playwright, args.profile)
        page = context.pages[0] if context.pages else context.new_page()
        print(f"Using {browser_name}. A browser window should be open.")
        try:
            page.goto(LOGIN_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            if session_is_ready(page.url):
                confirm_existing_session(page)
            else:
                prompt_manual_login(page, args.login_timeout)

            if args.max is None:
                print(f"Opening your email and reading every job match from the last {args.months} month(s).")
            else:
                print(f"Opening your email and reading up to {args.max} job matches from the last {args.months} month(s).")
            run_gmail_search(page, args.account, args.query)
            stubs = collect_search_results(page, args.max)
            if not stubs:
                print("No emails matched that search.")
                print("Try a broader query, for example:")
                print("  python scrape_job_applications.py --months 6")
                return

            print(f"Found {len(stubs)} matching conversation(s).")
            scrape_messages(
                page,
                args.account,
                stubs,
                args.pause,
                args.list_only,
                records,
                on_progress=lambda rows: write_outputs(rows, args.output),
            )
        finally:
            if records:
                json_path, csv_path = write_outputs(records, args.output)
                counts = Counter(record["status"] for record in records)
                print()
                print(f"Saved {len(records)} email(s).")
                for status, count in counts.most_common():
                    print(f"  {status}: {count}")
                print(f"JSON: {json_path}")
                print(f"CSV:  {csv_path}")
            context.close()


if __name__ == "__main__":
    main()
