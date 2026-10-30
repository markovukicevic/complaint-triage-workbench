"""Prepare the reviewed Consumer Financial Protection Bureau discovery sample locally.

The raw archive and generated records are private local data. Generated output is
written under data/private/, which is excluded by .gitignore.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/cfpb-jan-feb-2025.zip"
OUTPUT = ROOT / "data/private/development_pilot_20.jsonl"
MANIFEST = ROOT / "data/private/development_pilot_manifest.json"

SOURCE_GROUPS = {
    "Problem with a purchase shown on your statement": "Purchase dispute or unauthorized transaction",
    "Fees or interest": "Fees or interest",
    "Closing your account": "Account closure or cancellation",
    "Problem when making payments": "Payment or billing process",
    "Incorrect information on your report": "Credit-report information accuracy",
}
REVIEWED_IDS_BY_ISSUE = {
    "Problem with a purchase shown on your statement": ["12065504", "11477365", "11780324", "11346692"],
    "Fees or interest": ["12124830", "11464372", "11772617", "11729141"],
    "Closing your account": ["11839004", "11502381", "12083282", "11904528"],
    "Problem when making payments": ["11576673", "11468604", "11577001", "11561050"],
    "Incorrect information on your report": ["12073383", "12190306", "11349165", "11894340"],
}
# Independent human review of the five paid pilot cases, completed before model calls.
PILOT_CASES = {
    "11477365": ("Purchase dispute or unauthorized transaction", "Clear unauthorized account/purchase allegation."),
    "11464372": ("Fees or interest", "Clear disputed late fee and interest charge."),
    "11904528": ("Account closure or cancellation", "Clear account closure and withheld rewards complaint."),
    "11468604": ("Payment or billing process", "Autopay payments were returned and fees followed."),
    "11349165": ("Credit-report information accuracy", "Clear dispute about inaccurate account-status reporting."),
}

REDACTIONS = [
    (re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"), "[REDACTED EMAIL]"),
    (re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), "[REDACTED PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED SSN]"),
    (re.compile(r"(?i)\b(?:account|card|case|claim|reference|confirmation)\s+(?:number|no\.?|#)\s*(?:(?:is|was)\s+)?[:#]?\s*[A-Z0-9-]{6,}\b"), "[REDACTED IDENTIFIER]"),
    (re.compile(r"\b\d{12,19}\b"), "[REDACTED LONG NUMBER]"),
    (re.compile(r"(?i)\b\d{1,6}\s+(?:[A-Z0-9.'-]+\s+){0,5}(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|court|ct\.?|way)\b[^\n,;]*"), "[REDACTED STREET ADDRESS]"),
]


def main() -> None:
    ids = {cid for group in REVIEWED_IDS_BY_ISSUE.values() for cid in group}
    found: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                complaint_id = (row.get("Complaint ID") or "").strip()
                if complaint_id in ids:
                    found[complaint_id] = row
    if set(found) != ids:
        raise RuntimeError(f"Sample mismatch: expected {len(ids)} source records, found {len(found)}")

    records = []
    redaction_counts: dict[str, int] = {}
    for source_issue, complaint_ids in REVIEWED_IDS_BY_ISSUE.items():
        for complaint_id in complaint_ids:
            row = found[complaint_id]
            if row.get("Product") != "Credit card" or not (row.get("Consumer complaint narrative") or "").strip():
                raise RuntimeError(f"Selected source record {complaint_id} is not a narrative-bearing credit-card row")
            text = row["Consumer complaint narrative"].strip()
            counts = 0
            for pattern, replacement in REDACTIONS:
                text, n = pattern.subn(replacement, text)
                counts += n
            redaction_counts[complaint_id] = counts
            record = {
                "case_id": f"pilot-{len(records) + 1:02d}",
                "source_complaint_id": complaint_id,
                "source_date_received": row.get("Date received", ""),
                "source_issue": source_issue,
                "source_sub_issue": row.get("Sub-issue", ""),
                "cfpb_issue_as_reference_only": SOURCE_GROUPS[source_issue],
                "reviewed_target_category": PILOT_CASES[complaint_id][0] if complaint_id in PILOT_CASES else None,
                "pilot_review_note": PILOT_CASES[complaint_id][1] if complaint_id in PILOT_CASES else None,
                "narrative_redacted": text,
            }
            records.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    archive_sha256 = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    manifest = {
        "dataset": "development_pilot_20",
        "source": "Consumer Financial Protection Bureau January-February 2025 export",
        "archive_file": "data/raw/cfpb-jan-feb-2025.zip",
        "archive_sha256": archive_sha256,
        "csv_member": csv_name,
        "record_count": len(records),
        "selection": "The 20 discovery records already reviewed in MVP-SPEC.md: four per each of five source Issue labels, selected by SHA-256 of triage-jan-feb-2025-v1: plus Complaint ID.",
        "use": "Private development pilot only; these records are not held-out evaluation data and not training data.",
        "privacy_review": "All 20 narratives were manually inspected. CFPB redaction placeholders were preserved. Automated patterns additionally redact email addresses, phone numbers, SSNs, explicit long identifiers, long digit sequences, and street addresses. Brand/company names and dates/amounts remain. This process does not guarantee anonymization.",
        "redactions_by_case": redaction_counts,
        "pilot_cases": PILOT_CASES,
        "model_requests_authorized": 5,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} records in ignored private storage.")
    print(f"Records with added pattern redactions: {sum(v > 0 for v in redaction_counts.values())}.")
    print(f"Pilot categories: {len(PILOT_CASES)}; source text lengths: {min(map(lambda r: len(r['narrative_redacted']), records))}-{max(map(lambda r: len(r['narrative_redacted']), records))} characters.")
    print("Narrative text and source complaint IDs were not printed.")


if __name__ == "__main__":
    main()
