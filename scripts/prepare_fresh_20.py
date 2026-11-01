"""Prepare a fresh private sample without printing complaint text or identifiers."""
from __future__ import annotations

import csv
import argparse
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

from prepare_private_pilot import ARCHIVE, REDACTIONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/private/fresh_taxonomy_v2_20.jsonl"
DEFAULT_MANIFEST = ROOT / "data/private/fresh_taxonomy_v2_20_manifest.json"
DEFAULT_SEED = "taxonomy-v2-fresh-20-20261030"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--exclude-file", action="append", default=[])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--case-prefix", default="fresh")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    old_ids = {
        json.loads(line)["source_complaint_id"]
        for line in (ROOT / "data/private/development_pilot_20.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for exclude_file in args.exclude_file:
        exclude_path = Path(exclude_file).resolve()
        old_ids.update(
            json.loads(line)["source_complaint_id"]
            for line in exclude_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    candidates = []
    csv_name = ""
    with zipfile.ZipFile(ARCHIVE) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                complaint_id = (row.get("Complaint ID") or "").strip()
                narrative = (row.get("Consumer complaint narrative") or "").strip()
                if row.get("Product") == "Credit card" and narrative and complaint_id not in old_ids:
                    candidates.append((complaint_id, row))
    rng = random.Random(args.seed)
    if args.count < 1 or args.count > len(candidates):
        raise SystemExit("Requested count is outside the eligible candidate range.")
    selected = rng.sample(candidates, args.count)
    records = []
    redaction_counts = {}
    for index, (complaint_id, row) in enumerate(selected, 1):
        text = (row.get("Consumer complaint narrative") or "").strip()
        count = 0
        for pattern, replacement in REDACTIONS:
            text, matches = pattern.subn(replacement, text)
            count += matches
        case_id = f"{args.case_prefix}-{index:02d}"
        redaction_counts[case_id] = count
        records.append({
            "case_id": case_id,
            "source_complaint_id": complaint_id,
            "source_date_received": row.get("Date received", ""),
            "source_issue": row.get("Issue", ""),
            "source_sub_issue": row.get("Sub-issue", ""),
            "cfpb_issue_as_reference_only": row.get("Issue", ""),
            "narrative_redacted": text,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    manifest = {
        "dataset": "fresh_taxonomy_v2_20",
        "source": "Consumer Financial Protection Bureau January-February 2025 export",
        "archive_file": "data/raw/cfpb-jan-feb-2025.zip",
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "csv_member": csv_name,
        "record_count": args.count,
        "selection_seed": args.seed,
        "selection": f"Random sample of {args.count} narrative-bearing credit-card records after excluding all source IDs in the supplied exclusion files.",
        "use": "Fresh taxonomy-v2 development check; not held-out evaluation and not training data.",
        "privacy_review": "Automated redaction patterns applied; manual review is required before external transmission. This does not guarantee anonymization.",
        "redactions_by_case": redaction_counts,
        "external_model_requests_authorized": 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} fresh records from {len(candidates)} eligible candidates.")
    print(f"Selection seed: {args.seed}; records written to ignored private storage.")
    print("Narrative text and source complaint IDs were not printed.")


if __name__ == "__main__":
    main()
