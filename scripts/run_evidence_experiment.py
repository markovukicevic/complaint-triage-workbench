"""Run the evidence-only extractor for explicitly selected private cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_service import EvidenceCallError, extract_evidence
from triage import IssueCategory, inspect_evidence

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-dataset-sha256", required=True)
    args = parser.parse_args()
    input_path = (ROOT / args.input).resolve()
    output = (ROOT / args.output).resolve()
    private_root = (ROOT / "data/private").resolve()
    if output.exists() or private_root not in output.parents:
        raise SystemExit("Output must be a new file under data/private/.")
    dataset_bytes = input_path.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_hash != args.confirm_dataset_sha256:
        raise SystemExit("Dataset fingerprint does not match.")
    records = {item["case_id"]: item for item in (json.loads(line) for line in dataset_bytes.decode().splitlines() if line.strip())}
    baseline = {}
    for path in sorted(ROOT.glob(args.baseline_glob)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            if result.get("request_status") == "success":
                baseline[result["case_id"]] = result["model_category"]
    results = []
    for case_id in args.case_ids:
        if case_id not in records or case_id not in baseline:
            raise SystemExit(f"Missing input or baseline record: {case_id}")
        category = IssueCategory(baseline[case_id])
        try:
            response = extract_evidence(records[case_id]["narrative_redacted"], category)
            validation = inspect_evidence(
                type("Analysis", (), {"evidence": response.evidence})(),
                records[case_id]["narrative_redacted"],
            )
            results.append({
                "case_id": case_id,
                "request_status": "success",
                "selected_category": category.value,
                "evidence": response.evidence,
                "evidence_validation": asdict(validation),
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            })
        except EvidenceCallError as exc:
            results.append({
                "case_id": case_id,
                "request_status": "failed",
                "selected_category": category.value,
                "error_code": exc.code,
                "finish_reason": exc.finish_reason,
                "usage": exc.usage,
            })
        print(f"{case_id}: {results[-1]['request_status']}")
    output.write_text(json.dumps({"dataset_sha256": dataset_hash, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(results)} results to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
