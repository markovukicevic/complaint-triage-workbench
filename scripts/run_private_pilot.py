"""Run one to three explicitly selected OpenRouter pilot requests, without retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_service import ModelCallError, analyze_with_diagnostics, configuration_snapshot, PROMPT_VERSION
from triage import queue_for, validate_grounding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "data/private/development_pilot_20.jsonl"
REFERENCE = ROOT / "data/private/evaluation-reference-v1.jsonl"
APPROVED_CASE_IDS = {"pilot-01", "pilot-02", "pilot-03", "pilot-04", "pilot-05", "pilot-06", "pilot-07", "pilot-08", "pilot-09", "pilot-10", "pilot-11", "pilot-12", "pilot-13", "pilot-14", "pilot-15", "pilot-16", "pilot-17", "pilot-18", "pilot-19", "pilot-20"}
load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicitly selected private pilot cases once each.")
    parser.add_argument("--case-ids", nargs="+", required=True, choices=sorted(APPROVED_CASE_IDS))
    parser.add_argument("--output", required=True, help="New output path under data/private/")
    parser.add_argument("--confirm-dataset-sha256", required=True, help="Expected input dataset fingerprint")
    parser.add_argument("--confirm-reference-sha256", required=True, help="Expected approved reference fingerprint")
    parser.add_argument("--prompt-version", choices=("triage-v1", "triage-v2"), default=PROMPT_VERSION)
    args = parser.parse_args()
    if len(args.case_ids) > 3 or len(set(args.case_ids)) != len(args.case_ids):
        raise SystemExit("Select one to three distinct cases; requests are never retried.")
    output = (ROOT / args.output).resolve()
    private_root = (ROOT / "data/private").resolve()
    if private_root not in output.parents or output.exists():
        raise SystemExit("Output must be a new file under data/private/.")
    dataset_bytes = INPUT.read_bytes()
    reference_bytes = REFERENCE.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    if args.confirm_dataset_sha256 != dataset_hash:
        raise SystemExit("Dataset fingerprint does not match; review the current prepared data before running.")
    if args.confirm_reference_sha256 != reference_hash:
        raise SystemExit("Reference fingerprint does not match; review the current annotations before running.")
    records = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    by_id = {item["case_id"]: item for item in records}
    reference_rows = [json.loads(line) for line in reference_bytes.decode("utf-8").splitlines() if line.strip()]
    references = {item["case_id"]: item for item in reference_rows}
    if len(by_id) != 20 or len(records) != 20 or len(references) != 20 or len(reference_rows) != 20:
        raise SystemExit("The standardized evaluation pack must contain 20 unique data and reference rows.")
    if set(by_id) != set(references):
        raise SystemExit("Dataset and reference case IDs do not match; no requests were sent.")
    if any(row.get("annotation_status") != "approved_by_marko" for row in reference_rows):
        raise SystemExit("All reference annotations must be reviewed and marked approved_by_marko before running.")
    if any(cid not in by_id for cid in args.case_ids):
        raise SystemExit("A selected pilot case is missing from the prepared dataset.")
    results: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, case_id in enumerate(args.case_ids, 1):
        item = by_id[case_id]
        started = time.perf_counter()
        try:
            response = analyze_with_diagnostics(item["narrative_redacted"], prompt_version=args.prompt_version)
            analysis, grounding_problems = validate_grounding(response.analysis, item["narrative_redacted"])
            result = {
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "case_id": item["case_id"],
                "reference_category_draft": references[case_id]["reference_category_draft"],
                "reference_annotation_status": references[case_id]["annotation_status"],
                "model_category": analysis.issue_category.value,
                "queue": queue_for(analysis.issue_category),
                "summary": analysis.summary,
                "requested_outcome": analysis.requested_outcome,
                "key_facts": analysis.key_facts,
                "missing_information": analysis.missing_information,
                "evidence": analysis.evidence,
                "uncertainty_reason": analysis.uncertainty_reason,
                "grounding_problems": grounding_problems,
                "latency_seconds": round(time.perf_counter() - started, 2),
                "request_status": "success",
            }
        except ModelCallError as exc:
            # Persist only allowlisted diagnostics. Never serialize provider or validation text.
            result = {
                "case_id": item["case_id"],
                "reference_category_draft": references[case_id]["reference_category_draft"],
                "reference_annotation_status": references[case_id]["annotation_status"],
                "request_status": "failed",
                "error_code": exc.code,
                "finish_reason": exc.finish_reason,
                "validation_issues": list(exc.validation_issues),
                "provider_status": exc.provider_status,
                "usage": exc.usage,
                "latency_seconds": round(time.perf_counter() - started, 2),
            }
        except Exception:
            # Unexpected failures are deliberately reduced to a fixed label.
            result = {
                "case_id": item["case_id"],
                "reference_category_draft": references[case_id]["reference_category_draft"],
                "reference_annotation_status": references[case_id]["annotation_status"],
                "request_status": "failed",
                "error_code": "unexpected_local_error",
                "latency_seconds": round(time.perf_counter() - started, 2),
            }
        results.append(result)
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        run_metadata = {
            "configuration": configuration_snapshot(args.prompt_version),
            "dataset_sha256": dataset_hash,
            "reference_sha256": reference_hash,
            "source_revision": revision_result.stdout.strip() if revision_result.returncode == 0 else None,
            "source_branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
            "working_tree_dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
            "python_version": sys.version.split()[0],
            "reference_statuses": sorted({row["annotation_status"] for row in reference_rows}),
        }
        output.write_text(json.dumps({"run_metadata": run_metadata, "pilot_attempts": len(results), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Pilot request {index}/{len(args.case_ids)}: case {item['case_id']} — {result['request_status']}; {result['latency_seconds']} seconds")
    print(f"Private pilot results saved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
