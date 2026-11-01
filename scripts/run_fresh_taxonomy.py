"""Run explicitly selected fresh private cases once each; never retries or overwrites."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_service import ModelCallError, analyze_with_diagnostics, configuration_snapshot
from triage import assess_analysis

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DEFAULT_INPUT = ROOT / "data/private/fresh_taxonomy_v2_20.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--prompt-version", default="taxonomy-v2a", choices=("taxonomy-v2a", "taxonomy-v2b", "taxonomy-v2c"))
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-dataset-sha256", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    if len(args.case_ids) > 3 or len(set(args.case_ids)) != len(args.case_ids):
        raise SystemExit("Select one to three distinct cases; requests are never retried.")
    output = (ROOT / args.output).resolve()
    private_root = (ROOT / "data/private").resolve()
    if private_root not in output.parents or output.exists():
        raise SystemExit("Output must be a new file under data/private/.")
    dataset_bytes = input_path.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_hash != args.confirm_dataset_sha256:
        raise SystemExit("Fresh dataset fingerprint does not match.")
    records = [json.loads(line) for line in dataset_bytes.decode().splitlines() if line.strip()]
    by_id = {item["case_id"]: item for item in records}
    if len(records) < 1 or len(by_id) != len(records) or not set(args.case_ids).issubset(by_id):
        raise SystemExit("Input dataset must contain unique case IDs and all selected cases.")
    results = []
    for case_id in args.case_ids:
        item = by_id[case_id]
        started = time.perf_counter()
        try:
            response = analyze_with_diagnostics(item["narrative_redacted"], prompt_version=args.prompt_version)
            decision = assess_analysis(response.analysis, item["narrative_redacted"])
            result = {
                "result_schema_version": 2,
                "case_id": case_id,
                "request_status": "success",
                "configuration": configuration_snapshot(args.prompt_version),
                "model_category": decision.model_analysis.issue_category.value,
                "evidence_validation": asdict(decision.evidence_validation),
                "final_system_category": decision.final_analysis.issue_category.value,
                "final_queue": decision.final_queue,
                "summary": decision.final_analysis.summary,
                "requested_outcome": decision.final_analysis.requested_outcome,
                "key_facts": decision.final_analysis.key_facts,
                "missing_information": decision.final_analysis.missing_information,
                "evidence": decision.final_analysis.evidence,
                "uncertainty_reason": decision.final_analysis.uncertainty_reason,
                "grounding_problems": list(decision.grounding_problems),
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "latency_seconds": round(time.perf_counter() - started, 2),
            }
        except ModelCallError as exc:
            result = {
                "result_schema_version": 2,
                "case_id": case_id,
                "request_status": "failed",
                "configuration": configuration_snapshot(args.prompt_version),
                "error_code": exc.code,
                "finish_reason": exc.finish_reason,
                "validation_issues": list(exc.validation_issues),
                "provider_status": exc.provider_status,
                "usage": exc.usage,
                "latency_seconds": round(time.perf_counter() - started, 2),
            }
        results.append(result)
        print(f"{case_id}: {result['request_status']}")
    output.write_text(json.dumps({"dataset_sha256": dataset_hash, "prompt_version": args.prompt_version, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(results)} results to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
