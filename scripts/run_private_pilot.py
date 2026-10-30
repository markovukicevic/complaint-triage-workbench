"""Make the five approved, sequential OpenRouter pilot requests once each."""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_service import analyze_with_model
from triage import queue_for, validate_grounding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "data/private/development_pilot_20.jsonl"
OUTPUT = ROOT / "data/private/pilot_results.json"
PILOT_IDS = ["11477365", "11464372", "11904528", "11468604", "11349165"]
load_dotenv(ROOT / ".env")


def main() -> None:
    records = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {item["source_complaint_id"]: item for item in records}
    if any(cid not in by_id for cid in PILOT_IDS):
        raise SystemExit("A selected pilot case is missing from the prepared dataset.")
    results: list[dict] = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for index, complaint_id in enumerate(PILOT_IDS, 1):
        item = by_id[complaint_id]
        started = time.perf_counter()
        try:
            analysis = analyze_with_model(item["narrative_redacted"])
            analysis, grounding_problems = validate_grounding(analysis, item["narrative_redacted"])
            result = {
                "case_id": item["case_id"],
                "reference_category": item["reviewed_target_category"],
                "reference_note": item["pilot_review_note"],
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
        except Exception as exc:
            # Provider exceptions can contain request details. Persist only their class.
            result = {
                "case_id": item["case_id"],
                "reference_category": item["reviewed_target_category"],
                "request_status": "failed",
                "error_type": type(exc).__name__,
                "latency_seconds": round(time.perf_counter() - started, 2),
            }
        results.append(result)
        OUTPUT.write_text(json.dumps({"model_id": "z-ai/glm-5.3-flash", "provider": "OpenRouter", "pilot_attempts": len(results), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Pilot request {index}/5: case {item['case_id']} — {result['request_status']}; {result['latency_seconds']} seconds")
    print(f"Private pilot results saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
