from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from model_service import MODEL_ID, REASONING_EFFORT, REQUEST_TIMEOUT_SECONDS, _safe_finish_reason, _safe_usage
from triage import IssueCategory

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))


MIN_SPAN_CHARS = 24


def split_spans(text: str) -> list[str]:
    """Split text into exact spans while merging short punctuation fragments."""
    boundaries = [match for match in re.finditer(r"(?<=[.!?])\s+|\n+", text)]
    ranges: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        if text[start:boundary.start()].strip():
            ranges.append((start, boundary.start()))
        start = boundary.end()
    if text[start:].strip():
        ranges.append((start, len(text)))

    merged: list[tuple[int, int]] = []
    for span_start, span_end in ranges:
        if merged and span_end - span_start < MIN_SPAN_CHARS:
            previous_start, _ = merged.pop()
            merged.append((previous_start, span_end))
        else:
            merged.append((span_start, span_end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_SPAN_CHARS:
        _, next_end = merged[1]
        merged = [(merged[0][0], next_end), *merged[2:]]
    return [text[start:end].strip() for start, end in merged if text[start:end].strip()]


class SpanSelection(BaseModel):
    evidence_span_ids: list[int] = Field(max_length=3)


SPAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "complaint_evidence_spans",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"evidence_span_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}},
            "required": ["evidence_span_ids"],
        },
    },
}

SPAN_SYSTEM_PROMPT = """Select evidence spans for a human reviewer. The complaint has already been classified; do not classify or summarize it. Return exactly one JSON object with one field, evidence_span_ids.

Select one to three strongest numbered spans that directly support the supplied category. Do not select general background, legal conclusions, or a related issue that does not support the selected category. Do not invent IDs, rewrite text, or return quotations. If no span directly supports the category, return an empty array. Preserve the order of selected spans.
"""

CATEGORY_SUPPORT_RULES = {
    IssueCategory.PURCHASE_DISPUTE: "Select spans that state an unauthorized/disputed card transaction, purchase, misuse, or the handling of that specific dispute.",
    IssueCategory.FEES_INTEREST: "Select spans that state a fee, interest charge, interest rate, or interest calculation on the card.",
    IssueCategory.PAYMENT_BILLING: "Select spans that state payment posting, allocation, processing, billing, statement delivery, or refund/credit handling.",
    IssueCategory.ACCOUNT_CLOSURE: "Select spans that state account closure, cancellation, inability to close, reopening, or the stated reason for closure.",
    IssueCategory.CREDIT_REPORT: "Select spans that state inaccurate information reported to a credit bureau or on a credit report.",
    IssueCategory.MANUAL_REVIEW: "Select spans that show why the case is unclear, outside the credit-card scope, sparse, or has materially overlapping issues.",
}


@dataclass(frozen=True)
class SpanEvidenceResponse:
    evidence: list[str]
    selected_ids: list[int]
    span_count: int
    finish_reason: str
    usage: dict[str, int] | None


class SpanEvidenceCallError(Exception):
    def __init__(self, code: str, *, finish_reason: str = "unknown", usage: dict[str, int] | None = None):
        self.code = code
        self.finish_reason = finish_reason
        self.usage = usage
        super().__init__(code)


def extract_span_evidence(complaint: str, category: IssueCategory) -> SpanEvidenceResponse:
    spans = split_spans(complaint)
    api_key = os.getenv("OPENROUTER_GLM_API_KEY")
    if not api_key:
        raise SpanEvidenceCallError("missing_api_key")
    numbered = "\n".join(f"[{index}] {span}" for index, span in enumerate(spans))
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=float(REQUEST_TIMEOUT_SECONDS),
        max_retries=0,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Complaint Triage Workbench"},
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SPAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Selected category: {category.value}\nCategory support rule: {CATEGORY_SUPPORT_RULES[category]}\nNumbered complaint spans:\n{numbered}"},
            ],
            response_format=SPAN_RESPONSE_FORMAT,
            temperature=0,
            max_tokens=1024,
            extra_body={"reasoning": {"effort": REASONING_EFFORT}, "provider": {"require_parameters": True}},
        )
    except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
        raise SpanEvidenceCallError(f"provider_error_{getattr(exc, 'status_code', None) or 'unknown'}") from None
    except Exception:
        raise SpanEvidenceCallError("provider_error") from None
    choices = getattr(response, "choices", None)
    finish_reason = _safe_finish_reason(getattr(choices[0], "finish_reason", None)) if choices else "unknown"
    usage = _safe_usage(response)
    if not choices:
        raise SpanEvidenceCallError("no_choices", finish_reason=finish_reason, usage=usage)
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(content, str) or not content.strip():
        raise SpanEvidenceCallError("empty_content", finish_reason=finish_reason, usage=usage)
    try:
        selected = SpanSelection.model_validate_json(content).evidence_span_ids
    except ValidationError:
        raise SpanEvidenceCallError("schema_validation_error", finish_reason=finish_reason, usage=usage) from None
    if any(index < 0 or index >= len(spans) for index in selected):
        raise SpanEvidenceCallError("invalid_span_id", finish_reason=finish_reason, usage=usage)
    selected = [index for index in selected if len(spans[index]) >= MIN_SPAN_CHARS]
    return SpanEvidenceResponse(
        evidence=[spans[index] for index in selected],
        selected_ids=selected,
        span_count=len(spans),
        finish_reason=finish_reason,
        usage=usage,
    )
