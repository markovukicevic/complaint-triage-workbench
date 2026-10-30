from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    PURCHASE_DISPUTE = "Purchase dispute or unauthorized transaction"
    FEES_INTEREST = "Fees or interest"
    PAYMENT_BILLING = "Payment or billing process"
    ACCOUNT_CLOSURE = "Account closure or cancellation"
    CREDIT_REPORT = "Credit-report information accuracy"
    MANUAL_REVIEW = "Manual review"


QUEUE_BY_CATEGORY = {
    IssueCategory.PURCHASE_DISPUTE: "Transactions and disputes",
    IssueCategory.FEES_INTEREST: "Fees and interest",
    IssueCategory.PAYMENT_BILLING: "Payment servicing",
    IssueCategory.ACCOUNT_CLOSURE: "Account closure",
    IssueCategory.CREDIT_REPORT: "Credit reporting",
}


class ComplaintAnalysis(BaseModel):
    issue_category: IssueCategory
    summary: str = Field(min_length=1, max_length=1200)
    requested_outcome: str = Field(max_length=800)
    key_facts: list[str] = Field(max_length=12)
    missing_information: list[str] = Field(max_length=12)
    evidence: list[str] = Field(max_length=8)
    uncertainty_reason: str = Field(max_length=600)


class ReviewRecord(BaseModel):
    complaint_text: str
    issue_category: IssueCategory
    queue: str
    summary: str
    requested_outcome: str
    key_facts: list[str]
    missing_information: list[str]
    evidence: list[str]
    uncertainty_reason: str
    review_status: str


def queue_for(category: IssueCategory) -> str:
    return QUEUE_BY_CATEGORY.get(category, "Manual review")


def validate_grounding(analysis: ComplaintAnalysis, complaint: str) -> tuple[ComplaintAnalysis, list[str]]:
    """Keep only exact evidence excerpts and route unsupported output to a person."""
    valid = [quote for quote in analysis.evidence if quote.strip() and quote in complaint]
    problems: list[str] = []
    if len(valid) != len(analysis.evidence):
        problems.append("One or more evidence excerpts did not exactly match the complaint text.")
    if not valid:
        problems.append("The model did not provide a verifiable excerpt from the complaint.")
    if problems:
        analysis = analysis.model_copy(
            update={
                "issue_category": IssueCategory.MANUAL_REVIEW,
                "evidence": valid,
                "uncertainty_reason": " ".join(filter(None, [analysis.uncertainty_reason, *problems])),
            }
        )
    return analysis, problems



def load_examples(path: str = "examples/demo_cases.jsonl") -> list[dict[str, str]]:
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise ValueError("Every demo example must include non-empty text.")
                examples.append(item)
    return examples
