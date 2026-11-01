from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass
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
    result_schema_version: int = 2
    complaint_text: str
    original_model_category: IssueCategory
    evidence_validation: dict[str, Any]
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


@dataclass(frozen=True)
class EvidenceValidation:
    """Safe accounting of quote checks; quote text itself is not copied here."""

    checked_count: int
    valid_count: int
    invalid_count: int
    invalid_indexes: tuple[int, ...]
    missing_evidence: bool

    @property
    def has_warnings(self) -> bool:
        return self.invalid_count > 0 or self.missing_evidence


@dataclass(frozen=True)
class TriageDecision:
    model_analysis: ComplaintAnalysis
    evidence_validation: EvidenceValidation
    final_analysis: ComplaintAnalysis
    grounding_problems: tuple[str, ...]
    final_queue: str


def inspect_evidence(analysis: ComplaintAnalysis, complaint: str) -> EvidenceValidation:
    valid_indexes = tuple(
        index for index, quote in enumerate(analysis.evidence)
        if quote.strip() and quote in complaint
    )
    return EvidenceValidation(
        checked_count=len(analysis.evidence),
        valid_count=len(valid_indexes),
        invalid_count=len(analysis.evidence) - len(valid_indexes),
        invalid_indexes=tuple(index for index in range(len(analysis.evidence)) if index not in valid_indexes),
        missing_evidence=len(valid_indexes) == 0,
    )


def assess_analysis(analysis: ComplaintAnalysis, complaint: str) -> TriageDecision:
    """Preserve the model proposal and separately derive evidence and routing outcomes."""
    validation = inspect_evidence(analysis, complaint)
    valid = [quote for index, quote in enumerate(analysis.evidence) if index not in validation.invalid_indexes]
    problems: list[str] = []
    if validation.invalid_count:
        problems.append("One or more evidence excerpts did not exactly match the complaint text.")
    if not valid:
        problems.append("The model did not provide a verifiable excerpt from the complaint.")
    final_analysis = analysis
    if problems:
        # Preserve the model's category and expose evidence problems separately.
        # Evidence quality is a review signal, not a category override.
        final_analysis = analysis.model_copy(
            update={
                "issue_category": analysis.issue_category,
                "evidence": valid,
                "uncertainty_reason": " ".join(filter(None, [analysis.uncertainty_reason, *problems])),
            }
        )
    return TriageDecision(
        model_analysis=analysis,
        evidence_validation=validation,
        final_analysis=final_analysis,
        grounding_problems=tuple(problems),
        final_queue=queue_for(final_analysis.issue_category),
    )


def validate_grounding(analysis: ComplaintAnalysis, complaint: str) -> tuple[ComplaintAnalysis, list[str]]:
    """Compatibility helper returning the evidence-filtered final decision."""
    decision = assess_analysis(analysis, complaint)
    return decision.final_analysis, list(decision.grounding_problems)



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
