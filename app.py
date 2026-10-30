from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from model_service import ModelCallError, analyze_with_model

from triage import (
    ComplaintAnalysis,
    IssueCategory,
    ReviewRecord,
    load_examples,
    queue_for,
    validate_grounding,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
MAX_COMPLAINT_CHARS = 12_000

def _analysis_to_state(analysis: ComplaintAnalysis, complaint: str) -> dict[str, Any]:
    analysis, grounding_problems = validate_grounding(analysis, complaint)
    return {
        "analysis": analysis.model_dump(mode="json"),
        "queue": queue_for(analysis.issue_category),
        "review_status": "pending",
        "grounding_problems": grounding_problems,
    }


def clear_review_export() -> None:
    st.session_state.pop("reviewed_record", None)


def clear_case_outputs() -> None:
    for key in ("triage_result", "triage_complaint", "reviewed_record", "review_category", "review_summary", "review_requested", "review_facts", "review_missing", "review_evidence", "review_uncertainty"):
        st.session_state.pop(key, None)


def update_example() -> None:
    selected_title = st.session_state.get("example_choice", "Paste your own complaint")
    selected_example = next((item for item in st.session_state["demo_examples"] if item["title"] == selected_title), None)
    st.session_state["complaint_input"] = selected_example["text"] if selected_example else ""
    clear_case_outputs()


def show_list_editor(title: str, values: list[str], key: str) -> list[str]:
    text = st.text_area(title, value="\n".join(values), key=key, height=100, help="Use one item per line.", on_change=clear_review_export)
    return [line.strip() for line in text.splitlines() if line.strip()]


st.set_page_config(page_title="Complaint Triage Workbench", page_icon="🧾", layout="wide")
st.title("Complaint Triage Workbench")
st.caption("AI-assisted first review of consumer complaints. A human reviewer checks every suggestion.")

with st.expander("About this prototype and your text", expanded=True):
    st.write("This portfolio prototype analyses complaint text and suggests one of five issue groups. It does not contact a company, determine whether a claim is true, or make a case decision.")
    st.warning("When you click **Analyse complaint**, the text is sent to OpenRouter for GLM 5.3 Flash processing. Avoid pasting names, account numbers, or other sensitive details. The app does not save pasted text to disk; the current result remains in this browser session until cleared or the session ends. Model/API charges may apply.")

try:
    examples = load_examples(os.path.join(ROOT, "examples", "demo_cases.jsonl"))
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Could not load demo examples: {exc}")
    st.stop()

st.session_state["demo_examples"] = examples
st.session_state.setdefault("complaint_input", "")
example_names = ["Paste your own complaint"] + [item["title"] for item in examples]
st.selectbox("Choose a fictional example or paste text", example_names, key="example_choice", on_change=update_example)
complaint = st.text_area(
    "Complaint text",
    key="complaint_input",
    height=240,
    max_chars=MAX_COMPLAINT_CHARS,
    placeholder="Paste one complaint here…",
    help=f"Up to {MAX_COMPLAINT_CHARS:,} characters. The text is sent to the selected model only when you click Analyse complaint.",
    on_change=clear_case_outputs,
)

if len(complaint) > MAX_COMPLAINT_CHARS:
    st.error(f"Please shorten the complaint to {MAX_COMPLAINT_CHARS:,} characters.")

if st.button("Analyse complaint", type="primary", disabled=not complaint.strip() or len(complaint) > MAX_COMPLAINT_CHARS):
    clear_case_outputs()
    with st.spinner("Analysing with GLM 5.3 Flash via OpenRouter…"):
        try:
            result = analyze_with_model(complaint.strip())
            st.session_state["triage_result"] = _analysis_to_state(result, complaint.strip())
            st.session_state["triage_complaint"] = complaint.strip()
        except ModelCallError as exc:
            st.session_state.pop("triage_result", None)
            if exc.code == "schema_validation_error":
                fields = sorted({str(item["field"]) for item in exc.validation_issues})
                detail = f" Affected fields: {', '.join(fields)}." if fields else ""
                st.error("The model response did not match the required fields." + detail + " Treat this case as manual review.")
            elif exc.code == "invalid_json":
                st.error("The model response was not valid JSON. Treat this case as manual review.")
            elif exc.code == "empty_content":
                st.error("The model returned no response content. No analysis was produced.")
            elif exc.code == "output_truncated":
                st.error("The model response reached its output limit and may be incomplete. Treat this case as manual review.")
            elif exc.code == "provider_filtered_output":
                st.error("The provider did not return an analysis for this request.")
            elif exc.code == "provider_auth_error":
                st.error("OpenRouter rejected authentication. Check the API key configuration.")
            elif exc.code == "provider_rate_limited":
                st.error("OpenRouter rate-limited the request. Wait before trying again.")
            elif exc.code == "provider_timeout":
                st.error("The OpenRouter request timed out.")
            elif exc.code == "provider_connection_error":
                st.error("Could not connect to OpenRouter. Check the network connection.")
            elif exc.code == "missing_api_key":
                st.error("The OpenRouter API key is not configured in the project .env file.")
            else:
                st.error("The OpenRouter request failed. Check the request configuration and try again.")
        except Exception:
            # Provider exceptions can contain request details. Do not echo them into the app.
            st.session_state.pop("triage_result", None)
            st.error("The analysis could not be completed. Treat this case as manual review.")

state = st.session_state.get("triage_result")
if state:
    model_data = state["analysis"]
    st.divider()
    st.subheader("Review the draft")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original complaint**")
        st.info(st.session_state.get("triage_complaint", ""))
        st.caption("Check every suggestion against the original text before accepting it.")
    with right:
        categories = [category.value for category in IssueCategory]
        chosen_category = st.selectbox(
            "Issue category",
            categories,
            index=categories.index(model_data["issue_category"]),
            key="review_category",
            on_change=clear_review_export,
        )
        current_category = IssueCategory(chosen_category)
        st.text_input("Suggested queue (determined by category)", value=queue_for(current_category), disabled=True)
        summary = st.text_area("Summary", value=model_data["summary"], key="review_summary", height=120, on_change=clear_review_export)
        requested = st.text_area("Requested outcome", value=model_data["requested_outcome"], key="review_requested", height=80, on_change=clear_review_export)
        key_facts = show_list_editor("Key facts (one per line)", model_data["key_facts"], "review_facts")
        missing = show_list_editor("Missing information (one per line)", model_data["missing_information"], "review_missing")
        evidence = show_list_editor("Evidence excerpts (one per line; exact quotes only)", model_data["evidence"], "review_evidence")
        uncertainty = st.text_area("Uncertainty or reason for manual review", value=model_data["uncertainty_reason"], key="review_uncertainty", height=70, on_change=clear_review_export)

    for problem in state.get("grounding_problems", []):
        st.warning(problem + " The case was sent to manual review.")
    invalid_quotes = [quote for quote in evidence if quote not in st.session_state.get("triage_complaint", "")]
    if invalid_quotes:
        st.warning("One or more edited excerpts are not exact text from the complaint; correct or remove them before accepting.")

    if current_category == IssueCategory.MANUAL_REVIEW and not uncertainty.strip():
        st.info("Add a short reason, or keep the case flagged for manual review.")

    review_choice = st.radio("Reviewer decision", ["Pending", "Accepted", "Corrected", "Needs review"], horizontal=True, index=0, on_change=clear_review_export)
    missing_supported_evidence = not evidence and current_category != IssueCategory.MANUAL_REVIEW and review_choice != "Needs review"
    if missing_supported_evidence:
        st.warning("Keep at least one exact evidence excerpt for a case marked as accepted or corrected.")
    if st.button(
        "Prepare reviewed record",
        disabled=bool(invalid_quotes)
        or not summary.strip()
        or (current_category == IssueCategory.MANUAL_REVIEW and not uncertainty.strip())
        or missing_supported_evidence,
    ):
        final_analysis = ComplaintAnalysis(
            issue_category=current_category,
            summary=summary.strip(),
            requested_outcome=requested.strip(),
            key_facts=key_facts,
            missing_information=missing,
            evidence=evidence,
            uncertainty_reason=uncertainty.strip(),
        )
        final_queue = "Manual review" if review_choice == "Needs review" else queue_for(current_category)
        final_state = ReviewRecord(
            complaint_text=st.session_state.get("triage_complaint", ""),
            **final_analysis.model_dump(),
            queue=final_queue,
            review_status=review_choice.lower(),
        )
        st.session_state["reviewed_record"] = final_state.model_dump(mode="json")

    reviewed = st.session_state.get("reviewed_record")
    if reviewed:
        st.success("Reviewed record is ready to export. It has not been saved to the server.")
        st.download_button(
            "Download reviewed record (JSON)",
            data=json.dumps(reviewed, ensure_ascii=False, indent=2),
            file_name="reviewed-complaint.json",
            mime="application/json",
        )
