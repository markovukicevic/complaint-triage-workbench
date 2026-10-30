import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from model_service import analyze_with_model
from streamlit.testing.v1 import AppTest

from triage import ComplaintAnalysis, IssueCategory, load_examples, queue_for, validate_grounding


class TriageTests(unittest.TestCase):
    def test_demo_examples_are_fictional_and_load(self):
        examples = load_examples()
        self.assertEqual(len(examples), 6)
        self.assertTrue(all("fictional" in item["title"].lower() for item in examples))

    def test_each_supported_category_has_fixed_queue(self):
        expected = {
            IssueCategory.PURCHASE_DISPUTE: "Transactions and disputes",
            IssueCategory.FEES_INTEREST: "Fees and interest",
            IssueCategory.PAYMENT_BILLING: "Payment servicing",
            IssueCategory.ACCOUNT_CLOSURE: "Account closure",
            IssueCategory.CREDIT_REPORT: "Credit reporting",
        }
        for category, queue in expected.items():
            with self.subTest(category=category):
                self.assertEqual(queue_for(category), queue)
        self.assertEqual(queue_for(IssueCategory.MANUAL_REVIEW), "Manual review")

    def test_exact_evidence_keeps_category(self):
        complaint = "I paid $75 on June 4, but the payment is missing."
        analysis = ComplaintAnalysis(
            issue_category=IssueCategory.PAYMENT_BILLING,
            summary="The reported payment is missing.",
            requested_outcome="Credit the payment.",
            key_facts=["$75", "June 4"],
            missing_information=[],
            evidence=["I paid $75 on June 4"],
            uncertainty_reason="",
        )
        checked, problems = validate_grounding(analysis, complaint)
        self.assertEqual(checked.issue_category, IssueCategory.PAYMENT_BILLING)
        self.assertEqual(checked.evidence, ["I paid $75 on June 4"])
        self.assertEqual(problems, [])

    def test_missing_or_invented_evidence_routes_to_manual_review(self):
        complaint = "I paid $75 on June 4, but the payment is missing."
        for evidence in ([], ["The bank refused to help."]):
            with self.subTest(evidence=evidence):
                analysis = ComplaintAnalysis(
                    issue_category=IssueCategory.PAYMENT_BILLING,
                    summary="The reported payment is missing.",
                    requested_outcome="",
                    key_facts=[],
                    missing_information=[],
                    evidence=evidence,
                    uncertainty_reason="",
                )
                checked, problems = validate_grounding(analysis, complaint)
                self.assertEqual(checked.issue_category, IssueCategory.MANUAL_REVIEW)
                self.assertEqual(queue_for(checked.issue_category), "Manual review")
                self.assertTrue(problems)

    def test_schema_rejects_unapproved_categories(self):
        with self.assertRaises(ValidationError):
            ComplaintAnalysis.model_validate(
                {
                    "issue_category": "Loan servicing",
                    "summary": "Example",
                    "requested_outcome": "",
                    "key_facts": [],
                    "missing_information": [],
                    "evidence": [],
                    "uncertainty_reason": "",
                }
            )


    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_openrouter_response_is_parsed_without_network(self, openai_class):
        payload = {
            "issue_category": IssueCategory.PAYMENT_BILLING.value,
            "summary": "The payment is missing.",
            "requested_outcome": "Credit the payment.",
            "key_facts": ["$75", "June 4"],
            "missing_information": [],
            "evidence": ["I paid $75 on June 4"],
            "uncertainty_reason": "",
        }
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        analysis = analyze_with_model("I paid $75 on June 4, but the payment is missing.")
        self.assertEqual(analysis.issue_category, IssueCategory.PAYMENT_BILLING)
        self.assertEqual(analysis.evidence, ["I paid $75 on June 4"])
        call = mock_client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(call.kwargs["max_tokens"], 1400)
        self.assertEqual(openai_class.call_args.kwargs["max_retries"], 0)

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_invalid_model_json_is_rejected_without_network(self, openai_class):
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"issue_category":"made up"}'))]
        )
        with self.assertRaises(ValidationError):
            analyze_with_model("fictional complaint")

    @patch.dict(os.environ, {}, clear=True)
    @patch("model_service.OpenAI")
    def test_missing_api_key_fails_before_client_creation(self, openai_class):
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_GLM_API_KEY"):
            analyze_with_model("fictional complaint")
        openai_class.assert_not_called()

    def test_reviewer_changes_are_exported_and_needs_review_routes_manually(self):
        complaint = "I paid $75 on June 4, but the payment is missing."
        analysis = ComplaintAnalysis(
            issue_category=IssueCategory.PAYMENT_BILLING,
            summary="Payment missing.",
            requested_outcome="Credit payment.",
            key_facts=["$75"],
            missing_information=[],
            evidence=["I paid $75 on June 4"],
            uncertainty_reason="",
        )
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=10)
        app.session_state["triage_result"] = {
            "analysis": analysis.model_dump(mode="json"),
            "queue": queue_for(analysis.issue_category),
            "review_status": "pending",
            "grounding_problems": [],
        }
        app.session_state["triage_complaint"] = complaint
        app.run()
        app.selectbox(key="review_category").select(IssueCategory.FEES_INTEREST.value).run()
        self.assertEqual(app.text_input[0].value, "Fees and interest")
        app.text_area(key="review_summary").set_value("Reviewer corrected the payment summary.").run()
        app.radio[0].set_value("Needs review").run()
        app.button[1].click().run()
        exported = app.session_state["reviewed_record"]
        self.assertEqual(exported["summary"], "Reviewer corrected the payment summary.")
        self.assertEqual(exported["issue_category"], IssueCategory.FEES_INTEREST.value)
        self.assertEqual(exported["queue"], "Manual review")
        self.assertEqual(exported["review_status"], "needs review")
        self.assertEqual(exported["complaint_text"], complaint)

    def test_changing_input_clears_stale_analysis_and_export(self):
        complaint = "I paid $75 on June 4, but the payment is missing."
        analysis = ComplaintAnalysis(
            issue_category=IssueCategory.PAYMENT_BILLING,
            summary="Payment missing.",
            requested_outcome="",
            key_facts=[],
            missing_information=[],
            evidence=["I paid $75"],
            uncertainty_reason="",
        )
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=10)
        app.session_state["triage_result"] = {
            "analysis": analysis.model_dump(mode="json"),
            "queue": queue_for(analysis.issue_category),
            "review_status": "pending",
            "grounding_problems": [],
        }
        app.session_state["triage_complaint"] = complaint
        app.session_state["reviewed_record"] = {"old": "record"}
        app.run()
        app.text_area(key="complaint_input").set_value("A different complaint.").run()
        self.assertNotIn("triage_result", app.session_state)
        self.assertNotIn("reviewed_record", app.session_state)

    def test_ui_selecting_demo_fills_complaint_and_does_not_call_model(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=10).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.button), 1)
        self.assertTrue(app.button[0].disabled)
        app.selectbox[0].select("Unfamiliar card purchase (fictional)").run()
        self.assertFalse(app.exception)
        self.assertIn("$42.18", app.text_area[0].value)
        self.assertEqual(len(app.button), 1)
        self.assertTrue(app.button[0].disabled is False)
        self.assertEqual(len(app.get("success")), 0)


if __name__ == "__main__":
    unittest.main()
