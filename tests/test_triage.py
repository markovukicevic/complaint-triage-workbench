import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai import APIStatusError
from pydantic import ValidationError

from model_service import (
    MAX_OUTPUT_TOKENS, MAX_RETRIES, MODEL_ID, PROMPT_SHA256_V1, PROMPT_SHA256_V2, PROMPT_VERSION,
    REASONING_EFFORT, REQUEST_TIMEOUT_SECONDS, RESPONSE_FORMAT, SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    TEMPERATURE, ModelCallError, analyze_with_diagnostics, analyze_with_model,
    configuration_snapshot, prompt_for_version, prompt_sha256,
)
from streamlit.testing.v1 import AppTest

from triage import ComplaintAnalysis, IssueCategory, load_examples, queue_for, validate_grounding


class TriageTests(unittest.TestCase):
    def test_prompt_does_not_infer_an_unstated_requested_outcome(self):
        self.assertIn("state the concrete action the consumer asks the company to take", SYSTEM_PROMPT)
        self.assertIn("Do not infer a remedy from the problem, harm, or facts", SYSTEM_PROMPT)
        self.assertIn("read the entire complaint", SYSTEM_PROMPT)
        self.assertIn("Requests may be phrased indirectly", SYSTEM_PROMPT)
        self.assertIn("If the complaint states no clear action, return an empty string", SYSTEM_PROMPT)

    def test_v1_prompt_and_model_configuration_are_frozen(self):
        self.assertEqual(PROMPT_VERSION, "triage-v1")
        self.assertEqual(prompt_sha256(), PROMPT_SHA256_V1)
        self.assertEqual(
            configuration_snapshot(),
            {
                "provider": "OpenRouter",
                "model_id": "z-ai/glm-5.3-flash",
                "prompt_version": "triage-v1",
                "prompt_sha256": PROMPT_SHA256_V1,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_output_tokens": 4096,
                "reasoning_effort": "low",
                "timeout_seconds": 45,
                "max_retries": 0,
            },
        )
        self.assertEqual((MODEL_ID, TEMPERATURE, MAX_OUTPUT_TOKENS), ("z-ai/glm-5.3-flash", 0, 4096))
        self.assertEqual((REASONING_EFFORT, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES), ("low", 45, 0))
        self.assertEqual(RESPONSE_FORMAT, {"type": "json_object"})

    def test_v1_prompt_guards_against_known_inference_failures(self):
        self.assertIn("Every factual claim must be supported", SYSTEM_PROMPT)
        self.assertIn("Do not add a fee, enforcement action", SYSTEM_PROMPT)
        self.assertIn("Never infer what a masked placeholder", SYSTEM_PROMPT)
        self.assertIn("outside the credit-card scope", SYSTEM_PROMPT)

    def test_v2_prompt_targets_actor_scope_and_unsupported_claim_failures(self):
        self.assertEqual(prompt_sha256("triage-v1"), PROMPT_SHA256_V1)
        self.assertEqual(prompt_sha256("triage-v2"), PROMPT_SHA256_V2)
        self.assertIn("consumer, employer, bank, merchant, and regulator are different actors", SYSTEM_PROMPT_V2)
        self.assertIn("Do not transfer an event from one actor to another", SYSTEM_PROMPT_V2)
        self.assertIn("Do not infer gender, identity", SYSTEM_PROMPT_V2)
        self.assertIn("outside the credit-card scope", SYSTEM_PROMPT_V2)
        self.assertIn("Never infer what a masked placeholder", SYSTEM_PROMPT_V2)
        self.assertIn("return an empty string", SYSTEM_PROMPT_V2)
        self.assertIn("Always include this field", SYSTEM_PROMPT_V2)

    def test_prompt_versions_are_selectable_without_changing_default(self):
        self.assertEqual(PROMPT_VERSION, "triage-v1")
        self.assertIs(prompt_for_version("triage-v1"), SYSTEM_PROMPT)
        self.assertIs(prompt_for_version("triage-v2"), SYSTEM_PROMPT_V2)
        self.assertEqual(configuration_snapshot("triage-v2")["prompt_sha256"], PROMPT_SHA256_V2)
        self.assertEqual(configuration_snapshot("triage-v2")["max_output_tokens"], 4096)
        with self.assertRaises(ValueError):
            prompt_for_version("triage-v3")

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
        self.assertEqual(call.kwargs["max_tokens"], 4096)
        self.assertEqual(call.kwargs["extra_body"], {"reasoning": {"effort": "low"}})
        self.assertEqual(openai_class.call_args.kwargs["max_retries"], 0)

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_v2_request_uses_v2_prompt_and_keeps_baseline_settings(self, openai_class):
        payload = {
            "issue_category": IssueCategory.MANUAL_REVIEW.value,
            "summary": "The complaint describes an unclear product.",
            "requested_outcome": "",
            "key_facts": [],
            "missing_information": [],
            "evidence": [],
            "uncertainty_reason": "The product is unclear.",
        }
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))],
            usage=None,
        )
        response = analyze_with_diagnostics("Fictional unclear product complaint.", prompt_version="triage-v2")
        self.assertEqual(response.analysis.issue_category, IssueCategory.MANUAL_REVIEW)
        call = mock_client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["messages"][0]["content"], SYSTEM_PROMPT_V2)
        self.assertEqual(call.kwargs["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(call.kwargs["temperature"], 0)
        self.assertEqual(call.kwargs["max_tokens"], 4096)
        self.assertEqual(call.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(call.kwargs["extra_body"], {"reasoning": {"effort": "low"}})

    @patch("model_service.OpenAI")
    def test_unknown_prompt_version_fails_before_client_creation(self, openai_class):
        with self.assertRaises(ValueError):
            analyze_with_diagnostics("Fictional complaint", prompt_version="triage-v3")
        openai_class.assert_not_called()

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_invalid_model_json_is_rejected_without_network(self, openai_class):
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"issue_category":"made up"}'))]
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_model("fictional complaint")
        self.assertEqual(caught.exception.code, "schema_validation_error")
        self.assertIn("issue_category", {issue["field"] for issue in caught.exception.validation_issues})

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_malformed_json_has_distinct_safe_diagnostic(self, openai_class):
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"summary": '))],
            usage=None,
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("fictional complaint")
        self.assertEqual(caught.exception.code, "invalid_json")
        self.assertTrue(all(set(item) <= {"field", "error", "index"} for item in caught.exception.validation_issues))

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_unexpected_provider_failure_does_not_expose_message(self, openai_class):
        secret_detail = "PRIVATE_ACCOUNT_987654321"
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError(secret_detail)
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("fictional complaint")
        self.assertEqual(caught.exception.code, "provider_error")
        self.assertNotIn(secret_detail, repr(caught.exception.__dict__))

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_empty_model_response_has_safe_diagnostic(self, openai_class):
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="  "))],
            usage=None,
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("private complaint sentinel")
        self.assertEqual(caught.exception.code, "empty_content")
        self.assertEqual(caught.exception.finish_reason, "stop")
        self.assertNotIn("private complaint sentinel", repr(caught.exception.__dict__))

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_truncated_model_response_is_not_parsed(self, openai_class):
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"summary":"partial"}'))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4096, total_tokens=4107),
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("fictional complaint")
        self.assertEqual(caught.exception.code, "output_truncated")
        self.assertEqual(caught.exception.finish_reason, "length")
        self.assertEqual(caught.exception.usage, {"input_tokens": 11, "output_tokens": 4096, "total_tokens": 4107})

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_schema_diagnostics_only_keep_allowlisted_field_and_error(self, openai_class):
        mock_client = openai_class.return_value
        model_secret = "PRIVATE_ACCOUNT_987654321"
        response_content = json.dumps({
            "issue_category": IssueCategory.PAYMENT_BILLING.value,
            "summary": model_secret,
            "requested_outcome": "",
            "key_facts": "wrong type",
            "missing_information": [],
            "evidence": ["I paid $75 on June 4"],
            "uncertainty_reason": "",
        })
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=response_content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("fictional complaint")
        diagnostic = repr(caught.exception.__dict__)
        self.assertEqual(caught.exception.code, "schema_validation_error")
        self.assertNotIn(model_secret, diagnostic)
        self.assertEqual({item["field"] for item in caught.exception.validation_issues}, {"key_facts"})
        self.assertTrue(all(set(item) <= {"field", "error", "index"} for item in caught.exception.validation_issues))

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_provider_status_is_classified_without_error_body(self, openai_class):
        from httpx import Request, Response

        mock_client = openai_class.return_value
        secret_body = "PRIVATE_ACCOUNT_987654321"
        request = Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        mock_client.chat.completions.create.side_effect = APIStatusError(
            "provider response contained " + secret_body,
            response=Response(429, request=request),
            body={"message": secret_body},
        )
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_diagnostics("fictional complaint")
        self.assertEqual(caught.exception.code, "provider_rate_limited")
        self.assertEqual(caught.exception.provider_status, 429)
        self.assertNotIn(secret_body, repr(caught.exception.__dict__))

    @patch.dict(os.environ, {"OPENROUTER_GLM_API_KEY": "test-key"})
    @patch("model_service.OpenAI")
    def test_success_returns_only_safe_completion_metadata(self, openai_class):
        payload = {
            "issue_category": IssueCategory.PAYMENT_BILLING.value,
            "summary": "The payment is missing.",
            "requested_outcome": "",
            "key_facts": [],
            "missing_information": [],
            "evidence": ["The payment is missing."],
            "uncertainty_reason": "",
        }
        mock_client = openai_class.return_value
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=7, secret="private"),
                raw_response="secret",
            ),
        )
        response = analyze_with_diagnostics("The payment is missing.")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage, {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "reasoning_tokens": 7})

    @patch.dict(os.environ, {}, clear=True)
    @patch("model_service.OpenAI")
    def test_missing_api_key_fails_before_client_creation(self, openai_class):
        with self.assertRaises(ModelCallError) as caught:
            analyze_with_model("fictional complaint")
        self.assertEqual(caught.exception.code, "missing_api_key")
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
