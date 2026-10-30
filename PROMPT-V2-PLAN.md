# Prompt v2 proposal

## State

`triage-v2` is prepared for offline review. It has not been sent to the model or evaluated on the 20-case set. The Streamlit app and pilot runner continue to use `triage-v1` by default. Select v2 explicitly in the pilot runner only after deciding to evaluate it.

## Why prepare a second prompt

The v1 baseline found recurring risks: confusing which actor did something, inventing a remedy or other detail, treating masked text as a specific identifier, and assigning a specific category to an out-of-scope complaint. One response also omitted the required `uncertainty_reason` field.

## Changes in v2

- Identify the product and actors from the complaint before classifying. Route non-credit-card or unclear products to Manual review.
- Keep each action attached to the named actor; describe allegations as reports, not verified findings.
- Limit summaries and key facts to stated information. Do not infer fees, outcomes, account status, identity, or the meaning of masked placeholders.
- Record only an explicitly requested action. Leave the field empty when the complaint gives no clear action.
- Require the existing schema fields, including `uncertainty_reason`, and require evidence quotations to support the associated point.

The existing Pydantic schema, categories, deterministic queue mapping, model, and request settings are unchanged. Prompt SHA-256: `9f92d7e0dee6b804b21bf27bff34aae5d85dbd95e43e1ef05671160d6ab529b6`. The frozen v1 prompt hash remains `2a8b1d90649d16491c01900287b130b25dc99ac0fe3822b37eeb9e2e5b5b3276`.

## Offline checks

Mock-based tests check that v1 remains frozen, v2 includes the intended instructions, selecting a prompt version sends that prompt, unknown versions fail before the client is created, and request settings remain fixed. These tests do not establish that the live model follows the instructions or produces factually correct summaries.

## Evaluation procedure

Keep the approved v1 references and baseline unchanged. If a v2 comparison is approved, run the same 20 cases once each under the same model settings, record v2's prompt hash and run metadata separately, then assess structure, category and manual-review decisions, requested outcomes, summary claims, coverage, evidence exactness and support. Do not combine v1 and v2 results or describe this development sample as a held-out or representative benchmark. Continue human review.

The next evaluation run requires separate authorization for the 20 complaint texts to be sent to OpenRouter. No model requests are made by preparing this proposal.
