# Complaint Triage Workbench — Project Plan

## Project summary

**Working name:** Complaint Triage Workbench
**Proposed GitHub repository:** `complaint-triage-workbench`

This personal portfolio project explores how an AI assistant can help a human reviewer turn a long consumer complaint into a structured case. It will suggest an issue category and a review queue, prepare a concise summary, point to supporting text, and allow the reviewer to correct the result.

The project’s value is a repeatable and inspectable review workflow, with quality measured on examples that were not used to develop it. It does not claim that AI is inherently better than using ChatGPT directly; the project should demonstrate whether its fixed workflow, evidence display, and evaluation are useful.

The Consumer Financial Protection Bureau complaint data is historical and not representative of all consumers. Consumer Financial Protection Bureau labels will be treated as reference labels, not unquestionable ground truth. Complaints describe consumer-reported allegations and do not establish that an allegation is true.

## Problem and intended workflow

A reviewer reading a consumer’s free-text account may need to identify what happened, what issue is central, what the consumer has already tried, what outcome they want, which team should review the case, and what details are missing.

The application will present the original complaint beside a structured draft containing:

- A suggested issue category.
- A suggested demonstration review queue, selected through documented rules.
- A concise summary of the problem, earlier resolution attempts, and requested outcome.
- Dates, amounts, or other key facts when stated.
- Supporting excerpts from the complaint.
- Missing information and ambiguity flags.

The reviewer can accept, edit, or mark the draft for further review, then export the reviewed case. The reviewer remains responsible for decisions.

A **narrative** is the consumer’s free-text description of what happened. **Metadata** is structured information associated with a complaint, such as its product, issue label, company, or date. An **issue category** describes the problem; a **queue** is a worklist for the team that might review it. Queues in this prototype are illustrative and do not connect to real organizations.

## Proposed MVP scope

Start with credit-card complaints and approximately four or five issue groups. Confirm the specific groups after inspecting the available archive and examples.

The MVP will:

1. Let a user select a curated example or paste complaint text.
2. Classify the complaint into one of the supported issue groups.
3. Suggest a queue using an explicit category-to-queue mapping.
4. Summarise the complaint and identify stated outcomes, dates, and amounts.
5. Show evidence excerpts and flag missing or ambiguous information.
6. Let a human reviewer accept, correct, or flag the result.
7. Export the reviewed result in a simple structured format.

Pasted text will not be saved by default. The first version will not include persistent case management, team accounts, assignments, integrations, actual Consumer Financial Protection Bureau submission, or live routing to a financial institution.

## Data plan

Use a selected historical export from the Consumer Financial Protection Bureau’s Consumer Complaint Database Narratives Archive. The Consumer Financial Protection Bureau announcement and archive should be checked again when data preparation begins, because archive coverage and publication policies can change.

- [Consumer Financial Protection Bureau Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Consumer Financial Protection Bureau Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
- [Consumer Financial Protection Bureau announcement on narrative publication](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/)

Prepare a compact set of approximately 120 complaints, subject to inspection of available records:

1. Download a specific archive export and record its date range and source.
2. Inspect the fields and select records relevant to the chosen credit-card issue groups.
3. Remove records with no usable narrative, duplicates, or unsuitable content.
4. Keep source identifiers and reference categories for provenance and evaluation.
5. Save the prepared examples in versioned local JSONL files; each line represents one complaint and its fields.
6. Separate development examples from the held-out evaluation examples.

The application will use these local files for curated examples; it will not query the Consumer Financial Protection Bureau archive at runtime. The selected records support prompt and workflow development and evaluation; they are not used to train a new model. The model will not receive the reference category when it makes a classification prediction.

The public demo may also include synthetic examples. Any selected real narratives should be handled with care, and obvious personal identifiers should be removed from demo fixtures where practical. Document the source, selection method, date range, and limitations.

## Evaluation plan

A **held-out set** is a group of examples kept separate from development so it can provide an independent check on performance. A possible initial split is 90 development examples and 30 held-out examples, revised if the available data or category distribution makes that unsuitable. Repeatedly tuning against the held-out cases would make them development data; a fresh set would then be needed for an independent check.

Evaluate classification and queue suggestions separately from summary quality:

- Compare issue predictions with Consumer Financial Protection Bureau reference labels; report per-category precision and recall, macro-F1, and a confusion matrix where the sample supports them.
- Document the queue mapping and measure queue assignment against it, including cases that should be flagged for human review.
- Manually review a smaller summary sample for factual support, coverage of the problem and requested outcome, and unsupported details.
- Record ambiguous cases and meaningful failure examples.
- Measure processing time and model cost. Claim time savings only if a suitable comparison supports them.

Publish the sample selection method, annotation guidance, mapping rules, model and prompt configuration, results, and limitations. Small sample results are an initial demonstration, not proof of reliable financial complaint handling.

## Technical direction — local MVP approved

Marko approved the local MVP stack on 2026-10-29: Streamlit, Python 3.13.15, Pydantic, local JSONL fixtures, the OpenAI-compatible Python SDK through OpenRouter, and GLM 5.3 Flash (`z-ai/glm-5.3-flash`). Privacy terms and provider cost controls still need review. Public deployment and real-data preparation remain unapproved; obtain explicit approval before those later steps.

- **Interface (approved 2026-10-29):** Streamlit, for a compact Python-based interactive demo.
- **Application language and validation (approved 2026-10-29):** Python 3.13.15 with Pydantic for structured output validation.
- **Data fixtures (approved 2026-10-29):** local JSONL files; no database for the first version. The initial milestone uses fictional examples.
- **Model integration (approved 2026-10-29):** the OpenAI-compatible Python SDK through the OpenRouter API.
- **Model (selected 2026-10-29):** GLM 5.3 Flash, model ID `z-ai/glm-5.3-flash`. Keep model credentials server-side. Privacy terms and provider cost controls remain under review.
- **Deployment packaging (proposed):** Docker on a Netcup VPS, with HTTPS through a reverse proxy. Requires separate approval before deployment.
- **Public address (proposed):** a subdomain such as `triage.markovu.com`, subject to DNS and hosting setup; requires separate approval.

GitHub Pages can host static project documentation, screenshots, and evaluation results, but it cannot host the Python backend for a Streamlit app. The interactive app is therefore planned for a VPS, subject to final technical approval. Model credentials must remain server-side. A public demo should include saved example outputs and reasonable limits on live analysis to control API costs. VPS size should be chosen after measuring the application’s needs.

## RAG extension — later phase

RAG (retrieval-augmented generation) is not needed for the first version, which classifies and summarises information supplied in the complaint. Consider it as a later extension if the app should consult handling guidance.

A safe demonstration could use a small, clearly labelled fictional support-policy handbook. The app could retrieve relevant passages and suggest a next step with citations, while keeping consumer statements separate from procedural guidance. Evaluate whether the retrieved guidance is relevant and whether the suggestion follows the cited text. This extension also requires explicit technical approval before implementation.

## Delivery sequence

1. **Local MVP (implemented):** reviewer workflow using fictional examples, structured response checks, fixed queue mapping, human edits, and JSON export. Real-data sampling and redaction will be proposed and approved separately before those records are used. Automated checks use mocked model responses; a live provider call has not been tested.
2. **Measured evaluation:** assess classification, queue mapping, summary quality, failures, latency, and cost.
3. **Public VPS demo:** publish the source and documentation on GitHub, then deploy the interactive demo to the approved VPS and subdomain.
4. **RAG extension:** add and evaluate cited procedural guidance only if its value is demonstrated and the extension is approved.

## GitHub presentation

The README should explain the problem and intended user, show how to run and try the demo, document data provenance, describe the workflow and evaluation, and link to the Consumer Financial Protection Bureau sources. Include screenshots or a short demo, selected examples, measured results, and failure cases. Clearly state that this is an independent portfolio prototype using historical, non-representative complaint data; it is not Consumer Financial Protection Bureau software, legal advice, a case-outcome decision system, or a live complaint-routing service.
