# Complaint Triage Workbench — Initial MVP Specification

**Status:** Starting functional scope, Streamlit, Python 3.13.15, Pydantic output validation, local JSONL fixtures, the OpenAI-compatible Python SDK through OpenRouter, and GLM 5.3 Flash (`z-ai/glm-5.3-flash`) were approved by Marko on 2026-10-29. The first local implementation uses fictional examples; live API behavior has not yet been verified against the provider. Privacy terms, provider cost controls, real-data sampling/redaction, and public deployment remain open.

## Purpose

The app will help a human reviewer turn a consumer's free-text complaint into a concise, checkable draft case: identify the main issue, suggest a demonstration review queue, summarize stated facts and requested outcome, and show the text supporting those suggestions. The reviewer remains responsible for corrections and next steps.

The project demonstrates a repeatable workflow around an LLM, not a claim that the app is inherently more capable than ChatGPT. Its value should be assessed through consistency, evidence, human review, export, and measured quality.

## Source inspection

The Consumer Financial Protection Bureau [Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) describes previously published complaint records received from December 1, 2011 through August 14, 2026 and provides full-record ZIP exports by period.

Two exports were inspected:

| Archive export | Downloaded CSV size | Credit-card records inspected | Credit-card records with narrative |
|---|---:|---:|---:|
| August 2026 | 177,212,547 bytes uncompressed | 7,996 | 0 |
| January–February 2025 | 491,908,736 bytes uncompressed | 15,286 | 6,958 |

The January–February 2025 export contains records with `Date received` from 2025-01-01 through 2025-02-28. The archive's August 2026 export contained no narrative text in its credit-card rows, so the older batch is the useful candidate source for this prototype. Archive export labels describe publication batches; the complaint dates above were checked in the CSV itself.

The CSV has 16 fields: `Date received`, `Product`, `Sub-product`, `Issue`, `Sub-issue`, `Consumer complaint narrative`, `Company public response`, `Company`, `State`, `ZIP code`, `Tags`, `Submitted via`, `Date sent to company`, `Company response to consumer`, `Timely response?`, and `Complaint ID`.

The two company-response fields do not provide detailed agent reply text in this export. `Company response to consumer` contains outcome/status values such as “Closed with explanation” or “Closed with monetary relief”; `Company public response` contains short categorical statements. They are not a benchmark for drafting agent replies.

Consumer Financial Protection Bureau categories are the labels consumers selected in the submission process. They are useful reference labels for evaluation, but they may not perfectly describe the narrative. Consumer Financial Protection Bureau also cautions that complaint data is not representative of all consumers' experiences. These records contain consumer-reported allegations, not independently established findings.

## Initial narrative review

A reproducible discovery sample of 20 records was reviewed: four non-empty credit-card narratives from each of five candidate Consumer Financial Protection Bureau `Issue` labels. For each label, records were ordered by SHA-256 of `triage-jan-feb-2025-v1:` plus `Complaint ID`; the first four were selected. This provides a repeatable first look, not a representative sample or a model evaluation set.

| Consumer Financial Protection Bureau reference issue | Credit-card records with narrative in this export | Four reviewed complaint IDs |
|---|---:|---|
| Problem with a purchase shown on your statement | 1,628 | 12065504, 11477365, 11780324, 11346692 |
| Fees or interest | 984 | 12124830, 11464372, 11772617, 11729141 |
| Closing your account | 576 | 11839004, 11502381, 12083282, 11904528 |
| Problem when making payments | 500 | 11576673, 11468604, 11577001, 11561050 |
| Incorrect information on your report | 252 | 12073383, 12190306, 11349165, 11894340 |

The five candidate reference issues account for 3,940 of the 6,958 credit-card narratives in this batch (about 57%). The rest would need to be out of scope or routed to human review in a five-category MVP.

The review found useful examples of purchase disputes and unauthorized charges, fees and interest, problems making payments, account closure, and credit-report inaccuracies. It also found mismatches and overlap: one payment-labeled narrative was primarily about a credit-report error; a fees-labeled example concerned a savings account interest-rate issue; and account-closure cases included disputes about payments or withheld rewards. This reinforces the need to show evidence, let a reviewer correct the result, and treat Consumer Financial Protection Bureau labels as fallible references.

No complaint narrative text is copied into this specification. The identifiers above allow the inspected records to be located in the cited export. Before including any real narrative in a public demo fixture, review and redact it; keep the full evaluation text out of the public repository by default.

## Proposed narrow MVP task

**Input:** one complaint narrative, selected from a small reviewed demo set or pasted by the user.

**Approved starting categories (2026-10-29):**

1. Purchase dispute or unauthorized transaction — based primarily on `Problem with a purchase shown on your statement`.
2. Fees or interest — based primarily on `Fees or interest`.
3. Payment or billing process — based primarily on `Problem when making payments`.
4. Account closure or cancellation — based primarily on `Closing your account`.
5. Credit-report information accuracy — based primarily on `Incorrect information on your report`.

These five task labels are the approved starting taxonomy, informed by observed Consumer Financial Protection Bureau categories. The app may preserve the source `Issue` and `Sub-issue` as reference labels for evaluation, while asking the model to classify from narrative text alone. Category boundaries may be revisited based on broader review, but that does not change this approval record.

**Proposed demonstration queues** (illustrative worklists, with no connection to a real company or regulator):

| Suggested category | Demonstration queue |
|---|---|
| Purchase dispute or unauthorized transaction | Transactions and disputes |
| Fees or interest | Fees and interest |
| Payment or billing process | Payment servicing |
| Account closure or cancellation | Account closure |
| Credit-report information accuracy | Credit reporting |
| Unsupported, overlapping, or unclear case | Manual review |

Queue routing should be a documented deterministic mapping from the app's accepted category, not a separate unconstrained model decision. A reviewer can override it.

## Approved expected analysis fields (2026-10-29)

For each complaint, the app should return:

- `issue_category`: one supported category, or `manual_review`.
- `queue`: the queue determined by the mapping, or `manual_review`.
- `summary`: brief account of the problem and what the consumer says they already tried.
- `requested_outcome`: what the consumer asks to happen, if stated.
- `key_facts`: dates, amounts, account/product details, and other relevant facts explicitly present.
- `missing_information`: details that appear necessary but are absent; do not invent them.
- `evidence`: short verbatim excerpts from the input supporting category and summary.
- `uncertainty_reason`: a plain-language reason when the case is ambiguous, outside scope, or unsupported.
- `review_status`: pending, accepted, corrected, or needs review.

The original narrative should remain visible beside the draft. The user can accept, edit, or flag the result and export a reviewed record. Pasted text should not be saved by default.

## Data preparation and evaluation proposal

The January–February 2025 archive export is a feasible source to investigate further. Download it only when preparing the dataset; keep the raw export and full narrative evaluation sample local and out of version control. Provide a reproducible preparation script or documented steps plus complaint IDs and category labels. The public repository could contain a few manually reviewed, appropriately redacted sample fixtures and synthetic examples to make the demo easy to try.

A 120-record working set remains a planning estimate, not an approved dataset size. First check per-category counts, duplicate narratives, and content suitability; sample with category balance if the goal is comparing categories. Split development and evaluation records before prompt tuning. A 90/30 split is only a starting suggestion and would yield limited evidence per category. Do not repeatedly tune against evaluation cases.

Evaluate issue-category predictions against Consumer Financial Protection Bureau reference labels with per-category precision/recall, macro-F1, and a confusion matrix where sample sizes support them. Assess summaries separately on a human-reviewed subset for factual support, coverage of the problem/request, and unsupported claims. Record queue results against the documented mapping, abstentions, notable failures, latency, and model cost. Do not claim improved handling or time savings without a suitable comparison.

## Out of scope and open decisions

The MVP does not submit complaints, route them to real organizations, make consumer-rights or case-outcome decisions, draft agent replies, or provide a persistent team inbox. The Consumer Financial Protection Bureau archive does not offer a continuous feed of newly published narratives in the inspected export; the app will use locally prepared examples rather than querying the site at runtime.

Starting functional scope approval on 2026-10-29 covers the five categories above, the analysis fields above, and manual review for ambiguous or out-of-scope cases. Streamlit, Python 3.13.15, Pydantic, local JSONL fixtures, and the OpenAI-compatible Python SDK through OpenRouter were also explicitly approved on 2026-10-29 for the local MVP. GLM 5.3 Flash through OpenRouter was selected on 2026-10-29 (model ID: `z-ai/glm-5.3-flash`; verified against the public OpenRouter model catalog/API on 2026-10-29). Privacy terms and provider cost controls remain open. Real-data sampling and redaction, Docker, VPS, and subdomain are outside the current local implementation approval.


## Private real-data pilot (2026-10-30)

A private 20-record development sample was prepared from the existing deterministic discovery sample with `scripts/prepare_private_pilot.py`; generated complaint text and outputs are stored under ignored `data/private/`. The app still loads only the six fictional examples. The first five category-balanced model requests were attempted once each. Three intact-input analyses matched their manually reviewed category and deterministic queue, with exact evidence excerpts. One request used a malformed redacted input and is excluded; one returned an empty response. In one usable purchase-dispute result, the model inferred an unstated requested outcome, so that field needs human correction. This three-case qualitative review is not a performance estimate. No retry was made, and token usage/API cost was not captured. The redaction rule was corrected and the 20-record dataset regenerated; the pre-fix data and outputs remain ignored local audit snapshots. Pattern scans found no email, phone, SSN, or long digit sequences in the corrected narratives, but the limited review does not guarantee anonymization.

## References

- [Consumer Financial Protection Bureau Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Consumer Financial Protection Bureau Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
- Archive export inspected: [January 2025 through February 2025 ZIP](https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip)
- Latest-period export inspected: [August 2026 ZIP](https://files.consumerfinance.gov/f/documents/CCDB_Export_21_August_2026.zip)
