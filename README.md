# Complaint Triage Workbench

A local portfolio prototype that turns a consumer complaint into a checkable draft for a human reviewer. It suggests one of five issue categories, maps that category to a demonstration review queue, summarizes the reported problem, and displays exact supporting excerpts. The reviewer can edit or flag the draft and export the reviewed record as JSON.

This is not Consumer Financial Protection Bureau software, legal advice, a case decision system, or a live routing service. Consumer Financial Protection Bureau categories are imperfect reference labels, and consumer narratives are allegations rather than verified findings.

## Run locally

Requirements: Python 3.13.15.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Ensure `.env` contains `OPENROUTER_GLM_API_KEY=...`. If `.env` does not exist, create it from `.env.example`; do not overwrite an existing `.env`. Then start the app:

```bash
streamlit run app.py
```

The development configuration binds to `127.0.0.1` so this local milestone is reachable only from your computer.

The app uses OpenRouter model `z-ai/glm-5.3-flash`. Each analysis sends the complaint text to OpenRouter and may incur API charges. The key stays server-side. Avoid submitting sensitive information. The app does not write pasted complaint text to disk; it remains in the active browser session and can be exported by the reviewer.

## Try it

Choose one of the fictional examples, click **Analyse complaint**, inspect the original text beside the proposed result, correct or flag the result, and download the reviewed JSON record. Or paste a complaint you are comfortable sending to the model provider.

## Current scope

- Five approved starting issue categories with fixed queue mapping.
- Summary, requested outcome, key facts, missing information, evidence excerpts, and uncertainty reason.
- Manual-review routing for unsupported or unclear cases and evidence quotations that fail exact matching.
- Human review fields and JSON export.
- Fictional examples only; Consumer Financial Protection Bureau source data and evaluation cases are not loaded into the app yet.

## Data source and limitations

The project is informed by the Consumer Financial Protection Bureau's [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) and its [Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). The archive is historical and Consumer Financial Protection Bureau says its complaints are not representative of all consumers. See `MVP-SPEC.md` for the discovery sample and data findings. Raw source archives are kept locally under `data/raw/` and excluded from Git.

## Privacy and cost

Text is sent to OpenRouter only after the user clicks **Analyse complaint**. Do not paste names, full account numbers, or other sensitive information. The app keeps the current case in Streamlit session memory; it does not persist it on the server. OpenRouter usage may incur charges. Automated checks use fictional examples and do not make live model calls.

## Technical choices

Approved: Streamlit, Python 3.13.15, Pydantic, JSONL fixtures, OpenAI-compatible SDK access through OpenRouter, and GLM 5.3 Flash. Exact installed package versions are recorded in `requirements-lock.txt`. Public deployment, Consumer Financial Protection Bureau data sampling/redaction, and persistent storage remain out of scope for this local milestone.
