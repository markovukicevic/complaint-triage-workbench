# Frozen evaluation baseline

The current evaluation baseline is **Taxonomy V2A** (`taxonomy-v2a`). It is separate from the Streamlit application default, which remains `triage-v1` until a future release is explicitly approved.

The frozen baseline consists of:

- Taxonomy V2A prompt and recorded prompt hash;
- strict JSON-schema response validation;
- GLM 5.3 Flash with temperature `0`, low reasoning effort, and a 4,096-token limit;
- evidence exact-match validation;
- category preservation when at least one evidence excerpt is valid;
- Evidence warnings are shown separately; evidence failure does not override the model category.

Taxonomy V2B, V2C, and repair-retry experiments remain non-baseline experiments. Their outputs must not be combined with the V2A scorecards.
