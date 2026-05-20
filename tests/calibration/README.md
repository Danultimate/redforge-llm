# Calibration Tests

The false-positive defense for RedForge lives in this directory.

For each `(module, variant)` pair, this directory holds:

- A **labeled set** of `(prompt, response, expected_severity)` tuples — minimum 20 per variant, stratified across response shapes (plain / RAG-augmented / agent/tool-use / multi-turn-summary).
- A test that runs the current rubric against the labeled set and asserts **per-severity precision/recall floors** (not a single accuracy number).
- A **distribution-shift held-out set** drawn from a different response distribution; CI emits a warning (not failure) if heldout precision drops >10% below calibration set.
- A **heuristic-only labeled set** with its own (relaxed) precision/recall floors, since `judge=None` mode has different characteristics.

CI also runs a **monthly re-calibration job** against the current default judge model versions; failures open an issue rather than break the build.
