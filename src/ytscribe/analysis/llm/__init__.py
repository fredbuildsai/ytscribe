"""LLM layer: structured per-talk annotation and stance, built on the deterministic pre-labels. Runs only on a human-approved
taxonomy; every quoted evidence string is checked against the transcript; results are cached and versioned."""
