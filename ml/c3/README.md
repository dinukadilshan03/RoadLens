# C3 component

Put C3-specific preprocessing and inference adapters here. Keep reusable
production Python code importable, document its contract, and do not commit
private datasets or model weights.

The supplied trained ZIP is imported here with separate `geometry`,
`pothole_visual_severity`, `crack_visual_severity`, `integration`, and
`research_only` folders. Original root geometry files remain in place for
existing imports. ZIP provenance and the SHA-256 manifest are in
`bundle_evidence/`. All 45 manifest entries were verified at import.

Live assessment uses the root CRDL model, C3-22 pothole visual classifier,
and C3-27 RGB-only crack visual classifier. C3-28 learned reconstruction,
C3-07 development length, and crack morphology fusion stay research-only.
Weights and joblib files exist locally and are excluded from Git.

See `docs/c3-workspace.md` for the API, UI, deployment and validation contract.
