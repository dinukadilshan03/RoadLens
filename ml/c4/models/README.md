# C4 model folder (not in Git)

Put one model here, in either format:

| Format | Files | Notes |
|---|---|---|
| Portable (recommended) | `c4_model_portable.json`, `c4_model_portable.xgb.json` | Made by `scripts/export_portable_model.py`; works with any recent xgboost |
| Joblib bundle | `c4_deployment_model.joblib` | From Notebook 07; needs the same scikit-learn/xgboost versions as the Colab session |

The adapter records the SHA-256 of the model files as part of the model version in every result.
`python -m ml.c4.scripts.make_demo_model --out ml/c4/models/demo` creates a synthetic demo model for
local testing; its results are marked `source_kind: "mock"`.
