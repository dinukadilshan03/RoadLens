
from pathlib import Path
import numpy as np
import joblib

def predict_dataset_label_cm(feature_record, model_path):

    bundle = joblib.load(Path(model_path))

    assert bundle["contract"] == (
        "C3_SUT_CRACK_ANNOTATED_LENGTH_DEV_V1"
    )

    names = bundle["input_features"]

    x = np.array([[
        float(feature_record[k])
        for k in names
    ]], dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("Nonfinite feature")

    result = float(
        bundle["estimator"].predict(x)[0]
    )

    return {
        "predicted_dataset_length_label_cm": result,
        "deployment_authorized": False,
        "severity_label": None,
        "note": (
            "Experimental same-dataset learned estimate, "
            "NOT calibrated cross-phone measurement"
        )
    }
