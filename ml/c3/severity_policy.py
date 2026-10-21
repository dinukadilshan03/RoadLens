
import math

POLICY_VERSION = "C3-20.1.0"

STANDARD_ID = (
    "FHWA-HRT-13-092_ACP_8_POTHOLES_2014"
)

STANDARD_URL = (
    "https://www.fhwa.dot.gov/publications/"
    "research/infrastructure/pavements/ltpp/"
    "13092/001.cfm"
)


def severity_band(depth_mm):

    if (
        isinstance(depth_mm, bool)
        or not isinstance(
            depth_mm,
            (int, float)
        )
    ):
        raise ValueError(
            "depth_mm must be a numeric "
            "maximum physical depth"
        )

    d = float(depth_mm)

    if not math.isfinite(d) or d <= 0:

        raise ValueError(
            "depth_mm must be finite and > 0"
        )

    if d < 25.0:
        return "LOW"

    if d <= 50.0:
        return "MODERATE"

    return "HIGH"


def _number(
    value,
    field,
    *,
    zero_allowed=False
):

    if value is None or value == "":
        return None

    if isinstance(value, bool):

        raise ValueError(
            field + " must be numeric"
        )

    try:
        z = float(value)

    except (ValueError, TypeError):

        raise ValueError(
            field + " must be numeric"
        ) from None

    if (
        not math.isfinite(z)
        or z < 0
        or (z == 0 and not zero_allowed)
    ):

        raise ValueError(
            field + " has invalid value"
        )

    return z


def _true(v):

    # Blank fields are never automatically
    # treated as verified evidence.

    return (
        v is True
        or (
            isinstance(v, str)
            and v.strip().lower() == "true"
        )
    )


def evaluate_field_reference(field=None):

    result = {

        "source":
            "MANUAL_FIELD_REFERENCE_ONLY",

        "standard_id":
            STANDARD_ID,

        "standard_url":
            STANDARD_URL,

        "status":
            "NOT_PROVIDED",

        "reference_severity_nominal":
            None,

        "reference_severity_uncertainty_robust":
            None,

        "model_predicted_severity":
            None,

        "max_depth_mm":
            None,

        "uncertainty_mm":
            None,

        "nine_point_depths_available":
            False,

        "nine_point_mean_mm":
            None,

        "nine_point_max_mm":
            None,

        "road_plan_definition_verified":
            False,

        "measurement_quality_warnings":
            [],

        "not_a_model_prediction":
            True
    }

    if field is None:
        return result

    if not isinstance(field, dict):

        raise ValueError(
            "Field reference must be a dictionary"
        )

    # ------------------------------------------------
    # VERIFY DECLARED FIELD-MEASUREMENT SOURCE
    # ------------------------------------------------

    if field.get("source") != (
        "MANUAL_FIELD_MEASUREMENT"
    ):

        result["status"] = (
            "UNVERIFIED_FIELD_SOURCE"
        )

        return result

    # ------------------------------------------------
    # FHWA ASPHALT-PAVEMENT APPLICABILITY
    # ------------------------------------------------

    if field.get("road_surface_type") != (
        "ASPHALT_CONCRETE"
    ):

        result["status"] = (
            "FHWA_ACP_STANDARD_NOT_APPLICABLE"
            "_OR_UNCONFIRMED"
        )

        return result

    # ------------------------------------------------
    # FHWA MINIMUM POTHOLE PLAN DIMENSION
    # ------------------------------------------------

    if not _true(
        field.get(
            "fhwa_150mm_inscribed_circle_confirmed"
        )
    ):

        result["status"] = (
            "FHWA_POTHOLE_SIZE_DEFINITION"
            "_NOT_CONFIRMED"
        )

        return result

    result[
        "road_plan_definition_verified"
    ] = True

    # ------------------------------------------------
    # VERIFY ROAD-SURFACE DEPTH REFERENCE
    # ------------------------------------------------

    if not _true(
        field.get(
            "healthy_road_reference_verified"
        )
    ):

        result["status"] = (
            "FIELD_ROAD_REFERENCE_NOT_CONFIRMED"
        )

        return result

    # ------------------------------------------------
    # REQUIRE DOCUMENTED MEASUREMENT METHOD
    # ------------------------------------------------

    if field.get("measurement_method") not in (
        "STRAIGHTEDGE_RULER",
        "ROAD_REFERENCE_DEPTH_GAUGE"
    ):

        result["status"] = (
            "FIELD_MEASUREMENT_METHOD"
            "_NOT_DOCUMENTED"
        )

        return result

    # ------------------------------------------------
    # PHYSICALLY MEASURED DEEPEST POINT
    # ------------------------------------------------

    depth = _number(
        field.get("deepest_point_mm"),
        "deepest_point_mm"
    )

    if depth is None:

        result["status"] = (
            "MISSING_DEEPEST_POINT_MEASUREMENT"
        )

        return result

    result["max_depth_mm"] = depth

    # ------------------------------------------------
    # OPTIONAL NINE-POINT DEPTH EVIDENCE
    # ------------------------------------------------

    points = field.get(
        "nine_point_depths_mm"
    )

    if points not in (None, ""):

        if (
            not isinstance(
                points,
                (list, tuple)
            )
            or len(points) != 9
        ):

            raise ValueError(
                "nine_point_depths_mm must "
                "contain exactly 9 readings"
            )

        values = [

            _number(
                x,
                "nine_point_depths_mm",
                zero_allowed=True
            )

            for x in points
        ]

        if any(x is None for x in values):

            raise ValueError(
                "Nine-point readings cannot "
                "contain blanks"
            )

        result[
            "nine_point_depths_available"
        ] = True

        result["nine_point_mean_mm"] = (
            sum(values) / 9
        )

        result["nine_point_max_mm"] = max(
            values
        )

        if max(values) > depth + 1e-9:

            result["status"] = (
                "CONTRADICTORY_DEPTH_MEASUREMENTS"
            )

            result[
                "measurement_quality_warnings"
            ].append(
                "A sampled point exceeds the "
                "recorded deepest point"
            )

            return result

    # ------------------------------------------------
    # NOMINAL FHWA REFERENCE SEVERITY
    # ------------------------------------------------

    result[
        "reference_severity_nominal"
    ] = severity_band(depth)

    # ------------------------------------------------
    # OPTIONAL MEASUREMENT UNCERTAINTY
    # ------------------------------------------------

    uncertainty = _number(

        field.get(
            "depth_uncertainty_mm"
        ),

        "depth_uncertainty_mm",

        zero_allowed=True
    )

    result["uncertainty_mm"] = uncertainty

    if uncertainty is None:

        result["status"] = (
            "REFERENCE_NOMINAL_"
            "UNCERTAINTY_UNREPORTED"
        )

        result[
            "measurement_quality_warnings"
        ].append(
            "Measurement uncertainty missing; "
            "nominal class is not "
            "uncertainty-robust"
        )

        return result

    lower = max(
        0.0,
        depth - uncertainty
    )

    upper = depth + uncertainty

    # ------------------------------------------------
    # CHECK WHETHER UNCERTAINTY CROSSES A BOUNDARY
    # ------------------------------------------------

    crosses_25 = (
        lower < 25.0 <= upper
    )

    crosses_50 = (
        lower <= 50.0 < upper
    )

    if crosses_25 or crosses_50:

        result["status"] = (
            "REFERENCE_THRESHOLD_AMBIGUOUS"
        )

        return result

    # ------------------------------------------------
    # STABLE LABEL UNDER REPORTED UNCERTAINTY
    # ------------------------------------------------

    result["status"] = (
        "REFERENCE_LABEL_STABLE_WITH_"
        "REPORTED_UNCERTAINTY"
    )

    result[
        "reference_severity_uncertainty_robust"
    ] = severity_band(depth)

    return result
