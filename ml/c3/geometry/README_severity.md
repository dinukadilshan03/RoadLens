
# CRDL C3-20 — Severity Policy

The FHWA ACP 8 pothole severity reference uses
independently measured maximum depression depth:

- LOW: below 25 mm
- MODERATE: 25 to 50 mm
- HIGH: above 50 mm

The applicable pavement and pothole-size criteria
must be checked.

This module does not classify CRDL RAW outputs
or independently certify real-world field accuracy.

The seven-row CSV is blank.
Record actual field measurements later.

For backend integration:

from component3_severity_bridge import Component3SeverityBridge

bridge = Component3SeverityBridge(
    bundle_root,
    device="cpu"
)

result = bridge.predict(
    rgb_uint8,
    detections,
    field_measurements=None
)
