"""Generate reports from server-computed findings and supplied road imagery."""

import base64
import io
from datetime import datetime
from zoneinfo import ZoneInfo
from threading import Lock

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

REPORT_LOCK = Lock()
LIMITATION = (
    "Image-based damage assessment. The surface view shows the estimated depression pattern."
)


def layer(url):
    return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))


def append_surface_page(pdf, defect):
    surface = defect.get("relative_surface")
    if not surface:
        return
    fig = Figure(figsize=(8.27, 11.69), facecolor="#f7f7f0")
    ax = fig.add_subplot(111, projection="3d")
    x, y = np.meshgrid(surface["x"], surface["y"])
    z = np.asarray(surface["z"])
    color = np.asarray(surface["depression"])
    cmap = LinearSegmentedColormap.from_list(
        "relative",
        [(0, "#d4d8ce"), (0.003, "#d4d8ce"), (0.004, "#5b9991"),
         (0.2, "#57ad9b"), (0.45, "#d7bb69"), (0.7, "#ed8a43"), (1, "#cc3830")],
    )
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=cmap(color),
        shade=False,
        linewidth=0,
        antialiased=True,
    )
    ax.view_init(elev=31, azim=-63)
    ax.plot_surface(x, y, np.zeros_like(z), color="#5b9991", alpha=0.18, shade=False)
    aspect = surface["aspect_ratio"]
    ax.set_box_aspect((aspect["x"], aspect["y"], aspect["z"]))
    ax.set(
        xlabel="Across image (%)",
        ylabel="Along image (%)",
        zlabel="Relative level",
        zlim=(-1.14, 0.04),
        title="Estimated Road Depression - 3D View",
    )
    ax.invert_yaxis()
    ax.set_zticks([-1, 0], ["Deeper", "Road reference"])
    ax.set_facecolor("#f7f7f0")
    fig.suptitle(defect["presentation"]["label"], fontsize=18, y=0.94)
    fig.text(
        0.1,
        0.15,
        "Colors show relative depression: cooler is shallower; warmer is deeper.\n"
        "The surface follows the depression map in image coordinates.\n"
        "The translucent plane marks the surrounding road level.",
        fontsize=11,
        linespacing=1.7,
        color="#526355",
    )
    pdf.savefig(fig)
    fig.clear()


def generate_report(rgb, result, sources=None):
    sources = sources or {"frame": rgb}
    stream = io.BytesIO()
    with REPORT_LOCK, PdfPages(stream) as pdf:
        for group, source_rgb in sources.items():
            fig = Figure(figsize=(8.27, 11.69), layout="constrained")
            axes = fig.subplots(2, 1, height_ratios=[3, 1])
            axes[0].imshow(source_rgb)
            group_defects = [d for d in result["defects"] if d.get("source_group", "frame") == group]
            for defect in group_defects:
                axes[0].imshow(layer(defect["visuals"]["outline"]))
                x, y = defect["source_bbox_xyxy"][:2]
                axes[0].text(x, y, defect["presentation"]["label"], color="white",
                             bbox={"facecolor": "#243531", "alpha": 0.85})
            source_label = "Crack mask" if group_defects and group_defects[0].get("source_kind") == "mask" else "Road photo"
            axes[0].set_title("RoadLens - Your Road Damage Report / " + source_label + "\n"
                + datetime.now(ZoneInfo("Asia/Colombo")).strftime("%d %B %Y, %H:%M %Z"))
            axes[0].axis("off")
            axes[1].axis("off")
            axes[1].text(0, 0.9, result["summary"] + "\n\n"
                + "\n\n".join(item["message"] for item in result.get("input_warnings", []))
                + "\n\n" + LIMITATION, wrap=True, va="top", fontsize=11, transform=axes[1].transAxes)
            pdf.savefig(fig)
            fig.clear()
        for defect in result["defects"]:
            rgb = sources[defect.get("source_group", "frame")]
            p = defect["presentation"]
            fig = Figure(figsize=(8.27, 11.69), layout="constrained")
            fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.93))
            axes = fig.subplots(3, 2)
            fig.suptitle(p["label"], fontsize=18)
            for ax in axes.flat:
                ax.axis("off")
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("Uploaded crack mask" if defect.get("source_kind") == "mask" else "Original road image")
            axes[0, 1].imshow(rgb)
            axes[0, 1].imshow(layer(defect["visuals"]["outline"]))
            axes[0, 1].set_title("Highlighted damaged area")
            if defect["damage_type"] == "pothole" and defect.get(
                "raw_axial_depression_map"
            ):
                depression = np.asarray(defect["raw_axial_depression_map"])
                axes[1, 0].imshow(depression, cmap="YlOrRd")
                axes[1, 0].set_title("Estimated depression pattern")
                axes[1, 1].imshow(rgb)
                axes[1, 1].imshow(
                    depression,
                    cmap="YlOrRd",
                    alpha=0.6,
                    extent=(-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5),
                )
                axes[1, 1].set_title("Depression overlay")
                box = defect["source_bbox_xyxy"]
                row = min(
                    depression.shape[0] - 1,
                    int((box[1] + box[3]) / 2 / rgb.shape[0] * depression.shape[0]),
                )
                axes[1, 1].axhline(
                    row / depression.shape[0] * rgb.shape[0], color="white"
                )
                axes[2, 0].set_axis_on()
                values = depression[row]
                maximum = max(float(depression.max()), 1e-12)
                axes[2, 0].plot(np.linspace(0, 100, len(values)), -values / maximum)
                axes[2, 0].axhline(0, color="#759278", linewidth=0.8)
                axes[2, 0].set(
                    title="Road Depression Profile",
                    xlabel="Position across image (%)",
                    ylabel="Relative surface level",
                )
            elif defect["damage_type"] == "crack":
                axes[1, 0].imshow(layer(defect["visuals"]["centerline"]))
                axes[1, 0].set_title("Crack centerline")
                axes[1, 1].imshow(layer(defect["visuals"]["width"]))
                axes[1, 1].set_title(
                    "Visible width variation (cooler: narrower; warmer: wider)"
                )
                axes[2, 0].text(
                    0,
                    1,
                    "Crack Pattern Complexity: " + defect["crack_network"]["complexity"]["label"] + "\n\n"
                    + defect["crack_network"]["complexity"]["explanation"] + "\n\n"
                    + "Regional visual findings (not whole-network severity):\n"
                    + "\n".join("Region " + str(item["region_id"]) + ": " + str((item["prediction"] or {}).get("visual_crack_severity", "Unavailable")) for item in defect["crack_network"]["regional_visual_findings"]),
                    va="top",
                    wrap=True,
                )
            text = "\n\n".join(
                [
                    p["measurement"],
                    "AI visual severity: " + p["severity"],
                    p["shape"],
                    p["explanation"],
                    " ".join(p["review_reasons"]),
                    p["measurement_note"],
                    p["reliability"],
                    p["recommendation"],
                ]
            )
            axes[2, 1].text(0, 1, text, va="top", wrap=True, fontsize=10)
            fig.text(
                0.08,
                0.015,
                "Estimated surface shape from the uploaded image and damage mask.",
                fontsize=9,
            )
            pdf.savefig(fig)
            fig.clear()
            append_surface_page(pdf, defect)
    return stream.getvalue()
