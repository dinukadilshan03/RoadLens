import type { Metadata } from "next";
import C2Dashboard from "./C2Dashboard";

export const metadata: Metadata = {
  title: "Road Damage Segmentation Engine (C2) · RoadLens",
  description:
    "Component 2: Noise-Resilient Road Damage Segmentation Engine using Pix2Pix cGAN Latent Thermal Radiometry, Dual MobileNetV3 + LTCA, and Dynamic Alpha Gating.",
};

export default function C2Page() {
  return <C2Dashboard />;
}
