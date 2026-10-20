import type { Metadata } from "next";
import C1Dashboard from "./C1Dashboard";
import "./c1.css";

export const metadata: Metadata = {
  title: "Road Damage Detection · RoadLens",
  description:
    "Component 1: Road distress localization, classification, and downstream C2 segmentation feed using YOLOv8m (B0).",
};

export default function C1Page() {
  return <C1Dashboard />;
}
