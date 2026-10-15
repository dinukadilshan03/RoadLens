import type { Metadata } from "next";
import C4Dashboard from "./C4Dashboard";

export const metadata: Metadata = {
  title: "Road roughness · RoadLens",
  description: "Component 4: smartphone road-roughness assessment (good / regular / bad).",
};

export default function C4Page() {
  return <C4Dashboard />;
}
