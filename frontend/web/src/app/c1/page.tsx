import type { Metadata } from "next";
import C1Dashboard from "./C1Dashboard";

export const metadata: Metadata = {
  title: "Road Damage Detection (C1) · RoadLens",
  description: "Component 1: Temporal Consistency-Guided Road Damage Detection and Mapping.",
};

export default function C1Page() {
  return <C1Dashboard />;
}
