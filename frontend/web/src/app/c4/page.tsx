import type { Metadata } from "next";
import C4Dashboard from "./C4Dashboard";
import "./c4.css";

export const metadata: Metadata = {
  title: "Road condition · RoadLens",
  description: "Component 4: road condition (good, regular, bad) from smartphone motion sensors.",
};

export default function C4Page() {
  return <C4Dashboard />;
}
