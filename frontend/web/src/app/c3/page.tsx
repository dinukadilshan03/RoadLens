import type { Metadata } from "next";
import C3Dashboard from "./C3Dashboard";
import "./c3.css";
export const metadata: Metadata = {
  title: "Road Surface Intelligence · RoadLens",
  description:
    "Inspect model-predicted road depression, pixel geometry and measurement reliability.",
};
export default function Page() {
  return <C3Dashboard />;
}
