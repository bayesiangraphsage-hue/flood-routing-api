import dynamic from "next/dynamic";

const MapComponent = dynamic(
  () => import("../src/MapComponent"),
  {
    ssr: false,
  }
);

export default function Home() {
  return (
    <div>
      <h1 style={{ textAlign: "center" }}>
        Flood Routing System
      </h1>

      <MapComponent />
    </div>
  );
}
