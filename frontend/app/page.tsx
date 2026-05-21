"use client";

import dynamic from "next/dynamic";
import { Toaster } from "sonner";

const MapComponent = dynamic(
  () => import("@/components/MapComponent"),
  {
    ssr: false,
  }
);

export default function Page() {
  return (
    <>
      <Toaster richColors position="top-right" />

      <MapComponent />
    </>
  );
}