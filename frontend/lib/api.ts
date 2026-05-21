import { RouteResponse } from "@/types/route";

export async function fetchRoute(
  origin: [number, number],
  destination: [number, number]
): Promise<RouteResponse> {
  const response = await fetch(
    "https://flood-routing-api-1.onrender.com/predict_route",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        origin,
        destination,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("No safe route found.");
  }

  return response.json();
}