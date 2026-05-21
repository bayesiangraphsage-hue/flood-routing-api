import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  useMapEvents,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";

function ClickHandler({ setOrigin, setDestination, origin, destination }) {
  useMapEvents({
    click(e) {
      const coords = [e.latlng.lat, e.latlng.lng];

      if (!origin) {
        setOrigin(coords);
      } else if (!destination) {
        setDestination(coords);
      }
    },
  });

  return null;
}

export default function MapComponent() {
  const [origin, setOrigin] = useState(null);
  const [destination, setDestination] = useState(null);
  const [route, setRoute] = useState([]);

  const getRoute = async () => {
    try {
      const response = await fetch(
        "https://flood-routing-api-1.onrender.com/predict_route"
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

      const data = await response.json();

      if (data.route) {
        setRoute(data.route);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to get route");
    }
  };

  const resetMap = () => {
    setOrigin(null);
    setDestination(null);
    setRoute([]);
  };

  return (
    <div>
      <div style={{ marginBottom: "10px", textAlign: "center" }}>
        <button onClick={getRoute}>
          Get Route
        </button>

        <button onClick={resetMap} style={{ marginLeft: "10px" }}>
          Reset
        </button>
      </div>

      <MapContainer
        center={[10.3157, 123.8854]}
        zoom={13}
        style={{ height: "80vh", width: "100%" }}
      >
        <TileLayer
          attribution="OpenStreetMap"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <ClickHandler
          origin={origin}
          destination={destination}
          setOrigin={setOrigin}
          setDestination={setDestination}
        />

        {origin && <Marker position={origin} />}
        {destination && <Marker position={destination} />}

        {route.length > 0 && (
          <Polyline positions={route} />
        )}
      </MapContainer>
    </div>
  );
}