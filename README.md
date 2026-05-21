# Intelligent Flood Routing System

An uncertainty-aware flood routing system using:

- FastAPI
- Bayesian GraphSAGE-GRU
- MC Dropout
- GeoPandas
- NetworkX
- React + Next.js
- Leaflet/OpenStreetMap

The system predicts safer flood-aware routes and visualizes them on an interactive map.

---

# Project Structure

```text
flood-routing-api/
│
├── api/
│   └── index.py
│
├── outputs/
│   ├── best_bayesian_graphsage_gru_mc_dropout_weighted_huber_fixed.pt
│   ├── graph_tensors_nc_snapshots.pt
│   ├── processed_nodes.gpkg
│   └── processed_edges.gpkg
│
├── frontend/
│   ├── pages/
│   ├── src/
│   ├── styles/
│   ├── package.json
│   └── next.config.js
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Backend Deployment (Render)

The backend uses:

- FastAPI
- GeoPandas
- PyTorch
- PyTorch Geometric

and is hosted on Render.

Backend URL:

```text
https://flood-routing-api-1.onrender.com
```

---

# Frontend Deployment (Vercel)

The frontend uses:

- Next.js
- React Leaflet
- OpenStreetMap

and is hosted on Vercel.

---

# How the System Works

1. User clicks origin point on map
2. User clicks destination point
3. Frontend sends coordinates to backend
4. Backend computes safest route
5. Route is displayed on map

---

# API Endpoint

## POST `/predict_route`

Example request:

```json
{
  "origin": [10.323, 123.922],
  "destination": [10.315, 123.885]
}
```

Example response:

```json
{
  "status": "success",
  "route": [
    [10.323, 123.922],
    [10.322, 123.921]
  ]
}
```

---

# Frontend Setup

## Install Dependencies

Open terminal inside `frontend/`

```bash
npm install
```

---

## Run Frontend Locally

```bash
npm run dev
```

Frontend runs at:

```text
http://localhost:3000
```

---

# Backend Setup

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend Locally

```bash
uvicorn api.index:app --reload
```

Backend runs at:

```text
http://localhost:8000
```

---

# Render Deployment Steps

1. Push repository to GitHub
2. Open Render Dashboard
3. Create New Web Service
4. Select Docker environment
5. Connect GitHub repository
6. Deploy

---

# Vercel Deployment Steps

1. Open Vercel Dashboard
2. Import GitHub repository
3. Set Root Directory:

```text
frontend
```

4. Deploy

---

# Technologies Used

- FastAPI
- PyTorch
- PyTorch Geometric
- GeoPandas
- NetworkX
- Next.js
- React
- Leaflet
- OpenStreetMap
- Render
- Vercel

---

# Notes

- Render free tier may sleep after inactivity
- First backend request may take 30–60 seconds
- GeoPandas and torch-geometric are hosted on Render because Vercel serverless functions are not ideal for heavy ML/GIS workloads

---

# Author

Lyndrian Shalom Baclayon
