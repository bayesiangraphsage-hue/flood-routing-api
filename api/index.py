import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import networkx as nx
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class GraphSAGEGRU(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, gru_dim=64, dropout=0.3):
        super().__init__()
        self.sage1 = SAGEConv(node_in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_dim, batch_first=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear((gru_dim * 2) + edge_in_dim, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.dropout = dropout

    def forward(self, x_seq, message_edge_index, edge_label_index, edge_attr_seq):
        L, N, _ = x_seq.shape
        node_embeds = []
        for t in range(L):
            x_t = x_seq[t]
            h = self.sage1(x_t, message_edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(h)
            h = self.sage2(h, message_edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(h)
            node_embeds.append(h)
        node_seq = torch.stack(node_embeds, dim=0).permute(1, 0, 2)
        gru_out, _ = self.gru(node_seq)
        node_final = gru_out[:, -1, :]
        node_final = F.dropout(node_final, p=self.dropout, training=self.training)
        u = edge_label_index[0]
        v = edge_label_index[1]
        edge_node_emb = torch.cat([node_final[u], node_final[v]], dim=1)
        edge_attr_last = edge_attr_seq[-1]
        edge_input = torch.cat([edge_node_emb, edge_attr_last], dim=1)
        pred = self.edge_mlp(edge_input).squeeze()
        return pred

# ==========================================
# 2. API SETUP & STATE
# ==========================================
app = FastAPI(title="Intelligent Flood Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    origin: Tuple[float, float]
    destination: Tuple[float, float]

# Ensure paths work in Vercel (points to the root outputs folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(DATA_DIR, "best_bayesian_graphsage_gru_mc_dropout_weighted_huber_fixed.pt")
TENSORS_PATH = os.path.join(DATA_DIR, "graph_tensors_nc_snapshots.pt")
NODES_PATH = os.path.join(DATA_DIR, "processed_nodes.gpkg")
EDGES_PATH = os.path.join(DATA_DIR, "processed_edges.gpkg")

device = torch.device("cpu")  # Vercel does not have GPUs

# Global state
model = None
nodes_gdf = None
edges_gdf = None
osmid_to_latlon = {}
tensors = None
R_interactive = None

ROUTING_ALPHA = 50.0
CVAR_BETA = 0.95
ASSUME_BIDIRECTIONAL = False

def build_routing_graph(edges_input_gdf, penalty_col="pred_flood_penalty", uncertainty_col="uncertainty"):
    R = nx.MultiDiGraph()
    z = norm.ppf(CVAR_BETA)
    cvar_multiplier = norm.pdf(z) / (1.0 - CVAR_BETA)

    for _, row in edges_input_gdf.iterrows():
        u = str(row["u"])
        v = str(row["v"])
        if u.endswith('.0'): u = u[:-2]
        if v.endswith('.0'): v = v[:-2]

        base_time = float(row.get("travel_time", row["length"] / (30 * 1000 / 3600)))
        mu = float(row.get(penalty_col, 0.0))
        sigma = float(row.get(uncertainty_col, 0.0))
        planned_penalty = mu + cvar_multiplier * sigma
        planned_penalty = float(np.clip(planned_penalty, 0, 1))

        planned_cost = base_time * (1 + ROUTING_ALPHA * planned_penalty)
        edge_attrs = dict(
            length=float(row["length"]),
            travel_time=base_time,
            planned_cost=planned_cost,
            planned_penalty=planned_penalty
        )
        R.add_edge(u, v, **edge_attrs)

        oneway_val = str(row.get("oneway", "no")).strip().lower()
        is_oneway = oneway_val in ["yes", "true", "1", "y", "f"]
        if not is_oneway and ASSUME_BIDIRECTIONAL:
            R.add_edge(v, u, **edge_attrs)
    return R

@app.on_event("startup")
def load_resources():
    global model, nodes_gdf, edges_gdf, osmid_to_latlon, tensors, R_interactive
    try:
        nodes_gdf = gpd.read_file(NODES_PATH)
        edges_gdf = gpd.read_file(EDGES_PATH)

        nodes_wgs84 = nodes_gdf.to_crs("EPSG:4326")
        for _, row in nodes_wgs84.iterrows():
            n_id = str(row['osmid'])
            if n_id.endswith('.0'): n_id = n_id[:-2]
            osmid_to_latlon[n_id] = (row['geometry'].y, row['geometry'].x)

        tensors = torch.load(TENSORS_PATH, map_location=device)
        node_in_dim = tensors["node_static_tensor"].shape[-1] + len(tensors["rainfall_feature_names"])
        edge_in_dim = tensors["edge_static_tensor"].shape[-1] + len(tensors["rainfall_feature_names"])

        model = GraphSAGEGRU(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.train() # MC Dropout active

        R_interactive = build_routing_graph(edges_gdf)
    except Exception as e:
        print(f"Startup warning: {e}")

@app.get("/")
def read_root():
    return {"message": "Flood Routing ML Backend is running!"}

@app.post("/predict_route")
def predict_route(req: RouteRequest):
    if nodes_gdf is None or model is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded.")

    try:
        orig_pt = gpd.GeoSeries([Point(req.origin[1], req.origin[0])], crs="EPSG:4326").to_crs(nodes_gdf.crs).iloc[0]
        dest_pt = gpd.GeoSeries([Point(req.destination[1], req.destination[0])], crs="EPSG:4326").to_crs(nodes_gdf.crs).iloc[0]

        orig_idx = nodes_gdf.geometry.distance(orig_pt).idxmin()
        dest_idx = nodes_gdf.geometry.distance(dest_pt).idxmin()

        orig_node = str(nodes_gdf.iloc[orig_idx]['osmid'])
        dest_node = str(nodes_gdf.iloc[dest_idx]['osmid'])
        if orig_node.endswith('.0'): orig_node = orig_node[:-2]
        if dest_node.endswith('.0'): dest_node = dest_node[:-2]

        route_nodes = nx.shortest_path(
            R_interactive,
            source=orig_node,
            target=dest_node,
            weight="planned_cost"
        )

        route_coords = [osmid_to_latlon[n] for n in route_nodes]

        return {
            "status": "success",
            "route": route_coords,
            "message": f"Routed {len(route_nodes)} nodes from {orig_node} to {dest_node}"
        }
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found between the selected points.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))