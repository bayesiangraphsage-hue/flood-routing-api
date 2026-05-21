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

warnings.filterwarnings("ignore")

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Flood Routing AI API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL
# =====================================================

class GraphSAGEGRU(nn.Module):

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim=64,
        gru_dim=64,
        dropout=0.3,
    ):
        super().__init__()

        self.sage1 = SAGEConv(
            node_in_dim,
            hidden_dim,
        )

        self.sage2 = SAGEConv(
            hidden_dim,
            hidden_dim,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_dim,
            batch_first=True,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                (gru_dim * 2) + edge_in_dim,
                128,
            ),

            nn.Dropout(dropout),

            nn.ReLU(),

            nn.Linear(
                128,
                64,
            ),

            nn.Dropout(dropout),

            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )

        self.dropout = dropout

    def forward(
        self,
        x_seq,
        message_edge_index,
        edge_label_index,
        edge_attr_seq,
    ):

        L, N, _ = x_seq.shape

        node_embeds = []

        for t in range(L):

            x_t = x_seq[t]

            h = self.sage1(
                x_t,
                message_edge_index,
            )

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            h = F.relu(h)

            h = self.sage2(
                h,
                message_edge_index,
            )

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            h = F.relu(h)

            node_embeds.append(h)

        node_seq = torch.stack(
            node_embeds,
            dim=0,
        ).permute(1, 0, 2)

        gru_out, _ = self.gru(node_seq)

        node_final = gru_out[:, -1, :]

        node_final = F.dropout(
            node_final,
            p=self.dropout,
            training=self.training,
        )

        u = edge_label_index[0]

        v = edge_label_index[1]

        edge_node_emb = torch.cat(
            [
                node_final[u],
                node_final[v],
            ],
            dim=1,
        )

        edge_attr_last = edge_attr_seq[-1]

        edge_input = torch.cat(
            [
                edge_node_emb,
                edge_attr_last,
            ],
            dim=1,
        )

        pred = self.edge_mlp(
            edge_input
        ).squeeze()

        return pred


# =====================================================
# REQUEST MODEL
# =====================================================

class RouteRequest(BaseModel):

    origin: Tuple[float, float]

    destination: Tuple[float, float]

    model: str = "Bayesian MC Dropout"

    risk: float = 0.95


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "best_bayesian_graphsage_gru_mc_dropout_weighted_huber_fixed.pt",
)

TENSORS_PATH = os.path.join(
    DATA_DIR,
    "graph_tensors_nc_snapshots.pt",
)

NODES_PATH = os.path.join(
    DATA_DIR,
    "processed_nodes.gpkg",
)

EDGES_PATH = os.path.join(
    DATA_DIR,
    "processed_edges.gpkg",
)

device = torch.device("cpu")

# =====================================================
# GLOBALS
# =====================================================

model = None

nodes_gdf = None

edges_gdf = None

tensors = None

osmid_to_latlon = {}

ROUTING_ALPHA = 50.0

CVAR_BETA = 0.95

ASSUME_BIDIRECTIONAL = False

# =====================================================
# BUILD GRAPH
# =====================================================

def build_routing_graph(
    edges_input_gdf,
    penalty_col="pred_flood_penalty",
    uncertainty_col="uncertainty",
):

    R = nx.MultiDiGraph()

    z = norm.ppf(CVAR_BETA)

    cvar_multiplier = (
        norm.pdf(z)
        / (1.0 - CVAR_BETA)
    )

    for _, row in edges_input_gdf.iterrows():

        u = str(row["u"])

        v = str(row["v"])

        if u.endswith(".0"):
            u = u[:-2]

        if v.endswith(".0"):
            v = v[:-2]

        # =================================
        # FORCE LENGTH
        # =================================

        length_m = float(
            row.get("length", 100)
        )

        # =================================
        # FORCE TRAVEL TIME
        # =================================

        if (
            "travel_time" in row
            and row["travel_time"] is not None
            and not np.isnan(row["travel_time"])
        ):

            base_time = float(
                row["travel_time"]
            )

        else:

            # fallback
            base_time = (
                length_m / 8.33
            )

        # =================================
        # RISK VALUES
        # =================================

        mu = float(
            row.get(
                penalty_col,
                0.05,
            )
        )

        sigma = float(
            row.get(
                uncertainty_col,
                0.02,
            )
        )

        planned_penalty = (
            mu
            + cvar_multiplier * sigma
        )

        planned_penalty = float(
            np.clip(
                planned_penalty,
                0,
                1,
            )
        )

        planned_cost = (
            base_time
            * (
                1
                + ROUTING_ALPHA
                * planned_penalty
            )
        )

        edge_attrs = dict(
            length=length_m,
            travel_time=base_time,
            planned_cost=planned_cost,
            planned_penalty=planned_penalty,
        )

        R.add_edge(
            u,
            v,
            **edge_attrs,
        )

        oneway_val = str(
            row.get(
                "oneway",
                "no",
            )
        ).strip().lower()

        is_oneway = oneway_val in [
            "yes",
            "true",
            "1",
            "y",
            "f",
        ]

        if (
            not is_oneway
            and ASSUME_BIDIRECTIONAL
        ):

            R.add_edge(
                v,
                u,
                **edge_attrs,
            )

    return R


# =====================================================
# STARTUP
# =====================================================

@app.on_event("startup")
def startup_event():

    global model

    global nodes_gdf

    global edges_gdf

    global tensors

    global osmid_to_latlon

    try:

        print("Loading nodes...")

        nodes_gdf = gpd.read_file(
            NODES_PATH
        )

        print("Loading edges...")

        edges_gdf = gpd.read_file(
            EDGES_PATH
        )

        print("Converting CRS...")

        nodes_wgs84 = nodes_gdf.to_crs(
            "EPSG:4326"
        )

        for _, row in nodes_wgs84.iterrows():

            node_id = str(row["osmid"])

            if node_id.endswith(".0"):
                node_id = node_id[:-2]

            osmid_to_latlon[node_id] = (
                row.geometry.y,
                row.geometry.x,
            )

        print("Loading tensors...")

        tensors = torch.load(
            TENSORS_PATH,
            map_location=device,
        )

        node_in_dim = (
            tensors["node_static_tensor"].shape[-1]
            + len(
                tensors["rainfall_feature_names"]
            )
        )

        edge_in_dim = (
            tensors["edge_static_tensor"].shape[-1]
            + len(
                tensors["rainfall_feature_names"]
            )
        )

        print("Loading model...")

        model = GraphSAGEGRU(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
        ).to(device)

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device,
            )
        )

        model.train()

        print("Backend ready.")

    except Exception as e:

        print(
            f"STARTUP ERROR: {str(e)}"
        )


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "message":
            "Flood Routing Backend Running"
    }


# =====================================================
# ROUTING ENDPOINT
# =====================================================

@app.post("/predict_route")
def predict_route(
    req: RouteRequest
):

    global CVAR_BETA

    try:

        # =================================
        # UPDATE CVAR
        # =================================

        CVAR_BETA = req.risk

        # =================================
        # REBUILD GRAPH
        # =================================

        R_dynamic = build_routing_graph(
            edges_gdf
        )

        # =================================
        # POINTS
        # =================================

        orig_pt = gpd.GeoSeries(
            [
                Point(
                    req.origin[1],
                    req.origin[0],
                )
            ],
            crs="EPSG:4326",
        ).to_crs(
            nodes_gdf.crs
        ).iloc[0]

        dest_pt = gpd.GeoSeries(
            [
                Point(
                    req.destination[1],
                    req.destination[0],
                )
            ],
            crs="EPSG:4326",
        ).to_crs(
            nodes_gdf.crs
        ).iloc[0]

        # =================================
        # NEAREST NODES
        # =================================

        orig_idx = (
            nodes_gdf.geometry.distance(
                orig_pt
            ).idxmin()
        )

        dest_idx = (
            nodes_gdf.geometry.distance(
                dest_pt
            ).idxmin()
        )

        orig_node = str(
            nodes_gdf.iloc[orig_idx]["osmid"]
        )

        dest_node = str(
            nodes_gdf.iloc[dest_idx]["osmid"]
        )

        if orig_node.endswith(".0"):
            orig_node = orig_node[:-2]

        if dest_node.endswith(".0"):
            dest_node = dest_node[:-2]

        # =================================
        # SHORTEST PATH
        # =================================

        route_nodes = nx.shortest_path(
            R_dynamic,
            source=orig_node,
            target=dest_node,
            weight="planned_cost",
        )

        # =================================
        # ROUTE COORDS
        # =================================

        route_coords = []

        for n in route_nodes:

            if n in osmid_to_latlon:

                route_coords.append(
                    osmid_to_latlon[n]
                )

        # =================================
        # ANALYTICS
        # =================================

        distance_km = 0.0

        estimated_time_min = 0.0

        risk_values = []

        risk_edges = []

        for i in range(
            len(route_nodes) - 1
        ):

            u = route_nodes[i]

            v = route_nodes[i + 1]

            edge_data = (
                R_dynamic.get_edge_data(
                    u,
                    v,
                )
            )

            if not edge_data:
                continue

            edge_attrs = list(
                edge_data.values()
            )[0]

            # =============================
            # DISTANCE
            # =============================

            edge_length = float(
                edge_attrs.get(
                    "length",
                    0,
                )
            )

            distance_km += (
                edge_length / 1000
            )

            # =============================
            # TRAVEL TIME
            # =============================

            edge_time = float(
                edge_attrs.get(
                    "travel_time",
                    0,
                )
            )

            estimated_time_min += (
                edge_time / 60
            )

            # =============================
            # RISK
            # =============================

            penalty = float(
                edge_attrs.get(
                    "planned_penalty",
                    0,
                )
            )

            risk_values.append(
                penalty
            )

            # =============================
            # HIGH RISK
            # =============================

            if penalty > 0.2:

                if (
                    u in osmid_to_latlon
                    and v in osmid_to_latlon
                ):

                    risk_edges.append([
                        osmid_to_latlon[u],
                        osmid_to_latlon[v],
                    ])

        # =================================
        # FINAL RISK SCORE
        # =================================

        if len(risk_values) > 0:

            risk_score = float(
                np.mean(risk_values)
            )

        else:

            risk_score = 0.0

        print("===================================")
        print("DISTANCE:", distance_km)
        print("TIME:", estimated_time_min)
        print("RISK:", risk_score)
        print("===================================")

        # =================================
        # RESPONSE
        # =================================

        return {

            "status": "success",

            "route": route_coords,

            "risk_edges": risk_edges,

            "distance_km": float(
                round(
                    distance_km,
                    2,
                )
            ),

            "estimated_time_min": float(
                round(
                    estimated_time_min,
                    2,
                )
            ),

            "risk_score": float(
                round(
                    risk_score,
                    3,
                )
            ),

            "model_used": req.model,

            "risk_beta": req.risk,

            "message":
                f"Safe route generated "
                f"using {req.model}",
        }

    except nx.NetworkXNoPath:

        raise HTTPException(
            status_code=404,
            detail="No safe route found.",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
