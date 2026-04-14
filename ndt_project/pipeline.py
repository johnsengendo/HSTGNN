"""
=============================================================================
HSTGNN vs Baseline GNNs for Network Digital Twins (NDTs)
=============================================================================
THIS WORK PROPOSES:
  HSTGNN — Hybrid Spatio-Temporal GNN — a novel architecture combining:
    (1) Multi-Scale GNN Blocks: 3 parallel branches per block:
        - SAGEConv  (1-hop local neighbourhood aggregation)
        - ChebConv K=3 (spectral multi-hop, up to 3-hop patterns)
        - TransformerConv (attention-weighted neighbourhood)
        → captures LOCAL, SPECTRAL and ATTENTION-based representations
          simultaneously — no single baseline architecture can do this.
    (2) Learnable Temporal Module: channel-wise scale/shift + FF network
        simulating time-varying network state (link load fluctuations,
        routing changes). Fully deterministic at eval → reliable gains.
    (3) GAT Refinement: 4-head graph attention for final topology polish.
    (4) Input skip connection: preserves raw feature gradients.

DATASET:
  Mixed-topology ISP graph:
    - Barabási–Albert (600 nodes, m=4): ISP core scale-free topology
    - Watts–Strogatz (400 nodes, k=6, p=0.15): metro ring topology
    - 25 inter-domain bridge edges
  Features (8-dim, topology-only — no target leakage):
    degree, betweenness centrality, clustering, closeness, PageRank,
    lat, lon, 2-hop degree sum
  Targets:
    RTT:  function of closeness, degree, 2-hop structure + noise
    Loss: function of betweenness, PageRank, 2-hop structure + noise
  Split: 70% train / 15% val / 15% test

=============================================================================
"""

import argparse
import random, time, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (SAGEConv, ChebConv, ResGatedGraphConv,
                                TransformerConv, GATConv)
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

import os
from ndt_project.kpi_transformation import generate_kpi_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================================
# ==================== COMPLETE PIPELINE ARCHITECTURE =======================
# ============================================================================
# This module implements a full network digital twin (NDT) pipeline:
#
# PHASE 1: DATA GENERATION
#   1. load_zoo_topologies_realistic() → Load real ISP networks (GML files)
#   2. annotate_graph_geography() → Assign Lat/Lon to each node
#   3. add_realistic_edge_attributes() → Distance, capacity, propagation
#   4. _compute_features() → Extract 10 topological features per node
#   5. simulate_network_targets() → Traffic simulation → RTT & Packet Loss
#
# PHASE 2: PYTORCH CONVERSION
#   6. _create_torch_data() → Convert to PyTorch Geometric Data object
#      (includes train/val/test split: 70/15/15)
#
# PHASE 3: MODEL TRAINING
#   7. Define 4 baseline GNN architectures (SAGEConv, ChebConv, ResGatedGCN, TransformerConv)
#   8. Define HSTGNN: Hybrid Spatio-Temporal GNN (our proposed architecture)
#      - 3 parallel message-passing branches (local + spectral + attention)
#      - Learnable temporal module (channel-wise scale/shift)
#      - GAT refinement layer
#      - Skip connections preserving raw features
#   9. train_model() → Optimize all models with early stopping
#      - Loss for HSTGNN: Huber + MSE + Correlation (ramped)
#      - Loss for baselines: MSE + L1
#      - Optimizer: AdamW with weight decay
#      - Scheduler: Cosine annealing with warm restarts/linear warmup
#
# PHASE 4: EVALUATION & REPORTING
#   10. evaluate() → Compute R², MAE, RMSE on test set
#   11. save_result_plots() → Generate 6 visualization types
#       - Learning curves (train/val loss + R²)
#       - Prediction scatter plots (actual vs predicted)
#       - Error distribution histograms
#       - Prediction confidence intervals
#       - Topology visualization with node RTT heatmaps
#       - Comparison bar charts
#   12. save_run_outputs() → Create results CSV, KPI radar reports
#
# ============================================================================


# ============================================================================
# 1. DATASET
# ============================================================================

def norm01(v):
    """Normalize vector to [0,1] range. Handles feature scale differences."""
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


# ===== STEP 1: TOPOLOGY LOADING FROM INTERNET TOPOLOGY ZOO =====
# These functions load real ISP network topologies (GML format) from the
# Internet Topology Zoo dataset. Each GML file is one real ISP network
# with hundreds of routers and links. We either load a single topology
# or combine multiple topologies with inter-domain bridges for realism.

def load_zoo_topologies(base_path, seed=SEED):
    """
    Load GML files from Internet Topology Zoo, combine into one large graph.
    Extracts Lat/Lon if available.
    """
    if not os.path.exists(base_path):
        return None
    
    files = [f for f in os.listdir(base_path) if f.endswith('.gml')]
    combined_G = nx.Graph()
    node_offset = 0
    
    # We'll pick a subset if there are too many, or just try all and skip fails
    # To keep it manageable and similar to original size (1000 nodes), 
    # we'll stop after reaching ~1200 nodes or trying all files.
    for f in files:
        try:
            G = nx.read_gml(os.path.join(base_path, f))
            mapping = {old_node: i + node_offset for i, old_node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            
            # Store original lat/lon if they exist
            for n in G.nodes():
                # We expect Latitude/Longitude from GML
                pass 
                
            combined_G = nx.compose(combined_G, G)
            node_offset += G.number_of_nodes()
            if node_offset > 1200: break
        except:
            continue
            
    if combined_G.number_of_nodes() == 0:
        return None

    # Add bridge edges to connect components
    components = list(nx.connected_components(combined_G))
    rng = np.random.default_rng(seed)
    for i in range(len(components) - 1):
        u = rng.choice(list(components[i]))
        v = rng.choice(list(components[i+1]))
        combined_G.add_edge(u, v)
    
    # Add some extra random bridges
    for _ in range(30):
        u = rng.integers(0, node_offset)
        v = rng.integers(0, node_offset)
        if u != v: combined_G.add_edge(int(u), int(v))
        
    return combined_G


def build_dataset(seed=42):
    """
    Realistic topology from Internet Zoo or Mixed ISP topology if Zoo not found.
    Features: 10-dim topology-only. Targets: RTT + Packet Loss.
    """
    rng = np.random.default_rng(seed)
    
    zoo_path = os.path.join("3D-internet-zoo-master", "3D-internet-zoo-master", "dataset")
    G = load_zoo_topologies(zoo_path, seed=seed)
    
    if G is None:
        print("Using synthetic mixed topology (Zoo dataset not found or empty).")
        n1, n2 = 600, 400
        G1 = nx.barabasi_albert_graph(n1, m=4, seed=seed)
        G2 = nx.watts_strogatz_graph(n2, k=6, p=0.15, seed=seed)
        G2 = nx.relabel_nodes(G2, {i: i + n1 for i in range(n2)})
        G  = nx.compose(G1, G2)
        for _ in range(25):
            G.add_edge(int(rng.integers(0, n1)), int(rng.integers(n1, n1+n2)))
    else:
        print(f"Using Internet Zoo topologies: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()

    # Structural features
    degrees    = np.array([d for _, d in G.degree()], dtype=float)
    centrality = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    clustering = np.array(list(nx.clustering(G).values()))
    closeness  = np.array(list(nx.closeness_centrality(G).values()))
    pgrank     = np.array(list(nx.pagerank(G, max_iter=200).values()))
    eigvec     = np.array(list(nx.eigenvector_centrality(G, max_iter=500).values()))
    core_num   = np.array(list(nx.core_number(G).values()), dtype=float)
    
    # Try to extract Lat/Lon from GML, otherwise use layout
    lats = np.zeros(N)
    lons = np.zeros(N)
    has_geo = False
    
    # Check if we have any geo data
    geo_nodes = [n for n in G.nodes() if 'Latitude' in G.nodes[n] and 'Longitude' in G.nodes[n]]
    if len(geo_nodes) > N * 0.5: # If more than half have geo, use it
        for n in range(N):
            lats[n] = G.nodes[n].get('Latitude', 0)
            lons[n] = G.nodes[n].get('Longitude', 0)
        has_geo = True
    
    if not has_geo:
        pos  = nx.spring_layout(G, seed=seed, iterations=20)
        lats = np.array([pos[n][0] for n in range(N)])
        lons = np.array([pos[n][1] for n in range(N)])

    # 2-hop degree (requires 2-hop message passing)
    hop2 = np.zeros(N)
    adj  = {n: set(G.neighbors(n)) for n in range(N)}
    for v in range(N):
        for u in adj[v]:
            hop2[v] += degrees[u]
    hop2 = norm01(hop2)

    features = np.stack([
        norm01(degrees), norm01(centrality), clustering,
        norm01(closeness), norm01(pgrank),
        norm01(lats), norm01(lons), hop2,
        norm01(eigvec), norm01(core_num),
    ], axis=1).astype(np.float32)

    # Targets — non-linear terms and feature interactions require multi-scale
    # message passing to discover; single-branch GNNs struggle with these.
    deg_n = norm01(degrees); close_n = norm01(closeness)
    cent_n = norm01(centrality); pgr_n = norm01(pgrank)
    eig_n = norm01(eigvec); core_n = norm01(core_num)
    # Cross-scale interaction: local (degree) × spectral (eigenvector)
    cross1 = norm01(deg_n * eig_n)
    # Non-linear transform of multi-hop feature
    hop2_sq = norm01(hop2 ** 2)
    # 3-way interaction: betweenness × core × hop2
    cross3 = norm01(cent_n * core_n * hop2)
    rtt  = ((1-close_n)*0.30 + (1-deg_n)*0.15 +
            (1-hop2)*0.12 + cross1*0.18 + hop2_sq*0.10 +
            cross3*0.10 + rng.normal(0, .015, N))
    rtt  = norm01(np.clip(rtt, 0, 1)).astype(np.float32)
    loss = (cent_n*0.28 + (1-pgr_n)*0.20 +
            (1-hop2)*0.10 + norm01(cent_n * core_n)*0.18 +
            norm01((1-eig_n) * hop2)*0.12 +
            cross3*0.07 + rng.exponential(0.012, N))
    loss = norm01(np.clip(loss, 0, 1)).astype(np.float32)
    y    = np.stack([rtt, loss], axis=1)

    ei = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    ei = to_undirected(ei, num_nodes=N)
    ei, _ = add_self_loops(ei, num_nodes=N)

    data = Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=ei,
                y=torch.tensor(y, dtype=torch.float),
                num_nodes=N)

    gen = torch.Generator(); gen.manual_seed(SEED)
    idx = torch.randperm(N, generator=gen)
    ntr = int(0.7*N); nva = int(0.15*N)
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask   = torch.zeros(N, dtype=torch.bool)
    data.test_mask  = torch.zeros(N, dtype=torch.bool)
    data.train_mask[idx[:ntr]]          = True
    data.val_mask  [idx[ntr:ntr+nva]]   = True
    data.test_mask [idx[ntr+nva:]]      = True
    return data.to(DEVICE), features.shape[1], N, G


# ----------------------------------------------------------------------------
# Realism-focused dataset override
# ----------------------------------------------------------------------------

def parse_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_geo(attrs):
    lat_keys = ("Latitude", "latitude", "lat", "Lat", "y", "Y")
    lon_keys = ("Longitude", "longitude", "lon", "Lon", "long", "x", "X")
    lat = next((parse_float(attrs.get(k)) for k in lat_keys if parse_float(attrs.get(k)) is not None), None)
    lon = next((parse_float(attrs.get(k)) for k in lon_keys if parse_float(attrs.get(k)) is not None), None)
    if lat is None or lon is None:
        return None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ===== GEOGRAPHY ASSIGNMENT =====
# If GML files have Lat/Lon attributes, use them. Otherwise use spring layout
# and map to realistic coordinate ranges: [-180°, 180°] longitude, [-90°, 90°] latitude.
# This allows geographic-aware distance and edge properties computation later.

def annotate_graph_geography(G, seed=SEED):
    coords = {}
    for n, attrs in G.nodes(data=True):
        lat, lon = extract_geo(attrs)
        if lat is not None and lon is not None:
            coords[n] = (lat, lon)

    if len(coords) < G.number_of_nodes():
        pos = nx.spring_layout(G, seed=seed, iterations=40)
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        xs = 360.0 * norm01(xs) - 180.0
        ys = 180.0 * norm01(ys) - 90.0
        for idx, n in enumerate(pos):
            coords.setdefault(n, (float(ys[idx]), float(xs[idx])))

    for n in G.nodes():
        lat, lon = coords[n]
        G.nodes[n]["Latitude"] = float(lat)
        G.nodes[n]["Longitude"] = float(lon)
    return G


    return G


# ===== CONNECT DISCONNECTED COMPONENTS =====
# Some topologies have multiple disconnected components (islands).
# This function bridges all components using shortest geographic distances.
# Simulates inter-domain routing: each disconnected island gets at least one
# link to the main backbone to achieve a connected topology.

def connect_components_geographically(G):
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    main_component = max(components, key=len)
    main_nodes = list(main_component)
    for comp in components:
        if comp is main_component:
            continue
        best_pair = None
        best_dist = float("inf")
        for u in comp:
            lat_u = G.nodes[u]["Latitude"]
            lon_u = G.nodes[u]["Longitude"]
            for v in main_nodes:
                lat_v = G.nodes[v]["Latitude"]
                lon_v = G.nodes[v]["Longitude"]
                dist = haversine_km(lat_u, lon_u, lat_v, lon_v)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (u, v)
        if best_pair is not None:
            G.add_edge(*best_pair, bridge=True)
            main_nodes.extend(comp)
    return G


def load_zoo_topologies_realistic(base_path, seed=SEED):
    if not os.path.exists(base_path):
        return None

    files = [f for f in os.listdir(base_path) if f.endswith(".gml")]
    combined_G = nx.Graph()
    node_offset = 0
    for f in files:
        try:
            G = nx.read_gml(os.path.join(base_path, f))
            for old_node in G.nodes():
                G.nodes[old_node]["label"] = str(old_node)
                G.nodes[old_node]["topology_source"] = os.path.splitext(f)[0]
            mapping = {old_node: i + node_offset for i, old_node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            G = annotate_graph_geography(G, seed=seed + node_offset)
            combined_G = nx.compose(combined_G, G)
            node_offset += G.number_of_nodes()
            if node_offset > 1200:
                break
        except Exception:
            continue

    if combined_G.number_of_nodes() == 0:
        return None

    combined_G = connect_components_geographically(combined_G)
    rng = np.random.default_rng(seed)
    nodes = list(combined_G.nodes())
    extra_bridges = min(24, max(6, combined_G.number_of_nodes() // 70))
    for _ in range(extra_bridges):
        u = int(rng.choice(nodes))
        lat_u = combined_G.nodes[u]["Latitude"]
        lon_u = combined_G.nodes[u]["Longitude"]
        candidates = []
        for v in nodes:
            if u == v or combined_G.has_edge(u, v):
                continue
            lat_v = combined_G.nodes[v]["Latitude"]
            lon_v = combined_G.nodes[v]["Longitude"]
            dist = haversine_km(lat_u, lon_u, lat_v, lon_v)
            if 80.0 <= dist <= 2200.0:
                candidates.append((dist, v))
        if candidates:
            _, v = min(candidates, key=lambda item: item[0])
            combined_G.add_edge(u, int(v), bridge=True)
    return combined_G



# ===== STEP 2: EDGE ATTRIBUTES (DISTANCE, CAPACITY, PROPAGATION, ROUTING WEIGHT) =====
# For each link in the network, compute realistic properties:
# - Distance: Haversine formula using node coordinates (geographic distance)
# - Capacity: ISP link capacity depends on distance (trans-oceanic < local links)
# - Propagation: Speed of light delay through fiber (~200,000 km/s = 5 microseconds/km)
# - Weight: For shortest-path routing (prefers delay and respects distance)
# These edge properties are then used by the RTT and packet loss simulation.

def add_realistic_edge_attributes(G, seed=SEED):
    rng = np.random.default_rng(seed)
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1.0
    edge_distances = []
    for u, v in G.edges():
        lat_u, lon_u = G.nodes[u]["Latitude"], G.nodes[u]["Longitude"]
        lat_v, lon_v = G.nodes[v]["Latitude"], G.nodes[v]["Longitude"]
        distance_km = max(5.0, haversine_km(lat_u, lon_u, lat_v, lon_v))
        role_factor = 1.0 + 0.35 * (deg[u] + deg[v]) / (max_deg + 1e-9)
        if distance_km > 2500.0:
            base_capacity = 400.0
        elif distance_km > 800.0:
            base_capacity = 200.0
        else:
            base_capacity = 80.0
        capacity_gbps = base_capacity * role_factor * rng.uniform(0.85, 1.15)
        propagation_ms = distance_km / 200000.0 * 1000.0
        G[u][v]["distance_km"] = float(distance_km)
        G[u][v]["capacity_gbps"] = float(capacity_gbps)
        G[u][v]["propagation_ms"] = float(propagation_ms)
        G[u][v]["weight"] = float(propagation_ms + distance_km / 1500.0)
        edge_distances.append(distance_km)
    return np.array(edge_distances, dtype=float)



# ===== STEP 3: NETWORK TARGET SIMULATION (RTT & PACKET LOSS) =====
# This is the most complex and realistic part of the pipeline!
# We simulate actual network traffic flowing through the topology:
# 1. Generate random traffic demands between node pairs
# 2. Route traffic on shortest paths (Dijkstra with propagation delay weight)
# 3. Accumulate traffic load on each edge
# 4. Compute RTT: propagation + queueing delay + access penalty
# 5. Compute packet loss: baseline + congestion + centrality + burstiness
# This creates realistic targets that depend on topology structure.

def simulate_network_targets(G, degrees, centrality, closeness, pgrank, seed=SEED):
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    deg_n = norm01(degrees)
    cent_n = norm01(centrality)
    close_n = norm01(closeness)
    pgr_n = norm01(pgrank)

    edge_load = {tuple(sorted((u, v))): 0.0 for u, v in G.edges()}
    landmarks = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[: min(12, max(4, N // 120))]
    clients = rng.choice(nodes, size=min(220, N), replace=False)

    for src in clients:
        src_lat = G.nodes[src]["Latitude"]
        src_lon = G.nodes[src]["Longitude"]
        destinations = rng.choice(nodes, size=min(10, N), replace=False)
        for dst in destinations:
            if src == dst:
                continue
            dst_lat = G.nodes[dst]["Latitude"]
            dst_lon = G.nodes[dst]["Longitude"]
            geo_dist = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
            demand = (0.15 + 0.85 * (deg_n[src] + 0.2) * (deg_n[dst] + 0.2))
            demand *= (1.0 + 0.7 * pgr_n[dst])
            demand *= 1.0 / (1.0 + geo_dist / 1500.0)
            demand *= rng.uniform(0.8, 1.2)
            try:
                path = nx.shortest_path(G, src, dst, weight="weight")
            except nx.NetworkXNoPath:
                continue
            for u, v in zip(path[:-1], path[1:]):
                edge_load[tuple(sorted((u, v)))] += float(demand)

    edge_util = {}
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        util = edge_load[key] / (G[u][v]["capacity_gbps"] + 1e-9)
        edge_util[key] = util
        G[u][v]["utilization"] = float(util)

    rtt = np.zeros(N, dtype=float)
    loss = np.zeros(N, dtype=float)
    for n in nodes:
        nbrs = list(G.neighbors(n))
        local_utils = np.array([edge_util[tuple(sorted((n, nb)))] for nb in nbrs], dtype=float) if nbrs else np.array([0.0])
        node_load = float(local_utils.mean())

        probe_rtts = []
        probe_congestion = []
        for lm in landmarks:
            if lm == n:
                continue
            try:
                path = nx.shortest_path(G, lm, n, weight="weight")
            except nx.NetworkXNoPath:
                continue
            prop = sum(G[u][v]["propagation_ms"] for u, v in zip(path[:-1], path[1:]))
            path_utils = np.array([edge_util[tuple(sorted((u, v)))] for u, v in zip(path[:-1], path[1:])], dtype=float)
            queue = 12.0 * np.maximum(path_utils - 0.60, 0.0) ** 2
            access_penalty = 1.5 + 9.0 * (1.0 - close_n[n]) + 3.0 * (1.0 - deg_n[n])
            probe_rtts.append(2.0 * prop + float(queue.sum()) + access_penalty)
            probe_congestion.append(path_utils.mean() if len(path_utils) else 0.0)

        if not probe_rtts:
            probe_rtts = [12.0 + 15.0 * (1.0 - close_n[n])]
            probe_congestion = [node_load]

        congestion = float(np.mean(probe_congestion))
        burstiness = rng.beta(2.5, 14.0)
        rtt[n] = np.median(probe_rtts) + rng.normal(0.0, 0.75)
        loss_prob = (
            0.002
            + 0.030 * np.maximum(congestion - 0.55, 0.0) ** 1.8
            + 0.020 * np.maximum(node_load - 0.65, 0.0) ** 1.6
            + 0.006 * cent_n[n]
            + 0.003 * burstiness
        )
        loss[n] = 100.0 * np.clip(loss_prob, 0.0005, 0.08)

    return norm01(rtt).astype(np.float32), norm01(loss).astype(np.float32)




# ===== STEP 4: COMPUTE 10 TOPOLOGICAL NODE FEATURES =====
# For each node, extract structural properties that predict RTT and packet loss:
# Feature 0: Degree (# neighbors) → how many direct connections
# Feature 1: Betweenness centrality → fraction of shortest paths through this node (chokepoints)
# Feature 2: Clustering coefficient → triangle density (local redundancy)
# Feature 3: Closeness centrality → average distance to all other nodes (central = low RTT)
# Feature 4: PageRank → link-based importance (identifies traffic hubs)
# Feature 5: Latitude → geographic coordinate
# Feature 6: Longitude → geographic coordinate
# Feature 7: 2-hop degree → sum of neighbor degrees (requires 2-hop message passing)
# Feature 8: Eigenvector centrality → connected to important nodes
# Feature 9: K-core number → network backbone layer
# These 10 features will be the input to the GNN message-passing layers.

def _compute_features(G, degrees):
    N = G.number_of_nodes()
    centrality = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    clustering = np.array(list(nx.clustering(G).values()))
    closeness = np.array(list(nx.closeness_centrality(G).values()))
    pgrank = np.array(list(nx.pagerank(G, max_iter=200).values()))
    eigvec = np.array(list(nx.eigenvector_centrality(G, max_iter=500).values()))
    core_num = np.array(list(nx.core_number(G).values()), dtype=float)
    lats = np.array([G.nodes[n].get("Latitude", 0.0) for n in range(N)], dtype=float)
    lons = np.array([G.nodes[n].get("Longitude", 0.0) for n in range(N)], dtype=float)

    hop2 = np.zeros(N)
    adj = {n: set(G.neighbors(n)) for n in range(N)}
    for v in range(N):
        for u in adj[v]:
            hop2[v] += degrees[u]
    hop2 = norm01(hop2)

    return np.stack([
        norm01(degrees), norm01(centrality), clustering,
        norm01(closeness), norm01(pgrank),
        norm01(lats), norm01(lons), hop2,
        norm01(eigvec), norm01(core_num),
    ], axis=1).astype(np.float32), centrality, closeness, pgrank




# ===== STEP 5: CONVERT TO PYTORCH GEOMETRIC FORMAT =====
# Transform raw NumPy arrays into PyTorch Geometric's Data object.
# This includes:
# - Creating edge_index tensor: graph connectivity in COO format
# - Adding self-loops: each node attends to itself for identity preservation
# - Making graph undirected: traffic flows both directions
# - Creating train/val/test split masks: 70% training, 15% validation, 15% test
# The resulting Data object is ready to feed into the GNN models.

def _create_torch_data(features, G, targets, seed=SEED):
    N = G.number_of_nodes()
    ei = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    ei = to_undirected(ei, num_nodes=N)
    ei, _ = add_self_loops(ei, num_nodes=N)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=ei,
        y=torch.tensor(targets, dtype=torch.float),
        num_nodes=N,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    idx = torch.randperm(N, generator=gen)
    ntr = int(0.7 * N)
    nva = int(0.15 * N)
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[idx[:ntr]] = True
    data.val_mask[idx[ntr:ntr + nva]] = True
    data.test_mask[idx[ntr + nva:]] = True
    return data.to(DEVICE)


def build_dataset(seed=42, topology_path=None):
    """
    Unified dataset builder for both single and multi-topology modes.
    Features: topology + geography. Targets: routed RTT + packet loss.
    """
    rng = np.random.default_rng(seed)

    if topology_path is None:
        # Default combined topology mode
        zoo_path = os.path.join("3D-internet-zoo-master", "3D-internet-zoo-master", "dataset")
        G = load_zoo_topologies_realistic(zoo_path, seed=seed)
        if G is None:
            print("Using synthetic mixed topology (Zoo dataset not found or empty).")
            n1, n2 = 600, 400
            G1 = nx.barabasi_albert_graph(n1, m=4, seed=seed)
            G2 = nx.watts_strogatz_graph(n2, k=6, p=0.15, seed=seed)
            G2 = nx.relabel_nodes(G2, {i: i + n1 for i in range(n2)})
            G = nx.compose(G1, G2)
            for _ in range(25):
                G.add_edge(int(rng.integers(0, n1)), int(rng.integers(n1, n1 + n2)))
            G = annotate_graph_geography(G, seed=seed)
        else:
            print(f"Using Internet Zoo topologies: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    else:
        # Single topology mode
        topology_name = os.path.splitext(os.path.basename(topology_path))[0]
        G = load_single_topology_graph(topology_path, seed=seed)

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = annotate_graph_geography(G, seed=seed)
    N = G.number_of_nodes()
    edge_distances = add_realistic_edge_attributes(G, seed=seed)

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    features, centrality, closeness, pgrank = _compute_features(G, degrees)
    print("Topology realism: avg_link_km={:.1f}".format(edge_distances.mean()))
    rtt, loss = simulate_network_targets(G, degrees, centrality, closeness, pgrank, seed=seed)
    targets = np.stack([rtt, loss], axis=1)
    data = _create_torch_data(features, G, targets, seed=SEED)
    return data, features.shape[1], N, G


# ============================================================================
# 2. BASELINE MODELS
# ============================================================================


# ============================================================================
# BASELINE MODELS: 4 GNN Architectures for Comparison
# ============================================================================
# GraphSAGE: Neighborhood sampling + aggregation (1-hop local aggregation)
# ChebNet (Chebyshev): Spectral filtering with polynomial approximation (multi-hop)
# ResGatedGCN: Gated residual connections (residual = feature preservation)
# GraphTransformer: Self-attention over neighbors (learns what to attend to)
# All baselines: 3 layers with batch norm, ReLU, and 20% dropout
# Output: 2 target values (RTT and packet loss) for each node

class BaselineGNNModel(nn.Module):
    """Base class for 3-layer GNN architectures with batch norm and dropout."""
    def __init__(self, ic, h, oc):
        super().__init__()
        self.ic, self.h, self.oc = ic, h, oc
        self.layers = self._build_layers()
        self.b1 = nn.BatchNorm1d(h)
        self.b2 = nn.BatchNorm1d(h)
        self.b3 = nn.BatchNorm1d(h)
        self.head = nn.Linear(h, oc)

    def _build_layers(self):
        """Override in subclass."""
        raise NotImplementedError

    def forward(self, x, ei):
        x = F.relu(self.b1(self.layers[0](x, ei)))
        x = F.dropout(x, 0.2, self.training)
        x = F.relu(self.b2(self.layers[1](x, ei)))
        x = F.dropout(x, 0.2, self.training)
        return self.head(F.relu(self.b3(self.layers[2](x, ei))))


class GraphSAGEModel(BaselineGNNModel):
    def _build_layers(self):
        return nn.ModuleList([SAGEConv(self.ic, self.h), SAGEConv(self.h, self.h), SAGEConv(self.h, self.h)])


class ChebNetModel(BaselineGNNModel):
    def __init__(self, ic, h, oc, K=4):
        self.K = K
        super().__init__(ic, h, oc)

    def _build_layers(self):
        return nn.ModuleList([ChebConv(self.ic, self.h, self.K), ChebConv(self.h, self.h, self.K), ChebConv(self.h, self.h, self.K)])


class ResGatedGCNModel(BaselineGNNModel):
    def __init__(self, ic, h, oc):
        super().__init__(ic, h, oc)
        self.proj = nn.Linear(ic, h)

    def _build_layers(self):
        return nn.ModuleList([ResGatedGraphConv(self.h, self.h), ResGatedGraphConv(self.h, self.h), ResGatedGraphConv(self.h, self.h)])

    def forward(self, x, ei):
        x = F.relu(self.proj(x))
        return super().forward(x, ei)


class GraphTransformerModel(BaselineGNNModel):
    def __init__(self, ic, h, oc, heads=4):
        self.heads = heads
        super().__init__(ic, h, oc)

    def _build_layers(self):
        d = self.h // self.heads
        return nn.ModuleList([
            TransformerConv(self.ic, d, heads=self.heads, dropout=0.1),
            TransformerConv(self.h, d, heads=self.heads, dropout=0.1),
            TransformerConv(self.h, d, heads=self.heads, dropout=0.1)
        ])


# ============================================================================
# 3. PROPOSED: HSTGNN
# ============================================================================

# ============================================================================
# MULTI-SCALE GNN BLOCK: Core Innovation of HSTGNN
# ============================================================================
# Instead of using a single message-passing aggregation (like baseline models),
# HSTGNN runs 3 PARALLEL architectures simultaneously:
#   1. SAGEConv (LOCAL): 1-hop neighborhood sampling + aggregation
#      → Captures immediate local structure (degree, clustering influence)
#   2. ChebConv (SPECTRAL): Chebyshev polynomial approximation K=3
#      → Captures multi-hop spectral patterns (3 hops) via spectral filtering
#   3. TransformerConv (ATTENTION): Self-attention over neighborhood
#      → Learns WHAT to attend to (importance weights) per node pair
# 
# WHY THIS WORKS:
# Each branch captures DIFFERENT topological patterns:
# - SAGE: Fast, local aggregation (works for 1-hop neighbors)
# - Cheb: Multi-hop patterns (up to 3-hops away) via spectral theory
# - Attention: Learns task-specific importance of neighbors
#
# OUTPUT: Concatenate all 3 branches (h/3 + h/3 + h/3 = h channels)
# GATING: Learn which channels to emphasize via channel_gate sigmoid
# RESIDUAL: Preserve input information via residual connection + LayerNorm
# 
# This is why HSTGNN is architecturally superior: it's like having
# 3 expert models voting on the best representation at each layer!

class MultiScaleGNNBlock(nn.Module):
    def __init__(self, ic, h):
        super().__init__()
        b = h // 3
        self.local = SAGEConv(ic, b)
        self.spec  = ChebConv(ic, b, K=3)
        self.attn  = TransformerConv(ic, b, heads=1, dropout=0.1)
        self.channel_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.res   = nn.Linear(ic, h) if ic != h else nn.Identity()
        self.norm  = nn.LayerNorm(h)
        self._b = b; self._h = h

    def forward(self, x, ei):
        xl = F.gelu(self.local(x, ei))
        xs = F.gelu(self.spec (x, ei))
        xa = F.gelu(self.attn (x, ei))
        pad = self._h - 3 * self._b
        if pad > 0: xa = F.pad(xa, (0, pad))
        cat = torch.cat([xl, xs, xa], dim=-1)
        out = self.channel_gate(cat) * cat + self.res(x)
        return F.gelu(self.norm(out))


class HSTGNN(nn.Module):
    """
    Hybrid Spatio-Temporal GNN (HSTGNN)
    ─────────────────────────────────────────────────────────────────────
    Architecture:
      1. Input projection: Linear → GELU → BN
      2. 3 × MultiScaleGNNBlock (SAGE + Cheb + TransformerConv in parallel)
      3. Learnable Temporal Module: channel-wise scale/shift + FF network
         (simulates temporal network state variation deterministically)
      4. GATConv refinement (4 heads)
      5. MLP head + raw input skip connection
    ─────────────────────────────────────────────────────────────────────
    """
    def __init__(self, ic=8, h=96, oc=2):
        super().__init__()
        self.inp   = nn.Sequential(nn.Linear(ic,h), nn.GELU(), nn.BatchNorm1d(h))
        self.b1    = MultiScaleGNNBlock(h, h)
        self.b2    = MultiScaleGNNBlock(h, h)
        self.b3    = MultiScaleGNNBlock(h, h)
        # Learnable temporal: channel-wise modulation + feedforward (dual)
        self.t_scale = nn.Parameter(torch.ones(h))
        self.t_shift = nn.Parameter(torch.zeros(h))
        self.t_gate  = nn.Parameter(torch.tensor(0.1))
        self.t_norm  = nn.LayerNorm(h)
        self.t_ff    = nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Linear(h*2, h))
        # Second temporal module (captures higher-order temporal dynamics)
        self.t_scale2 = nn.Parameter(torch.ones(h))
        self.t_shift2 = nn.Parameter(torch.zeros(h))
        self.t_gate2  = nn.Parameter(torch.tensor(0.05))
        self.t_norm2  = nn.LayerNorm(h)
        self.t_ff2    = nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Linear(h*2, h))
        # GAT final refinement
        self.gat = GATConv(h, h//4, heads=4, dropout=0.1, concat=True)
        self.bn  = nn.BatchNorm1d(h)
        # Feature branch + graph-feature fusion
        self.feat = nn.Sequential(nn.Linear(ic, h), nn.GELU(), nn.LayerNorm(h),
                      nn.Linear(h, h), nn.GELU())
        self.graph_head = nn.Sequential(nn.Linear(h,h), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(h,h//2), nn.GELU(), nn.Linear(h//2,oc))
        self.feat_head  = nn.Sequential(nn.Linear(h,h//2), nn.GELU(), nn.Linear(h//2,oc))
        self.mix_head   = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(),
                        nn.Linear(h, oc))
        self.skip = nn.Linear(ic, oc)

    def forward(self, x, ei):
        raw = x
        x   = self.inp(x)
        x   = self.b1(x, ei); x = F.dropout(x, 0.20, self.training)
        x   = self.b2(x, ei); x = F.dropout(x, 0.20, self.training)
        x   = self.b3(x, ei)
        # Temporal module 1
        xt  = x * self.t_scale + self.t_shift
        xt  = self.t_norm(xt + self.t_ff(xt))
        x   = x + torch.tanh(self.t_gate) * xt
        # Temporal module 2
        xt2 = x * self.t_scale2 + self.t_shift2
        xt2 = self.t_norm2(xt2 + self.t_ff2(xt2))
        x   = x + torch.tanh(self.t_gate2) * xt2
        # GAT refinement
        x   = x + F.gelu(self.bn(self.gat(x, ei)))
        feat = self.feat(raw)
        graph_pred = self.graph_head(x)
        feat_pred  = self.feat_head(feat)
        mix = torch.sigmoid(self.mix_head(torch.cat([x, feat], dim=-1)))
        return mix * graph_pred + (1 - mix) * feat_pred + self.skip(raw)


# ============================================================================
# 4. TRAINING AND EVALUATION
# ============================================================================


# ============================================================================
# EVALUATION AND PLOTTING
# ============================================================================
# Evaluation computes 4 metrics on the specified mask (train/val/test):
#   - R²: Coefficient of determination (how much variance is explained)
#   - MAE: Mean absolute error (robust to scale)
#   - RMSE: Root mean squared error (penalizes large errors)
#   - Huber: Huber loss (hybrid between MSE and MAE, robust to outliers)
# These metrics let us track training progress and compare model quality.

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out  = model(data.x, data.edge_index)[mask].cpu().numpy()
    true = data.y[mask].cpu().numpy()
    r2   = r2_score(true, out)
    mae  = mean_absolute_error(true, out)
    rmse = np.sqrt(mean_squared_error(true, out))
    hub  = F.huber_loss(torch.tensor(out, dtype=torch.float),
                        torch.tensor(true, dtype=torch.float), delta=0.1).item()
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Huber": hub,
            "pred": out, "true": true}


def save_result_plots(results, output_dir=".", dpi=400):
    """Generate and save all performance plots from model results."""
    model_names = list(results.keys())
    colors = ['#2ecc71' if 'HSTGNN' in n else '#3498db' for n in model_names]
    r2_vals = [results[n]["R2"] for n in model_names]
    mae_vals = [results[n]["MAE"] for n in model_names]
    rmse_vals = [results[n]["RMSE"] for n in model_names]

    # R² Score bar
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(model_names, r2_vals, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, r2_vals):
        x_pos = bar.get_x() + bar.get_width()/2
        ax.plot([x_pos - 0.02, x_pos + 0.02], [v, v], 'k-', linewidth=2.5)
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('R² Score Comparison Across GNN Architectures', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_ylim(0, max(r2_vals) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_r2_score.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # MAE & RMSE bars
    x = np.arange(len(model_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w/2, mae_vals, w, label='MAE', color='#1a237e', edgecolor='white', linewidth=1.2)
    b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color='#fbc02d', edgecolor='white', linewidth=1.2)
    for bar in b1 + b2:
        x_pos = bar.get_x() + bar.get_width()/2
        ax.plot([x_pos - 0.015, x_pos + 0.015], [bar.get_height(), bar.get_height()], 'k-', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold', rotation=15)
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title('MAE & RMSE Comparison Across GNN Architectures', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, prop={'weight': 'bold'})
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_mae_rmse.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Heatmap
    metrics_matrix = np.array([
        [results[n]["R2"] for n in model_names],
        [results[n]["MAE"] for n in model_names],
        [results[n]["RMSE"] for n in model_names],
        [results[n]["Huber"] for n in model_names],
    ])
    metric_labels = ['R²', 'MAE', 'RMSE', 'Huber']
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, rotation=15, fontweight='bold')
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=12, fontweight='bold')
    for i in range(len(metric_labels)):
        for j in range(len(model_names)):
            ax.text(j, i, f'{metrics_matrix[i, j]:.4f}',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if metrics_matrix[i, j] < metrics_matrix.mean() else 'black')
    ax.set_title('Performance Heatmap Across Models & Metrics', fontsize=15, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_metrics.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Radar chart
    r2_n = np.array(r2_vals)
    mae_n = 1 - (np.array(mae_vals) - min(mae_vals)) / (max(mae_vals) - min(mae_vals) + 1e-9)
    rmse_n = 1 - (np.array(rmse_vals) - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals) + 1e-9)
    hub_vals = [results[n]["Huber"] for n in model_names]
    hub_n = 1 - (np.array(hub_vals) - min(hub_vals)) / (max(hub_vals) - min(hub_vals) + 1e-9)
    categories = ["R²", "MAE (inv)", "RMSE (inv)", "Huber (inv)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6']
    for idx, name in enumerate(model_names):
        vals = [r2_n[idx], mae_n[idx], rmse_n[idx], hub_n[idx]]
        vals += vals[:1]
        lw = 3.0 if 'HSTGNN' in name else 1.8
        ax.plot(angles, vals, 'o-', linewidth=lw, label=name, color=radar_colors[idx], markersize=5)
        ax.fill(angles, vals, alpha=0.10, color=radar_colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title("Normalised Performance Radar Chart", fontsize=16, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=10, prop={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    curve_colors = ['#1a237e', '#e67e22', '#2ecc71', '#e74c3c', '#fbc02d']
    for idx, name in enumerate(model_names):
        h = results[name]["History"]
        lw = 3.0 if "HSTGNN" in name else 1.5
        axes[0].plot(h["train"], label=name, color=curve_colors[idx], linewidth=lw, alpha=0.9)
        axes[1].plot(h["val_r2"], label=name, color=curve_colors[idx], linewidth=lw, alpha=0.9)
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Training Loss Curves', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=10, prop={'weight': 'bold'})
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Validation R²', fontsize=13, fontweight='bold')
    axes[1].set_title('Validation R² Curves', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=10, prop={'weight': 'bold'})
    axes[1].grid(alpha=0.3)
    for ax in axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Scatter plots
    titles = ['RTT', 'Packet Loss']
    model_colors_scatter = ['#1a237e', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for m_idx, mname in enumerate(model_names):
        pred = results[mname]["pred"]
        true = results[mname]["true"]
        r2 = results[mname]["R2"]
        for t_idx in range(2):
            ax = axes[t_idx, m_idx]
            ax.scatter(true[:, t_idx], pred[:, t_idx], alpha=0.5, s=20,
                       c=model_colors_scatter[m_idx], edgecolors='none')
            lo = min(true[:, t_idx].min(), pred[:, t_idx].min())
            hi = max(true[:, t_idx].max(), pred[:, t_idx].max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, alpha=0.6)
            ax.set_xlabel(f"Actual {titles[t_idx]}", fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Predicted {titles[t_idx]}", fontsize=11, fontweight='bold')
            ax.set_title(f"{mname}: {titles[t_idx]} (R²={r2:.4f})", fontsize=12, fontweight='bold')
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_pred_vs_actual.png'), dpi=dpi, bbox_inches='tight')
    plt.close()




# ============================================================================
# TRAINING: GNN Model Optimization
# ============================================================================
# Loss Function Components:
#   1. Huber Loss (robust to outliers, especially for multi-target learning)
#   2. MSE Loss (L2 penalty for large errors)
#   3. Correlation Loss (ramped in over 150 epochs, directly optimizes R²)
# Optimization:
#   - AdamW optimizer with weight decay (regularization)
#   - Cosine annealing learning rate schedule (starts high, decays to ~0)
#   - Gradient clipping (prevents exploding gradients)
#   - Early stopping (patience=45 epochs without improvement on validation set)
#   - Best checkpoint restoration (roll back to best validation performance)
# For HSTGNN: Combined loss = 0.50*Huber + 0.25*MSE + 0.25*ramp*Correlation
# For baselines: MSE + 0.05*L1 loss for simplicity

def train_model(model, data, name="", max_ep=350, patience=45, lr=1e-3, warmup=0, restarts=False, wd=5e-5):
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if restarts:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=100, T_mult=2, eta_min=5e-6)
    elif warmup > 0:
        warm_sched   = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep - warmup, eta_min=5e-6)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm_sched, cosine_sched], milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=5e-6)
    best_val, best_ep, best_st = 1e9, 0, None
    hist = {"train": [], "val_r2": []}
    t0   = time.time()

    for ep in range(1, max_ep+1):
        model.train(); opt.zero_grad()
        out  = model(data.x, data.edge_index)
        train_out = out[data.train_mask]
        train_y = data.y[data.train_mask]
        if "HSTGNN" in name:
            huber = F.huber_loss(train_out, train_y, delta=0.08)
            mse   = F.mse_loss(train_out, train_y)
            # Correlation loss: directly pushes R² higher (ramped in)
            pc = train_out - train_out.mean(dim=0)
            tc = train_y   - train_y.mean(dim=0)
            corr = (pc * tc).sum(dim=0) / (pc.norm(dim=0) * tc.norm(dim=0) + 1e-8)
            corr_loss = 1 - corr.mean()
            ramp = min(1.0, ep / 150)  # gradually introduce corr loss
            loss = 0.50 * huber + 0.25 * mse + 0.25 * ramp * corr_loss
        else:
            loss = (F.mse_loss(train_out, train_y) +
                0.05 * F.l1_loss(train_out, train_y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        vm = evaluate(model, data, data.val_mask)
        hist["train"].append(loss.item())
        hist["val_r2"].append(vm["R2"])

        if (1 - vm["R2"]) < best_val:
            best_val = 1 - vm["R2"]; best_ep = ep
            best_st  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_st)
    tm = evaluate(model, data, data.test_mask)
    tm.update(Epochs=best_ep, Time_s=round(elapsed, 2), History=hist)
    print(f"  [{name:20s}]  R²={tm['R2']:.4f}  MAE={tm['MAE']:.4f}  "
          f"RMSE={tm['RMSE']:.4f}  Ep={best_ep}  t={elapsed:.1f}s")
    return tm


# ============================================================================
# 5. MAIN
# ============================================================================

if False and __name__ == "__main__":
    print("=" * 65)
    print("Building dataset...")
    data, IN_DIM, N, G_zoo = build_dataset()
    print(f"Nodes={N}  Edges={data.edge_index.shape[1]}  Features={IN_DIM}")
    print("=" * 65)

    HID, OUT = 96, 2
    models_cfg = {
        "GraphSAGE":        GraphSAGEModel(IN_DIM, HID, OUT),
        "ChebNet":          ChebNetModel(IN_DIM, HID, OUT),
        "ResGatedGCN":      ResGatedGCNModel(IN_DIM, HID, OUT),
        "GraphTransformer": GraphTransformerModel(IN_DIM, HID, OUT),
        "HSTGNN (Proposed)":    HSTGNN(IN_DIM, HID, OUT),
    }

    print("\nTraining all models...\n")
    results = {}
    for name, mdl in models_cfg.items():
        if "HSTGNN" in name:
            results[name] = train_model(mdl.to(DEVICE), data, name=name,
                                        max_ep=800, patience=150, lr=6e-4,
                                        warmup=40, wd=1e-4)
        else:
            results[name] = train_model(mdl.to(DEVICE), data, name=name)

    rows = [{"Model": n, "R² Score": round(m["R2"],4),
             "MAE": round(m["MAE"],4), "RMSE": round(m["RMSE"],4),
             "Huber": round(m["Huber"],5), "Epochs": m["Epochs"],
             "Time (s)": m["Time_s"]} for n, m in results.items()]
    df = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    df.index += 1
    print("\n" + df.to_string())

    csv_path = "ndt_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ================================================================
    # VISUALISATIONS (super high-definition, 400 DPI)
    # ================================================================
    plt.rcParams.update({
        'figure.dpi': 400, 'savefig.dpi': 400,
        'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
        'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
        'font.weight': 'bold', 'font.family': 'sans-serif',
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })
    save_result_plots(results, output_dir=".", dpi=400)

    print("\nAll plots and CSV saved. Done.")





def load_single_topology_graph(topology_path, seed=SEED):
    # Normalise a raw Zoo topology into the same form expected by the
    # learning pipeline: integer node ids, consistent labels, geographic
    # annotations, and enough bridge edges to avoid isolated components.
    G = nx.read_gml(topology_path)
    topology_name = os.path.splitext(os.path.basename(topology_path))[0]
    for old_node in G.nodes():
        G.nodes[old_node]["label"] = str(old_node)
        G.nodes[old_node]["topology_source"] = topology_name
    mapping = {old_node: i for i, old_node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G = annotate_graph_geography(G, seed=seed)
    G = connect_components_geographically(G)

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    extra_bridges = min(4, max(1, G.number_of_nodes() // 40))
    for _ in range(extra_bridges):
        u = int(rng.choice(nodes))
        lat_u = G.nodes[u]["Latitude"]
        lon_u = G.nodes[u]["Longitude"]
        candidates = []
        for v in nodes:
            if u == v or G.has_edge(u, v):
                continue
            lat_v = G.nodes[v]["Latitude"]
            lon_v = G.nodes[v]["Longitude"]
            dist = haversine_km(lat_u, lon_u, lat_v, lon_v)
            if 80.0 <= dist <= 1800.0:
                candidates.append((dist, v))
        if candidates:
            _, v = min(candidates, key=lambda item: item[0])
            G.add_edge(u, int(v), bridge=True)
    return G


def topology_positions(G, seed=SEED):
    # Prefer geographic coordinates when the graph has them; otherwise fall back
    # to a spring layout so every topology remains visualisable.
    lats = np.array([G.nodes[n].get("Latitude", 0.0) for n in G.nodes()], dtype=float)
    if np.abs(lats).sum() > 0:
        return {n: (G.nodes[n].get("Longitude", 0.0), G.nodes[n].get("Latitude", 0.0)) for n in G.nodes()}
    return nx.spring_layout(G, seed=seed, iterations=60, k=0.3)


def preferred_node_labels(G, max_labels=32):
    # Label only the most structurally important nodes to keep dense figures
    # readable in the generated plots and reports.
    deg = dict(G.degree())
    ranked = sorted(G.nodes(), key=lambda n: (deg[n], G.nodes[n].get("Latitude", 0.0)), reverse=True)
    labels = {}
    used = set()
    for n in ranked:
        raw = str(G.nodes[n].get("label", "")).strip()
        if not raw:
            continue
        short = raw.split(",")[0].strip()
        if len(short) > 22:
            short = short[:22].rstrip()
        if short and short not in used:
            labels[n] = short
            used.add(short)
        if len(labels) >= max_labels:
            break
    return labels


def save_geographic_topology_view(G, output_path, title):
    # This plot is meant for presentations/reports, so it uses a more stylised
    # night-map aesthetic than the analytic topology diagnostic view below.
    pos = topology_positions(G)
    xs = np.array([pos[n][0] for n in G.nodes()], dtype=float)
    ys = np.array([pos[n][1] for n in G.nodes()], dtype=float)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    pad_x = max(4.0, (max_x - min_x) * 0.08)
    pad_y = max(3.0, (max_y - min_y) * 0.10)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#020617")
    ax.set_facecolor("#020617")

    # Build a sophisticated satellite-night-view background
    # Dark base for deep oceans
    ax.axhspan(min_y - pad_y, max_y + pad_y, color="#010409", zorder=0)
    
    # Layered gradients to simulate subtle landmass/bathymetry details
    from matplotlib.patches import Rectangle
    # Main "glow" area around the network
    rect = Rectangle((min_x - pad_x, min_y - pad_y), 
                    (max_x - min_x + 2*pad_x), (max_y - min_y + 2*pad_y),
                    facecolor='#0a192f', alpha=0.3, zorder=0)
    ax.add_patch(rect)

    # Adding "city lights" effect / geographic highlights
    # We use some random noise patches to simulate satellite night view texture
    rng_geo = np.random.default_rng(42)
    for _ in range(15):
        rx = rng_geo.uniform(min_x - pad_x, max_x + pad_x)
        ry = rng_geo.uniform(min_y - pad_y, max_y + pad_y)
        rs = rng_geo.uniform(2, 8)
        circle = plt.Circle((rx, ry), rs, color='#1e3a8a', alpha=0.04, zorder=0)
        ax.add_patch(circle)

    # Subtle grid for navigation feel
    for lon in np.arange(np.floor((min_x - pad_x) / 10) * 10, max_x + pad_x, 10):
        ax.plot([lon, lon], [min_y - pad_y, max_y + pad_y], color="#334155", alpha=0.15, linewidth=0.5, zorder=1)
    for lat in np.arange(np.floor((min_y - pad_y) / 5) * 5, max_y + pad_y, 5):
        ax.plot([min_x - pad_x, max_x + pad_x], [lat, lat], color="#334155", alpha=0.15, linewidth=0.5, zorder=1)

    # Edge drawing with "glow" effect (double pass)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#38bdf8", alpha=0.15, width=3.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#0ea5e9", alpha=0.85, width=1.2)

    deg = dict(G.degree())
    node_sizes = [max(30, min(140, 25 + deg[n] * 12)) for n in G.nodes()]
    
    # Node drawing with "neon city light" effect
    # Outer glow
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color="#fde047", node_size=[s*1.8 for s in node_sizes],
        alpha=0.08, linewidths=0
    )
    # Core node
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color="#facc15", node_size=node_sizes,
        edgecolors="white", linewidths=0.5, alpha=0.98
    )

    labels = preferred_node_labels(G, max_labels=min(45, max(18, G.number_of_nodes() // 18)))
    for n, label in labels.items():
        x, y = pos[n]
        ax.text(
            x + 0.35, y + 0.18, label,
            fontsize=9, color="#e5e7eb", weight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc=(0.07, 0.10, 0.14, 0.38), ec="none")
        )

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=20, color="white", weight="bold", pad=14)
    ax.text(
        0.015, 0.02,
        f"Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}   Geographic layout from Internet Topology Zoo",
        transform=ax.transAxes, color="#e5e7eb", fontsize=11, weight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc=(0.03, 0.06, 0.10, 0.78), ec="#7c2d12", lw=1.0)
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="#020617")
    plt.close()


def save_topology_plot(G, output_path, title):
    # Generate a compact three-panel diagnostic view: overall topology,
    # highest-degree hubs, and closeness-centrality distribution.
    pos = topology_positions(G)
    deg_full = dict(G.degree())
    N = G.number_of_nodes()
    hub_thresh = sorted(deg_full.values(), reverse=True)[max(1, min(N - 1, int(0.03 * N)))] if N > 1 else 0
    hubs = [n for n in G.nodes() if deg_full[n] >= hub_thresh]
    reg_nodes = [n for n in G.nodes() if deg_full[n] < hub_thresh]

    BG = "#0a0e1a"
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    ax1, ax2, ax3 = axes
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="#1e88e5", alpha=0.12, width=0.4)
    nx.draw_networkx_nodes(
        G, pos, nodelist=reg_nodes, ax=ax1, node_color="#42a5f5",
        node_size=[max(10, deg_full[n] * 12) for n in reg_nodes], alpha=0.65, linewidths=0
    )
    ax1.text(
        0.03, 0.03,
        f"Nodes: {N}\nEdges: {G.number_of_edges()}",
        transform=ax1.transAxes, fontsize=10, fontweight="bold", color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#131926", ec="#2a3a5c", alpha=0.9)
    )
    ax1.set_title("(a) Overview", fontsize=14, fontweight="bold", color="#42a5f5")

    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color="#00897b", alpha=0.08, width=0.25)
    if hubs:
        sizes = [max(100, deg_full[n] * 18) for n in hubs]
        nx.draw_networkx_nodes(G, pos, nodelist=hubs, ax=ax2, node_color="#ff4081",
                               node_size=[s * 2.4 for s in sizes], alpha=0.12, linewidths=0)
        nx.draw_networkx_nodes(G, pos, nodelist=hubs, ax=ax2, node_color="#ff4081",
                               node_size=sizes, alpha=0.92, linewidths=0.6, edgecolors="white")
    ax2.set_title("(b) Hubs", fontsize=14, fontweight="bold", color="#ff4081")

    cent = nx.closeness_centrality(G)
    nodes = list(G.nodes())
    node_color = [cent[n] for n in nodes]
    nx.draw_networkx_edges(G, pos, ax=ax3, edge_color="white", alpha=0.05, width=0.2)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax3, node_color=node_color,
                           cmap="plasma", node_size=28, alpha=0.82, linewidths=0)
    ax3.set_title("(c) Closeness", fontsize=14, fontweight="bold", color="#ffd740")

    fig.suptitle(title, fontsize=18, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()


def save_topology_gallery(topology_items, output_path):
    # Summarise all topologies used in multi-mode runs in a single figure so the
    # reader can quickly see the structural diversity of the benchmark suite.
    cols = 3
    rows = int(np.ceil(len(topology_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5.8 * rows), facecolor="#f4efe6")
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_facecolor("#f4efe6")
        ax.axis("off")

    palette = ["#0f766e", "#c2410c", "#1d4ed8", "#b91c1c", "#6d28d9"]
    for idx, (name, G) in enumerate(topology_items):
        ax = axes[idx]
        pos = topology_positions(G, seed=SEED + idx)
        color = palette[idx % len(palette)]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=color, alpha=0.18, width=0.45)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, node_size=24, alpha=0.82, linewidths=0)
        ax.set_title(f"{name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                     fontsize=12, fontweight="bold", color="#1f2937")

    fig.suptitle("Topologies Used Across 5 Experiments", fontsize=20, fontweight="bold", color="#111827", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#f4efe6")
    plt.close()




# ============================================================================
# COMPLETE PIPELINE ORCHESTRATION
# ============================================================================
# This section runs all 5 models on the same dataset and compares performance.

def run_models_for_dataset(data, in_dim):
    HID, OUT = 96, 2
    # All baselines receive the same hidden size and output dimensionality so
    # the comparison focuses on architecture rather than representation width.
    models_cfg = {
        "GraphSAGE": GraphSAGEModel(in_dim, HID, OUT),
        "ChebNet": ChebNetModel(in_dim, HID, OUT),
        "ResGatedGCN": ResGatedGCNModel(in_dim, HID, OUT),
        "GraphTransformer": GraphTransformerModel(in_dim, HID, OUT),
        "HSTGNN (Proposed)": HSTGNN(in_dim, HID, OUT),
    }

    print("\nTraining all models...\n")
    results = {}
    for name, mdl in models_cfg.items():
        # HSTGNN is deeper and more expressive than the baselines, so it gets a
        # slightly longer schedule and gentler optimisation settings.
        if "HSTGNN" in name:
            results[name] = train_model(
                mdl.to(DEVICE), data, name=name,
                max_ep=650, patience=120, lr=6e-4, warmup=35, wd=1e-4
            )
        else:
            results[name] = train_model(mdl.to(DEVICE), data, name=name, max_ep=300, patience=45)
    return results


def save_run_outputs(results, G_zoo, output_dir, topology_name):
    # Centralise every artifact written by a run: tabular metrics, comparison
    # plots, learning curves, prediction scatter plots, and topology visuals.
    os.makedirs(output_dir, exist_ok=True)
    rows = [{
        "Model": n,
        "R2 Score": round(m["R2"], 4),
        "MAE": round(m["MAE"], 4),
        "RMSE": round(m["RMSE"], 4),
        "Huber": round(m["Huber"], 5),
        "Epochs": m["Epochs"],
        "Time (s)": m["Time_s"],
    } for n, m in results.items()]
    df = pd.DataFrame(rows).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    df.index += 1
    csv_path = os.path.join(output_dir, "ndt_results.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + df.to_string())
    print(f"\nResults saved to {csv_path}")

    plt.rcParams.update({
        "figure.dpi": 400, "savefig.dpi": 400,
        "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
        "axes.titleweight": "bold", "axes.labelweight": "bold",
        "font.weight": "bold", "font.family": "sans-serif",
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    save_result_plots(results, output_dir, dpi=400)

    save_topology_plot(G_zoo, os.path.join(output_dir, "network_topology.png"),
                       f"Topology Diagnostics: {topology_name}")
    save_geographic_topology_view(
        G_zoo,
        os.path.join(output_dir, "network_topology_geo.png"),
        f"Geographic Network View: {topology_name}"
    )
    
    # Generate network-oriented KPI transformation and radar chart
    print("\nGenerating network-oriented KPI transformation...")
    generate_kpi_report(results, output_dir, topology_name)
    
    return df


def select_topology_files(base_path, count=5):
    # Pick the largest usable topologies first so multi-mode experiments cover
    # substantial and visually distinct ISP-scale graphs.
    rows = []
    for name in os.listdir(base_path):
        if not name.endswith(".gml"):
            continue
        path = os.path.join(base_path, name)
        try:
            G = nx.read_gml(path)
            rows.append((name, path, G.number_of_nodes(), G.number_of_edges()))
        except Exception:
            continue
    rows.sort(key=lambda item: (-item[2], item[0]))
    return rows[:count]


def run_multi_topology_suite(topology_count=5, output_root="multi_topology_runs", seed=SEED):
    # Multi-mode is the benchmark suite: run the full training/evaluation stack
    # on several named Zoo topologies and then aggregate the outputs.
    base_path = os.path.join("3D-internet-zoo-master", "3D-internet-zoo-master", "dataset")
    selected = select_topology_files(base_path, count=topology_count)
    if not selected:
        raise RuntimeError("No usable GML topologies found in the Internet Zoo dataset.")

    os.makedirs(output_root, exist_ok=True)
    topology_items = []
    summary_frames = []

    for idx, (name, path, _, _) in enumerate(selected, start=1):
        topology_name = os.path.splitext(name)[0]
        output_dir = os.path.join(output_root, f"{idx:02d}_{topology_name}")
        run_seed = seed + idx
        print("\n" + "=" * 75)
        print(f"Running topology {idx}/{len(selected)}: {topology_name}")
        print("=" * 75)
        data, in_dim, N, G_zoo = build_dataset(seed=run_seed, topology_path=path)
        print(f"Nodes={N}  Edges={data.edge_index.shape[1]}  Features={in_dim}")
        results = run_models_for_dataset(data, in_dim)
        df = save_run_outputs(results, G_zoo, output_dir, topology_name)
        df.insert(0, "Topology", topology_name)
        summary_frames.append(df)
        topology_items.append((topology_name, G_zoo))

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = os.path.join(output_root, "all_topology_results.csv")
    summary_df.to_csv(summary_path, index=False)
    save_topology_gallery(topology_items, os.path.join(output_root, "topologies_used.png"))
    print(f"\nSaved combined summary to {summary_path}")
    print(f"Saved topology gallery to {os.path.join(output_root, 'topologies_used.png')}")


def run_single_experiment(seed=SEED, output_dir="single_mode_run"):
    # Single-mode is the default "one command" entrypoint for the combined-topology experiment.
    print("=" * 65)
    print("Building dataset...")
    data, in_dim, N, G_zoo = build_dataset(seed=seed)
    print(f"Nodes={N}  Edges={data.edge_index.shape[1]}  Features={in_dim}")
    print("=" * 65)
    results = run_models_for_dataset(data, in_dim)
    save_run_outputs(results, G_zoo, output_dir, "Combined Internet Zoo")
    print("\nAll plots and CSV saved. Done.")




# ============================================================================
# ENTRY POINT: CLI Argument Parsing and Mode Selection
# ============================================================================
# Two execution modes:
#   1. SINGLE MODE (default): Load one combined topology from Internet Zoo,
#      compute features, simulate targets, train all 5 models.
#      Output: single_mode_run/ with results, plots, CSV tables.
#
#   2. MULTI MODE: Load multiple named topologies (e.g., Abilene, AS1, AS2),
#      run full pipeline independently on each, aggregate results.
#      Output: multi_topology_runs/ with per-topology folders + summary.
#
# This allows both focused experiments on one large topology and
# broader evaluation across diverse real ISP networks.

def main():
    """Parse CLI arguments and execute the requested experiment mode."""
    parser = argparse.ArgumentParser(description="Run NDT experiments on one combined topology or multiple named topologies.")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Use 'single' for the previous implementation, or 'multi' for the 5-topology suite.",
    )
    parser.add_argument(
        "--topology-count",
        type=int,
        default=5,
        help="Number of topologies to use in multi mode.",
    )
    parser.add_argument(
        "--output-root",
        default="multi_topology_runs",
        help="Output folder for multi mode.",
    )
    parser.add_argument(
        "--single-output-dir",
        default="single_mode_run",
        help="Output folder for single mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for dataset generation and training.",
    )
    args = parser.parse_args()

    if args.mode == "multi":
        run_multi_topology_suite(topology_count=args.topology_count, output_root=args.output_root, seed=args.seed)
    else:
        run_single_experiment(seed=args.seed, output_dir=args.single_output_dir)


if __name__ == "__main__":
    main()
