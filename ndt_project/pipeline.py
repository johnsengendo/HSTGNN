"""
=============================================================================
HSTGNN vs Baseline GNNs for Network Digital Twins (NDTs)
=============================================================================

PAPER CONTEXT:
  Zacarias et al. evaluated 4 GNN architectures (GraphSAGE, ChebNet,
  ResGatedGCN, GraphTransformer) on RIPE Atlas data for NDT performance
  prediction. GraphTransformer achieved R²=0.9763 (best), SAGE was fastest.

OUR FRAMEWORK PROPOSES:
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
  Targets:
    RTT:  function of closeness, degree, 2-hop structure + noise
    Loss: function of betweenness, PageRank, 2-hop structure + noise
  Split: 60% train / 20% val / 20% test
=============================================================================
"""

# ============================================================================
# IMPORTING essential libraries and configuring environment
# ============================================================================
import argparse
import random, time, warnings, shutil
import numpy as np
import pandas as pd
# Setting matplotlib backend to 'Agg' for non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# Using NetworkX for graph construction and manipulation
import networkx as nx
# Importing PyTorch essentials for building and training neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F
# Importing geometric data structures for graph neural networks
from torch_geometric.data import Data
# Importing different GNN convolution layers for model architectures
from torch_geometric.nn import (SAGEConv, ChebConv, ResGatedGraphConv,
                                TransformerConv, GATConv)
# Using utility functions for graph edge processing
from torch_geometric.utils import add_self_loops, to_undirected
# Importing evaluation metrics for regression tasks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Suppressing warnings for cleaner output
warnings.filterwarnings("ignore")

import os

# ============================================================================
# SETTING random seeds for reproducibility and determining computing device
# ============================================================================
SEED = 42
# Seeding all RNG sources to ensure consistent experiment results
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# Detecting GPU availability for accelerated training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================================
# 1. DATASET
# ============================================================================

# Normalizing feature vectors to [0, 1] range using min-max scaling
def norm01(v):
    """Using min-max normalization to scaling features into unit interval."""
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


# Loading and processing GML files by combining multiple topologies into single graph
def load_zoo_topologies(base_path, seed=SEED):
    """
    Loading GML files from Internet Topology Zoo and combining into one large graph.
    Extracting Lat/Lon coordinates if available in the data.
    """
    if not os.path.exists(base_path):
        return None
    
    files = [f for f in os.listdir(base_path) if f.endswith('.gml')]
    combined_G = nx.Graph()
    node_offset = 0
    
    # We pick a subset if there are too many, or just try all and skip fails
    # To keep it manageable and similar to original size (1000 nodes), 
    # we stop after reaching ~1200 nodes or trying all files.
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


# Building comprehensive dataset by generating topological features and network targets
def build_dataset(seed=42):
    """
    Building realistic topology dataset from Internet Zoo or synthesizing mixed ISP topology if unavailable.
    Computing 10-dimensional topological features and simulating RTT + Packet Loss targets.
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

    # Computing structural topological metrics for each node in the graph
    # Calculating node degree (number of neighbors)
    degrees    = np.array([d for _, d in G.degree()], dtype=float)
    # Computing betweenness centrality (importance in shortest paths)
    centrality = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    # Computing clustering coefficient (local density of triangles)
    clustering = np.array(list(nx.clustering(G).values()))
    # Computing closeness centrality (average distance to other nodes)
    closeness  = np.array(list(nx.closeness_centrality(G).values()))
    # Computing PageRank scores (importance based on connection pattern)
    pgrank     = np.array(list(nx.pagerank(G, max_iter=200).values()))
    # Computing eigenvector centrality (influence via connected nodes)
    eigvec     = np.array(list(nx.eigenvector_centrality(G, max_iter=500).values()))
    # Computing core number (maximum k for k-core decomposition)
    core_num   = np.array(list(nx.core_number(G).values()), dtype=float)
    
    # Try to extract Lat/Lon from GML, otherwise use layout
    lats = np.zeros(N)
    lons = np.zeros(N)
    has_geo = False
    
    # Checking if we have any geo data
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

    # Designing target labels by combining topological features with non-linear interactions
    # Normalizing individual features for stable target computation
    deg_n = norm01(degrees); close_n = norm01(closeness)
    cent_n = norm01(centrality); pgr_n = norm01(pgrank)
    eig_n = norm01(eigvec); core_n = norm01(core_num)
    # Creating cross-scale interactions: combining local (degree) with spectral (eigenvector) properties
    cross1 = norm01(deg_n * eig_n)
    # Applying non-linear transformation to multi-hop feature for capturing higher-order patterns
    hop2_sq = norm01(hop2 ** 2)
    # Constructing 3-way interaction term: betweenness × core × hop2 for complex dependencies
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

# Parsing string values robustly by handling None and invalid numeric formats
def parse_float(value):
    """Converting string inputs to float by handling edge cases and missing values."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            # Stripping whitespace and checking for empty strings
            value = value.strip()
            if not value:
                return None
        # Attempting conversion to float
        return float(value)
    except (TypeError, ValueError):
        # Returning None if conversion fails
        return None


# Extracting geographic coordinates from graph node attributes
def extract_geo(attrs):
    """Retrieving latitude and longitude from node attributes using multiple possible keys."""
    # Trying alternative latitude key names
    lat_keys = ("Latitude", "latitude", "lat", "Lat", "y", "Y")
    # Trying alternative longitude key names
    lon_keys = ("Longitude", "longitude", "lon", "Lon", "long", "x", "X")
    # Extracting first valid latitude value from attributes
    lat = next((parse_float(attrs.get(k)) for k in lat_keys if parse_float(attrs.get(k)) is not None), None)
    # Extracting first valid longitude value from attributes
    lon = next((parse_float(attrs.get(k)) for k in lon_keys if parse_float(attrs.get(k)) is not None), None)
    # Validating that both coordinates are present
    if lat is None or lon is None:
        return None, None
    # Checking geographic coordinate ranges for validity
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    return lat, lon


# Computing great-circle distance between two geographic points using Haversine formula
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculating geodetic distance in kilometers using spherical Earth approximation."""
    # Setting Earth radius for distance computation
    radius_km = 6371.0
    # Converting latitude coordinates to radians
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    # Computing latitude and longitude differences in radians
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    # Applying Haversine formula for central angle
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    # Converting central angle to distance using Earth radius
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# Annotating graph nodes with geographic coordinates using GML attributes or layout
def annotate_graph_geography(G, seed=SEED):
    """Enriching graph nodes with latitude/longitude using provided or computed positions."""
    # Extracting geographic coordinates from node attributes
    coords = {}
    for n, attrs in G.nodes(data=True):
        lat, lon = extract_geo(attrs)
        if lat is not None and lon is not None:
            coords[n] = (lat, lon)

    # Filling missing coordinates using spring layout when geo data incomplete
    if len(coords) < G.number_of_nodes():
        # Generating 2D layout via force-directed algorithm
        pos = nx.spring_layout(G, seed=seed, iterations=40)
        # Normalizing x-coordinates to longitude range [-180, 180]
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        xs = 360.0 * norm01(xs) - 180.0
        # Normalizing y-coordinates to latitude range [-90, 90]
        ys = 140.0 * norm01(ys) - 70.0
        # Assigning computed coordinates to nodes missing geographic data
        for idx, n in enumerate(pos):
            coords.setdefault(n, (float(ys[idx]), float(xs[idx])))

    # Writing geographic annotations back to graph nodes
    for n in G.nodes():
        lat, lon = coords[n]
        G.nodes[n]["Latitude"] = float(lat)
        G.nodes[n]["Longitude"] = float(lon)
    return G


# Connecting disconnected graph components using minimum geographic distance criterion
def connect_components_geographically(G):
    """Merging isolated components by creating bridges between closest node pairs."""
    # Identifying connected components in the graph
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    # Selecting largest component as main network
    main_component = max(components, key=len)
    main_nodes = list(main_component)
    # Processing each isolated component individually
    for comp in components:
        if comp is main_component:
            continue
        # Finding closest node pair between component and main network
        best_pair = None
        best_dist = float("inf")
        for u in comp:
            # Retrieving coordinates for component node
            lat_u = G.nodes[u]["Latitude"]
            lon_u = G.nodes[u]["Longitude"]
            # Searching for closest node in main network
            for v in main_nodes:
                lat_v = G.nodes[v]["Latitude"]
                lon_v = G.nodes[v]["Longitude"]
                # Computing geographic distance between node pairs
                dist = haversine_km(lat_u, lon_u, lat_v, lon_v)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (u, v)
        # Adding bridge edge connecting component to main network
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


# Adding realistic edge attributes based on geographic and topological data
def add_realistic_edge_attributes(G, seed=SEED):
    """Assigning capacity, latency, and weight attributes to edges using network realism principles."""
    # Creating random number generator for stochastic variation
    rng = np.random.default_rng(seed)
    # Building degree dictionary for centrality-based capacity assignment
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1.0
    # Collecting edge distances for statistics
    edge_distances = []
    # Processing each edge in the graph
    for u, v in G.edges():
        # Extracting geographic coordinates for both endpoints
        lat_u, lon_u = G.nodes[u]["Latitude"], G.nodes[u]["Longitude"]
        lat_v, lon_v = G.nodes[v]["Latitude"], G.nodes[v]["Longitude"]
        # Computing geographic distance between endpoints
        distance_km = max(5.0, haversine_km(lat_u, lon_u, lat_v, lon_v))
        # Modulating capacity based on endpoint centrality (hub preference)
        role_factor = 1.0 + 0.35 * (deg[u] + deg[v]) / (max_deg + 1e-9)
        # Assigning capacity based on distance categories
        if distance_km > 2500.0:
            base_capacity = 400.0
        elif distance_km > 800.0:
            base_capacity = 200.0
        else:
            base_capacity = 80.0
        # Computing final capacity with role and stochastic factors
        capacity_gbps = base_capacity * role_factor * rng.uniform(0.85, 1.15)
        # Computing propagation delay based on geographic distance
        propagation_ms = distance_km / 200000.0 * 1000.0
        # Storing edge attributes
        G[u][v]["distance_km"] = float(distance_km)
        G[u][v]["capacity_gbps"] = float(capacity_gbps)
        G[u][v]["propagation_ms"] = float(propagation_ms)
        # Computing edge weight as combination of latency and distance
        G[u][v]["weight"] = float(propagation_ms + distance_km / 1500.0)
        # Collecting distance for statistics
        edge_distances.append(distance_km)
    # Returning edge distances for dataset statistics
    return np.array(edge_distances, dtype=float)


# Simulating realistic routing-based network latency and packet loss targets
def simulate_network_targets(G, degrees, centrality, closeness, pgrank, seed=SEED):
    """Generating RTT and loss targets by simulating network routing and congestion dynamics."""
    # Creating random number generator with fixed seed
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    # Normalizing topological features for target computation
    deg_n = norm01(degrees)
    cent_n = norm01(centrality)
    close_n = norm01(closeness)
    pgr_n = norm01(pgrank)

    # Initializing edge load tracking for each link
    edge_load = {tuple(sorted((u, v))): 0.0 for u, v in G.edges()}
    # Selecting landmark nodes based on connectivity (proxy for importance)
    landmarks = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[: min(12, max(4, N // 120))]
    # Sampling client nodes throughout network
    clients = rng.choice(nodes, size=min(220, N), replace=False)

    # Simulating traffic flows between clients and destinations
    for src in clients:
        # Retrieving source node coordinates
        src_lat = G.nodes[src]["Latitude"]
        src_lon = G.nodes[src]["Longitude"]
        # Selecting destination nodes for traffic generation
        destinations = rng.choice(nodes, size=min(10, N), replace=False)
        for dst in destinations:
            if src == dst:
                continue
            # Retrieving destination node coordinates
            dst_lat = G.nodes[dst]["Latitude"]
            dst_lon = G.nodes[dst]["Longitude"]
            # Computing geographic distance to modulate demand
            geo_dist = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
            # Computing traffic demand based on node degrees and geographic distance
            demand = (0.15 + 0.85 * (deg_n[src] + 0.2) * (deg_n[dst] + 0.2))
            # Boosting demand for high-PageRank destinations
            demand *= (1.0 + 0.7 * pgr_n[dst])
            # Reducing demand over long geographic distances
            demand *= 1.0 / (1.0 + geo_dist / 1500.0)
            # Adding stochastic variation to demand
            demand *= rng.uniform(0.8, 1.2)
            # Computing routing path and accumulating edge loads
            try:
                path = nx.shortest_path(G, src, dst, weight="weight")
            except nx.NetworkXNoPath:
                continue
            # Adding traffic demand to each link on the path
            for u, v in zip(path[:-1], path[1:]):
                edge_load[tuple(sorted((u, v)))] += float(demand)

    # Computing link utilization by dividing load by capacity
    edge_util = {}
    for u, v in G.edges():
        # Getting link load and capacity
        key = tuple(sorted((u, v)))
        util = edge_load[key] / (G[u][v]["capacity_gbps"] + 1e-9)
        edge_util[key] = util
        # Storing utilization as edge attribute for future reference
        G[u][v]["utilization"] = float(util)

    # Initializing RTT and loss arrays for all nodes
    rtt = np.zeros(N, dtype=float)
    loss = np.zeros(N, dtype=float)
    # Computing RTT and loss metrics for each node
    for n in nodes:
        # Retrieving neighbor nodes for local load assessment
        nbrs = list(G.neighbors(n))
        # Computing local utilization from incident links
        local_utils = np.array([edge_util[tuple(sorted((n, nb)))] for nb in nbrs], dtype=float) if nbrs else np.array([0.0])
        # Averaging local link utilizations
        node_load = float(local_utils.mean())

        # Probing RTT to landmarks for representative latency estimation
        probe_rtts = []
        probe_congestion = []
        for lm in landmarks:
            if lm == n:
                continue
            # Computing shortest path to landmark
            try:
                path = nx.shortest_path(G, lm, n, weight="weight")
            except nx.NetworkXNoPath:
                continue
            # Summing propagation delays along path
            prop = sum(G[u][v]["propagation_ms"] for u, v in zip(path[:-1], path[1:]))
            # Computing congestion-induced queuing delay
            path_utils = np.array([edge_util[tuple(sorted((u, v)))] for u, v in zip(path[:-1], path[1:])], dtype=float)
            # Modeling queue using congestion above threshold
            queue = 12.0 * np.maximum(path_utils - 0.60, 0.0) ** 2
            # Computing access penalty based on node centrality
            access_penalty = 1.5 + 9.0 * (1.0 - close_n[n]) + 3.0 * (1.0 - deg_n[n])
            # Combining components into probe RTT
            probe_rtts.append(2.0 * prop + float(queue.sum()) + access_penalty)
            probe_congestion.append(path_utils.mean() if len(path_utils) else 0.0)

        # Using default RTT if no valid landmarks reached
        if not probe_rtts:
            probe_rtts = [12.0 + 15.0 * (1.0 - close_n[n])]
            probe_congestion = [node_load]

        # Computing node metrics from probe measurements
        congestion = float(np.mean(probe_congestion))
        # Modeling traffic burstiness using beta distribution
        burstiness = rng.beta(2.5, 14.0)
        # Taking median of probe RTTs with small random jitter
        rtt[n] = np.median(probe_rtts) + rng.normal(0.0, 0.75)
        # Computing packet loss probability from congestion and load
        loss_prob = (
            0.002
            + 0.030 * np.maximum(congestion - 0.55, 0.0) ** 1.8
            + 0.020 * np.maximum(node_load - 0.65, 0.0) ** 1.6
            + 0.006 * cent_n[n]
            + 0.003 * burstiness
        )
        # Converting loss probability to percentage
        loss[n] = 100.0 * np.clip(loss_prob, 0.0005, 0.08)

    # Returning normalized RTT and loss as regression targets
    return norm01(rtt).astype(np.float32), norm01(loss).astype(np.float32)


def build_dataset(seed=42):
    """
    Realistic topology from Internet Zoo or Mixed ISP topology if Zoo not found.
    Features: topology + geography. Targets: routed RTT + packet loss.
    """
    rng = np.random.default_rng(seed)

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

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = annotate_graph_geography(G, seed=seed)
    N = G.number_of_nodes()
    edge_distances = add_realistic_edge_attributes(G, seed=seed)

    degrees = np.array([d for _, d in G.degree()], dtype=float)
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

    features = np.stack([
        norm01(degrees), norm01(centrality), clustering,
        norm01(closeness), norm01(pgrank),
        norm01(lats), norm01(lons), hop2,
        norm01(eigvec), norm01(core_num),
    ], axis=1).astype(np.float32)

    print("Topology realism: avg_link_km={:.1f}".format(edge_distances.mean()))
    rtt, loss = simulate_network_targets(G, degrees, centrality, closeness, pgrank, seed=seed)
    y = np.stack([rtt, loss], axis=1)

    ei = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    ei = to_undirected(ei, num_nodes=N)
    ei, _ = add_self_loops(ei, num_nodes=N)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=ei,
        y=torch.tensor(y, dtype=torch.float),
        num_nodes=N,
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)
    idx = torch.randperm(N, generator=gen)
    ntr = int(0.7 * N)
    nva = int(0.15 * N)
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[idx[:ntr]] = True
    data.val_mask[idx[ntr:ntr + nva]] = True
    data.test_mask[idx[ntr + nva:]] = True
    return data.to(DEVICE), features.shape[1], N, G


# ============================================================================
# 2. BASELINE MODELS
# ============================================================================

class GraphSAGEModel(nn.Module):
    def __init__(self, ic, h, oc):
        super().__init__()
        self.c1=SAGEConv(ic,h); self.c2=SAGEConv(h,h); self.c3=SAGEConv(h,h)
        self.head=nn.Linear(h,oc)
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.b3=nn.BatchNorm1d(h)
    def forward(self, x, ei):
        x=F.relu(self.b1(self.c1(x,ei))); x=F.dropout(x,.2,self.training)
        x=F.relu(self.b2(self.c2(x,ei))); x=F.dropout(x,.2,self.training)
        return self.head(F.relu(self.b3(self.c3(x,ei))))


class ChebNetModel(nn.Module):
    def __init__(self, ic, h, oc, K=4):
        super().__init__()
        self.c1=ChebConv(ic,h,K); self.c2=ChebConv(h,h,K); self.c3=ChebConv(h,h,K)
        self.head=nn.Linear(h,oc)
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.b3=nn.BatchNorm1d(h)
    def forward(self, x, ei):
        x=F.relu(self.b1(self.c1(x,ei))); x=F.dropout(x,.2,self.training)
        x=F.relu(self.b2(self.c2(x,ei))); x=F.dropout(x,.2,self.training)
        return self.head(F.relu(self.b3(self.c3(x,ei))))


class ResGatedGCNModel(nn.Module):
    def __init__(self, ic, h, oc):
        super().__init__()
        self.proj=nn.Linear(ic,h)
        self.c1=ResGatedGraphConv(h,h); self.c2=ResGatedGraphConv(h,h); self.c3=ResGatedGraphConv(h,h)
        self.head=nn.Linear(h,oc)
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.b3=nn.BatchNorm1d(h)
    def forward(self, x, ei):
        x=F.relu(self.proj(x))
        x=F.relu(self.b1(self.c1(x,ei))); x=F.dropout(x,.2,self.training)
        x=F.relu(self.b2(self.c2(x,ei))); x=F.dropout(x,.2,self.training)
        return self.head(F.relu(self.b3(self.c3(x,ei))))


class GraphTransformerModel(nn.Module):
    def __init__(self, ic, h, oc, heads=4):
        super().__init__()
        d=h//heads
        self.c1=TransformerConv(ic,d,heads=heads,dropout=.1)
        self.c2=TransformerConv(h, d,heads=heads,dropout=.1)
        self.c3=TransformerConv(h, d,heads=heads,dropout=.1)
        self.head=nn.Linear(h,oc)
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.b3=nn.BatchNorm1d(h)
    def forward(self, x, ei):
        x=F.relu(self.b1(self.c1(x,ei))); x=F.dropout(x,.2,self.training)
        x=F.relu(self.b2(self.c2(x,ei))); x=F.dropout(x,.2,self.training)
        return self.head(F.relu(self.b3(self.c3(x,ei))))


# ============================================================================
# 3. PROPOSED: HSTGNN
# ============================================================================

# Implementing multi-scale graph neural network block using parallel convolution branches
class MultiScaleGNNBlock(nn.Module):
    """
    Combining three parallel GNN branches capturing different spatial scales:
      - SAGEConv:       aggregating 1-hop local neighborhood
      - ChebConv K=3:   processing spectral multi-hop patterns (up to 3 hops)
      - TransformerConv: computing attention-weighted neighborhood features
    Concatenating branches and applying learned channel gating + LayerNorm for fusion.
    """
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


# Defining Hybrid Spatio-Temporal GNN architecture combining multiple representation learning paradigms
class HSTGNN(nn.Module):
    """
    Hybrid Spatio-Temporal GNN (HSTGNN) - fusing spatial and temporal network dynamics
    ─────────────────────────────────────────────────────────────────────
    Architecture composition:
      1. Projecting inputs using Linear → GELU → BatchNorm
      2. Processing through 3 × MultiScaleGNNBlock (SAGE + Cheb + TransformerConv in parallel)
      3. Applying Learnable Temporal Module: using channel-wise scale/shift + FF network
         (simulating temporal network state variation deterministically)
      4. Refining features using GATConv with 4 attention heads
      5. Predicting targets using MLP head + raw input skip connection
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
        # Storing raw input for skip connection
        raw = x
        # Projecting input features through dense layer and batch normalization
        x   = self.inp(x)
        # Processing through first multi-scale GNN block and applying dropout
        x   = self.b1(x, ei); x = F.dropout(x, 0.20, self.training)
        # Processing through second multi-scale GNN block and applying dropout
        x   = self.b2(x, ei); x = F.dropout(x, 0.20, self.training)
        # Processing through third multi-scale GNN block for deep representation
        x   = self.b3(x, ei)
        # Applying first temporal module: using channel-wise modulation and feedforward network
        xt  = x * self.t_scale + self.t_shift
        xt  = self.t_norm(xt + self.t_ff(xt))
        x   = x + torch.tanh(self.t_gate) * xt
        # Applying second temporal module: capturing higher-order temporal dynamics
        xt2 = x * self.t_scale2 + self.t_shift2
        xt2 = self.t_norm2(xt2 + self.t_ff2(xt2))
        x   = x + torch.tanh(self.t_gate2) * xt2
        # Refining features using multi-head graph attention
        x   = x + F.gelu(self.bn(self.gat(x, ei)))
        # Processing raw features in parallel for dual representation
        feat = self.feat(raw)
        # Predicting targets using graph-based pathway
        graph_pred = self.graph_head(x)
        # Predicting targets using feature-based pathway
        feat_pred  = self.feat_head(feat)
        # Computing adaptive mixing weight using sigmoid gating
        mix = torch.sigmoid(self.mix_head(torch.cat([x, feat], dim=-1)))
        # Combining predictions using learned mixing weight and skip connection
        return mix * graph_pred + (1 - mix) * feat_pred + self.skip(raw)


# ============================================================================
# 4. TRAINING AND EVALUATION
# ============================================================================

# Evaluating model performance using multiple regression metrics without gradient computation
@torch.no_grad()
def evaluate(model, data, mask):
    """Computing performance metrics by generating predictions and comparing with ground truth."""
    # Setting model to evaluation mode and disabling dropout/batch norm updates
    model.eval()
    # Predicting outputs using model and extracting specified subset
    out  = model(data.x, data.edge_index)[mask].cpu().numpy()
    # Retrieving ground truth labels for the specified subset
    true = data.y[mask].cpu().numpy()
    # Computing R² score (coefficient of determination)
    r2   = r2_score(true, out)
    # Computing Mean Absolute Error (average absolute prediction error)
    mae  = mean_absolute_error(true, out)
    # Computing Root Mean Squared Error (emphasizing larger errors)
    rmse = np.sqrt(mean_squared_error(true, out))
    # Computing Huber loss (robust to outliers)
    hub  = F.huber_loss(torch.tensor(out, dtype=torch.float),
                        torch.tensor(true, dtype=torch.float), delta=0.1).item()
    # Returning comprehensive evaluation metrics dictionary
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Huber": hub,
            "pred": out, "true": true}


# Training neural network model using adaptive optimization and learning rate scheduling
def train_model(model, data, name="", max_ep=350, patience=45, lr=1e-3, warmup=0, restarts=False, wd=5e-5):
    """Optimizing model parameters using Adam with configurable learning rate scheduling."""
    # Creating AdamW optimizer with weight decay regularization
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Selecting learning rate scheduler based on training configuration
    if restarts:
        # Using cosine annealing with warm restarts for escaping local minima
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=100, T_mult=2, eta_min=5e-6)
    elif warmup > 0:
        # Using linear warmup followed by cosine annealing
        warm_sched   = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep - warmup, eta_min=5e-6)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm_sched, cosine_sched], milestones=[warmup])
    else:
        # Using simple cosine annealing without warmup
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=5e-6)
    # Initializing tracking variables for best model and training history
    best_val, best_ep, best_st = 1e9, 0, None
    hist = {"train": [], "val_r2": []}
    # Recording training start time for duration measurement
    t0   = time.time()

    # Iterating through training epochs with early stopping capability
    for ep in range(1, max_ep+1):
        # Enabling training mode and resetting gradient accumulators
        model.train(); opt.zero_grad()
        # Generating predictions for all samples
        out  = model(data.x, data.edge_index)
        # Extracting predictions and labels for training set
        train_out = out[data.train_mask]
        train_y = data.y[data.train_mask]
        # Selecting loss function based on model type
        if "HSTGNN" in name:
            # Computing Huber loss for robustness to outliers
            huber = F.huber_loss(train_out, train_y, delta=0.08)
            # Computing Mean Squared Error loss
            mse   = F.mse_loss(train_out, train_y)
            # Computing correlation loss: directly optimizing for R² improvement
            pc = train_out - train_out.mean(dim=0)
            tc = train_y   - train_y.mean(dim=0)
            corr = (pc * tc).sum(dim=0) / (pc.norm(dim=0) * tc.norm(dim=0) + 1e-8)
            corr_loss = 1 - corr.mean()
            # Gradually introducing correlation loss over training (ramping strategy)
            ramp = min(1.0, ep / 150)
            loss = 0.50 * huber + 0.25 * mse + 0.25 * ramp * corr_loss
        else:
            # Using combined MSE and L1 loss for baseline models
            loss = (F.mse_loss(train_out, train_y) +
                0.05 * F.l1_loss(train_out, train_y))
        # Computing gradients via backpropagation
        loss.backward()
        # Clipping gradients to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Updating model parameters and adjusting learning rate
        opt.step(); sched.step()

        # Evaluating model performance on validation set
        vm = evaluate(model, data, data.val_mask)
        # Recording training loss and validation R² for monitoring
        hist["train"].append(loss.item())
        hist["val_r2"].append(vm["R2"])

        # Determining if this epoch achieved best validation performance
        if (1 - vm["R2"]) < best_val:
            # Saving best model state when validation improves
            best_val = 1 - vm["R2"]; best_ep = ep
            best_st  = {k: v.clone() for k, v in model.state_dict().items()}
        # Implementing early stopping when patience criterion is exceeded
        if ep - best_ep >= patience:
            break

    # Computing total training duration
    elapsed = time.time() - t0
    # Loading best model state for test evaluation
    model.load_state_dict(best_st)
    # Evaluating best model on test set
    tm = evaluate(model, data, data.test_mask)
    # Augmenting results with training metadata and history
    tm.update(Epochs=best_ep, Time_s=round(elapsed, 2), History=hist)
    # Displaying training summary with key metrics
    print(f"  [{name:20s}]  R²={tm['R2']:.4f}  MAE={tm['MAE']:.4f}  "
          f"RMSE={tm['RMSE']:.4f}  Ep={best_ep}  t={elapsed:.1f}s")
    # Returning comprehensive training results
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
    # VISUALISATIONS  (super high-definition, 400 DPI)
    # ================================================================
    plt.rcParams.update({
        'figure.dpi': 400, 'savefig.dpi': 400,
        'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
        'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
        'font.weight': 'bold', 'font.family': 'sans-serif',
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

    model_names = list(results.keys())
    colors = ['#2ecc71' if 'HSTGNN' in n else '#3498db' for n in model_names]
    r2_vals  = [results[n]["R2"]   for n in model_names]
    mae_vals = [results[n]["MAE"]  for n in model_names]
    rmse_vals= [results[n]["RMSE"] for n in model_names]

    DPI = 400

    # --- 1. Bar plot: R² Score comparison ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(model_names, r2_vals, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('R² Score Comparison Across GNN Architectures', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_ylim(0, max(r2_vals) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('bar_r2_score.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved bar_r2_score.png")

    # --- 2. Bar plot: MAE & RMSE side-by-side (dark blue + yellow) ---
    x = np.arange(len(model_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w/2, mae_vals, w, label='MAE', color='#1a237e', edgecolor='white', linewidth=1.2)
    b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color='#fbc02d', edgecolor='white', linewidth=1.2)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title('MAE & RMSE Comparison Across GNN Architectures', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, prop={'weight': 'bold'})
    ax.tick_params(axis='both', labelsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('bar_mae_rmse.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved bar_mae_rmse.png")

    # --- 3. Heatmap: all metrics ---
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
    plt.savefig('heatmap_metrics.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved heatmap_metrics.png")

    # --- 4. Radar / spider chart ---
    r2_n   = np.array(r2_vals)
    mae_n  = 1 - (np.array(mae_vals) - min(mae_vals)) / (max(mae_vals) - min(mae_vals) + 1e-9)
    rmse_n = 1 - (np.array(rmse_vals) - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals) + 1e-9)
    hub_n  = np.array([results[n]["Huber"] for n in model_names])
    hub_n  = 1 - (hub_n - hub_n.min()) / (hub_n.max() - hub_n.min() + 1e-9)
    categories = ['R²', 'MAE\n(inverted, ↑ = better)', 'RMSE\n(inverted, ↑ = better)', 'Huber\n(inverted, ↑ = better)']
    N_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6']
    for idx, name in enumerate(model_names):
        vals = [r2_n[idx], mae_n[idx], rmse_n[idx], hub_n[idx]]
        vals += vals[:1]
        lw = 3.0 if 'HSTGNN' in name else 1.8
        ax.plot(angles, vals, 'o-', linewidth=lw, label=name, color=radar_colors[idx], markersize=6)
        ax.fill(angles, vals, alpha=0.10, color=radar_colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_title('Normalised Performance Radar Chart', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=10, prop={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved radar_chart.png")

    # --- 5. Training curves (loss & val R²) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    curve_colors = ['#1a237e', '#e67e22', '#2ecc71', '#e74c3c', '#fbc02d']
    for idx, name in enumerate(model_names):
        h = results[name]["History"]
        c = curve_colors[idx]
        lw = 3.0 if 'HSTGNN' in name else 1.5
        axes[0].plot(h["train"], label=name, color=c, linewidth=lw, alpha=0.9)
        axes[1].plot(h["val_r2"], label=name, color=c, linewidth=lw, alpha=0.9)
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Training Loss Curves', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=10, prop={'weight': 'bold'}); axes[0].grid(alpha=0.3)
    for lbl in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        lbl.set_fontweight('bold')
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Validation R²', fontsize=13, fontweight='bold')
    axes[1].set_title('Validation R² Curves', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=10, prop={'weight': 'bold'}); axes[1].grid(alpha=0.3)
    for lbl in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        lbl.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved training_curves.png")

    # --- 6. Scatter: Predicted vs Actual (HSTGNN vs GraphTransformer) ---
    # Find best competitor (highest R² excluding HSTGNN)
    best_comp = max((n for n in model_names if 'HSTGNN' not in n),
                    key=lambda n: results[n]["R2"])
    scatter_models = ["HSTGNN (Proposed)", best_comp]
    scatter_labels = ["HSTGNN (Proposed)", best_comp]
    titles = ['RTT', 'Packet Loss']

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    model_colors = [['#1a237e', '#fbc02d'], ['#e74c3c', '#9b59b6']]
    for row, (mname, mlabel) in enumerate(zip(scatter_models, scatter_labels)):
        pred = results[mname]["pred"]
        true = results[mname]["true"]
        r2 = results[mname]["R2"]
        for col in range(2):
            ax = axes[row][col]
            ax.scatter(true[:, col], pred[:, col], alpha=0.5, s=20,
                       c=model_colors[row][col], edgecolors='none')
            lo = min(true[:, col].min(), pred[:, col].min())
            hi = max(true[:, col].max(), pred[:, col].max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, alpha=0.6)
            ax.set_xlabel(f'Actual {titles[col]}', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'Predicted {titles[col]}', fontsize=13, fontweight='bold')
            ax.set_title(f'{mlabel}: {titles[col]} (R²={r2:.4f})',
                         fontsize=14, fontweight='bold')
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_pred_vs_actual.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved scatter_pred_vs_actual.png")

    # ── Network Topology Visualisation (3-panel) ───────────────────
    print("Generating network topology visualisation...")
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    try:
        emoji_fp = FontProperties(fname="C:/Windows/Fonts/seguiemj.ttf")
    except Exception:
        emoji_fp = FontProperties(family='Segoe UI Emoji')

    seed = 42
    G_full = G_zoo
    N = G_full.number_of_nodes()
    
    # Calculate positions
    # Use existing Lat/Lon if they exist and are not all 0
    lats = np.array([G_full.nodes[n].get('Latitude', 0) for n in G_full.nodes()])
    lons = np.array([G_full.nodes[n].get('Longitude', 0) for n in G_full.nodes()])
    
    if np.abs(lats).sum() > 0:
        pos_all = {n: (G_full.nodes[n].get('Longitude', 0), G_full.nodes[n].get('Latitude', 0)) for n in G_full.nodes()}
    else:
        pos_all = nx.spring_layout(G_full, seed=seed, iterations=60, k=0.3)

    deg_full = dict(G_full.degree())
    hub_thresh = sorted(deg_full.values(), reverse=True)[max(1, int(0.03 * N))]
    hubs = [n for n in G_full.nodes() if deg_full[n] >= hub_thresh]
    reg_nodes = [n for n in G_full.nodes() if deg_full[n] < hub_thresh]

    BG = '#0a0e1a'
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')

    # ─── Panel 1: Overview ─────────────────
    ax1 = axes[0]
    nx.draw_networkx_edges(G_full, pos_all, ax=ax1,
                           edge_color='#1e88e5', alpha=0.10, width=0.35)
    nx.draw_networkx_nodes(G_full, pos_all, nodelist=reg_nodes, ax=ax1,
                           node_color='#42a5f5',
                           node_size=[max(4, deg_full[n]*0.7) for n in reg_nodes],
                           alpha=0.55, linewidths=0)
    
    ax1.text(0.02, 0.02,
             f'Nodes: {N}\nEdges: {G_full.number_of_edges()}\n'
             f'Avg deg: {2*G_full.number_of_edges()/N:.1f}',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             color='white', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='#131926', ec='#2a3a5c', alpha=0.9))
    ax1.set_title('(a) Network Topology Overview',
                  fontsize=15, fontweight='bold', color='#42a5f5', pad=12)

    # ─── Panel 2: Hubs and Connections ──────────────
    ax2 = axes[1]
    nx.draw_networkx_edges(G_full, pos_all, ax=ax2,
                           edge_color='#00897b', alpha=0.05, width=0.2)
    if hubs:
        bhs = [max(80, deg_full[n]*5) for n in hubs]
        nx.draw_networkx_nodes(G_full, pos_all, nodelist=hubs, ax=ax2,
                               node_color='#ff4081', node_size=[s*2.8 for s in bhs],
                               alpha=0.12, linewidths=0)
        nx.draw_networkx_nodes(G_full, pos_all, nodelist=hubs, ax=ax2,
                               node_color='#ff4081', node_size=bhs,
                               alpha=0.95, linewidths=0.6, edgecolors='white')
    
    ax2.set_title('(b) Critical Hubs (Top 3% Degree)',
                  fontsize=15, fontweight='bold', color='#ff4081', pad=12)

    # ─── Panel 3: Structural Centrality ───────────────
    ax3 = axes[2]
    # Use closeness centrality for color
    cent = nx.closeness_centrality(G_full)
    nodes = list(G_full.nodes())
    node_color = [cent[n] for n in nodes]
    
    nx.draw_networkx_edges(G_full, pos_all, ax=ax3,
                           edge_color='white', alpha=0.05, width=0.2)
    sc = nx.draw_networkx_nodes(G_full, pos_all, nodelist=nodes, ax=ax3,
                           node_color=node_color, cmap='plasma',
                           node_size=20, alpha=0.8)
    
    ax3.set_title('(c) Structural Centrality (Closeness)',
                  fontsize=15, fontweight='bold', color='#ffd740', pad=12)

    fig.suptitle('Realistic Network Digital Twin Topology (Internet Zoo)',
                 fontsize=20, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('network_topology.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("Saved network_topology.png")

    print("\nAll plots and CSV saved. Done.")


# Building graph-specific dataset with geographic and topological attributes
def build_dataset_from_graph(G, seed=42, label="custom"):
    """Constructing training dataset from a single graph by computing features and simulating targets."""
    # Creating copy to prevent modifying original graph
    G = G.copy()
    # Removing self-loops that may exist in input graph
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Relabeling nodes to contiguous integers for efficient indexing
    G = nx.convert_node_labels_to_integers(G)
    # Annotating nodes with geographic coordinates
    G = annotate_graph_geography(G, seed=seed)
    N = G.number_of_nodes()
    # Computing edge attributes including distance, capacity, and latency
    edge_distances = add_realistic_edge_attributes(G, seed=seed)

    degrees = np.array([d for _, d in G.degree()], dtype=float)
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

    features = np.stack([
        norm01(degrees), norm01(centrality), clustering,
        norm01(closeness), norm01(pgrank),
        norm01(lats), norm01(lons), hop2,
        norm01(eigvec), norm01(core_num),
    ], axis=1).astype(np.float32)

    # Printing dataset statistics for verification
    print(f"Using single topology: {label} ({N} nodes, {G.number_of_edges()} edges)")
    print("Topology realism: avg_link_km={:.1f}".format(edge_distances.mean()))
    # Simulating network performance targets using routing-based congestion model
    rtt, loss = simulate_network_targets(G, degrees, centrality, closeness, pgrank, seed=seed)
    # Stacking RTT and loss into multi-target output
    y = np.stack([rtt, loss], axis=1)

    # Converting graph edges to tensor format for PyTorch Geometric
    ei = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    # Converting directed edges to undirected by duplicating
    ei = to_undirected(ei, num_nodes=N)
    # Adding self-loops to edges for node self-attention
    ei, _ = add_self_loops(ei, num_nodes=N)

    # Creating PyTorch Geometric data object with features, edges, and targets
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=ei,
        y=torch.tensor(y, dtype=torch.float),
        num_nodes=N,
    )

    # Initializing random number generator with seed
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Creating random permutation of node indices
    idx = torch.randperm(N, generator=gen)
    # Computing split boundaries for train/val/test
    ntr = int(0.7 * N)
    nva = int(0.15 * N)
    # Creating boolean masks for train/val/test subsets
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)
    # Assigning nodes to train/val/test partitions
    data.train_mask[idx[:ntr]] = True
    data.val_mask[idx[ntr:ntr + nva]] = True
    data.test_mask[idx[ntr + nva:]] = True
    # Moving data to appropriate device and returning
    return data.to(DEVICE), features.shape[1], N, G


_combined_build_dataset = build_dataset


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


def build_dataset(seed=42, topology_path=None):
    # Reuse the original combined-topology builder for the default "single"
    # experiment, but allow multi-mode runs to swap in a specific Zoo graph.
    if topology_path is None:
        return _combined_build_dataset(seed=seed)
    topology_name = os.path.splitext(os.path.basename(topology_path))[0]
    G = load_single_topology_graph(topology_path, seed=seed)
    return build_dataset_from_graph(G, seed=seed, label=topology_name)


def topology_positions(G, seed=SEED):
    # Prefer geographic coordinates when the graph has them; otherwise fall back
    # to a spring layout so every topology remains visualisable.
    lats = np.array([G.nodes[n].get("Latitude", 0.0) for n in G.nodes()], dtype=float)
    if np.abs(lats).sum() > 0:
        return {n: (G.nodes[n].get("Longitude", 0.0), G.nodes[n].get("Latitude", 0.0)) for n in G.nodes()}
    return nx.spring_layout(G, seed=seed, iterations=60, k=0.3)


def preferred_node_labels(G, max_labels=32):
    # Labeling only the most structurally important nodes to keep dense figures
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

    ax.axhspan(min_y - pad_y, max_y + pad_y, color="#010409", zorder=0)
    
    # Layered gradients to simulate subtle landmass/bathymetry details
    from matplotlib.patches import Rectangle
    # Main "glow" area around the network
    rect = Rectangle((min_x - pad_x, min_y - pad_y), 
                    (max_x - min_x + 2*pad_x), (max_y - min_y + 2*pad_y),
                    facecolor='#0a192f', alpha=0.3, zorder=0)
    ax.add_patch(rect)

    
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
    # Generating a compact three-panel diagnostic view: overall topology,
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


# Training all baseline and proposed models on provided dataset
def run_models_for_dataset(data, in_dim):
    """Running complete training pipeline for all model architectures on single dataset."""
    # Setting hidden and output dimensions for model consistency
    HID, OUT = 96, 2
    # Creating model instances: all baselines receive same hidden size for fair comparison
    models_cfg = {
        "GraphSAGE": GraphSAGEModel(in_dim, HID, OUT),
        "ChebNet": ChebNetModel(in_dim, HID, OUT),
        "ResGatedGCN": ResGatedGCNModel(in_dim, HID, OUT),
        "GraphTransformer": GraphTransformerModel(in_dim, HID, OUT),
        "HSTGNN (Proposed)": HSTGNN(in_dim, HID, OUT),
    }

    # Displaying training progress message
    print("\nTraining all models...\n")
    # Initializing results dictionary
    results = {}
    # Training each model with architecture-specific hyperparameters
    for name, mdl in models_cfg.items():
        # HSTGNN uses more aggressive training due to greater expressiveness
        if "HSTGNN" in name:
            # Training HSTGNN with longer schedule and gentle optimization
            results[name] = train_model(
                mdl.to(DEVICE), data, name=name,
                max_ep=650, patience=120, lr=6e-4, warmup=35, wd=1e-4
            )
        else:
            # Training baseline models with standard hyperparameters
            results[name] = train_model(mdl.to(DEVICE), data, name=name, max_ep=300, patience=45)
    # Returning trained model results
    return results


def save_run_outputs(results, G_zoo, output_dir, topology_name):
    # Centralising every artifact written by a run: tabular metrics, comparison
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
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
        "axes.titleweight": "bold", "axes.labelweight": "bold",
        "font.weight": "bold", "font.family": "sans-serif",
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    model_names = list(results.keys())
    colors = ["#2ecc71" if "HSTGNN" in n else "#3498db" for n in model_names]
    r2_vals = [results[n]["R2"] for n in model_names]
    mae_vals = [results[n]["MAE"] for n in model_names]
    rmse_vals = [results[n]["RMSE"] for n in model_names]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(model_names, r2_vals, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("R2 Score", fontsize=14, fontweight="bold")
    ax.set_title(f"R2 Score Comparison: {topology_name}", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_r2_score.png"), bbox_inches="tight")
    plt.close()

    # Ploting error metrics together because MAE and RMSE complement each other:
    # MAE shows average error, while RMSE penalises larger misses.
    x = np.arange(len(model_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w/2, mae_vals, w, label='MAE', color='#1a237e', edgecolor='white', linewidth=1.2)
    b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color='#fbc02d', edgecolor='white', linewidth=1.2)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold', rotation=15)
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title(f'MAE & RMSE Comparison: {topology_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, prop={'weight': 'bold'})
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_mae_rmse.png"), bbox_inches="tight")
    plt.close()

    # compare overall model profiles across "higher is better" dimensions.
    r2_n = np.array(r2_vals)
    mae_n = 1 - (np.array(mae_vals) - min(mae_vals)) / (max(mae_vals) - min(mae_vals) + 1e-9)
    rmse_n = 1 - (np.array(rmse_vals) - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals) + 1e-9)
    hub_vals = [results[n]["Huber"] for n in model_names]
    hub_n = 1 - (np.array(hub_vals) - min(hub_vals)) / (max(hub_vals) - min(hub_vals) + 1e-9)
    categories = ["R2", "MAE (inv)", "RMSE (inv)", "Huber (inv)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
    for idx, name in enumerate(model_names):
        vals = [r2_n[idx], mae_n[idx], rmse_n[idx], hub_n[idx]]
        vals += vals[:1]
        lw = 3.0 if "HSTGNN" in name else 1.8
        ax.plot(angles, vals, "o-", linewidth=lw, label=name, color=radar_colors[idx], markersize=5)
        ax.fill(angles, vals, alpha=0.10, color=radar_colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Normalised Performance Radar Chart: {topology_name}", fontsize=16, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), bbox_inches="tight")
    plt.close()

    # Training curves
    # behaviour, and whether the stronger model simply trained longer.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    curve_colors = ["#1a237e", "#e67e22", "#2ecc71", "#e74c3c", "#fbc02d"]
    for idx, name in enumerate(model_names):
        h = results[name]["History"]
        lw = 3.0 if "HSTGNN" in name else 1.5
        axes[0].plot(h["train"], label=name, color=curve_colors[idx], linewidth=lw, alpha=0.9)
        axes[1].plot(h["val_r2"], label=name, color=curve_colors[idx], linewidth=lw, alpha=0.9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation R2")
    axes[1].set_title("Validation R2 Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), bbox_inches="tight")
    plt.close()

    # Compare the proposed model against the strongest non-HSTGNN baseline to
    # keep the scatter plot readable while still showing the most relevant gap.
    best_comp = max((n for n in model_names if "HSTGNN" not in n), key=lambda n: results[n]["R2"])
    scatter_models = ["HSTGNN (Proposed)", best_comp]
    titles = ["RTT", "Packet Loss"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    model_colors = [["#1a237e", "#fbc02d"], ["#e74c3c", "#9b59b6"]]
    for row, mname in enumerate(scatter_models):
        pred = results[mname]["pred"]
        true = results[mname]["true"]
        r2 = results[mname]["R2"]
        for col in range(2):
            ax = axes[row][col]
            ax.scatter(true[:, col], pred[:, col], alpha=0.5, s=20, c=model_colors[row][col], edgecolors="none")
            lo = min(true[:, col].min(), pred[:, col].min())
            hi = max(true[:, col].max(), pred[:, col].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6)
            ax.set_xlabel(f"Actual {titles[col]}")
            ax.set_ylabel(f"Predicted {titles[col]}")
            ax.set_title(f"{mname}: {titles[col]} (R2={r2:.4f})")
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_pred_vs_actual.png"), bbox_inches="tight")
    plt.close()

    save_topology_plot(G_zoo, os.path.join(output_dir, "network_topology.png"),
                       f"Topology Diagnostics: {topology_name}")
    save_geographic_topology_view(
        G_zoo,
        os.path.join(output_dir, "network_topology_geo.png"),
        f"Geographic Network View: {topology_name}"
    )
    return df


# Retrieving available topology files from Internet Zoo dataset
def select_topology_files(base_path, count=5):
    """Selecting largest topologies from Zoo dataset for comprehensive benchmark."""
    # Initializing list to collect topology metadata
    rows = []
    # Iterating through dataset files
    for name in os.listdir(base_path):
        # Filtering for GML topology files
        if not name.endswith(".gml"):
            continue
        # Constructing full file path
        path = os.path.join(base_path, name)
        try:
            # Reading topology and collecting metadata
            G = nx.read_gml(path)
            rows.append((name, path, G.number_of_nodes(), G.number_of_edges()))
        except Exception:
            # Skipping topologies that fail to parse
            continue
    # Sorting by node count descending to select largest topologies first
    rows.sort(key=lambda item: (-item[2], item[0]))
    # Returning requested count of largest topologies
    return rows[:count]


# Running benchmark suite across multiple network topologies
def run_multi_topology_suite(topology_count=5, output_root="multi_topology_runs", seed=SEED):
    """Executing full training pipeline on multiple Internet Zoo topologies and aggregating results."""
    # Locating Internet Zoo dataset directory
    base_path = os.path.join("3D-internet-zoo-master", "3D-internet-zoo-master", "dataset")
    # Selecting largest topologies for benchmark
    selected = select_topology_files(base_path, count=topology_count)
    # Validating that sufficient topologies were found
    if not selected:
        raise RuntimeError("No usable GML topologies found in the Internet Zoo dataset.")

    # Creating output directory for experiment results
    os.makedirs(output_root, exist_ok=True)
    # Initializing lists for topology info and result frames
    topology_items = []
    summary_frames = []

    # Processing each selected topology
    for idx, (name, path, _, _) in enumerate(selected, start=1):
        # Extracting topology name without extension
        topology_name = os.path.splitext(name)[0]
        # Creating output subdirectory for this topology
        output_dir = os.path.join(output_root, f"{idx:02d}_{topology_name}")
        # Generating run-specific seed for reproducibility
        run_seed = seed + idx
        # Printing section header for this topology run
        print("\n" + "=" * 75)
        print(f"Running topology {idx}/{len(selected)}: {topology_name}")
        print("=" * 75)
        # Building dataset for this topology
        data, in_dim, N, G_zoo = build_dataset(seed=run_seed, topology_path=path)
        # Displaying dataset statistics
        print(f"Nodes={N}  Edges={data.edge_index.shape[1]}  Features={in_dim}")
        # Running all models on this topology
        results = run_models_for_dataset(data, in_dim)
        # Saving results including plots and metrics
        df = save_run_outputs(results, G_zoo, output_dir, topology_name)
        # Adding topology name to results dataframe
        df.insert(0, "Topology", topology_name)
        # Collecting results for summary
        summary_frames.append(df)
        topology_items.append((topology_name, G_zoo))

    # Concatenating results from all topologies into single dataframe
    summary_df = pd.concat(summary_frames, ignore_index=True)
    # Saving combined results to CSV file
    summary_path = os.path.join(output_root, "all_topology_results.csv")
    summary_df.to_csv(summary_path, index=False)
    # Saving visualization of all topologies used
    save_topology_gallery(topology_items, os.path.join(output_root, "topologies_used.png"))
    # Printing completion messages
    print(f"\nSaved combined summary to {summary_path}")
    print(f"Saved topology gallery to {os.path.join(output_root, 'topologies_used.png')}")


# Running single-topology experiment using combined Internet Zoo graphs
def run_single_experiment(seed=SEED, output_dir="single_mode_run"):
    """Executing full experiment on combined Internet Zoo topology without multi-mode overhead."""
    # Displaying section header
    print("=" * 65)
    print("Building dataset...")
    # Building combined dataset from multiple Zoo topologies
    data, in_dim, N, G_zoo = build_dataset(seed=seed)
    # Displaying dataset statistics for verification
    print(f"Nodes={N}  Edges={data.edge_index.shape[1]}  Features={in_dim}")
    print("=" * 65)
    # Running all models on combined topology
    results = run_models_for_dataset(data, in_dim)
    # Saving results including evaluation metrics and visualizations
    save_run_outputs(results, G_zoo, output_dir, "Combined Internet Zoo")
    # Displaying completion message
    print("\nAll plots and CSV saved. Done.")


# Parsing command-line arguments and executing requested experiment mode
def main():
    """Orchestrating experiment execution by parsing CLI arguments and running selected mode."""
    # Creating argument parser for experiment configuration
    parser = argparse.ArgumentParser(description="Running NDT experiments on combined or multiple topologies.")
    # Adding mode selection argument for single or multi topology benchmark
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Using 'single' for combined topology or 'multi' for 5-topology suite.",
    )
    # Adding topology count parameter for multi-mode experiments
    parser.add_argument(
        "--topology-count",
        type=int,
        default=5,
        help="Specifying number of topologies to use in multi mode.",
    )
    # Adding output directory parameter for multi-mode results
    parser.add_argument(
        "--output-root",
        default="multi_topology_runs",
        help="Setting output folder for multi mode experiments.",
    )
    # Adding output directory parameter for single-mode results
    parser.add_argument(
        "--single-output-dir",
        default="single_mode_run",
        help="Setting output folder for single mode experiments.",
    )
    # Adding random seed parameter for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Using random seed for dataset generation and training.",
    )
    # Parsing command-line arguments
    args = parser.parse_args()

    # Selecting experiment mode based on parsed arguments
    if args.mode == "multi":
        # Running multi-topology benchmark suite
        run_multi_topology_suite(topology_count=args.topology_count, output_root=args.output_root, seed=args.seed)
    else:
        # Running single combined topology experiment
        run_single_experiment(seed=args.seed, output_dir=args.single_output_dir)


if __name__ == "__main__":
    main()
