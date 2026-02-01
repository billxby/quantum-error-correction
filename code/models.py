"""
Quantum Error Correction Models and Utilities

This module contains reusable model classes and utility functions for
quantum error correction with surface codes, including:
- STIM circuit generation and MWPM decoding utilities
- Surface code sampling for training data generation
- Sparse graph representations for GNN-based decoders
- GCN (Graph Convolutional Network) decoder implementation
"""

# ============================================================
# IMPORTS
# ============================================================

import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, GINEConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import tempfile
import shutil
import os


# ============================================================
# STIM & MWPM UTILITIES
# ============================================================

def surface_code_circuit(p: float, d: int) -> stim.Circuit:
    """
    Generate a rotated surface code memory circuit.

    Args:
        p: Physical error rate
        d: Code distance (determines code size and rounds)

    Returns:
        stim.Circuit: The generated surface code circuit
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )


def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    """
    Count logical errors using MWPM decoder.

    Args:
        circuit: The stim circuit to sample from
        num_shots: Number of samples to take

    Returns:
        int: Number of logical errors (decoder mistakes)
    """
    # Sample the circuit
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


def ler_mwpm(p: float, d: int, num_shots: int = 100000) -> float:
    """
    Compute logical error rate using MWPM decoder.

    Args:
        p: Physical error rate
        d: Code distance
        num_shots: Number of samples (default: 100000)

    Returns:
        float: Logical error rate
    """
    circuit = surface_code_circuit(p, d)
    num_errors = count_logical_errors(circuit, num_shots)
    return num_errors / num_shots


def plot_mwpm(distances: list = None, noise_range: tuple = None, num_shots: int = 100000):
    """
    Plot MWPM decoder performance across distances and error rates.

    Args:
        distances: List of code distances (default: [3, 5, 7])
        noise_range: Tuple of (min, max, num_points) for noise values
        num_shots: Number of samples per point
    """
    if distances is None:
        distances = [3, 5, 7]
    if noise_range is None:
        noise_range = (0.001, 0.008, 8)

    for d in distances:
        xs = []
        ys = []
        yerrs = []
        for noise in np.linspace(*noise_range):
            ler = ler_mwpm(noise, d, num_shots)
            xs.append(noise)
            ys.append(ler)
            yerrs.append(np.sqrt(ler * (1 - ler) / num_shots))
        plt.errorbar(xs, ys, yerr=yerrs, label=f'd={d}', capsize=3)

    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()


# ============================================================
# SURFACE CODE SAMPLER
# ============================================================

class SurfaceCodeSampler:
    """
    Sampler class for surface code quantum error correction.

    This class generates surface code circuits and creates training datasets
    with configurable error rates. The distance d is specified at sample time,
    allowing sampling from different distance codes without creating multiple
    sampler instances.

    Attributes:
        default_p (float): Default physical error rate
        device (torch.device): Device to use for tensors
    """

    def __init__(self, p: float = 0.01, device: torch.device = None):
        """
        Initialize the sampler.

        Args:
            p (float): Default physical error rate (used if not overridden)
            device (torch.device): Device for tensors (defaults to CUDA if available)
        """
        self.default_p = p
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache circuits for different (d, p) values to avoid regenerating
        self._circuit_cache = {}

        print(f"SurfaceCodeSampler initialized:")
        print(f"  Default error rate: {p}")
        print(f"  Device: {self.device}")
        print(f"  Mode: Dynamic (supports any code distance)")

    def _generate_circuit(self, d: int, p: float) -> stim.Circuit:
        """Generate a surface code circuit with given distance and error rate."""
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p
        )

    def _get_circuit(self, d: int, p: float) -> stim.Circuit:
        """Get circuit for a given (d, p), using cache if available."""
        key = (d, p)
        if key not in self._circuit_cache:
            self._circuit_cache[key] = self._generate_circuit(d, p)
        return self._circuit_cache[key]

    def get_circuit(self, d: int, p: float = None) -> stim.Circuit:
        """Return a circuit for given distance and error rate."""
        if p is None:
            p = self.default_p
        return self._get_circuit(d, p)

    def sample(self,
               d: int,
               num_samples: int,
               p_values: list = None,
               p_weights: list = None,
               return_p_labels: bool = False) -> tuple:
        """
        Generate training data samples with configurable error rate distribution.

        This function generates syndrome detection data and observable flip labels.
        You can specify multiple error rates with weights to control what percentage
        of the dataset uses each error rate.

        Args:
            d (int): Code distance (determines code size and rounds)
            num_samples (int): Total number of samples to generate
            p_values (list[float], optional): Array of physical error rates to use.
                                              Defaults to [self.default_p].
            p_weights (list[float], optional): Array of weights (same length as p_values).
                                               Must sum to 1.0. Determines what fraction
                                               of samples use each error rate.
                                               Defaults to [1.0] (all samples at one rate).
            return_p_labels (bool): If True, also return which p was used for each sample.

        Returns:
            tuple: (detections, labels) or (detections, labels, p_indices) if return_p_labels
                - detections: torch.Tensor [num_samples, num_detectors] syndrome measurements (-1 or +1)
                - labels: torch.Tensor [num_samples] observable flip labels (0 or 1)
                - p_indices: torch.Tensor [num_samples] index into p_values for each sample

        Examples:
            # Single error rate (uses default p) for distance 3
            detections, labels = sampler.sample(d=3, num_samples=10000)

            # Single custom error rate for distance 5
            detections, labels = sampler.sample(d=5, num_samples=10000, p_values=[0.005], p_weights=[1.0])

            # Mixed error rates: 50% at p=0.001, 30% at p=0.003, 20% at p=0.005
            detections, labels = sampler.sample(
                d=3,
                num_samples=10000,
                p_values=[0.001, 0.003, 0.005],
                p_weights=[0.5, 0.3, 0.2]
            )
        """
        # Handle defaults
        if p_values is None:
            p_values = [self.default_p]
        if p_weights is None:
            p_weights = [1.0]

        # Validate inputs
        if len(p_values) != len(p_weights):
            raise ValueError(f"p_values and p_weights must have same length. "
                           f"Got {len(p_values)} and {len(p_weights)}")

        weight_sum = sum(p_weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"p_weights must sum to 1.0, got {weight_sum}")

        # Calculate samples per error rate
        samples_per_p = []
        remaining = num_samples
        for i, weight in enumerate(p_weights):
            if i == len(p_weights) - 1:
                # Last one gets remaining to handle rounding
                n = remaining
            else:
                n = int(round(num_samples * weight))
                remaining -= n
            samples_per_p.append(n)

        # Generate samples for each error rate
        all_detections = []
        all_labels = []
        all_p_indices = []

        for i, (p, n) in enumerate(zip(p_values, samples_per_p)):
            if n <= 0:
                continue

            circuit = self._get_circuit(d, p)
            sampler = circuit.compile_detector_sampler()
            detections, flips = sampler.sample(n, separate_observables=True)

            # Convert to tensors
            # Convert detections: 0 -> -1, 1 -> +1 for easier reading
            det_np = detections.astype(np.float32) * 2 - 1
            det_tensor = torch.from_numpy(det_np)
            label_tensor = torch.from_numpy(flips.astype(np.float32).flatten())

            all_detections.append(det_tensor)
            all_labels.append(label_tensor)
            all_p_indices.append(torch.full((n,), i, dtype=torch.long))

        # Concatenate all samples
        detections = torch.cat(all_detections, dim=0).to(self.device)
        labels = torch.cat(all_labels, dim=0).to(self.device)
        p_indices = torch.cat(all_p_indices, dim=0).to(self.device)

        # Shuffle the dataset so error rates are mixed
        perm = torch.randperm(num_samples, device=self.device)
        detections = detections[perm]
        labels = labels[perm]
        p_indices = p_indices[perm]

        if return_p_labels:
            return detections, labels, p_indices
        return detections, labels


# ============================================================
# SPARSE GRAPH REPRESENTATION
# ============================================================

class SparseGraph:
    """
    Converts syndrome detections into sparse PyTorch Geometric graphs.

    Following the paper's representation (Figure 3.1):
    - Only fired detectors become nodes (sparse graph)
    - Node features: [X?, Z?, d_North, d_West, d_time] (5 features)
    - Edge weights: e_ij = (max{|d_North_i - d_North_j|, |d_West_i - d_West_j|, |d_time_i - d_time_j|})^(-2)
    - K-nearest neighbor connectivity for scalability

    This class dynamically handles detections of any size by inferring the code
    distance from the number of detectors and generating coordinates on-the-fly.
    Supports variable-size batches where each sample may come from a different distance.

    Attributes:
        k_neighbors (int): Maximum number of neighbors per node
        device (torch.device): Device for output tensors
    """

    def __init__(self, k_neighbors: int = 6, device: torch.device = None):
        """
        Initialize the SparseGraph builder.

        Args:
            k_neighbors (int): Maximum number of neighbors per node (default 6)
            device (torch.device): Device for output tensors
        """
        self.k_neighbors = k_neighbors
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Caches for dynamic coordinate/feature computation
        self._coord_cache = {}    # num_detectors -> detector_coords dict
        self._feature_cache = {}  # num_detectors -> (all_features, normalization_bounds)

        print(f"SparseGraph initialized:")
        print(f"  K neighbors: {k_neighbors}")
        print(f"  Device: {self.device}")
        print(f"  Mode: Dynamic (supports any code distance)")

    @staticmethod
    def _infer_distance(num_detectors: int) -> int:
        """
        Infer code distance from number of detectors.
        """
        # Try known mappings first
        known = {24: 3, 120: 5, 336: 7, 680: 9, 1168: 11}
        if num_detectors in known:
            return known[num_detectors]

        # Otherwise solve: num_stabilizers * d = num_detectors
        # where num_stabilizers = d^2 - 1 for rotated surface code
        # So: (d^2 - 1) * d = num_detectors
        for d in range(3, 50, 2):  # odd distances only
            if (d * d - 1) * d == num_detectors:
                return d

        raise ValueError(f"Cannot infer distance from {num_detectors} detectors")

    @staticmethod
    def _generate_detector_coords(distance: int) -> dict:
        """
        Generate detector coordinates for a rotated surface code by creating
        a Stim circuit and extracting the actual detector coordinates.

        Args:
            distance: The code distance (must be odd >= 3)

        Returns:
            dict mapping detector_id -> [x, y, t, basis]
            where basis is 0 for Z-stabilizer, 1 for X-stabilizer
        """
        # Create a minimal Stim circuit for this distance to get correct coordinates
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001  # Minimal noise, just need structure
        )

        # Extract coordinates from Stim (returns dict: det_id -> tuple of coords)
        stim_coords = circuit.get_detector_coordinates()

        # Convert to our format: det_id -> [x, y, t, basis]
        detector_coords = {}
        for det_id, coords in stim_coords.items():
            x = coords[0]
            y = coords[1]
            t = coords[2] if len(coords) > 2 else 0.0
            # Determine basis from position (checkerboard pattern)
            # Convention: (x+y)/2 even -> Z stabilizer, odd -> X stabilizer
            basis = int(((x + y) / 2) % 2)
            detector_coords[det_id] = [float(x), float(y), float(t), float(basis)]

        return detector_coords

    def _get_coords_and_features(self, num_detectors: int):
        """
        Get detector coordinates and precomputed features for a given number of detectors.
        Uses caching to avoid recomputation.

        Args:
            num_detectors: Number of detectors in the detection sample

        Returns:
            tuple: (detector_coords, all_features, normalization_bounds, all_raw_coords)
                - detector_coords: dict mapping detector_id -> [x, y, t, basis]
                - all_features: Tensor [num_detectors, 5] with normalized node features
                - normalization_bounds: dict with x/y/t min/max values
                - all_raw_coords: Tensor [num_detectors, 3] with raw (north, west, time) for edge weights
        """
        if num_detectors in self._feature_cache:
            cached = self._feature_cache[num_detectors]
            # Handle legacy cache entries without raw_coords (length 3 vs 4)
            if len(cached) == 4:
                return cached
            # Rebuild cache with raw coords
            del self._feature_cache[num_detectors]

        # Cache miss: compute everything
        distance = self._infer_distance(num_detectors)

        # Generate or retrieve coordinates
        if num_detectors not in self._coord_cache:
            self._coord_cache[num_detectors] = self._generate_detector_coords(distance)

        detector_coords = self._coord_cache[num_detectors]

        # Compute normalization bounds
        coords = list(detector_coords.values())
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        t_vals = [c[2] for c in coords]

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        t_min, t_max = min(t_vals), max(t_vals)

        norm_bounds = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            't_min': t_min, 't_max': t_max
        }

        # Precompute features for all detectors
        features = []
        raw_coords = []  # Store raw (unnormalized) coordinates for edge weight computation
        for det_id in range(num_detectors):
            coord = detector_coords.get(det_id, [0, 0, 0, 0])
            x, y, t = coord[0], coord[1], coord[2]

            # Store raw coordinates (y=north, x=west, t=time)
            raw_coords.append([y, x, t])

            # Get basis from 4th coordinate
            if len(coord) >= 4:
                b = coord[3]
                is_x = 1.0 if b == 1 else 0.0
                is_z = 1.0 if b == 0 else 0.0
            else:
                # Fallback: checkerboard pattern
                stabilizer_type = int(((x + y) / 2) % 2)
                is_x = float(stabilizer_type)
                is_z = 1.0 - is_x

            # Normalized distances (0 to 1) for node features
            d_west = (x - x_min) / max(1, x_max - x_min)
            d_north = (y - y_min) / max(1, y_max - y_min)
            d_time = (t - t_min) / max(1, t_max - t_min)

            features.append([is_x, is_z, d_north, d_west, d_time])

        all_features = torch.tensor(features, dtype=torch.float32)
        all_raw_coords = torch.tensor(raw_coords, dtype=torch.float32)  # [num_detectors, 3]

        # Cache the result (now includes raw coordinates for edge weights)
        self._feature_cache[num_detectors] = (detector_coords, all_features, norm_bounds, all_raw_coords)

        return detector_coords, all_features, norm_bounds, all_raw_coords

    def _supremum_distance(self, feat_i: torch.Tensor, feat_j: torch.Tensor) -> float:
        """
        Compute supremum norm distance between two nodes.
        Uses d_North, d_West, d_time (indices 2, 3, 4 of features).
        """
        # Features are [X?, Z?, d_North, d_West, d_time]
        d_north_diff = abs(feat_i[2] - feat_j[2])
        d_west_diff = abs(feat_i[3] - feat_j[3])
        d_time_diff = abs(feat_i[4] - feat_j[4])

        return max(d_north_diff.item(), d_west_diff.item(), d_time_diff.item())

    def _compute_edge_weight(self, sup_dist: float) -> float:
        """
        Compute edge weight from supremum distance.
        e_ij = (max_distance)^(-2)
        """
        if sup_dist < 1e-9:
            return 1.0  # Same position, max weight
        return sup_dist ** (-2)

    def to_pyg(self, detections: torch.Tensor, label: torch.Tensor) -> Data:
        """
        Convert a single detection sample to a PyTorch Geometric Data object.

        Uses vectorized k-NN computation for ~10-30x speedup over naive loops.

        Args:
            detections: Tensor of shape [num_detectors] with values -1 or +1
            label: Scalar tensor (0 or 1) for the observable flip

        Returns:
            torch_geometric.data.Data with:
                - x: Node features [num_fired, 5]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge weights [num_edges, 1]
                - y: Label
        """
        # Move to CPU for processing
        detections_cpu = detections.cpu()
        num_detectors = detections_cpu.shape[0]

        # Get cached coordinates, features, and raw coords for edge weights
        _, all_features, _, all_raw_coords = self._get_coords_and_features(num_detectors)

        # Find fired detectors (value == +1)
        fired_mask = detections_cpu > 0
        fired_indices = torch.where(fired_mask)[0]
        num_nodes = fired_indices.shape[0]

        # Handle edge case: no fired detectors
        if num_nodes == 0:
            return Data(
                x=torch.zeros((0, 5), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1), dtype=torch.float32),
                y=label.cpu().clone()
            )

        # Get node features for fired detectors
        node_features = all_features[fired_indices]

        # Handle edge case: only one fired detector
        if num_nodes == 1:
            return Data(
                x=node_features,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1), dtype=torch.float32),
                y=label.cpu().clone()
            )

        # === VECTORIZED k-NN with supremum (L-infinity) distance ===
        # Use RAW (unnormalized) coordinates for edge weight computation
        # This ensures distances are integers (1, 2, 3...) so weights = dist^(-2) are meaningful
        raw_coords = all_raw_coords[fired_indices]  # [n, 3] - raw (north, west, time)

        # Compute pairwise L-infinity (supremum) distance using broadcasting
        # diff[i, j, k] = raw_coords[i, k] - raw_coords[j, k]
        diff = raw_coords.unsqueeze(0) - raw_coords.unsqueeze(1)  # [n, n, 3]
        sup_dist = diff.abs().max(dim=2).values  # [n, n] pairwise supremum distances

        # Exclude self-connections by setting diagonal to infinity
        sup_dist.fill_diagonal_(float('inf'))

        # Determine actual k (can't have more neighbors than nodes - 1)
        k = min(self.k_neighbors, num_nodes - 1)

        # Get k-nearest neighbors for each node (smallest distances)
        # topk with largest=False gives k smallest values
        _, knn_indices = sup_dist.topk(k, largest=False, dim=1)  # [n, k]

        # Build edge_index vectorized
        # src: each node repeated k times [0,0,...,0, 1,1,...,1, ..., n-1,n-1,...,n-1]
        # dst: the k nearest neighbors for each node
        src = torch.arange(num_nodes).unsqueeze(1).expand(-1, k).flatten()  # [n*k]
        dst = knn_indices.flatten()  # [n*k]
        edge_index = torch.stack([src, dst], dim=0).long()  # [2, n*k]

        # Compute edge weights: w = sup_dist^(-2)
        # With raw coords, distances are integers >= 1, so weights are in (0, 1]
        # dist=1 -> w=1.0, dist=2 -> w=0.25, dist=3 -> w=0.111, etc.
        edge_distances = sup_dist[src, dst]  # [n*k]
        edge_attr = (edge_distances ** -2).unsqueeze(1)  # [n*k, 1]

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label.cpu().clone()
        )

    def batch_to_pyg(self, detections: list, labels: list) -> list:
        """
        Convert a batch of detection samples to a list of PyG Data objects.
        Supports variable-size inputs where each detection may have different dimensions.

        Args:
            detections: List of tensors, each of shape [num_detectors_i]
            labels: List of scalar tensors

        Returns:
            List of torch_geometric.data.Data objects
        """
        return [self.to_pyg(det, lbl) for det, lbl in zip(detections, labels)]


def visualize_sparse_graph(graph: Data, title: str = "Sparse Graph Visualization"):
    """
    Visualize a PyG sparse graph with node features and edge weights.

    Node colors: Blue = Z-stabilizer, Red = X-stabilizer
    Node positions: Based on (d_West, d_North) spatial coordinates
    Edge thickness: Proportional to edge weight

    Args:
        graph: PyTorch Geometric Data object from SparseGraph
        title: Plot title
    """
    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Graph structure
    ax1 = axes[0]

    if graph.x.shape[0] == 0:
        ax1.text(0.5, 0.5, "No fired detectors\n(empty graph)",
                ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    else:
        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes with features
        for i in range(graph.x.shape[0]):
            features = graph.x[i].tolist()
            G.add_node(i,
                      is_x=features[0],
                      is_z=features[1],
                      d_north=features[2],
                      d_west=features[3],
                      d_time=features[4])

        # Add edges with weights
        edge_index = graph.edge_index.numpy()
        edge_weights = graph.edge_attr.numpy().flatten() if graph.edge_attr.shape[0] > 0 else []

        for idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, idx], edge_index[1, idx]
            weight = edge_weights[idx] if idx < len(edge_weights) else 1.0
            G.add_edge(src, dst, weight=weight)

        # Position nodes based on spatial coordinates (d_West, d_North)
        # Add time as a small offset to separate overlapping nodes
        pos = {}
        for i in range(graph.x.shape[0]):
            d_west = graph.x[i, 3].item()
            d_north = graph.x[i, 2].item()
            d_time = graph.x[i, 4].item()
            # Use west as x, north as y, with small time-based jitter
            pos[i] = (d_west + d_time * 0.05, d_north + d_time * 0.05)

        # Node colors based on stabilizer type
        node_colors = ['#e74c3c' if graph.x[i, 0].item() > 0.5 else '#3498db'
                      for i in range(graph.x.shape[0])]

        # Edge widths based on weights (scaled for visibility)
        if len(edge_weights) > 0:
            max_weight = max(edge_weights)
            edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
        else:
            edge_widths = []

        #Debug
        print("Edge width: ", *edge_widths)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                              node_size=500, alpha=0.9, edgecolors='black', linewidths=2)
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')

        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths,
                                  alpha=0.6, edge_color='gray',
                                  arrows=True, arrowsize=15,
                                  connectionstyle="arc3,rad=0.1")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', edgecolor='black', label='X-stabilizer'),
            Patch(facecolor='#3498db', edgecolor='black', label='Z-stabilizer')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        ax1.set_xlabel('d_West (spatial)')
        ax1.set_ylabel('d_North (spatial)')

    ax1.set_title(f'{title}\nNodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    # Right plot: Feature details table
    ax2 = axes[1]
    ax2.axis('off')

    if graph.x.shape[0] > 0:
        # Create feature table
        headers = ['Node', 'Type', 'd_North', 'd_West', 'd_Time']
        cell_data = []
        for i in range(graph.x.shape[0]):
            f = graph.x[i].tolist()
            node_type = 'X' if f[0] > 0.5 else 'Z'
            cell_data.append([i, node_type, f'{f[2]:.3f}', f'{f[3]:.3f}', f'{f[4]:.3f}'])

        table = ax2.table(cellText=cell_data, colLabels=headers,
                         loc='upper center', cellLoc='center',
                         colColours=['#f0f0f0']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Add edge info below the table
        info_text = f"Label: {graph.y.item():.0f} (Observable flip)\n\n"
        if graph.edge_index.shape[1] > 0:
            info_text += "Edge Weights (sample):\n"
            for idx in range(min(5, graph.edge_index.shape[1])):
                src = graph.edge_index[0, idx].item()
                dst = graph.edge_index[1, idx].item()
                w = graph.edge_attr[idx, 0].item()
                info_text += f"  {src} -> {dst}: {w:.4f}\n"
            if graph.edge_index.shape[1] > 5:
                info_text += f"  ... and {graph.edge_index.shape[1] - 5} more edges"

        ax2.text(0.5, 0.3, info_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                family='monospace')
    else:
        ax2.text(0.5, 0.5, "No nodes to display", ha='center', va='center', fontsize=12)

    ax2.set_title('Node Features & Graph Info')

    plt.tight_layout()
    plt.show()

# ============================================================
# DATASET CACHE
# ============================================================

def _save_to_gdrive(data, path):
    """
    Save data to path using temp file to avoid Google Drive sync conflicts.

    Google Drive's sync process can interfere with large file writes, causing
    corruption. This function writes to a local temp file first, then moves
    the completed file to the target path in one atomic operation.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            temp_path = tmp.name
        torch.save(data, temp_path)
        shutil.move(temp_path, path)
    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise


class DatasetCache:
    """
    Cache manager for pre-generated PyG graph datasets.

    This class handles the generation, storage, and retrieval of training datasets
    for GNN-based quantum error correction decoders. It eliminates the need to
    regenerate graphs on every training run by caching them to disk.

    Features:
    - Generate datasets with configurable distance, error rates, and sample sizes
    - Save/load datasets to/from disk with metadata
    - Incrementally grow datasets with ensure_size()
    - List and manage cached datasets

    Attributes:
        graphs (list): List of PyG Data objects
        config (dict): Dataset configuration metadata
        datasets_dir (Path): Directory for cached datasets
    """

    def __init__(self, base_path: Path = None, device: torch.device = None):
        """
        Initialize the DatasetCache.

        Args:
            base_path: Base path for dataset storage (defaults to current directory)
            device: Torch device for generation (defaults to CUDA if available)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets_dir = (base_path or Path(".")) / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Current dataset state
        self.graphs = []
        self.config = {}

        # Original detection arrays (saved alongside graphs)
        self.detections = None
        self.labels = None

    def generate(
        self,
        d: int,
        n_samples: int,
        p_values: list,
        p_weights: list,
        k_neighbors: int = 6,
        verbose: bool = True
    ) -> 'DatasetCache':
        """
        Generate a new dataset of PyG graphs.

        Args:
            d: Code distance
            n_samples: Number of samples to generate
            p_values: List of physical error rates
            p_weights: Weights for error rate distribution (must sum to 1.0)
            k_neighbors: K-neighbors for SparseGraph (default: 6)
            verbose: Print progress information

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating dataset: d={d}, n_samples={n_samples:,}")
            print(f"Error rates: {p_values} (weights: {p_weights})")
            print(f"{'='*60}")

        # Store configuration
        self.config = {
            'd': d,
            'n_samples': n_samples,
            'p_values': p_values,
            'p_weights': p_weights,
            'k_neighbors': k_neighbors,
            'generated_at': datetime.now().isoformat()
        }

        # Initialize sampler and graph builder
        sampler = SurfaceCodeSampler(p=p_values[0], device=self.device)
        graph_builder = SparseGraph(k_neighbors=k_neighbors, device=self.device)

        # Generate samples
        if verbose:
            print(f"\nSampling {n_samples:,} detection events...")

        detections, labels = sampler.sample(
            d=d,
            num_samples=n_samples,
            p_values=p_values,
            p_weights=p_weights
        )

        # Store original arrays (saved alongside graphs in save())
        self.detections = detections.cpu()
        self.labels = labels.cpu()

        # Convert to graphs with progress bar
        if verbose:
            print(f"Converting to PyG graphs...")

        self.graphs = []
        with tqdm(total=n_samples, desc="Converting", unit="graph",
                  disable=not verbose, dynamic_ncols=True) as pbar:
            for i in range(n_samples):
                graph = graph_builder.to_pyg(detections[i], labels[i])
                self.graphs.append(graph)
                pbar.update(1)

        if verbose:
            print(f"\nGenerated {len(self.graphs):,} graphs")

        return self

    def save(self, name: str) -> Path:
        """
        Save the dataset to disk with metadata.

        Args:
            name: Name for the dataset (e.g., 'd5_baseline')

        Returns:
            Path to the saved dataset file
        """
        if not self.graphs:
            raise ValueError("No graphs to save. Call generate() first.")

        # Save array format FIRST (smaller, faster, contains original data)
        # This ensures arrays are saved even if graph saving fails
        if self.detections is not None:
            nn_datasets_dir = self.datasets_dir.parent / "nn_datasets"
            nn_datasets_dir.mkdir(parents=True, exist_ok=True)

            array_data = {
                'detections': self.detections,
                'labels': self.labels,
                'd': self.config['d'],
                'n_samples': len(self.labels),
                'num_detectors': self.detections.shape[1],
                'p_values': self.config['p_values'],
                'generated_at': self.config['generated_at'],
            }
            array_path = nn_datasets_dir / f"{name}_array.pt"
            _save_to_gdrive(array_data, array_path)
            print(f"Array dataset saved: {array_path}")

        # Save graphs (larger file, takes longer)
        data_path = self.datasets_dir / f"{name}.pt"
        _save_to_gdrive(self.graphs, data_path)

        # Save metadata
        meta_path = self.datasets_dir / f"{name}.json"
        with open(meta_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Graph dataset saved: {data_path}")
        print(f"  Metadata: {meta_path}")
        print(f"  Samples: {len(self.graphs):,}")

        return data_path

    def load(self, name: str, verbose: bool = True) -> 'DatasetCache':
        """
        Load a dataset from disk.

        Args:
            name: Name of the dataset to load
            verbose: Print progress information (default: True)

        Returns:
            self (for chaining)
        """
        import os

        data_path = self.datasets_dir / f"{name}.pt"
        meta_path = self.datasets_dir / f"{name}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        # Get file size for progress display
        file_size_bytes = os.path.getsize(data_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

        if verbose:
            if file_size_gb >= 1:
                size_str = f"{file_size_gb:.2f} GB"
            else:
                size_str = f"{file_size_mb:.1f} MB"
            print(f"Loading dataset '{name}' ({size_str})...")

        # Load graphs
        if verbose:
            print(f"  Reading {name}.pt (this may take a while for large files)...")
        self.graphs = torch.load(str(data_path), weights_only=False)

        # Load metadata if exists
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {'name': name, 'n_samples': len(self.graphs)}

        if verbose:
            print(f"  Loaded {len(self.graphs):,} graphs")
            if 'd' in self.config:
                print(f"  Distance: d={self.config['d']}")
            if 'p_values' in self.config:
                print(f"  Error rates: {self.config['p_values']}")

        return self

    def ensure_size(self, n: int, verbose: bool = True) -> 'DatasetCache':
        """
        Ensure the dataset has at least n samples, generating more if needed.

        Args:
            n: Minimum number of samples required
            verbose: Print progress information

        Returns:
            self (for chaining)
        """
        current_size = len(self.graphs)

        if current_size >= n:
            if verbose:
                print(f"Dataset already has {current_size:,} samples (requested {n:,})")
            return self

        # Need to generate more
        needed = n - current_size

        if not self.config or 'd' not in self.config:
            raise ValueError("Cannot grow dataset without config. Load a dataset first or call generate().")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Growing dataset: {current_size:,} -> {n:,} (+{needed:,})")
            print(f"{'='*60}")

        # Generate additional samples
        d = self.config['d']
        p_values = self.config['p_values']
        p_weights = self.config['p_weights']
        k_neighbors = self.config.get('k_neighbors', 6)

        sampler = SurfaceCodeSampler(p=p_values[0], device=self.device)
        graph_builder = SparseGraph(k_neighbors=k_neighbors, device=self.device)

        if verbose:
            print(f"\nSampling {needed:,} additional detection events...")

        detections, labels = sampler.sample(
            d=d,
            num_samples=needed,
            p_values=p_values,
            p_weights=p_weights
        )

        if verbose:
            print(f"Converting to PyG graphs...")

        with tqdm(total=needed, desc="Converting", unit="graph",
                  disable=not verbose, dynamic_ncols=True) as pbar:
            for i in range(needed):
                graph = graph_builder.to_pyg(detections[i], labels[i])
                self.graphs.append(graph)
                pbar.update(1)

        # Update config
        self.config['n_samples'] = len(self.graphs)
        self.config['last_grown'] = datetime.now().isoformat()

        if verbose:
            print(f"\nDataset now has {len(self.graphs):,} samples")

        return self

    def get_graphs(self, n: int = None, shuffle: bool = False) -> list:
        """
        Get graphs from the dataset.

        Args:
            n: Number of graphs to return (None = all)
            shuffle: Whether to shuffle before returning

        Returns:
            List of PyG Data objects
        """
        import random

        graphs = self.graphs.copy() if shuffle else self.graphs

        if shuffle:
            random.shuffle(graphs)

        if n is not None:
            return graphs[:n]
        return graphs

    def size(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    @classmethod
    def list_datasets(cls, base_path: Path = None) -> list:
        """
        List all cached datasets.

        Args:
            base_path: Base path for dataset storage

        Returns:
            List of dicts with dataset info
        """
        datasets_dir = (base_path or Path(".")) / "datasets"

        if not datasets_dir.exists():
            return []

        datasets = []
        for pt_file in datasets_dir.glob("*.pt"):
            name = pt_file.stem
            meta_path = datasets_dir / f"{name}.json"

            info = {
                'name': name,
                'path': str(pt_file),
                'size_mb': pt_file.stat().st_size / (1024 * 1024)
            }

            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                info.update(meta)

            datasets.append(info)

        return datasets

    def __repr__(self) -> str:
        if self.config:
            return (f"DatasetCache(n_samples={len(self.graphs):,}, "
                    f"d={self.config.get('d', '?')}, "
                    f"p_values={self.config.get('p_values', '?')})")
        return f"DatasetCache(n_samples={len(self.graphs):,})"

    def __len__(self) -> int:
        return len(self.graphs)

# ============================================================
# GCN MODEL
# ============================================================

class GCNModel(torch.nn.Module):
    """
    Graph Convolutional Network that handles dynamic graph sizes.
    Uses global mean pooling to produce a single output regardless of input graph size.

    Supports edge weights (edge_attr) from SparseGraph which are passed through
    all GCN layers for weighted message passing.
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Classification head (compresses to single output)
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # Convert edge_attr [N,1] to edge_weight [N] for PyG's GCNConv
        edge_weight = data.edge_attr.view(-1) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graphs (no fired detectors) - return default prediction
        num_graphs = int(data.num_graphs) if hasattr(data, 'num_graphs') else (int(batch.max().item()) + 1 if x.size(0) > 0 else 1)
        if x.size(0) == 0:
            return torch.full((num_graphs, 1), 0.5, device=data.x.device if hasattr(data.x, 'device') else 'cpu')

        # Apply GCN layers with batch normalization and activation
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.silu(x)

        # Global mean pooling: aggregate node features to graph-level (optimized)
        x_pooled = global_mean_pool(x, batch, size=num_graphs)

        # Classification layers
        x = self.fc1(x_pooled)
        x = F.silu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class GCN:
    """
    GCN wrapper class with model lifecycle management.
    Provides init, train, save, and load functionality with human-readable model tracking.

    Designed to work with SparseGraph outputs:
    - Node features: [X?, Z?, d_North, d_West, d_time] (5 features)
    - Edge weights: distance-based weights

    Attributes:
        nickname: Human-readable name for this model instance
        model: The underlying GCNModel
        device: Torch device (cuda/cpu)
        models_dir: Directory for saving/loading models
        _loaded_from: Path of the loaded model (None if freshly initialized)
    """

    def __init__(self,
                 nickname: str = "gcn_model",
                 in_channels: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 device: torch.device = None,
                 base_path: Path = None,
                 seed: int = None):
        """
        Initialize a new GCN model.

        Args:
            nickname: Human-readable name for this model
            in_channels: Number of input features per node (default 5 for SparseGraph)
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers
            device: Torch device (defaults to CUDA if available)
            base_path: Base path for model storage (defaults to current directory)
            seed: Random seed for reproducibility (default: None, no seeding)
        """
        self.nickname = nickname
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_from = None

        # Set models directory based on base_path
        self.models_dir = (base_path or Path(".")) / "models" / "gcn"

        # Store config for saving/loading
        self._config = {
            'in_channels': in_channels,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seed': seed
        }

        # Set random seeds for reproducibility before model initialization
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize model
        self.model = GCNModel(in_channels, hidden_dim, num_layers).to(self.device)

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print(f"GCN initialized: {self}")

    def train(self,
              graphs: list,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              verbose: bool = True) -> list:
        """
        Train the model on a list of PyG graphs.

        Args:
            graphs: List of PyG Data objects (from SparseGraph.batch_to_pyg)
            epochs: Number of training epochs
            batch_size: Number of graphs per batch
            lr: Learning rate
            verbose: Print training progress

        Returns:
            List of loss values per epoch
        """
        from torch_geometric.loader import DataLoader

        # Announce training start
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {self}")
            print(f"{'='*50}")
            print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
            print(f"Training samples: {len(graphs)}")

        loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        total_batches = len(loader) * epochs

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0

        pbar = tqdm(total=total_batches, desc="Training", disable=not verbose)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data)
                y = batch_data.y.float().view(-1, 1)

                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Batch metrics
                batch_loss = loss.item()
                batch_acc = ((pred > 0.5).float() == y).float().mean().item() * 100

                # Running averages (exponential smoothing)
                running_loss = batch_loss * 0.1 + running_loss * 0.9 if running_loss else batch_loss
                running_acc = batch_acc * 0.1 + running_acc * 0.9 if running_acc else batch_acc

                # Accumulate epoch stats
                epoch_loss += batch_loss
                epoch_correct += ((pred > 0.5).float() == y).sum().item()
                epoch_total += y.size(0)

                batch_count += 1
                pbar.update(1)

                # Update display every 10 batches
                if batch_count % 10 == 0:
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{running_loss:.4f}',
                        'acc': f'{running_acc:.1f}%'
                    })

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

        pbar.close()

        final_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose:
            print(f"\nTraining complete! Final - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.1f}%")

        return epoch_losses

    def save(self, name: str) -> Path:
        """
        Save the model with a human-readable timestamp.

        Args:
            name: Base name for the saved model

        Returns:
            Path to the saved model file
        """
        # Create timestamp in human-readable format
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = self.models_dir / filename

        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self._config,
            'nickname': self.nickname,
            'timestamp': timestamp
        }
        torch.save(save_dict, filepath)

        print(f"Model saved to: {filepath}")
        return filepath

    def load(self, filepath: str) -> 'GCN':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model file (relative to models/gcn or absolute)

        Returns:
            self (for chaining)
        """
        # Handle relative paths
        path = Path(filepath)
        if not path.exists():
            path = self.models_dir / filepath

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate model with saved config
        config = save_dict['config']
        self._config = config
        self.model = GCNModel(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(save_dict['state_dict'])
        self.model.eval()

        # Update tracking
        self._loaded_from = path.name
        if 'nickname' in save_dict:
            self.nickname = save_dict['nickname']

        print(f"Model loaded: {self}")
        return self

    def predict(self, data) -> torch.Tensor:
        """Run inference on data."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self.model(data)

    def __repr__(self) -> str:
        loaded_info = f", loaded_from='{self._loaded_from}'" if self._loaded_from else ""
        return (f"GCN(nickname='{self.nickname}', "
                f"in_channels={self._config['in_channels']}, "
                f"hidden_dim={self._config['hidden_dim']}, "
                f"num_layers={self._config['num_layers']}"
                f"{loaded_info})")


# ============================================================
# GAT MODEL
# ============================================================

class GATModel(torch.nn.Module):
    """
    Graph Attention Network that handles dynamic graph sizes.
    Uses PyTorch Geometric's GATConv with multi-head attention.
    Uses global mean pooling to produce a single output regardless of input graph size.

    Supports edge weights (edge_attr) from SparseGraph which are flattened and passed
    as edge_weight to GATConv layers.
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 128, num_layers: int = 4,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # GAT layers with multi-head attention
        self.convs = torch.nn.ModuleList()

        # First layer: in_channels -> hidden_dim (with heads, output is heads * hidden_dim if concat=True)
        # We use hidden_dim // heads to get hidden_dim after concatenation
        head_dim = hidden_dim // heads
        self.convs.append(GATConv(in_channels, head_dim, heads=heads, concat=True,
                                   dropout=dropout, add_self_loops=True))

        # Middle and final layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, head_dim, heads=heads, concat=True,
                                       dropout=dropout, add_self_loops=True))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Classification head (compresses to single output)
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Handle edge weights: flatten edge_attr to 1D for GATConv
        edge_weight = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graphs (no fired detectors) - return default prediction
        num_graphs = int(data.num_graphs) if hasattr(data, 'num_graphs') else (int(batch.max().item()) + 1 if x.size(0) > 0 else 1)
        if x.size(0) == 0:
            return torch.full((num_graphs, 1), 0.5, device=data.x.device if hasattr(data.x, 'device') else 'cpu')

        # Apply GAT layers with batch normalization and activation
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_weight)
            x = bn(x)
            x = F.silu(x)

        # Global mean pooling: aggregate node features to graph-level (optimized)
        x_pooled = global_mean_pool(x, batch, size=num_graphs)

        # Classification layers
        x = self.fc1(x_pooled)
        x = F.silu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class GAT:
    """
    GAT wrapper class with model lifecycle management.
    Provides init, train, save, and load functionality with human-readable model tracking.

    Designed to work with SparseGraph outputs:
    - Node features: [X?, Z?, d_North, d_West, d_time] (5 features)
    - Edge weights: distance-based weights (flattened for GATConv)

    Attributes:
        nickname: Human-readable name for this model instance
        model: The underlying GATModel
        device: Torch device (cuda/cpu)
        models_dir: Directory for saving/loading models
        _loaded_from: Path of the loaded model (None if freshly initialized)
    """

    def __init__(self,
                 nickname: str = "gat_model",
                 in_channels: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 heads: int = 4,
                 dropout: float = 0.0,
                 device: torch.device = None,
                 base_path: Path = None,
                 seed: int = None):
        """
        Initialize a new GAT model.

        Args:
            nickname: Human-readable name for this model
            in_channels: Number of input features per node (default 5 for SparseGraph)
            hidden_dim: Hidden dimension size (should be divisible by heads)
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate for attention weights
            device: Torch device (defaults to CUDA if available)
            base_path: Base path for model storage (defaults to current directory)
            seed: Random seed for reproducibility (default: None, no seeding)
        """
        self.nickname = nickname
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_from = None

        # Set models directory based on base_path
        self.models_dir = (base_path or Path(".")) / "models" / "gat"

        # Store config for saving/loading
        self._config = {
            'in_channels': in_channels,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'heads': heads,
            'dropout': dropout,
            'seed': seed
        }

        # Set random seeds for reproducibility before model initialization
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize model
        self.model = GATModel(in_channels, hidden_dim, num_layers, heads, dropout).to(self.device)

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print(f"GAT initialized: {self}")

    def train(self,
              graphs: list,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              verbose: bool = True) -> list:
        """
        Train the model on a list of PyG graphs.

        Args:
            graphs: List of PyG Data objects (from SparseGraph.batch_to_pyg)
            epochs: Number of training epochs
            batch_size: Number of graphs per batch
            lr: Learning rate
            verbose: Print training progress

        Returns:
            List of loss values per epoch
        """
        from torch_geometric.loader import DataLoader

        # Announce training start
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {self}")
            print(f"{'='*50}")
            print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
            print(f"Training samples: {len(graphs)}")

        loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        total_batches = len(loader) * epochs

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0

        pbar = tqdm(total=total_batches, desc="Training", disable=not verbose)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data)
                y = batch_data.y.float().view(-1, 1)

                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Batch metrics
                batch_loss = loss.item()
                batch_acc = ((pred > 0.5).float() == y).float().mean().item() * 100

                # Running averages (exponential smoothing)
                running_loss = batch_loss * 0.1 + running_loss * 0.9 if running_loss else batch_loss
                running_acc = batch_acc * 0.1 + running_acc * 0.9 if running_acc else batch_acc

                # Accumulate epoch stats
                epoch_loss += batch_loss
                epoch_correct += ((pred > 0.5).float() == y).sum().item()
                epoch_total += y.size(0)

                batch_count += 1
                pbar.update(1)

                # Update display every 10 batches
                if batch_count % 10 == 0:
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{running_loss:.4f}',
                        'acc': f'{running_acc:.1f}%'
                    })

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

        pbar.close()

        final_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose:
            print(f"\nTraining complete! Final - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.1f}%")

        return epoch_losses

    def save(self, name: str) -> Path:
        """
        Save the model with a human-readable timestamp.

        Args:
            name: Base name for the saved model

        Returns:
            Path to the saved model file
        """
        # Create timestamp in human-readable format
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = self.models_dir / filename

        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self._config,
            'nickname': self.nickname,
            'timestamp': timestamp
        }
        torch.save(save_dict, filepath)

        print(f"Model saved to: {filepath}")
        return filepath

    def load(self, filepath: str) -> 'GAT':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model file (relative to models/gat or absolute)

        Returns:
            self (for chaining)
        """
        # Handle relative paths
        path = Path(filepath)
        if not path.exists():
            path = self.models_dir / filepath

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate model with saved config
        config = save_dict['config']
        self._config = config
        self.model = GATModel(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            heads=config.get('heads', 4),
            dropout=config.get('dropout', 0.0)
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(save_dict['state_dict'])
        self.model.eval()

        # Update tracking
        self._loaded_from = path.name
        if 'nickname' in save_dict:
            self.nickname = save_dict['nickname']

        print(f"Model loaded: {self}")
        return self

    def predict(self, data) -> torch.Tensor:
        """Run inference on data."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self.model(data)

    def __repr__(self) -> str:
        loaded_info = f", loaded_from='{self._loaded_from}'" if self._loaded_from else ""
        return (f"GAT(nickname='{self.nickname}', "
                f"in_channels={self._config['in_channels']}, "
                f"hidden_dim={self._config['hidden_dim']}, "
                f"num_layers={self._config['num_layers']}, "
                f"heads={self._config['heads']}"
                f"{loaded_info})")


# ============================================================
# GraphSAGE MODEL
# ============================================================

class WeightedSAGEConv(MessagePassing):
    """
    GraphSAGE convolution with edge weight support and configurable aggregation.

    Standard SAGEConv does not support edge weights. This custom layer follows
    the GraphSAGE approach (aggregate neighbors then concatenate with self)
    but incorporates edge weights into the aggregation.

    Supports three aggregation types:
    - 'mean': Weighted mean aggregation (default)
    - 'max': Element-wise max aggregation (edge weights used for weighting before max)
    - 'lstm': LSTM-based aggregation (sequential processing of neighbors)

    Formula:
    h_v = W * [h_v || weighted_agg(h_u * w_uv for u in N(v))]
    """

    def __init__(self, in_channels: int, out_channels: int, aggr_type: str = 'mean',
                 normalize: bool = False, root_weight: bool = True, bias: bool = True, **kwargs):
        # For LSTM we use a custom aggregation, otherwise use 'add' for weighted mean/max
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_type = aggr_type
        self.normalize = normalize
        self.root_weight = root_weight

        # Linear transformation for concatenated [self || neighbors]
        if root_weight:
            self.lin = torch.nn.Linear(in_channels * 2, out_channels, bias=bias)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # LSTM for lstm aggregation
        if aggr_type == 'lstm':
            self.lstm = torch.nn.LSTM(in_channels, in_channels, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        if self.aggr_type == 'lstm':
            # LSTM aggregation: process neighbors sequentially
            out = self._lstm_aggregate(x, edge_index, edge_weight)
        elif self.aggr_type == 'max':
            # Max aggregation with edge weights
            out = self._max_aggregate(x, edge_index, edge_weight)
        else:
            # Mean aggregation (default)
            out = self._mean_aggregate(x, edge_index, edge_weight)

        # Concatenate self with aggregated neighbors (GraphSAGE style)
        if self.root_weight:
            out = torch.cat([x, out], dim=-1)

        out = self.lin(out)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def _mean_aggregate(self, x, edge_index, edge_weight):
        """Weighted mean aggregation."""
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, aggr_mode='mean')

        # Normalize by degree (weighted degree if edge_weight provided)
        if edge_weight is not None:
            row, col = edge_index
            deg = torch.zeros(x.size(0), device=x.device)
            deg.scatter_add_(0, row, edge_weight)
            deg = deg.clamp(min=1)
            out = out / deg.view(-1, 1)
        else:
            from torch_geometric.utils import degree
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg = deg.clamp(min=1)
            out = out / deg.view(-1, 1)

        return out

    def _max_aggregate(self, x, edge_index, edge_weight):
        """Max aggregation with optional edge weight scaling."""
        num_nodes = x.size(0)
        row, col = edge_index

        # Get neighbor features, optionally scaled by edge weights
        neighbor_features = x[col]
        if edge_weight is not None:
            neighbor_features = neighbor_features * edge_weight.view(-1, 1)

        # Initialize output with very negative values for max
        out = torch.full((num_nodes, x.size(1)), float('-inf'), device=x.device)

        # Scatter max: for each node, take element-wise max of all neighbor features
        out = out.scatter_reduce(0, row.unsqueeze(1).expand(-1, x.size(1)),
                                  neighbor_features, reduce='amax', include_self=False)

        # Replace -inf with 0 for nodes with no neighbors
        out = torch.where(out == float('-inf'), torch.zeros_like(out), out)

        return out

    def _lstm_aggregate(self, x, edge_index, edge_weight):
        """LSTM-based aggregation of neighbors (batched implementation).

        Uses pack_padded_sequence for efficient batched LSTM processing,
        avoiding the slow Python for-loop over nodes.
        """
        from torch_geometric.utils import degree

        num_nodes = x.size(0)
        row, col = edge_index

        # Handle empty edge case
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, self.in_channels, device=x.device)

        # Get neighbor features
        neighbor_feats = x[col]  # [num_edges, in_channels]

        # Apply edge weights if provided
        if edge_weight is not None:
            neighbor_feats = neighbor_feats * edge_weight.view(-1, 1)

        # Sort edges by source node for grouping
        perm = torch.argsort(row)
        row_sorted = row[perm]
        neighbor_feats_sorted = neighbor_feats[perm]

        # Get counts per node (degree)
        deg = degree(row, num_nodes, dtype=torch.long)

        # Split into list of tensors (one per node)
        splits = deg.tolist()
        neighbor_groups = torch.split(neighbor_feats_sorted, splits)

        # Initialize output
        out = torch.zeros(num_nodes, self.in_channels, device=x.device)

        # Find nodes that have neighbors
        nodes_with_neighbors = (deg > 0).nonzero(as_tuple=True)[0]

        if len(nodes_with_neighbors) == 0:
            return out

        # Pad sequences and batch them
        max_neighbors = deg.max().item()
        batch_size = len(nodes_with_neighbors)

        # Create padded tensor for batched LSTM
        padded = torch.zeros(batch_size, max_neighbors, self.in_channels, device=x.device)
        lengths = []

        for i, node_idx in enumerate(nodes_with_neighbors):
            seq = neighbor_groups[node_idx.item()]
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq
            lengths.append(seq_len)

        # Pack for efficient LSTM processing (lengths must be on CPU)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device='cpu')
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths_tensor, batch_first=True, enforce_sorted=False
        )

        # Run LSTM on entire batch at once
        _, (h_n, _) = self.lstm(packed)  # h_n: [1, batch_size, hidden]

        # Place results back into output tensor
        out[nodes_with_neighbors] = h_n.squeeze(0)

        return out

    def message(self, x_j, edge_weight, aggr_mode=None):
        """Weight neighbor features by edge weight."""
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, aggr={self.aggr_type})'


class GraphSAGEModel(torch.nn.Module):
    """
    GraphSAGE (Graph SAmple and aggreGatE) model that handles dynamic graph sizes.
    Uses custom WeightedSAGEConv layers that support edge weights.
    Uses global mean pooling to produce a single output regardless of input graph size.

    Key difference from GCN: GraphSAGE concatenates aggregated neighbor features
    with the node's own features, then applies a linear transformation.

    Supports edge weights from SparseGraph for weighted neighbor aggregation.

    Args:
        in_channels: Number of input features per node (default 5 for SparseGraph)
        hidden_dim: Hidden dimension size
        num_layers: Number of WeightedSAGEConv layers
        dropout: Dropout rate applied after each GNN layer (default 0.0)
        aggr: Aggregation function: 'mean', 'max', or 'lstm' (default 'mean')
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 128, num_layers: int = 4,
                 dropout: float = 0.0, aggr: str = 'mean'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.aggr = aggr

        # Validate aggregation type
        if aggr not in ['mean', 'max', 'lstm']:
            raise ValueError(f"aggr must be 'mean', 'max', or 'lstm', got '{aggr}'")

        # WeightedSAGEConv layers (custom layer with edge weight support)
        self.convs = torch.nn.ModuleList()
        self.convs.append(WeightedSAGEConv(in_channels, hidden_dim, aggr_type=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(WeightedSAGEConv(hidden_dim, hidden_dim, aggr_type=aggr))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # Classification head (compresses to single output)
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # Convert edge_attr [N,1] to edge_weight [N] for weighted aggregation
        edge_weight = data.edge_attr.view(-1) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graphs (no fired detectors) - return default prediction
        num_graphs = int(data.num_graphs) if hasattr(data, 'num_graphs') else (int(batch.max().item()) + 1 if x.size(0) > 0 else 1)
        if x.size(0) == 0:
            return torch.full((num_graphs, 1), 0.5, device=data.x.device if hasattr(data.x, 'device') else 'cpu')

        # Apply WeightedSAGEConv layers with batch normalization, activation, and dropout
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.silu(x)
            x = self.dropout(x)

        # Global mean pooling: aggregate node features to graph-level (optimized)
        x_pooled = global_mean_pool(x, batch, size=num_graphs)

        # Classification layers
        x = self.fc1(x_pooled)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class GraphSAGE:
    """
    GraphSAGE wrapper class with model lifecycle management.
    Provides init, train, save, and load functionality with human-readable model tracking.

    Designed to work with SparseGraph outputs:
    - Node features: [X?, Z?, d_North, d_West, d_time] (5 features)
    - Edge weights: distance-based weights (used for weighted neighbor aggregation)

    Uses custom WeightedSAGEConv layers that incorporate edge weights into the
    GraphSAGE aggregation mechanism.

    Supports configurable:
    - dropout: Dropout rate after each GNN layer
    - aggr: Aggregation function ('mean', 'max', 'lstm')

    Attributes:
        nickname: Human-readable name for this model instance
        model: The underlying GraphSAGEModel
        device: Torch device (cuda/cpu)
        models_dir: Directory for saving/loading models
        _loaded_from: Path of the loaded model (None if freshly initialized)
    """

    def __init__(self,
                 nickname: str = "gsage_model",
                 in_channels: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.0,
                 aggr: str = 'mean',
                 device: torch.device = None,
                 base_path: Path = None,
                 seed: int = None):
        """
        Initialize a new GraphSAGE model.

        Args:
            nickname: Human-readable name for this model
            in_channels: Number of input features per node (default 5 for SparseGraph)
            hidden_dim: Hidden dimension size
            num_layers: Number of WeightedSAGEConv layers
            dropout: Dropout rate after each GNN layer (default 0.0)
            aggr: Aggregation function: 'mean', 'max', or 'lstm' (default 'mean')
            device: Torch device (defaults to CUDA if available)
            base_path: Base path for model storage (defaults to current directory)
            seed: Random seed for reproducibility (default: None, no seeding)
        """
        self.nickname = nickname
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_from = None

        # Set models directory based on base_path
        self.models_dir = (base_path or Path(".")) / "models" / "gsage" / "revised_training"

        # Store config for saving/loading
        self._config = {
            'in_channels': in_channels,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'aggr': aggr,
            'seed': seed
        }

        # Set random seeds for reproducibility before model initialization
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize model
        self.model = GraphSAGEModel(in_channels, hidden_dim, num_layers, dropout, aggr).to(self.device)

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print(f"GraphSAGE initialized: {self}")

    def train(self,
              graphs: list,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              verbose: bool = True) -> list:
        """
        Train the model on a list of PyG graphs.

        Args:
            graphs: List of PyG Data objects (from SparseGraph.batch_to_pyg)
            epochs: Number of training epochs
            batch_size: Number of graphs per batch
            lr: Learning rate
            verbose: Print training progress

        Returns:
            List of loss values per epoch
        """
        from torch_geometric.loader import DataLoader

        # Announce training start
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {self}")
            print(f"{'='*50}")
            print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
            print(f"Training samples: {len(graphs)}")

        loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        total_batches = len(loader) * epochs

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0

        pbar = tqdm(total=total_batches, desc="Training", disable=not verbose)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data)
                y = batch_data.y.float().view(-1, 1)

                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Batch metrics
                batch_loss = loss.item()
                batch_acc = ((pred > 0.5).float() == y).float().mean().item() * 100

                # Running averages (exponential smoothing)
                running_loss = batch_loss * 0.1 + running_loss * 0.9 if running_loss else batch_loss
                running_acc = batch_acc * 0.1 + running_acc * 0.9 if running_acc else batch_acc

                # Accumulate epoch stats
                epoch_loss += batch_loss
                epoch_correct += ((pred > 0.5).float() == y).sum().item()
                epoch_total += y.size(0)

                batch_count += 1
                pbar.update(1)

                # Update display every 10 batches
                if batch_count % 10 == 0:
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{running_loss:.4f}',
                        'acc': f'{running_acc:.1f}%'
                    })

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

        pbar.close()

        final_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose:
            print(f"\nTraining complete! Final - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.1f}%")

        return epoch_losses

    def save(self, name: str) -> Path:
        """
        Save the model with a human-readable timestamp.

        Args:
            name: Base name for the saved model

        Returns:
            Path to the saved model file
        """
        # Create timestamp in human-readable format
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = self.models_dir / filename

        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self._config,
            'nickname': self.nickname,
            'timestamp': timestamp
        }
        torch.save(save_dict, filepath)

        print(f"Model saved to: {filepath}")
        return filepath

    def load(self, filepath: str) -> 'GraphSAGE':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model file (relative to models/gsage or absolute)

        Returns:
            self (for chaining)
        """
        # Handle relative paths
        path = Path(filepath)
        if not path.exists():
            path = self.models_dir / filepath

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate model with saved config
        config = save_dict['config']
        self._config = config
        self.model = GraphSAGEModel(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.0),
            aggr=config.get('aggr', 'mean')
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(save_dict['state_dict'])
        self.model.eval()

        # Update tracking
        self._loaded_from = path.name
        if 'nickname' in save_dict:
            self.nickname = save_dict['nickname']

        print(f"Model loaded: {self}")
        return self

    def predict(self, data) -> torch.Tensor:
        """Run inference on data."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self.model(data)

    def __repr__(self) -> str:
        loaded_info = f", loaded_from='{self._loaded_from}'" if self._loaded_from else ""
        dropout_info = f", dropout={self._config.get('dropout', 0.0)}" if self._config.get('dropout', 0.0) > 0 else ""
        aggr_info = f", aggr='{self._config.get('aggr', 'mean')}'" if self._config.get('aggr', 'mean') != 'mean' else ""
        return (f"GraphSAGE(nickname='{self.nickname}', "
                f"in_channels={self._config['in_channels']}, "
                f"hidden_dim={self._config['hidden_dim']}, "
                f"num_layers={self._config['num_layers']}"
                f"{dropout_info}{aggr_info}{loaded_info})")


# ============================================================
# GIN MODEL (Graph Isomorphism Network with Edge features)
# ============================================================

class GINModel(torch.nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE) that handles dynamic graph sizes.
    Uses PyTorch Geometric's GINEConv which incorporates edge features into message passing.
    Uses global mean pooling to produce a single output regardless of input graph size.

    GIN is provably the most expressive GNN under the message-passing framework,
    achieving the same discriminative power as the Weisfeiler-Lehman graph isomorphism test.

    GINE formula:
    h_v^(k) = MLP^(k)((1 + eps^(k)) * h_v^(k-1) + sum_{u in N(v)} ReLU(h_u^(k-1) + e_uv))

    Key features:
    - Learnable epsilon (eps) parameter for self-loop weighting
    - MLP applied to aggregated features (not just linear transform)
    - Sum aggregation (critical for expressiveness)
    - Edge features incorporated via addition before ReLU
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 128, num_layers: int = 4,
                 train_eps: bool = True, edge_dim: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.train_eps = train_eps
        self.edge_dim = edge_dim

        # GINEConv layers with MLPs
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer: in_channels -> hidden_dim
        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINEConv(mlp1, train_eps=train_eps, edge_dim=edge_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=train_eps, edge_dim=edge_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Classification head (compresses to single output)
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Get edge attributes for GINEConv
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graphs (no fired detectors) - return default prediction
        num_graphs = int(data.num_graphs) if hasattr(data, 'num_graphs') else (int(batch.max().item()) + 1 if x.size(0) > 0 else 1)
        if x.size(0) == 0:
            return torch.full((num_graphs, 1), 0.5, device=data.x.device if hasattr(data.x, 'device') else 'cpu')

        # Apply GINEConv layers with batch normalization and activation
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.silu(x)

        # Global mean pooling: aggregate node features to graph-level (optimized)
        x_pooled = global_mean_pool(x, batch, size=num_graphs)

        # Classification layers
        x = self.fc1(x_pooled)
        x = F.silu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class GIN:
    """
    GIN (Graph Isomorphism Network) wrapper class with model lifecycle management.
    Provides init, train, save, and load functionality with human-readable model tracking.

    Uses GINEConv (edge-aware variant) to incorporate edge weights from SparseGraph.

    Designed to work with SparseGraph outputs:
    - Node features: [X?, Z?, d_North, d_West, d_time] (5 features)
    - Edge weights: distance-based weights (1-dim edge features)

    Attributes:
        nickname: Human-readable name for this model instance
        model: The underlying GINModel
        device: Torch device (cuda/cpu)
        models_dir: Directory for saving/loading models
        _loaded_from: Path of the loaded model (None if freshly initialized)
    """

    def __init__(self,
                 nickname: str = "gin_model",
                 in_channels: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 train_eps: bool = True,
                 edge_dim: int = 1,
                 device: torch.device = None,
                 base_path: Path = None,
                 seed: int = None):
        """
        Initialize a new GIN model.

        Args:
            nickname: Human-readable name for this model
            in_channels: Number of input features per node (default 5 for SparseGraph)
            hidden_dim: Hidden dimension size
            num_layers: Number of GINEConv layers
            train_eps: Whether epsilon is learnable (default True)
            edge_dim: Dimension of edge features (default 1 for SparseGraph)
            device: Torch device (defaults to CUDA if available)
            base_path: Base path for model storage (defaults to current directory)
            seed: Random seed for reproducibility (default: None, no seeding)
        """
        self.nickname = nickname
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_from = None

        # Set models directory based on base_path
        self.models_dir = (base_path or Path(".")) / "models" / "gin"

        # Store config for saving/loading
        self._config = {
            'in_channels': in_channels,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'train_eps': train_eps,
            'edge_dim': edge_dim,
            'seed': seed
        }

        # Set random seeds for reproducibility before model initialization
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize model
        self.model = GINModel(in_channels, hidden_dim, num_layers, train_eps, edge_dim).to(self.device)

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print(f"GIN initialized: {self}")

    def train(self,
              graphs: list,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              verbose: bool = True) -> list:
        """
        Train the model on a list of PyG graphs.

        Args:
            graphs: List of PyG Data objects (from SparseGraph.batch_to_pyg)
            epochs: Number of training epochs
            batch_size: Number of graphs per batch
            lr: Learning rate
            verbose: Print training progress

        Returns:
            List of loss values per epoch
        """
        from torch_geometric.loader import DataLoader

        # Announce training start
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {self}")
            print(f"{'='*50}")
            print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
            print(f"Training samples: {len(graphs)}")

        loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        total_batches = len(loader) * epochs

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0

        pbar = tqdm(total=total_batches, desc="Training", disable=not verbose)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data)
                y = batch_data.y.float().view(-1, 1)

                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Batch metrics
                batch_loss = loss.item()
                batch_acc = ((pred > 0.5).float() == y).float().mean().item() * 100

                # Running averages (exponential smoothing)
                running_loss = batch_loss * 0.1 + running_loss * 0.9 if running_loss else batch_loss
                running_acc = batch_acc * 0.1 + running_acc * 0.9 if running_acc else batch_acc

                # Accumulate epoch stats
                epoch_loss += batch_loss
                epoch_correct += ((pred > 0.5).float() == y).sum().item()
                epoch_total += y.size(0)

                batch_count += 1
                pbar.update(1)

                # Update display every 10 batches
                if batch_count % 10 == 0:
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{running_loss:.4f}',
                        'acc': f'{running_acc:.1f}%'
                    })

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

        pbar.close()

        final_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose:
            print(f"\nTraining complete! Final - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.1f}%")

        return epoch_losses

    def save(self, name: str) -> Path:
        """
        Save the model with a human-readable timestamp.

        Args:
            name: Base name for the saved model

        Returns:
            Path to the saved model file
        """
        # Create timestamp in human-readable format
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = self.models_dir / filename

        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self._config,
            'nickname': self.nickname,
            'timestamp': timestamp
        }
        torch.save(save_dict, filepath)

        print(f"Model saved to: {filepath}")
        return filepath

    def load(self, filepath: str) -> 'GIN':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model file (relative to models/gin or absolute)

        Returns:
            self (for chaining)
        """
        # Handle relative paths
        path = Path(filepath)
        if not path.exists():
            path = self.models_dir / filepath

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate model with saved config
        config = save_dict['config']
        self._config = config
        self.model = GINModel(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            train_eps=config.get('train_eps', True),
            edge_dim=config.get('edge_dim', 1)
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(save_dict['state_dict'])
        self.model.eval()

        # Update tracking
        self._loaded_from = path.name
        if 'nickname' in save_dict:
            self.nickname = save_dict['nickname']

        print(f"Model loaded: {self}")
        return self

    def predict(self, data) -> torch.Tensor:
        """Run inference on data."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self.model(data)

    def __repr__(self) -> str:
        loaded_info = f", loaded_from='{self._loaded_from}'" if self._loaded_from else ""
        return (f"GIN(nickname='{self.nickname}', "
                f"in_channels={self._config['in_channels']}, "
                f"hidden_dim={self._config['hidden_dim']}, "
                f"num_layers={self._config['num_layers']}, "
                f"train_eps={self._config['train_eps']}"
                f"{loaded_info})")


