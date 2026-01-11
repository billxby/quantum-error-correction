# ============================================================================
# COMPLETE GNN DECODER FOR QUANTUM ERROR CORRECTION
# Works with just: pip install torch torch-geometric stim pymatching numpy tqdm matplotlib
# ============================================================================

import stim
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
from tqdm.auto import trange
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# PART 1: SURFACE CODE CIRCUIT
# ============================================================================

def surface_code_circuit(p, d):
    """Generate surface code circuit with given error rate and distance"""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )

# ============================================================================
# PART 2: BUILD DETECTOR ADJACENCY MATRIX
# ============================================================================

def build_detector_adjacency_from_circuit(circuit):
    """
    Build adjacency graph where detectors are connected if they share data qubits.
    Uses detector coordinates to determine spatial connectivity.
    """
    num_detectors = circuit.num_detectors

    # Try to get detector coordinates
    try:
        detector_coords = circuit.get_detector_coordinates()

        # Build spatial adjacency graph
        edges = []
        coords_list = []

        for det_id in range(num_detectors):
            if det_id in detector_coords:
                coord = detector_coords[det_id]
                coords_list.append((det_id, coord))

        # Connect detectors that are spatially close
        for i, (id_i, coord_i) in enumerate(coords_list):
            for j, (id_j, coord_j) in enumerate(coords_list):
                if i >= j:
                    continue

                # Calculate spatial distance (use first 2 dimensions)
                dist = np.sqrt((coord_i[0] - coord_j[0])**2 +
                             (coord_i[1] - coord_j[1])**2)

                # Connect neighbors (distance < 2.0 means adjacent in grid)
                if dist < 2.0:
                    edges.append([id_i, id_j])
                    edges.append([id_j, id_i])  # Undirected

        if len(edges) == 0:
            print("Warning: No edges found via coordinates, using fallback")
            raise ValueError()

    except Exception as e:
        # Fallback: Create simple grid connectivity
        print(f"Using fallback adjacency (sequential connections)")
        edges = []
        for i in range(num_detectors - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

    # Convert to PyG edge_index format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, num_detectors

# ============================================================================
# PART 3: CONVERT DETECTOR MEASUREMENTS TO GRAPHS
# ============================================================================

def detections_to_graph(detections, edge_index, num_detectors):
    """
    Convert batch of detector measurements to PyG graph format.
    Each detector becomes a node with its measurement as feature.

    Args:
        detections: Tensor (batch_size, num_detectors) with values -1 or +1
        edge_index: Tensor (2, num_edges) defining graph connectivity
        num_detectors: Number of detector nodes

    Returns:
        List of PyG Data objects
    """
    batch_size = detections.shape[0]
    graphs = []

    for i in range(batch_size):
        # Node features: each detector's measurement
        node_features = detections[i, :num_detectors].unsqueeze(1).float()

        # Create graph
        graph = Data(x=node_features, edge_index=edge_index)
        graphs.append(graph)

    return graphs

# ============================================================================
# PART 4: SIMPLE GCN LAYER (NO COMPILED DEPENDENCIES)
# ============================================================================

class SimpleGCNConv(MessagePassing):
    """Graph Convolutional Layer - works without pyg-lib"""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # Add self-loops to adjacency
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Transform features
        x = self.lin(x)

        # Compute normalization (symmetric normalization)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        out = self.propagate(edge_index, x=x, norm=norm)

        return out + self.bias

    def message(self, x_j, norm):
        # Normalize messages from neighbors
        return norm.view(-1, 1) * x_j

# ============================================================================
# PART 5: GNN MODEL ARCHITECTURE
# ============================================================================

class SurfaceCodeGNN(torch.nn.Module):
    """Graph Neural Network for quantum error decoding"""

    def __init__(self, input_dim=1, hidden_dim=128, num_layers=4):
        super(SurfaceCodeGNN, self).__init__()

        # Graph convolutional layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SimpleGCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SimpleGCNConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Classification head
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers with batch normalization
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.silu(x)

        # Global mean pooling (aggregate node features to graph level)
        batch_size = int(batch.max().item()) + 1
        x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                x_pooled[i] = x[mask].mean(dim=0)

        # Classification layers
        x = self.fc1(x_pooled)
        x = F.silu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

# ============================================================================
# PART 6: BUILD DECODER
# ============================================================================

def build_decoder_gnn(circuit, device, hidden_dim=128, num_layers=4):
    """
    Build GNN decoder and extract graph structure from circuit.

    Returns:
        decoder: GNN model
        loss_fn: Loss function
        optimizer: Optimizer
        edge_index: Graph connectivity
        num_detectors: Number of detector nodes
    """
    # Build adjacency from circuit structure
    edge_index, num_detectors = build_detector_adjacency_from_circuit(circuit)
    edge_index = edge_index.to(device)

    print(f"Graph structure: {num_detectors} nodes, {edge_index.shape[1]} edges")
    print(f"Average degree: {edge_index.shape[1] / num_detectors:.2f}")

    # Create model
    decoder = SurfaceCodeGNN(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    # Loss and optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    return decoder, loss_fn, optimizer, edge_index, num_detectors

# ============================================================================
# PART 7: TRAINING LOOP
# ============================================================================

def train_loop_gnn(detections, flips, decoder, optimizer, loss_fn,
                   edge_index, num_detectors, train_size, bs=200, running_avg=0):
    """Train GNN decoder on detection/flip pairs (can be called multiple times)"""
    decoder.train()

    num_batches = train_size // bs

    with trange(num_batches) as pbar:
        for batch_idx in pbar:
            # Get batch
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, train_size)
            batch_detections = detections[start_idx:end_idx]
            batch_flips = flips[start_idx:end_idx]

            # Convert to graphs
            graphs = detections_to_graph(batch_detections, edge_index, num_detectors)
            batch_data = Batch.from_data_list(graphs).to(device)

            # Forward pass
            pred = decoder(batch_data)
            y = batch_flips.unsqueeze(1).float()

            # Compute loss and accuracy
            loss = loss_fn(pred, y)
            acc = torch.mean(((pred > 0.5) == y).float())

            # Update running average
            running_avg = acc * 0.01 + running_avg * 0.99 if running_avg != 0 else acc

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            if batch_idx % 100 == 0:
                pbar.set_description(f'Acc: {running_avg:.4f} Loss: {loss:.4f}')

    return running_avg

# ============================================================================
# PART 8: EVALUATION
# ============================================================================

def evaluate_model_gnn(decoder, circuit, edge_index, num_detectors, test_size=10000):
    """Evaluate GNN decoder accuracy on fresh test data"""
    sampler = circuit.compile_detector_sampler()
    detections, flips = sampler.sample(shots=test_size, separate_observables=True)
    detections = torch.Tensor(detections.astype(int) * 2 - 1).to(device)
    flips = torch.Tensor(flips.astype(int).flatten()).to(device)

    decoder.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 500
        all_preds = []

        for i in range(0, test_size, batch_size):
            batch_detections = detections[i:i+batch_size]
            graphs = detections_to_graph(batch_detections, edge_index, num_detectors)
            batch_data = Batch.from_data_list(graphs).to(device)

            pred = decoder(batch_data)
            all_preds.append((pred > 0.5).flatten())

        all_preds = torch.cat(all_preds)
        accuracy = torch.mean((all_preds == flips).float()).item()

    return accuracy

# ============================================================================
# PART 9: LOGICAL ERROR RATE CALCULATION
# ============================================================================

def ler_nn_gnn(decoder, p, d, edge_index, num_detectors, num_shots=100000):
    """Compute logical error rate for GNN decoder"""
    circuit = surface_code_circuit(p, d)
    sampler = circuit.compile_detector_sampler()
    detections, flips = sampler.sample(num_shots, separate_observables=True)
    detections = torch.Tensor(detections.astype(int) * 2 - 1).to(device)

    decoder.eval()
    with torch.no_grad():
        # Process in chunks
        chunk_size = 1000
        all_preds = []

        for i in range(0, num_shots, chunk_size):
            chunk_detections = detections[i:i+chunk_size]
            graphs = detections_to_graph(chunk_detections, edge_index, num_detectors)
            batch_data = Batch.from_data_list(graphs).to(device)
            pred = decoder(batch_data)
            all_preds.append((pred > 0.5).flatten().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        errors = (all_preds != flips.flatten())

    return errors.mean()

# ============================================================================
# PART 10: BASELINE MWPM FOR COMPARISON
# ============================================================================

def ler_mwpm(p, d):
    """Compute logical error rate using MWPM decoder"""
    import pymatching

    num_shots = 100000
    circuit = surface_code_circuit(p, d)

    # Sample circuit
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    # Build MWPM decoder
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Decode
    predictions = matcher.decode_batch(detection_events)

    # Count errors
    num_errors = 0
    for shot in range(num_shots):
        if not np.array_equal(observable_flips[shot], predictions[shot]):
            num_errors += 1

    return num_errors / num_shots

# ============================================================================
# PART 11: PROGRESSIVE TRAINING WITH CHUNKING
# ============================================================================

def train_until_beat_mwpm(p, d, max_train_size=10**8, chunk_size=10**7):
    """Train with progressively larger datasets until beating MWPM"""
    import gc

    print(f"\n{'='*60}")
    print(f"Progressive Training for d={d}, p={p}")
    print(f"{'='*60}")

    # Get MWPM baseline
    mwpm_accuracy = 1 - ler_mwpm(p, d)
    print(f"MWPM accuracy to beat: {mwpm_accuracy:.6f}")

    # Build circuit and get structure
    circuit = surface_code_circuit(p, d)
    edge_index, num_detectors = build_detector_adjacency_from_circuit(circuit)
    edge_index = edge_index.to(device)

    print(f"Graph structure: {num_detectors} nodes, {edge_index.shape[1]} edges")

    # Try progressively larger training sizes
    train_size = 100
    beat_mwpm = False
    decoder, loss_fn, optimizer = None, None, None

    while train_size <= max_train_size and not beat_mwpm:
        print(f"\nTrying train_size = {train_size:,}")

        # Build fresh model for this train_size
        decoder = SurfaceCodeGNN(
            input_dim=1,
            hidden_dim=128,
            num_layers=4
        ).to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

        # Train in chunks
        samples_trained = 0
        num_chunks = max(1, train_size // chunk_size)
        samples_per_chunk = min(train_size, chunk_size)
        running_avg = 0

        for chunk_idx in range(num_chunks):
            current_chunk_size = min(samples_per_chunk, train_size - samples_trained)
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: {current_chunk_size:,} samples")

            # Generate chunk data
            sampler = circuit.compile_detector_sampler()
            detections, flips = sampler.sample(shots=current_chunk_size, separate_observables=True)
            detections = torch.Tensor(detections.astype(int) * 2 - 1).to(device)
            flips = torch.Tensor(flips.astype(int).flatten()).to(device)

            # Train on this chunk (model retains learning)
            running_avg = train_loop_gnn(
                detections, flips, decoder, optimizer, loss_fn,
                edge_index, num_detectors, current_chunk_size,
                bs=min(256, current_chunk_size), running_avg=running_avg
            )

            samples_trained += current_chunk_size

            # Clear chunk data from memory
            del detections, flips, sampler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate after all chunks
        gnn_accuracy = evaluate_model_gnn(decoder, circuit, edge_index, num_detectors)
        print(f"GNN accuracy: {gnn_accuracy:.6f} vs MWPM: {mwpm_accuracy:.6f}")

        if gnn_accuracy > mwpm_accuracy:
            print(f"SUCCESS: GNN beats MWPM at train_size = {train_size:,}")
            beat_mwpm = True
        else:
            train_size *= 10
            # Clear model before trying next train_size
            del decoder, loss_fn, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not beat_mwpm:
        print(f"Did not beat MWPM within max train_size")
        return None, None, None, None

    return decoder, edge_index, num_detectors, train_size

# ============================================================================
# PART 12: MODEL SAVING AND LOADING
# ============================================================================

def save_gnn_model(decoder, p, d, train_size, filename=None):
    """Save trained GNN model to disk"""
    if filename is None:
        filename = f"gnn_decoder_d{d}_p{p}_n{train_size}.pt"
    torch.save(decoder.state_dict(), filename)
    print(f"Model saved to {filename}")
    return filename

def load_gnn_model(circuit, filename, device):
    """Load GNN model from disk"""
    edge_index, num_detectors = build_detector_adjacency_from_circuit(circuit)
    edge_index = edge_index.to(device)

    decoder = SurfaceCodeGNN(
        input_dim=1,
        hidden_dim=128,
        num_layers=4
    ).to(device)

    decoder.load_state_dict(torch.load(filename, weights_only=True))
    decoder.eval()

    return decoder, edge_index, num_detectors

# ============================================================================
# PART 13: EXTRAPOLATION TESTING
# ============================================================================

def test_extrapolation(decoder, edge_index, num_detectors, d,
                       train_p, test_p_values, num_shots=100000):
    """Test how well model trained at train_p generalizes to other error rates"""
    print(f"\n{'='*60}")
    print(f"Testing extrapolation for d={d} (trained at p={train_p})")
    print(f"{'='*60}")

    results = {'p': [], 'gnn_ler': [], 'gnn_yerr': []}

    for p in test_p_values:
        gnn_ler = ler_nn_gnn(decoder, p, d, edge_index, num_detectors, num_shots)
        yerr = np.sqrt(gnn_ler * (1 - gnn_ler) / num_shots)

        results['p'].append(p)
        results['gnn_ler'].append(gnn_ler)
        results['gnn_yerr'].append(yerr)

        print(f"  p = {p:.4f}: LER = {gnn_ler:.6f} ± {yerr:.6f}")

    return results

# ============================================================================
# PART 14: COMPLETE TRAINING EXAMPLE
# ============================================================================

def train_and_evaluate_gnn(p=0.005, d=5, train_size=100000):
    """Complete pipeline: train and evaluate GNN decoder"""

    print(f"\n{'='*60}")
    print(f"Training GNN Decoder for d={d}, p={p}")
    print(f"{'='*60}\n")

    # Build circuit
    circuit = surface_code_circuit(p, d)

    # Build GNN decoder
    decoder, loss_fn, optimizer, edge_index, num_detectors = build_decoder_gnn(
        circuit, device, hidden_dim=128, num_layers=4
    )

    # Generate training data
    print(f"\nGenerating {train_size:,} training samples...")
    sampler = circuit.compile_detector_sampler()
    detections, flips = sampler.sample(shots=train_size, separate_observables=True)
    detections = torch.Tensor(detections.astype(int) * 2 - 1).to(device)
    flips = torch.Tensor(flips.astype(int).flatten()).to(device)

    # Train
    print("\nTraining GNN...")
    train_loop_gnn(
        detections, flips, decoder, optimizer, loss_fn,
        edge_index, num_detectors, train_size, bs=256
    )

    # Evaluate
    print("\n\nEvaluating...")
    gnn_accuracy = evaluate_model_gnn(decoder, circuit, edge_index, num_detectors, test_size=10000)

    # Compare with MWPM
    print("\nComputing MWPM baseline...")
    mwpm_ler = ler_mwpm(p, d)
    mwpm_accuracy = 1 - mwpm_ler

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"GNN Accuracy:  {gnn_accuracy:.6f}")
    print(f"MWPM Accuracy: {mwpm_accuracy:.6f}")
    print(f"Improvement:   {(gnn_accuracy - mwpm_accuracy)*100:+.3f}%")
    print(f"{'='*60}\n")

    return decoder, edge_index, num_detectors, gnn_accuracy, mwpm_accuracy

# ============================================================================
# PART 12: RUN IT!
# ============================================================================

if __name__ == "__main__":
    import os

    # Configuration
    train_p = 0.005
    max_train_size = 10**8
    chunk_size = 10**7

    results = {}
    trained_models = {}

    # Progressive training for each distance
    for d in [3, 5, 7]:    
        model_path = f"gnn_decoder_d{d}_p{train_p}.pt"

        # Check if model already exists
        if os.path.exists(model_path):
            print(f"\nLoading existing model for d={d} from {model_path}")
            circuit = surface_code_circuit(train_p, d)
            decoder, edge_index, num_detectors = load_gnn_model(circuit, model_path, device)
            print(f"Model loaded successfully")

            # Get the train_size from filename if possible, otherwise set to None
            train_size = None
        else:
            print(f"\nTraining new model for d={d}...")
            decoder, edge_index, num_detectors, train_size = train_until_beat_mwpm(
                p=train_p, d=d, max_train_size=max_train_size, chunk_size=chunk_size
            )

            if decoder is not None:
                # Save the trained model
                save_gnn_model(decoder, train_p, d, train_size)
            else:
                print(f"Training failed for d={d}")
                continue

        trained_models[d] = (decoder, edge_index, num_detectors)
        results[d] = train_size

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING RESULTS SUMMARY")
    print(f"{'='*60}")
    for d, train_size in results.items():
        if train_size:
            print(f"Distance {d}: Succeeded with train_size = {train_size:,}")
        else:
            print(f"Distance {d}: Loaded from saved model")

    # Optional: Test extrapolation across physical error rates
    print(f"\n{'='*60}")
    print("TESTING EXTRAPOLATION (optional - uncomment to run)")
    print(f"{'='*60}")

    # Uncomment to test extrapolation:
    # test_p_values = np.linspace(0.001, 0.01, 10)
    # for d in [3, 5, 7]:
    #     if d in trained_models:
    #         decoder, edge_index, num_detectors = trained_models[d]
    #         extrapolation_results = test_extrapolation(
    #             decoder, edge_index, num_detectors, d,
    #             train_p=train_p, test_p_values=test_p_values,
    #             num_shots=100000
    #         )