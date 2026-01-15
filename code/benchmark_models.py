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
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, GINEConv
from torch_geometric.utils import add_self_loops
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json


# CUDA verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
# SIMPLE NN DECODER (Array-based, no graph structure)
# ============================================================

class SimpleNNModel(torch.nn.Module):
    """
    Simple feedforward neural network decoder for surface codes.

    Takes syndrome measurements as a flat array and predicts logical error.
    Does NOT use graph structure - serves as a baseline comparison for GNN decoders.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256, num_layers: int = 4):
        """
        Initialize SimpleNNModel.

        Args:
            in_channels: Number of input features (number of detectors in syndrome)
            hidden_dim: Base hidden dimension (will scale up in layers)
            num_layers: Not used directly, kept for API consistency
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers matching the reference implementation
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.SiLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, num_detectors]

        Returns:
            Predictions of shape [batch_size, 1]
        """
        return self.layers(x)


class SimpleNN:
    """
    SimpleNN wrapper class with model lifecycle management.
    Provides init, train, save, and load functionality with human-readable model tracking.

    This decoder takes raw syndrome arrays (not graphs) and predicts logical errors.
    Serves as a baseline comparison for GNN-based decoders.

    Attributes:
        nickname: Human-readable name for this model instance
        model: The underlying SimpleNNModel
        device: Torch device (cuda/cpu)
        models_dir: Directory for saving/loading models
        _loaded_from: Path of the loaded model (None if freshly initialized)
    """

    def __init__(self,
                 nickname: str = "simple_nn",
                 in_channels: int = None,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 device: torch.device = None,
                 base_path: Path = None,
                 seed: int = None):
        """
        Initialize a new SimpleNN model.

        Args:
            nickname: Human-readable name for this model
            in_channels: Number of input features (detectors). If None, must be set before training.
            hidden_dim: Hidden dimension size
            num_layers: Number of layers (for API consistency)
            device: Torch device (defaults to CUDA if available)
            base_path: Base path for model storage (defaults to current directory)
            seed: Random seed for reproducibility
        """
        self.nickname = nickname
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_from = None

        # Set models directory
        self.models_dir = (base_path or Path(".")) / "models" / "simple_nn"

        # Store config for saving/loading
        self._config = {
            'in_channels': in_channels,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seed': seed
        }

        # Set random seeds for reproducibility
        if seed is not None:
            import random
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize model (only if in_channels is known)
        if in_channels is not None:
            self.model = SimpleNNModel(in_channels, hidden_dim, num_layers).to(self.device)
        else:
            self.model = None

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print(f"SimpleNN initialized: {self}")

    def _get_num_detectors(self, d: int) -> int:
        """
        Calculate number of detectors for a given code distance.

        For rotated surface code with d rounds:
        - num_detectors = d * (d^2 - 1) / 2 * 2 + (d^2 - 1) / 2
        - Simplified: approximately d * (d^2 - 1)

        This is determined empirically from stim circuit generation.
        """
        # Generate a test circuit to get exact detector count
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=0.001,
        )
        return circuit.num_detectors

    def train(self,
              d: int,
              num_samples: int = 100000,
              epochs: int = 10,
              batch_size: int = 200,
              lr: float = 1e-3,
              p_values: list = None,
              p_weights: list = None,
              verbose: bool = True) -> list:
        """
        Train the model on surface code syndrome data.

        Args:
            d: Code distance
            num_samples: Number of training samples to generate
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            p_values: Physical error rates for training data
            p_weights: Weights for each error rate
            verbose: Print training progress

        Returns:
            List of loss values per epoch
        """
        # Default error rates
        if p_values is None:
            p_values = [0.005]
        if p_weights is None:
            p_weights = [1.0]

        # Get number of detectors and initialize model if needed
        num_detectors = self._get_num_detectors(d)

        if self.model is None or self._config['in_channels'] != num_detectors:
            self._config['in_channels'] = num_detectors
            self.model = SimpleNNModel(
                num_detectors,
                self._config['hidden_dim'],
                self._config['num_layers']
            ).to(self.device)
            if verbose:
                print(f"Model initialized for d={d} with {num_detectors} detectors")

        # Generate training data
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {self}")
            print(f"{'='*60}")
            print(f"Distance: d={d} | Detectors: {num_detectors}")
            print(f"Samples: {num_samples} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
            print(f"Error rates: {p_values} with weights {p_weights}")
            print(f"Generating training data...")

        sampler = SurfaceCodeSampler(p=p_values[0], device=self.device)
        detections, labels = sampler.sample(
            d=d,
            num_samples=num_samples,
            p_values=p_values,
            p_weights=p_weights
        )

        # Setup training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []
        num_batches = num_samples // batch_size

        if verbose:
            print(f"Starting training...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            running_avg_acc = 0.0

            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)

            for batch_idx in pbar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                X = detections[start_idx:end_idx]
                y = labels[start_idx:end_idx].unsqueeze(1)

                # Forward pass
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Compute accuracy
                acc = ((pred > 0.5).float() == y).float().mean().item()
                running_avg_acc = acc * 0.01 + running_avg_acc * 0.99 if running_avg_acc != 0 else acc

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update progress bar
                if batch_idx % 100 == 0:
                    pbar.set_postfix({
                        'acc': f'{running_avg_acc:.4f}',
                        'loss': f'{loss.item():.4f}'
                    })

            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}, Acc: {running_avg_acc:.4f}")

        if verbose:
            print(f"\nTraining complete!")

        return epoch_losses

    def save(self, name: str) -> Path:
        """
        Save the model with a human-readable timestamp.

        Args:
            name: Base name for the saved model

        Returns:
            Path to the saved model file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = self.models_dir / filename

        self.models_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self._config,
            'nickname': self.nickname,
            'timestamp': timestamp
        }
        torch.save(save_dict, filepath)

        print(f"Model saved to: {filepath}")
        return filepath

    def load(self, filepath: str) -> 'SimpleNN':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to saved model file (relative to models/simple_nn or absolute)

        Returns:
            self (for chaining)
        """
        path = Path(filepath)
        if not path.exists():
            path = self.models_dir / filepath

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        config = save_dict['config']
        self._config = config
        self.model = SimpleNNModel(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(self.device)

        self.model.load_state_dict(save_dict['state_dict'])
        self.model.eval()

        self._loaded_from = path.name
        if 'nickname' in save_dict:
            self.nickname = save_dict['nickname']

        print(f"Model loaded: {self}")
        return self

    def predict(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Run inference on syndrome data.

        Args:
            detections: Tensor of shape [batch_size, num_detectors]

        Returns:
            Predictions of shape [batch_size, 1]
        """
        self.model.eval()
        with torch.no_grad():
            detections = detections.to(self.device)
            return self.model(detections)

    def test_accuracy(self,
                      d: int,
                      num_samples: int = 10000,
                      p_values: list = None,
                      p_weights: list = None,
                      threshold: float = 0.5,
                      compare_mwpm: bool = True,
                      verbose: bool = True) -> dict:
        """
        Test model accuracy and compare with MWPM.

        Args:
            d: Code distance
            num_samples: Number of test samples
            p_values: Error rates to test
            p_weights: Weights for error rates
            threshold: Classification threshold
            compare_mwpm: Compare with MWPM decoder
            verbose: Print results

        Returns:
            Dictionary with accuracy metrics
        """
        if p_values is None:
            p_values = [0.005]
        if p_weights is None:
            p_weights = [1.0]

        # Generate test data
        sampler = SurfaceCodeSampler(p=p_values[0], device=self.device)
        detections, labels, p_indices = sampler.sample(
            d=d,
            num_samples=num_samples,
            p_values=p_values,
            p_weights=p_weights,
            return_p_labels=True
        )

        # Run predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(detections).squeeze()

        binary_preds = (predictions >= threshold).float()
        labels_cpu = labels.cpu()
        binary_preds_cpu = binary_preds.cpu()

        # Calculate metrics
        correct = (binary_preds_cpu == labels_cpu).sum().item()
        accuracy = correct / num_samples

        tp = ((binary_preds_cpu == 1) & (labels_cpu == 1)).sum().item()
        fp = ((binary_preds_cpu == 1) & (labels_cpu == 0)).sum().item()
        fn = ((binary_preds_cpu == 0) & (labels_cpu == 1)).sum().item()
        tn = ((binary_preds_cpu == 0) & (labels_cpu == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            'num_samples': num_samples,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'logical_error_rate': 1.0 - accuracy,
        }

        # MWPM comparison
        if compare_mwpm:
            mwpm_errors = 0
            for i, p in enumerate(p_values):
                mask = (p_indices.cpu() == i)
                if mask.sum() > 0:
                    circuit = sampler._get_circuit(d, p)
                    errors = count_logical_errors(circuit, int(mask.sum().item()))
                    mwpm_errors += errors

            mwpm_accuracy = 1.0 - (mwpm_errors / num_samples)
            results['mwpm_accuracy'] = mwpm_accuracy
            results['mwpm_logical_error_rate'] = mwpm_errors / num_samples
            results['improvement_over_mwpm'] = accuracy - mwpm_accuracy

        if verbose:
            print(f"\n{'='*60}")
            print(f"SimpleNN Accuracy Test Results")
            print(f"{'='*60}")
            print(f"Model: {self.nickname}")
            print(f"Distance: d={d} | Samples: {num_samples}")
            print(f"{'─'*60}")
            print(f"SimpleNN Accuracy: {accuracy:.4f} ({correct}/{num_samples})")
            print(f"SimpleNN LER: {results['logical_error_rate']:.4f}")
            print(f"{'─'*60}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            if compare_mwpm:
                print(f"{'─'*60}")
                print(f"MWPM Accuracy: {mwpm_accuracy:.4f}")
                print(f"MWPM LER: {results['mwpm_logical_error_rate']:.4f}")
                print(f"Improvement over MWPM: {results['improvement_over_mwpm']:+.4f}")
            print(f"{'='*60}")

        return results

    def __repr__(self) -> str:
        loaded_info = f", loaded_from='{self._loaded_from}'" if self._loaded_from else ""
        in_ch = self._config['in_channels'] if self._config['in_channels'] else "dynamic"
        return (f"SimpleNN(nickname='{self.nickname}', "
                f"in_channels={in_ch}, "
                f"hidden_dim={self._config['hidden_dim']}"
                f"{loaded_info})")


# ============================================================
# FLAT DATASET CACHE (for SimpleNN - array-based, not graphs)
# ============================================================

class FlatDatasetCache:
    """
    Cache manager for flat syndrome array datasets (for SimpleNN).

    Unlike DatasetCache which stores PyG graphs, this stores raw syndrome
    arrays and labels for use with simple feedforward neural networks.

    Features:
    - Generate datasets with configurable distance, error rates, and sample sizes
    - Save/load datasets to/from disk with metadata
    - Incrementally grow datasets with ensure_size()
    - List and manage cached datasets

    Attributes:
        detections (torch.Tensor): Syndrome arrays [n_samples, num_detectors]
        labels (torch.Tensor): Observable flip labels [n_samples]
        config (dict): Dataset configuration metadata
        datasets_dir (Path): Directory for cached datasets
    """

    def __init__(self, base_path: Path = None, device: torch.device = None):
        """
        Initialize the FlatDatasetCache.

        Args:
            base_path: Base path for dataset storage (defaults to current directory)
            device: Torch device for generation (defaults to CUDA if available)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets_dir = (base_path or Path(".")) / "datasets" / "flat"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Current dataset state
        self.detections = None
        self.labels = None
        self.config = {}

    def generate(
        self,
        d: int,
        n_samples: int,
        p_values: list,
        p_weights: list,
        verbose: bool = True
    ) -> 'FlatDatasetCache':
        """
        Generate a new dataset of flat syndrome arrays.

        Args:
            d: Code distance
            n_samples: Number of samples to generate
            p_values: List of physical error rates
            p_weights: Weights for error rate distribution (must sum to 1.0)
            verbose: Print progress information

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating flat dataset: d={d}, n_samples={n_samples:,}")
            print(f"Error rates: {p_values} (weights: {p_weights})")
            print(f"{'='*60}")

        # Initialize sampler
        sampler = SurfaceCodeSampler(p=p_values[0], device=self.device)

        # Get number of detectors
        circuit = sampler._get_circuit(d, p_values[0])
        num_detectors = circuit.num_detectors

        # Store configuration
        self.config = {
            'd': d,
            'n_samples': n_samples,
            'num_detectors': num_detectors,
            'p_values': p_values,
            'p_weights': p_weights,
            'generated_at': datetime.now().isoformat()
        }

        # Generate samples
        if verbose:
            print(f"\nSampling {n_samples:,} detection events...")
            print(f"  Detectors per sample: {num_detectors}")

        self.detections, self.labels = sampler.sample(
            d=d,
            num_samples=n_samples,
            p_values=p_values,
            p_weights=p_weights
        )

        if verbose:
            print(f"\nGenerated {n_samples:,} samples")
            print(f"  Detections shape: {self.detections.shape}")
            print(f"  Labels shape: {self.labels.shape}")
            print(f"  Positive rate: {self.labels.sum().item() / n_samples:.4f}")

        return self

    def save(self, name: str) -> Path:
        """
        Save the dataset to disk with metadata.

        Args:
            name: Name for the dataset (e.g., 'd5_baseline_flat')

        Returns:
            Path to the saved dataset file
        """
        if self.detections is None:
            raise ValueError("No data to save. Call generate() first.")

        # Save data as a dict
        data_path = self.datasets_dir / f"{name}.pt"
        torch.save({
            'detections': self.detections.cpu(),
            'labels': self.labels.cpu()
        }, data_path)

        # Save metadata
        meta_path = self.datasets_dir / f"{name}.json"
        with open(meta_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Flat dataset saved: {data_path}")
        print(f"  Metadata: {meta_path}")
        print(f"  Samples: {len(self.labels):,}")

        # Print file size
        import os
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")

        return data_path

    def load(self, name: str, verbose: bool = True) -> 'FlatDatasetCache':
        """
        Load a dataset from disk.

        Args:
            name: Name of the dataset to load
            verbose: Print progress information

        Returns:
            self (for chaining)
        """
        import os

        data_path = self.datasets_dir / f"{name}.pt"
        meta_path = self.datasets_dir / f"{name}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Flat dataset not found: {data_path}")

        # Get file size for progress display
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)

        if verbose:
            print(f"Loading flat dataset '{name}' ({file_size_mb:.1f} MB)...")

        # Load data
        data = torch.load(data_path, weights_only=False)
        self.detections = data['detections'].to(self.device)
        self.labels = data['labels'].to(self.device)

        # Convert to float if needed (for backwards compatibility with older datasets)
        if self.detections.dtype != torch.float32:
            self.detections = self.detections.float()
        if self.labels.dtype != torch.float32:
            self.labels = self.labels.float()

        # Load metadata if exists
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {'name': name, 'n_samples': len(self.labels)}

        if verbose:
            print(f"  Loaded {len(self.labels):,} samples")
            print(f"  Detections shape: {self.detections.shape}")
            if 'd' in self.config:
                print(f"  Distance: d={self.config['d']}")
            if 'p_values' in self.config:
                print(f"  Error rates: {self.config['p_values']}")

        return self

    def get_data(self, n: int = None, shuffle: bool = False) -> tuple:
        """
        Get detections and labels, optionally shuffled or limited.

        Args:
            n: Number of samples to return (None for all)
            shuffle: Whether to shuffle before returning

        Returns:
            tuple: (detections, labels)
        """
        if self.detections is None:
            raise ValueError("No data loaded. Call load() or generate() first.")

        detections = self.detections
        labels = self.labels

        if shuffle:
            perm = torch.randperm(len(labels), device=self.device)
            detections = detections[perm]
            labels = labels[perm]

        if n is not None:
            detections = detections[:n]
            labels = labels[:n]

        return detections, labels

    def __len__(self) -> int:
        return len(self.labels) if self.labels is not None else 0

    @staticmethod
    def list_datasets(base_path: Path = None) -> list:
        """
        List all cached flat datasets.

        Args:
            base_path: Base path to search for datasets

        Returns:
            List of dataset info dictionaries
        """
        import os

        datasets_dir = (base_path or Path(".")) / "datasets" / "flat"
        if not datasets_dir.exists():
            return []

        datasets = []
        for meta_file in datasets_dir.glob("*.json"):
            name = meta_file.stem
            data_file = datasets_dir / f"{name}.pt"

            if data_file.exists():
                with open(meta_file, 'r') as f:
                    config = json.load(f)

                size_mb = os.path.getsize(data_file) / (1024 * 1024)
                datasets.append({
                    'name': name,
                    'size_mb': size_mb,
                    **config
                })

        return sorted(datasets, key=lambda x: x['name'])
