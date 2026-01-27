"""
Script to build the complete DeepSets vs GNN comparison notebook
Properly handles MODEL_NAMES-based indexing for extrapolation models
"""

import json
from pathlib import Path

# Read the reference notebook
ref_path = Path(r"k:\Coding\projects\quantum-error-correction\code\results\nn_vs_gnn_comparison\comparison.ipynb")
with open(ref_path, 'r', encoding='utf-8') as f:
    ref_nb = json.load(f)

# Create new notebook based on reference
new_nb = {
    "cells": [],
    "metadata": ref_nb["metadata"],
    "nbformat": ref_nb["nbformat"],
    "nbformat_minor": ref_nb["nbformat_minor"]
}

def replace_simplenn_with_deepsets(text):
    """Replace SimpleNN references with DeepSets"""
    replacements = [
        ("SimpleNN", "DeepSets"),
        ("simplenn", "deepsets"),
        ("simple_nn", "deep_sets"),
        ("SNN", "DS"),
        ("NN (SimpleNN)", "DeepSets"),
        ("NN:", "DeepSets:"),
        ("'NN'", "'DeepSets'"),
        ('"NN"', '"DeepSets"'),
        ("nn_vs_gnn_comparison", "deepsets_vs_gnn_comparison"),
        ("from benchmark_models import SimpleNN", "from benchmark_models import DeepSets"),
        ('/ "nn" /', '/ "deepsets" /'),
        ("/ 'nn' /", "/ 'deepsets' /"),
        ('\"nn\"', '\"deepsets\"'),
        ("nn_params", "deepsets_params"),
        ("nn_size_mb", "deepsets_size_mb"),
        ("simplenn_params", "deepsets_params"),
        ("simplenn_size_mb", "deepsets_size_mb"),
        ("'simplenn'", "'deepsets'"),
        ('"simplenn"', '"deepsets"'),
        ("simplenn_models", "deepsets_models"),
        ("simplenn_info", "deepsets_info"),
        ("simplenn_training_results", "deepsets_training_results"),
        ("simplenn_results_path", "deepsets_results_path"),
        ("simplenn_all_accs", "deepsets_all_accs"),
        ("simplenn_accs", "deepsets_accs"),
        ("simplenn_preds", "deepsets_preds"),
        ("simplenn_throughputs", "deepsets_throughputs"),
        ("simplenn_acc", "deepsets_acc"),
        ("simplenn_latency", "deepsets_latency"),
        ("NEURAL NETWORK vs GRAPH", "DEEPSETS vs GRAPH"),
        ("Neural Network vs Graph Neural Network", "DeepSets vs Graph Neural Network"),
        ("load_simplenn_model", "load_deepsets_model"),
        ("NN better", "DeepSets better"),
        ("vs SimpleNN", "vs DeepSets"),
        ("GNN - NN", "GNN - DeepSets"),
        ("NN is", "DeepSets is"),
        ("label='NN", "label='DeepSets"),
        ("'nn_", "'deepsets_"),
    ]

    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result

def transform_cell(cell):
    """Transform a cell from SimpleNN to DeepSets"""
    new_cell = dict(cell)

    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            new_line = replace_simplenn_with_deepsets(line)
            new_source.append(new_line)
        new_cell["source"] = new_source
    elif cell["cell_type"] == "markdown":
        new_source = []
        for line in cell["source"]:
            new_line = replace_simplenn_with_deepsets(line)
            new_source.append(new_line)
        new_cell["source"] = new_source

    return new_cell

# Custom cells that need complete replacement
HELPER_FUNCTIONS_CELL = '''def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model_path: Path) -> float:
    """Get model file size in MB."""
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


# Model names for extrapolation experiments (16 total)
MODEL_NAMES = [
    "a1_d3_00", "a2_d3_10", "a3_d3_20", "a4_d3_40", "a5_d3_50",
    "b1_d5heavy", "b2_d5more", "b3_balanced", "b4_d7more", "b5_d7heavy",
    "c1_only_d3", "c2_only_d5", "c3_only_d7", "c4_no_d7",
    "equal_333333"
]


def load_deepsets_model(model_name: str, base_path: Path) -> Tuple[DeepSets, Dict]:
    """Load DeepSets model for a given model name."""
    model_path = base_path / "deepsets" / "extrapolation" / "models" / "revised_training" / f"{model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"DeepSets model not found: {model_name} at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = DeepSets(
        nickname=model_name,
        phi_hidden=config.get('phi_hidden', [64, 128, 256]),
        rho_hidden=config.get('rho_hidden', [256, 128, 64]),
        pool=config.get('pool', 'mean'),
        dropout=config.get('dropout', 0.1),
        device=device,
        base_path=base_path
    )

    if 'state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['state_dict'])

    model.model.eval()
    model._config = config

    info = {
        'name': model_name,
        'num_parameters': count_parameters(model.model),
        'model_size_mb': get_model_size_mb(model_path),
        'model_path': str(model_path),
        'config': config
    }

    return model, info


def load_graphsage_model(model_name: str, base_path: Path) -> Tuple[GraphSAGE, Dict]:
    """Load GraphSAGE model for a given model name."""
    model_path = base_path / "gSAGE" / "extrapolation" / "models" / "revised_training" / f"{model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"GraphSAGE model not found: {model_name} at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {
        'in_channels': 5, 'hidden_dim': 128, 'num_layers': 5, 'dropout': 0.0, 'aggr': 'max'
    })

    model = GraphSAGE(
        nickname=model_name,
        in_channels=config.get('in_channels', 5),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 5),
        dropout=config.get('dropout', 0.0),
        aggr=config.get('aggr', 'max'),
        device=device,
        base_path=base_path
    )

    if 'state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['state_dict'])

    model.model.eval()

    info = {
        'name': model_name,
        'num_parameters': count_parameters(model.model),
        'model_size_mb': get_model_size_mb(model_path),
        'model_path': str(model_path),
        'config': config
    }

    return model, info


print("Helper functions defined.")'''

LOAD_MODELS_CELL = '''# Load all models
deepsets_models = {}
graphsage_models = {}
deepsets_info = {}
graphsage_info = {}

print("Loading models...")
for model_name in MODEL_NAMES:
    # Load DeepSets
    try:
        model, info = load_deepsets_model(model_name, BASE_PATH)
        deepsets_models[model_name] = model
        deepsets_info[model_name] = info
        print(f"  ✓ DeepSets {model_name}: {info['num_parameters']:,} params, {info['model_size_mb']:.2f} MB")
    except Exception as e:
        print(f"  ✗ DeepSets {model_name}: Failed to load - {e}")

    # Load GraphSAGE
    try:
        model, info = load_graphsage_model(model_name, BASE_PATH)
        graphsage_models[model_name] = model
        graphsage_info[model_name] = info
        print(f"  ✓ GraphSAGE {model_name}: {info['num_parameters']:,} params, {info['model_size_mb']:.2f} MB")
    except Exception as e:
        print(f"  ✗ GraphSAGE {model_name}: Failed to load - {e}")

print(f"\\nLoaded {len(deepsets_models)} DeepSets models and {len(graphsage_models)} GraphSAGE models")'''

PARAM_COMPARISON_CELL = '''# Create parameter comparison table (for each model)
param_comparison = []
for model_name in MODEL_NAMES:
    if model_name in deepsets_info and model_name in graphsage_info:
        param_comparison.append({
            'model_name': model_name,
            'deepsets_params': deepsets_info[model_name]['num_parameters'],
            'graphsage_params': graphsage_info[model_name]['num_parameters'],
            'deepsets_size_mb': deepsets_info[model_name]['model_size_mb'],
            'graphsage_size_mb': graphsage_info[model_name]['model_size_mb'],
            'param_ratio': graphsage_info[model_name]['num_parameters'] / deepsets_info[model_name]['num_parameters'] if deepsets_info[model_name]['num_parameters'] > 0 else 0
        })

param_df = pd.DataFrame(param_comparison)
print("Parameter Budget Comparison:")
print(param_df.to_string(index=False))

if len(param_comparison) > 0:
    avg_ratio = param_df['param_ratio'].mean()
    print(f"\\nAverage parameter ratio (GraphSAGE/DeepSets): {avg_ratio:.2f}x")
    print(f"DeepSets avg params: {param_df['deepsets_params'].mean():,.0f}")
    print(f"GraphSAGE avg params: {param_df['graphsage_params'].mean():,.0f}")

    if avg_ratio > 1.5 or avg_ratio < 0.67:
        print(f"\\n⚠️  WARNING: Significant parameter budget difference ({avg_ratio:.2f}x)")
        print("   This should be documented in the comparison.")
    else:
        print(f"\\n✓ Parameter budgets are reasonably matched ({avg_ratio:.2f}x)")'''

GENERATE_TEST_DATA_CELL = '''# Generate shared test datasets for fair comparison
# Same random seed ensures identical data for both models
shared_test_data = {}
graph_builder = SparseGraph(k_neighbors=6, device=torch.device('cpu'))
sampler = SurfaceCodeSampler(p=0.005, device=torch.device('cpu'))

print("Generating shared test datasets...")
for d in DISTANCES:
    # Generate detections with fixed seed for reproducibility
    torch.manual_seed(SEED + d)
    np.random.seed(SEED + d)

    detections, labels = sampler.sample(
        d=d,
        num_samples=TEST_SAMPLES_PER_DISTANCE,
        p_values=[0.001, 0.003, 0.005, 0.007],
        p_weights=[0.25, 0.25, 0.25, 0.25]
    )

    graphs = graph_builder.batch_to_pyg(detections, labels)

    shared_test_data[d] = {
        'detections': detections.cpu(),
        'labels': labels.cpu(),
        'graphs': graphs,
        'num_samples': len(labels)
    }

    print(f"  ✓ d={d}: {len(labels):,} samples")

print(f"\\nGenerated shared test data for {len(shared_test_data)} distances")'''

INFERENCE_BENCHMARK_CELL = '''def benchmark_inference_speed(
    model,
    data,
    batch_sizes: List[int],
    num_warmup: int = 10,
    num_runs: int = 5,
    is_graph_model: bool = False,
    distance: int = None
) -> Dict:
    """Benchmark inference speed with multiple runs for statistical robustness."""
    model.model.eval()
    model.model.to(device)

    results = {}

    for batch_size in batch_sizes:
        if is_graph_model:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)
            num_samples = len(data)
        else:
            num_samples = len(data)

        # Warmup
        with torch.no_grad():
            warmup_count = 0
            if is_graph_model:
                for batch in loader:
                    if warmup_count >= num_warmup:
                        break
                    batch = batch.to(device)
                    _ = model.model(batch)
                    warmup_count += 1
            else:
                for i in range(0, min(num_warmup * batch_size, num_samples), batch_size):
                    batch = data[i:i+batch_size].to(device)
                    _ = model.predict(batch, distance)
                    warmup_count += 1

        if device.type == 'cuda':
            torch.cuda.synchronize()

        run_times = []
        for run in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                if is_graph_model:
                    for batch in loader:
                        batch = batch.to(device)
                        _ = model.model(batch)
                else:
                    for i in range(0, num_samples, batch_size):
                        batch = data[i:i+batch_size].to(device)
                        _ = model.predict(batch, distance)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            run_times.append(elapsed)

        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        throughput = num_samples / mean_time
        latency_per_sample = (mean_time / num_samples) * 1e6

        results[batch_size] = {
            'mean_time_seconds': mean_time,
            'std_time_seconds': std_time,
            'throughput_samples_per_sec': throughput,
            'latency_per_sample_us': latency_per_sample,
            'runs': run_times
        }

    return results


# Benchmark all models across all distances
print("Benchmarking inference speed...")
inference_benchmarks = {}

for model_name in MODEL_NAMES:
    if model_name not in deepsets_models or model_name not in graphsage_models:
        continue

    inference_benchmarks[model_name] = {}

    for d in DISTANCES:
        if d not in shared_test_data:
            continue

        if model_name not in inference_benchmarks:
            inference_benchmarks[model_name] = {}
        if d not in inference_benchmarks[model_name]:
            inference_benchmarks[model_name][d] = {}

        # Benchmark DeepSets
        print(f"  Benchmarking DeepSets {model_name} d={d}...")
        detections = shared_test_data[d]['detections']
        results = benchmark_inference_speed(
            deepsets_models[model_name],
            detections,
            BATCH_SIZES,
            NUM_WARMUP_RUNS,
            NUM_BENCHMARK_RUNS,
            is_graph_model=False,
            distance=d
        )
        inference_benchmarks[model_name][d]['deepsets'] = results

        # Benchmark GraphSAGE
        print(f"  Benchmarking GraphSAGE {model_name} d={d}...")
        graphs = shared_test_data[d]['graphs']
        results = benchmark_inference_speed(
            graphsage_models[model_name],
            graphs,
            BATCH_SIZES,
            NUM_WARMUP_RUNS,
            NUM_BENCHMARK_RUNS,
            is_graph_model=True
        )
        inference_benchmarks[model_name][d]['graphsage'] = results

print("\\n✓ Inference benchmarking complete")'''

ACCURACY_EVAL_CELL = '''def evaluate_accuracy_metrics(
    model,
    data,
    labels,
    num_runs: int = 4,
    threshold: float = 0.5,
    is_graph_model: bool = False,
    batch_size: int = 256,
    distance: int = None
) -> Dict:
    """Evaluate accuracy metrics with multiple runs for statistical robustness."""
    model.model.eval()
    model.model.to(device)

    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_predictions = []

    for run in range(num_runs):
        with torch.no_grad():
            predictions = []

            if is_graph_model:
                loader = DataLoader(data, batch_size=batch_size, shuffle=False)
                for batch in loader:
                    batch = batch.to(device)
                    pred = model.model(batch)
                    predictions.append(pred.cpu())
            else:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size].to(device)
                    pred = model.predict(batch, distance)
                    predictions.append(pred.cpu())

            predictions = torch.cat(predictions, dim=0).squeeze()
            binary_preds = (predictions >= threshold).float()
            labels_tensor = labels.float()

            correct = (binary_preds == labels_tensor).sum().item()
            accuracy = correct / len(labels_tensor)

            tp = ((binary_preds == 1) & (labels_tensor == 1)).sum().item()
            fp = ((binary_preds == 1) & (labels_tensor == 0)).sum().item()
            fn = ((binary_preds == 0) & (labels_tensor == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            if run == 0:
                all_predictions = binary_preds.numpy()

    return {
        'accuracy_mean': np.mean(all_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'accuracy_ci_95': 1.96 * np.std(all_accuracies) / np.sqrt(num_runs),
        'precision_mean': np.mean(all_precisions),
        'precision_std': np.std(all_precisions),
        'recall_mean': np.mean(all_recalls),
        'recall_std': np.std(all_recalls),
        'f1_mean': np.mean(all_f1s),
        'f1_std': np.std(all_f1s),
        'f1_ci_95': 1.96 * np.std(all_f1s) / np.sqrt(num_runs),
        'all_accuracies': all_accuracies,
        'all_f1s': all_f1s,
        'predictions': all_predictions
    }


# Evaluate all models on all distances
print("Evaluating accuracy metrics...")
accuracy_results = {}

for model_name in tqdm(MODEL_NAMES, desc="Models"):
    if model_name not in deepsets_models or model_name not in graphsage_models:
        continue

    accuracy_results[model_name] = {}

    for d in DISTANCES:
        if d not in shared_test_data:
            continue

        accuracy_results[model_name][d] = {}
        labels = shared_test_data[d]['labels']

        # Evaluate DeepSets
        detections = shared_test_data[d]['detections']
        results = evaluate_accuracy_metrics(
            deepsets_models[model_name],
            detections,
            labels,
            NUM_ACCURACY_RUNS,
            is_graph_model=False,
            distance=d
        )
        accuracy_results[model_name][d]['deepsets'] = results

        # Evaluate GraphSAGE
        graphs = shared_test_data[d]['graphs']
        results = evaluate_accuracy_metrics(
            graphsage_models[model_name],
            graphs,
            labels,
            NUM_ACCURACY_RUNS,
            is_graph_model=True
        )
        accuracy_results[model_name][d]['graphsage'] = results

print("\\n✓ Accuracy evaluation complete")'''

STATISTICAL_TESTS_CELL = '''def wilcoxon_test(deepsets_accs: List[float], graphsage_accs: List[float]) -> Dict:
    """Perform Wilcoxon signed-rank test."""
    if len(deepsets_accs) < 2 or len(graphsage_accs) < 2:
        return {'test': 'Wilcoxon', 'p_value': 1.0, 'significant': False, 'note': 'Insufficient samples'}

    statistic, p_value = wilcoxon(deepsets_accs, graphsage_accs, alternative='two-sided')
    n = len(deepsets_accs)
    z = stats.norm.ppf(p_value / 2) if p_value > 0 else 0
    r = abs(z) / np.sqrt(n)

    return {
        'test': 'Wilcoxon signed-rank',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < ALPHA,
        'effect_size_r': float(r),
        'interpretation': 'large' if r > 0.5 else ('medium' if r > 0.3 else 'small')
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


# Perform statistical tests for each model
print("Performing statistical tests...")
statistical_tests = {}

for model_name in MODEL_NAMES:
    if model_name not in accuracy_results:
        continue

    deepsets_all_accs = []
    graphsage_all_accs = []

    for d in DISTANCES:
        if d not in accuracy_results[model_name]:
            continue

        if 'deepsets' in accuracy_results[model_name][d]:
            deepsets_all_accs.extend(accuracy_results[model_name][d]['deepsets']['all_accuracies'])
        if 'graphsage' in accuracy_results[model_name][d]:
            graphsage_all_accs.extend(accuracy_results[model_name][d]['graphsage']['all_accuracies'])

    if len(deepsets_all_accs) >= 2 and len(graphsage_all_accs) >= 2:
        # Ensure equal length for paired test
        min_len = min(len(deepsets_all_accs), len(graphsage_all_accs))
        deepsets_paired = deepsets_all_accs[:min_len]
        graphsage_paired = graphsage_all_accs[:min_len]

        wilcoxon_result = wilcoxon_test(deepsets_paired, graphsage_paired)
        cohens_d_val = cohens_d(deepsets_paired, graphsage_paired)

        statistical_tests[model_name] = {
            'wilcoxon': wilcoxon_result,
            'cohens_d': cohens_d_val,
            'effect_size_interpretation': 'large' if abs(cohens_d_val) > 0.8 else ('medium' if abs(cohens_d_val) > 0.5 else 'small'),
            'deepsets_mean': np.mean(deepsets_paired),
            'graphsage_mean': np.mean(graphsage_paired),
            'diff': np.mean(graphsage_paired) - np.mean(deepsets_paired)
        }
        print(f"  {model_name}: DS={np.mean(deepsets_paired):.4f}, GS={np.mean(graphsage_paired):.4f}, p={wilcoxon_result['p_value']:.4f}")

print("\\n✓ Statistical testing complete")'''

# Transform all cells with necessary replacements
for i, cell in enumerate(ref_nb["cells"]):
    new_cell = transform_cell(cell)
    source_text = "".join(new_cell.get("source", []))

    # Replace specific cells with custom implementations
    if "def count_parameters" in source_text and "def load_deepsets_model" in source_text:
        new_cell["source"] = [HELPER_FUNCTIONS_CELL]
    elif "Loading DeepSets models" in source_text or ("Load all models" in source_text and "deepsets_models" in source_text):
        new_cell["source"] = [LOAD_MODELS_CELL]
    elif "Parameter Budget Comparison" in source_text and "param_comparison" in source_text:
        new_cell["source"] = [PARAM_COMPARISON_CELL]
    elif "Generate shared test datasets" in source_text:
        new_cell["source"] = [GENERATE_TEST_DATA_CELL]
    elif "def benchmark_inference_speed" in source_text:
        new_cell["source"] = [INFERENCE_BENCHMARK_CELL]
    elif "def evaluate_accuracy_metrics" in source_text:
        new_cell["source"] = [ACCURACY_EVAL_CELL]
    elif "def wilcoxon_test" in source_text:
        new_cell["source"] = [STATISTICAL_TESTS_CELL]

    new_nb["cells"].append(new_cell)

# Save the new notebook
output_path = Path(r"k:\Coding\projects\quantum-error-correction\code\results\deepsets_vs_gnn_comparison\comparison.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)

print(f"Created notebook at: {output_path}")
print(f"Total cells: {len(new_nb['cells'])}")
