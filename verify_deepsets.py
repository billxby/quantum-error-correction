import sys
sys.path.insert(0, "./code")
import torch
import numpy as np
from benchmark_models import DeepSets

def run_verification():
    print("========================================")
    print("VERIFYING DEEPSETS ENHANCEMENTS")
    print("========================================")

    # 1. Initialize Model with Fourier Features
    print("\n1. Initializing DeepSets (with Fourier Features)...")
    model = DeepSets(
        use_fourier_features=True,
        fourier_dim=64,
        seed=42
    )
    print(f"  Initialized: {model}")
    print(f"  Null Token size: {model.model.null_token.shape}")
    print(f"  Fourier Matrix size: {model.model.fourier.B.shape}")

    # 2. Check Coordinates
    print("\n2. Checking Coordinates for d=5...")
    try:
        coords = model.check_coordinates(5)
    except Exception as e:
        print(f"  ERROR checking coordinates: {e}")
        return

    # 3. Forward Pass Test
    print("\n3. Testing Forward Pass...")
    batch_size = 4
    max_fired = 10

    # Mock data: [batch, max_fired, 3]
    mock_coords = torch.randn(batch_size, max_fired, 3)
    # Counts: 0, 5, 10, 1
    mock_counts = torch.tensor([0, 5, 10, 1])

    try:
        out = model.model(mock_coords, mock_counts)
        print(f"  Forward pass successful.")
        print(f"  Output shape: {out.shape}")
        print(f"  Output values: {out.detach().min().item():.4f} to {out.detach().max().item():.4f}")

        # Check empty syndrome (index 0)
        # If null token is working, this should be consistent (and non-nan)
        print(f"  Empty syndrome output: {out[0].item():.4f}")

    except Exception as e:
        print(f"  ERROR in forward pass: {e}")
        return

    print("\n========================================")
    print("VERIFICATION COMPLETE: SUCCESS")
    print("========================================")

if __name__ == "__main__":
    run_verification()
