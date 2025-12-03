import sys
from pathlib import Path
import numpy as np
import torch
sys.path.append(str(Path.cwd()))

try:
    from experiments.ppi.models import SEALPredictor
    print("Successfully imported SEALPredictor.")
    
    # Create dummy graph
    # Graph: 0-1, 1-2, 2-3, 3-0 (Cycle)
    train_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    # Increase n_nodes to 10 to allow sufficient negative sampling
    n_nodes = 10
    
    # Use minimal parameters for speed
    params = {
        "num_hops": 1,
        "epochs": 1,
        "batch_size": 2,
        "hidden_channels": 16
    }
    
    predictor = SEALPredictor(seed=42, **params)
    print("Instantiated SEALPredictor.")
    
    predictor.fit(train_edges, n_nodes)
    print("Successfully fit SEALPredictor.")
    
    res = predictor.predict_all()
    print(f"Prediction shape: {res.shape}")
    print("Sample scores:", res[:2, :2])

except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
