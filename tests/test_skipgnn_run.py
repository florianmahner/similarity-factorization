import sys
from pathlib import Path
import numpy as np
import torch
sys.path.append(str(Path.cwd()))

try:
    from experiments.ppi.lib.models import SkipGNNPredictor
    print("Successfully imported SkipGNNPredictor.")
    
    # Create dummy graph
    # Graph: 0-1, 1-2, 2-3, 3-0 (Cycle) + isolated nodes to allow negative sampling
    train_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    # Increase n_nodes to 10. 
    # Max edges for n=10 is 45. We have 4. Remaining = 41.
    # We request 4 negatives. 41 > 4, so this will work.
    n_nodes = 10
    
    # Use minimal parameters for speed
    params = {
        "epochs": 1,
        "batch_size": 2,
        "hidden1": 4,
        "hidden2": 4,
        "hidden_decode1": 4
    }
    
    predictor = SkipGNNPredictor(seed=42, **params)
    print("Instantiated SkipGNNPredictor.")
    
    predictor.fit(train_edges, n_nodes)
    print("Successfully fit SkipGNNPredictor.")
    
    res = predictor.predict_all()
    print(f"Prediction shape: {res.shape}")
    print("Sample scores:", res[:2, :2])

except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
