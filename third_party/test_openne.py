"""Test OpenNE with TensorFlow compatibility mode."""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add OpenNE to path
sys.path.insert(0, str(Path(__file__).parent / "OpenNE/src"))

from openne.graph import Graph
from openne.node2vec import Node2vec
from openne.line import LINE

print("=" * 60)
print("Testing OpenNE with TensorFlow Compatibility Mode")
print("=" * 60)

# Create a small synthetic graph
print("\n1. Creating synthetic graph...")
g_nx = nx.karate_club_graph()
print(f"   Graph: {g_nx.number_of_nodes()} nodes, {g_nx.number_of_edges()} edges")

# Convert to OpenNE Graph format
print("\n2. Converting to OpenNE format...")
g = Graph()
g.read_g(g_nx)
print(f"   OpenNE Graph initialized")

# Test Node2Vec (gensim-based, should work)
print("\n3. Testing Node2Vec (gensim-based)...")
try:
    model = Node2vec(g, path_length=10, num_paths=5, dim=32, workers=1)
    print(f"   ✓ Node2Vec SUCCESS: {len(model.vectors)} embeddings")
    sample_node = list(model.vectors.keys())[0]
    print(f"   Sample embedding shape: {model.vectors[sample_node].shape}")
except Exception as e:
    print(f"   ✗ Node2Vec FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test DeepWalk (gensim-based, should work)
print("\n4. Testing DeepWalk (gensim-based)...")
try:
    model = Node2vec(g, path_length=10, num_paths=5, dim=32, workers=1, dw=True)
    print(f"   ✓ DeepWalk SUCCESS: {len(model.vectors)} embeddings")
    sample_node = list(model.vectors.keys())[0]
    print(f"   Sample embedding shape: {model.vectors[sample_node].shape}")
except Exception as e:
    print(f"   ✗ DeepWalk FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test LINE (TensorFlow-based)
print("\n5. Testing LINE (TensorFlow-based)...")
try:
    model = LINE(g, rep_size=32, epoch=1)
    print(f"   ✓ LINE SUCCESS: {len(model.vectors)} embeddings")
    sample_node = list(model.vectors.keys())[0]
    print(f"   Sample embedding shape: {model.vectors[sample_node].shape}")
except Exception as e:
    print(f"   ✗ LINE FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
