"""
Debug script to test LINE embeddings step-by-step.
Based on OpenNE README and examples.
"""

import sys
import os
from pathlib import Path

# Set up TensorFlow compatibility BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
sys.modules['tensorflow'] = tf.compat.v1
tf.compat.v1.disable_v2_behavior()

# Add paths
sys.path.append(str(Path(__file__).parent / "OpenNE/src"))
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from openne.graph import Graph
from openne.line import LINE
from openne.node2vec import Node2vec

print("="*60)
print("Step 1: Create a simple test graph")
print("="*60)

# Create a simple test graph with clear communities
g_nx = nx.karate_club_graph()
print(f"Test graph: {g_nx.number_of_nodes()} nodes, {g_nx.number_of_edges()} edges")

# Create synthetic labels based on club membership
labels_dict = {}
for node in g_nx.nodes():
    club = g_nx.nodes[node]['club']
    labels_dict[node] = [club]  # Multi-label format

print(f"Labels: {len(labels_dict)} nodes labeled")

# Convert to directed as OpenNE expects
g_nx = g_nx.to_directed()
for u, v in g_nx.edges():
    g_nx[u][v]['weight'] = 1.0

print("\n" + "="*60)
print("Step 2: Convert to OpenNE Graph format")
print("="*60)

g = Graph()
g.read_g(g_nx)
print(f"OpenNE Graph: {g.node_size} nodes")

print("\n" + "="*60)
print("Step 3: Test DeepWalk (baseline)")
print("="*60)

model_dw = Node2vec(g, path_length=10, num_paths=10, dim=16, workers=1, dw=True)
print(f"DeepWalk: {len(model_dw.vectors)} embeddings")
print(f"Sample embedding shape: {list(model_dw.vectors.values())[0].shape}")
print(f"Sample nodes: {list(model_dw.vectors.keys())[:5]}")

print("\n" + "="*60)
print("Step 4: Test LINE with different parameters")
print("="*60)

for order in [1, 2]:
    for epochs in [5, 10]:
        print(f"\n--- LINE order={order}, epochs={epochs} ---")
        model_line = LINE(g, rep_size=16, epoch=epochs, batch_size=100, order=order)
        print(f"LINE: {len(model_line.vectors)} embeddings")

        # Check if embeddings are valid
        sample_emb = list(model_line.vectors.values())[0]
        print(f"Sample embedding shape: {sample_emb.shape}")
        print(f"Sample embedding mean: {sample_emb.mean():.4f}, std: {sample_emb.std():.4f}")
        print(f"Are embeddings all zeros? {(sample_emb == 0).all()}")

print("\n" + "="*60)
print("Step 5: Test classification with LINE vs DeepWalk")
print("="*60)

for method_name, embeddings in [("DeepWalk", model_dw.vectors), ("LINE", model_line.vectors)]:
    print(f"\n--- Testing {method_name} ---")

    # Prepare data
    nodes = sorted(labels_dict.keys())
    X = np.array([embeddings.get(n, np.zeros(16)) for n in nodes])
    y_labels = [labels_dict[n] for n in nodes]

    # Binarize labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_labels)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Non-zero rows in X: {(X != 0).any(axis=1).sum()}/{len(X)}")
    print(f"X mean: {X.mean():.4f}, std: {X.std():.4f}")

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Train classifier
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train, Y_train)

    # Predict
    Y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(Y_test, Y_pred)
    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
