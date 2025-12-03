#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validate Grand Caricaie food web structure using community detection algorithms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

data_dir = Path("data/gateway/grand_caricaie")
adj_symmetric = np.load(data_dir / "adjacency_symmetric.npy")
adj_directed = np.load(data_dir / "adjacency_directed.npy")
species_df = pd.read_csv(data_dir / "species.csv")

def compute_trophic_levels(adj_directed: np.ndarray, species_df: pd.DataFrame) -> np.ndarray:
    n = len(species_df)
    levels = np.ones(n)
    
    basal_mask = np.sum(adj_directed, axis=1) == 0
    levels[basal_mask] = 1.0
    
    for _ in range(100):
        changed = False
        for i in range(n):
            if basal_mask[i]:
                continue
            prey_indices = np.where(adj_directed[i, :] > 0)[0]
            if len(prey_indices) > 0:
                new_level = 1.0 + np.mean(levels[prey_indices])
                if abs(new_level - levels[i]) > 1e-6:
                    levels[i] = new_level
                    changed = True
        if not changed:
            break
    
    return levels

species_df["trophic_level"] = compute_trophic_levels(adj_directed, species_df)

n = len(species_df)
print("=" * 80)
print("FOOD WEB STRUCTURE VALIDATION")
print("=" * 80)

G_undirected = nx.Graph()
G_undirected.add_nodes_from(range(n))
for i in range(n):
    for j in range(i + 1, n):
        if adj_symmetric[i, j] > 0:
            G_undirected.add_edge(i, j)

G_directed = nx.DiGraph()
G_directed.add_nodes_from(range(n))
for i in range(n):
    for j in range(n):
        if adj_directed[i, j] > 0:
            G_directed.add_edge(i, j)

print(f"\nNetwork: {n} species")
print(f"Undirected edges: {G_undirected.number_of_edges()}")
print(f"Directed edges: {G_directed.number_of_edges()}")
print(f"Density: {nx.density(G_undirected):.3f}")

print("\n" + "=" * 80)
print("LOUVAIN COMMUNITY DETECTION")
print("=" * 80)

communities_louvain = community.louvain_communities(G_undirected, seed=42)
print(f"\nFound {len(communities_louvain)} communities")

community_assignment = np.zeros(n, dtype=int)
for comm_id, comm_members in enumerate(communities_louvain):
    community_assignment[list(comm_members)] = comm_id
    print(f"\nCommunity {comm_id}: {len(comm_members)} species")
    
    members_df = species_df.iloc[list(comm_members)]
    
    body_mass = members_df["body_mass_g"].dropna()
    if len(body_mass) > 0:
        print(f"  Body mass: {np.log10(body_mass.min()):.2f} to {np.log10(body_mass.max()):.2f} (log10)")
    
    print(f"  Metabolic types: {members_df['metabolic_type'].value_counts().to_dict()}")
    print(f"  Movement types: {members_df['movement_type'].value_counts().to_dict()}")
    
    trophic = members_df["trophic_level"]
    if len(trophic) > 0:
        print(f"  Trophic level: {trophic.min():.2f} to {trophic.max():.2f}")
    
    if len(comm_members) <= 10:
        print(f"  Members: {', '.join(members_df['species'].head(10).tolist())}")
    else:
        print(f"  Top members: {', '.join(members_df['species'].head(5).tolist())}...")

modularity = community.modularity(G_undirected, communities_louvain)
print(f"\nModularity: {modularity:.4f}")

print("\n" + "=" * 80)
print("GREEDY MODULARITY COMMUNITY DETECTION")
print("=" * 80)

communities_greedy = community.greedy_modularity_communities(G_undirected)
print(f"\nFound {len(communities_greedy)} communities")

for comm_id, comm_members in enumerate(communities_greedy[:5]):
    members_df = species_df.iloc[list(comm_members)]
    print(f"\nCommunity {comm_id}: {len(comm_members)} species")
    print(f"  Top members: {', '.join(members_df['species'].head(5).tolist())}...")

modularity_greedy = community.modularity(G_undirected, communities_greedy)
print(f"\nModularity: {modularity_greedy:.4f}")

print("\n" + "=" * 80)
print("NETWORK CENTRALITY ANALYSIS")
print("=" * 80)

degree_centrality = nx.degree_centrality(G_undirected)
betweenness_centrality = nx.betweenness_centrality(G_undirected)
pagerank = nx.pagerank(G_directed)

species_df["degree_centrality"] = [degree_centrality[i] for i in range(n)]
species_df["betweenness_centrality"] = [betweenness_centrality[i] for i in range(n)]
species_df["pagerank"] = [pagerank[i] for i in range(n)]
species_df["louvain_community"] = community_assignment

print("\nTop 15 species by degree centrality:")
top_degree = species_df.nlargest(15, "degree_centrality")
for idx, row in top_degree.iterrows():
    print(f"  {row['species']:40s} | deg_cent={row['degree_centrality']:.3f} | TL={row['trophic_level']:.2f} | mass={row['body_mass_g']:.2e}g | comm={row['louvain_community']}")

print("\nTop 15 species by PageRank:")
top_pagerank = species_df.nlargest(15, "pagerank")
for idx, row in top_pagerank.iterrows():
    print(f"  {row['species']:40s} | PR={row['pagerank']:.4f} | TL={row['trophic_level']:.2f} | mass={row['body_mass_g']:.2e}g | comm={row['louvain_community']}")

print("\n" + "=" * 80)
print("BODY MASS vs COMMUNITY")
print("=" * 80)

for comm_id in range(min(5, len(communities_louvain))):
    comm_species = species_df[species_df["louvain_community"] == comm_id]
    body_mass = comm_species["body_mass_g"].dropna()
    if len(body_mass) > 1:
        log_mass = np.log10(body_mass)
        print(f"Community {comm_id}: n={len(comm_species)}, log mass: {log_mass.mean():.2f} ± {log_mass.std():.2f}")

output_path = Path("experiments/development/foodweb/outputs") / "structure_validation.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
species_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
comm_sizes = [len(c) for c in communities_louvain]
ax.bar(range(len(comm_sizes)), sorted(comm_sizes, reverse=True), color='steelblue', alpha=0.7)
ax.set_xlabel("Community rank")
ax.set_ylabel("Size (species)")
ax.set_title(f"Louvain communities (n={len(communities_louvain)}, Q={modularity:.3f})")
ax.grid(alpha=0.3, axis='y')

ax = axes[0, 1]
for comm_id in range(min(5, len(communities_louvain))):
    comm_species = species_df[species_df["louvain_community"] == comm_id]
    body_mass = comm_species["body_mass_g"].dropna()
    if len(body_mass) > 0:
        ax.hist(np.log10(body_mass), bins=20, alpha=0.5, label=f"C{comm_id}")
ax.set_xlabel("log10 body mass (g)")
ax.set_ylabel("Count")
ax.set_title("Body mass by community")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(species_df["degree_centrality"], species_df["pagerank"], 
          c=species_df["louvain_community"], cmap='tab10', alpha=0.6, s=30)
ax.set_xlabel("Degree centrality")
ax.set_ylabel("PageRank")
ax.set_title("Centrality measures colored by community")
ax.grid(alpha=0.3)

ax = axes[1, 1]
trophic_by_comm = []
for comm_id in range(min(5, len(communities_louvain))):
    comm_species = species_df[species_df["louvain_community"] == comm_id]
    trophic_by_comm.append(comm_species["trophic_level"].values)
ax.boxplot(trophic_by_comm, labels=[f"C{i}" for i in range(len(trophic_by_comm))])
ax.set_ylabel("Trophic level")
ax.set_title("Trophic level by community")
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plot_path = Path("experiments/development/foodweb/outputs") / "structure_validation.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

