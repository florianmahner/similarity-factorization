from abc import ABC, abstractmethod
import math
from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import shortest_path
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
from .utils import get_balanced_matrix, get_balanced_samples
from pysrf import SRF

# SEAL imports
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, SortAggregation
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, to_undirected
from torch_geometric.nn import GCNConv, global_sort_pool
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList, Embedding, Linear
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class BaseLinkPredictor(ABC):
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, train_edges: np.ndarray, n_nodes: int, mask_edges: np.ndarray = None):
        pass

    @abstractmethod
    def predict_all(self) -> np.ndarray:
        """Returns the full (N, N) score matrix for All-vs-All evaluation."""
        pass


class SRFPredictor(BaseLinkPredictor):
    def __init__(self, seed: int = 42, rank: int = 50, **kwargs):
        super().__init__(seed, **kwargs)
        self.rank = rank
        self.w = None

    def fit(self, train_edges, n_nodes):

        # Train on Balanced Sample (1s and sampled 0s)
        adj = get_balanced_matrix(train_edges, n_nodes, seed=self.seed, neg_ratio=1.0)

        model = SRF(
            rank=self.rank,
            rho=3.0,
            max_outer=150,
            max_inner=50,
            tol=1e-5,
            verbose=1,
            init="random_sqrt",
            random_state=self.seed,
            missing_values=np.nan,
            loss="frobenius",
        )
        self.w = model.fit_transform(adj)

    def predict_all(self) -> np.ndarray:
        return self.w @ self.w.T


class HeuristicPredictor(BaseLinkPredictor):
    def __init__(self, method: str, seed: int = 42, **kwargs):
        super().__init__(seed, **kwargs)
        self.method = method.upper()
        self.A = None

    def fit(self, train_edges, n_nodes):
        # 1. Build Sparse Adjacency Matrix (Symmetric)
        # We use float data to ensure matrix multiplications return floats
        rows = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
        cols = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)

        self.A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    def predict_all(self) -> np.ndarray:
        """
        Computes the full score matrix using fast sparse matrix math.
        Returns a Dense (N, N) numpy array.
        """
        # Common Neighbors: A * A
        # This counts paths of length 2 between all pairs
        if self.method == "CN":
            return (self.A @ self.A).toarray()

        # Pre-compute degrees for AA, RA, JC
        # .A1 converts matrix to flat 1D array
        degrees = np.array(self.A.sum(axis=1)).flatten()

        # Adamic-Adar (AA) and Resource Allocation (RA)
        # Formula: A * (1 / f(degree)) * A
        if self.method in ["AA", "RA"]:
            with np.errstate(divide="ignore"):
                if self.method == "AA":
                    w = 1.0 / np.log(degrees)
                else:  # RA
                    w = 1.0 / degrees
                w[np.isinf(w) | (w == 0)] = 0.0

            # Create diagonal weighting matrix
            D_inv = diags(w)
            return (self.A @ D_inv @ self.A).toarray()

        # Jaccard Coefficient (JC)
        # Formula: CN / (deg(u) + deg(v) - CN)
        if self.method == "JC":
            intersection = (self.A @ self.A).toarray()

            # Broadcast degrees to create (N, N) sum matrix
            # deg[u] + deg[v] for all u, v
            union_base = degrees[:, None] + degrees[None, :]

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = intersection / (union_base - intersection)
                scores[np.isnan(scores)] = 0.0
            return scores

        raise ValueError(f"Unknown heuristic: {self.method}")


# --- 3. Node2Vec (Using OpenNE) ---
class Node2VecPredictor(BaseLinkPredictor):
    def fit(self, train_edges, n_nodes):
        import networkx as nx
        from openne.graph import Graph
        from openne.node2vec import Node2vec

        # Create NetworkX DiGraph (OpenNE expects DiGraph for proper edge handling)
        G_nx = nx.DiGraph()
        G_nx.add_nodes_from(range(n_nodes))

        # Add edges in both directions for undirected graph
        for u, v in train_edges:
            G_nx.add_edge(u, v, weight=1.0)
            G_nx.add_edge(v, u, weight=1.0)

        # Wrap in OpenNE Graph
        G = Graph()
        G.read_g(G_nx)

        # Node2Vec parameters
        embedding_dim = self.kwargs.get("embedding_dim", 64)
        walk_length = self.kwargs.get("walk_length", 20)
        num_walks = self.kwargs.get("num_walks", 10)
        p = self.kwargs.get("p", 1.0)
        q = self.kwargs.get("q", 1.0)
        window_size = self.kwargs.get("window_size", 10)
        epochs = self.kwargs.get("epochs", 5)
        workers = self.kwargs.get("workers", 1)  # Use 1 worker for stability

        # Train Node2Vec
        model = Node2vec(
            graph=G,
            path_length=walk_length,
            num_paths=num_walks,
            dim=embedding_dim,
            p=p,
            q=q,
            workers=workers,
            window=window_size,
            iter=epochs,
        )

        # Extract embeddings in node order
        self.emb = np.zeros((n_nodes, embedding_dim))
        for node_id in range(n_nodes):
            # OpenNE uses original node IDs from G.look_back_list
            original_node = G.look_back_list[node_id]
            self.emb[node_id] = model.vectors[original_node]

    def predict_all(self):
        return self.emb @ self.emb.T


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    """
    Double-Radius Node Labeling (DRNL).
    Returns tensor of shape [num_subgraph_nodes] with integer labels.
    """
    # Convert to scipy sparse matrix for efficient path finding
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    # Distances to src
    dist2src = shortest_path(adj, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    # Distances to dst
    dist2dst = shortest_path(adj, directed=False, unweighted=True, indices=dst)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    # Calculate Z scores
    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)

    # Special handling for src and dst to ensure they are 1
    # (Note: In standard DRNL, src/dst usually get z=1.
    # The logic above usually results in 1, but we enforce it just in case of isolated components)
    z[src] = 1.0
    z[dst] = 1.0

    # Handle unreachable nodes
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)


# --- Dynamic Dataset ---
class SEALDynamicDataset(Dataset):
    """
    Extracts enclosing subgraphs on-the-fly to save RAM and allow multiprocessing.
    """

    def __init__(self, edge_index, links, labels, num_hops=1, max_z=1000):
        super().__init__()
        self.edge_index = edge_index
        self.links = links  # List of (src, dst) pairs
        self.labels = labels  # List of labels (0 or 1)
        self.num_hops = num_hops
        self.max_z = max_z

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        # 1. Extract k-hop subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], self.num_hops, self.edge_index, relabel_nodes=True
        )

        src_mapped, dst_mapped = mapping[0].item(), mapping[1].item()

        # 2. Remove the link between src and dst if it exists in the subgraph
        # (Crucial for link prediction to avoid leakage)
        mask1 = (sub_edge_index[0] != src_mapped) | (sub_edge_index[1] != dst_mapped)
        mask2 = (sub_edge_index[0] != dst_mapped) | (sub_edge_index[1] != src_mapped)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # 3. DRNL Labeling
        z = drnl_node_labeling(
            sub_edge_index, src_mapped, dst_mapped, num_nodes=subset.size(0)
        )

        # Clip max_z to prevent index out of bounds in embedding layer
        z = torch.clamp(z, 0, self.max_z)

        # 4. Construct Data object
        data = Data(
            x=None,
            z=z,
            edge_index=sub_edge_index,
            y=torch.tensor([y], dtype=torch.float),
        )
        return data


# --- Model: DGCNN ---
class DGCNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        max_z=1000,
        k=0.6,
        train_dataset=None,
        dynamic=False,
        GNN=GCNConv,
    ):
        super().__init__()
        self.k = k
        self.max_z = max_z

        # Embedding for the DRNL labels (z)
        self.z_embedding = Embedding(self.max_z + 1, hidden_channels)

        self.convs = ModuleList()
        # First layer takes embedding dim
        self.convs.append(GNN(hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))

        self.conv1d_params1 = Conv1d(1, 16, hidden_channels, hidden_channels)
        self.conv1d_params2 = Conv1d(16, 32, 5, 1)
        self.maxpool1d = MaxPool1d(2, 2)

        # Dense layers - we will calculate input dimension dynamically or based on k
        # Standard SEAL dense layer calculation:
        self.mlp = ModuleList(
            [
                Linear(
                    (
                        32 * (self.k * hidden_channels // 2 - 2) // 2 + 1
                        if isinstance(self.k, int)
                        else 0
                    ),
                    128,
                ),
                Linear(128, 1),
            ]
        )
        # Note: The linear dimension depends on k. In the forward pass, we handle k if it's a ratio.

        # If k is a ratio (0.6), we need to estimate integer k based on avg graph size or fix it.
        # For simplicity in this snippet, we will calculate the dense dim in the forward pass
        # or assume a fixed K if provided.
        self.fixed_k = None
        if k >= 1:
            self.fixed_k = int(k)
            # Re-init MLP with correct size
            dense_dim = self.fixed_k - 5 + 1  # after conv2
            dense_dim = dense_dim // 2  # after maxpool
            dense_dim = dense_dim - 5 + 1  # (approx, standard CNN math applies here)
            # To avoid shape math headaches, SEAL often uses a lazy Linear or we just view it in forward.

    def forward(self, z, edge_index, batch):
        x = self.z_embedding(z)

        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        x = torch.cat(xs[1:], dim=-1)

        # Global Sort Pooling
        # If k is < 1, it's a ratio. We usually pick a fixed k for the batch or dataset.
        # Here we hardcode a reasonable k for stability if not set, or use global_sort_pool's k
        k = self.fixed_k if self.fixed_k else 30

        x = global_sort_pool(x, batch, k)  # [num_graphs, k * hidden_channels]

        # Reshape for Conv1d: [num_graphs, 1, k * hidden]
        x = x.unsqueeze(1)

        x = self.conv1d_params1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv1d_params2(x).relu()

        x = x.view(x.size(0), -1)

        # Lazy initialization of MLP if dimension doesn't match (for safety)
        if self.mlp[0].in_features != x.size(1):
            self.mlp[0] = Linear(x.size(1), 128).to(x.device)

        x = self.mlp[0](x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp[1](x)
        return x


# --- The Predictor Class ---
class SEALPredictor:
    def __init__(self, seed: int = 42, **kwargs):
        self.kwargs = kwargs
        self.seed = seed
        self.num_hops = kwargs.get(
            "num_hops", 1
        )  # 1 hop usually suffices for SEAL and is faster
        self.hidden_channels = kwargs.get("hidden_channels", 32)
        self.num_layers = kwargs.get("num_layers", 3)
        self.batch_size = kwargs.get("batch_size", 32)
        self.lr = kwargs.get("lr", 0.0001)
        self.epochs = kwargs.get("epochs", 5)
        self.k = kwargs.get("k", 0.6)  # SortPooling ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Performance settings
        # Set num_workers > 0 to use parallel CPU processing for subgraph extraction
        self.num_workers = kwargs.get("num_workers", 4)

    def fit(self, train_edges, n_nodes):
        # 1. Setup Graph Structure (Undirected)
        # We need the full positive graph for subgraph extraction
        edge_index = torch.tensor(train_edges, dtype=torch.long).t()
        self.edge_index = to_undirected(edge_index, num_nodes=n_nodes)
        self.n_nodes = n_nodes

        # 2. Get Balanced Training Data (Streamlined!)
        # This now uses the EXACT same logic as SRF
        pos_edges_arr, neg_edges_arr = get_balanced_samples(
            train_edges, n_nodes, seed=self.seed, neg_ratio=1.0
        )

        # Convert to torch tensors
        pos_edges = torch.tensor(pos_edges_arr, dtype=torch.long)
        neg_edges = torch.tensor(neg_edges_arr, dtype=torch.long)

        # Combine into a list of links and labels
        # Positives (Label 1) + Negatives (Label 0)
        train_links = torch.cat([pos_edges, neg_edges], dim=0).tolist()
        train_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

        # 3. Create Dataset and Loader
        dataset = SEALDynamicDataset(
            self.edge_index, train_links, train_labels, num_hops=self.num_hops
        )

        # Use num_workers > 0 to parallelize subgraph extraction while GPU trains
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.kwargs.get("num_workers", 4),
        )

        # 4. Initialize Model
        self.model = DGCNN(
            hidden_channels=self.hidden_channels, num_layers=self.num_layers, k=self.k
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = BCEWithLogitsLoss()

        # 5. Training Loop
        print(
            f"Training SEAL on {self.device} with {len(train_links)} samples (Balanced)..."
        )
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data in loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                logits = self.model(data.z, data.edge_index, data.batch)
                loss = criterion(logits.view(-1), data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            print(
                f"Epoch {epoch+1}/{self.epochs} | Loss: {total_loss / len(dataset):.4f}"
            )

    def predict_all(self) -> np.ndarray:
        """
        Warning: SEAL is O(N^2) for full matrix prediction.
        This is extremely computationally expensive.
        """
        self.model.eval()
        scores = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)

        # Generate all pairs (Upper triangle)
        # For N=2000, this is 2 million pairs. For N=10000, this is 50 million.
        # We process in chunks to avoid generating a massive list in memory.

        print("Generating candidate pairs for All-vs-All prediction...")
        # Optimization: Only predict upper triangle, then mirror
        rows, cols = np.triu_indices(self.n_nodes, k=1)
        all_links = list(zip(rows, cols))

        # Use a larger batch size for inference
        inf_batch_size = self.batch_size * 2

        dataset = SEALDynamicDataset(
            self.edge_index, all_links, [0] * len(all_links), num_hops=self.num_hops
        )

        loader = DataLoader(
            dataset,
            batch_size=inf_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,  # Helps transfer to GPU faster
        )

        print(f"Predicting {len(all_links)} pairs. This may take a while...")

        with torch.no_grad():
            ptr = 0
            for data in loader:
                data = data.to(self.device)
                logits = self.model(data.z, data.edge_index, data.batch)
                preds = torch.sigmoid(logits.view(-1)).cpu().numpy()

                # Fill the matrix
                batch_len = len(preds)
                current_links = all_links[ptr : ptr + batch_len]

                for k, (u, v) in enumerate(current_links):
                    val = preds[k]
                    scores[u, v] = val
                    scores[v, u] = val

                ptr += batch_len

        return scores


def get_predictor(method, seed, params):
    m = method.lower()
    if m in ["cn", "aa", "ra", "jc"]:
        return HeuristicPredictor(m, seed)
    elif m == "srf":
        return SRFPredictor(seed, rank=params.get("rank", 64))
    elif m == "node2vec":
        return Node2VecPredictor(seed, **params.get("node2vec", {}))
    elif m == "seal":
        return SEALPredictor(seed, **params.get("seal", {}))
    raise ValueError(f"Unknown: {method}")
