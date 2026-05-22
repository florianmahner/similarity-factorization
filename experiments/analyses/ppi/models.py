from abc import ABC, abstractmethod
import math
from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import shortest_path
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList, Embedding, Linear

from .utils import get_balanced_matrix, get_balanced_samples
from pysrf import SRF

# SEAL imports
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, SortAggregation, global_sort_pool
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, to_undirected


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
    """Original SRF on binary adjacency (balanced 1s and 0s)."""

    def __init__(self, seed: int = 42, rank: int = 50, **kwargs):
        super().__init__(seed, **kwargs)
        self.rank = rank
        self.w = None

    def fit(self, train_edges, n_nodes):
        adj = get_balanced_matrix(train_edges, n_nodes, seed=self.seed, neg_ratio=1.0)

        model = SRF(
            rank=self.rank,
            rho=3.0,
            max_outer=500,
            max_inner=100,
            tol=1e-6,
            verbose=1,
            init="random_sqrt",
            random_state=self.seed,
            missing_values=np.nan,
            loss="frobenius",
        )
        self.w = model.fit_transform(adj)

    def predict_all(self) -> np.ndarray:
        return self.w @ self.w.T


class SRFPPMIPredictor(BaseLinkPredictor):
    """SRF on PPMI-transformed random walk similarity (NetMF-style)."""

    def __init__(self, seed: int = 42, rank: int = 50, window: int = 10, **kwargs):
        super().__init__(seed, **kwargs)
        self.rank = rank
        self.window = window
        self.w = None

    def _compute_ppmi(self, adj_sparse, n_nodes):
        """Compute PPMI matrix from adjacency (NetMF approach)."""
        # Degree matrix
        degrees = np.array(adj_sparse.sum(axis=1)).flatten()
        vol = degrees.sum()

        # Normalized adjacency: D^{-1} A
        with np.errstate(divide="ignore"):
            d_inv = 1.0 / degrees
            d_inv[np.isinf(d_inv)] = 0.0
        D_inv = diags(d_inv)
        M = D_inv @ adj_sparse

        # Approximate DeepWalk matrix: sum of M^k for k=1..window
        # This captures multi-hop random walk probabilities
        M_sum = M.copy()
        M_power = M.copy()
        for _ in range(2, self.window + 1):
            M_power = M_power @ M
            M_sum = M_sum + M_power

        # Convert to dense for PPMI computation
        M_sum = M_sum.toarray() / self.window

        # PPMI: log(M_sum * vol / (d_i * d_j)) - log(negative_samples)
        # Simplified: just use the random walk probabilities directly
        # Scale by volume and normalize by degree product
        d_outer = np.outer(degrees, degrees)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(M_sum * vol / (d_outer + 1e-10) + 1e-10)

        # Shift to PPMI (positive only)
        ppmi = np.maximum(pmi, 0)

        # Handle NaN/inf
        ppmi[~np.isfinite(ppmi)] = 0.0

        # Ensure perfect symmetry
        ppmi = (ppmi + ppmi.T) / 2

        # Diagonal must be NaN for SRF (missing values)
        np.fill_diagonal(ppmi, np.nan)

        return ppmi

    def fit(self, train_edges, n_nodes):
        # Build sparse adjacency
        rows = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
        cols = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj_sparse = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Compute PPMI similarity matrix
        print(f"  Computing PPMI matrix (window={self.window})...")
        ppmi = self._compute_ppmi(adj_sparse, n_nodes)

        print(f"  PPMI range: [{ppmi.min():.4f}, {ppmi.max():.4f}]")
        print(f"  PPMI mean: {ppmi.mean():.4f}")
        print(f"  PPMI non-zero: {(ppmi > 0).sum()}")

        # Fit SRF on continuous PPMI values
        model = SRF(
            rank=self.rank,
            rho=1.0,
            max_outer=100,
            max_inner=50,
            tol=1e-6,
            verbose=1,
            init="random_sqrt",
            random_state=self.seed,
            missing_values=np.nan,
            loss="frobenius",
        )
        self.w = model.fit_transform(ppmi)

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


# --- 3. OpenNE Models (Node2Vec, DeepWalk, LINE) ---
class OpenNEPredictor(BaseLinkPredictor):
    """Base class for OpenNE based embedding methods."""

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed, **kwargs)
        self.emb = None

    def _prepare_graph(self, train_edges, n_nodes):
        import networkx as nx
        from openne.graph import Graph

        # Create NetworkX DiGraph (OpenNE expects DiGraph for proper edge handling)
        G_nx = nx.DiGraph()
        G_nx.add_nodes_from(range(n_nodes))

        # Add edges in both directions for undirected graph representation
        for u, v in train_edges:
            G_nx.add_edge(u, v, weight=1.0)
            G_nx.add_edge(v, u, weight=1.0)

        G = Graph()
        G.read_g(G_nx)
        return G

    def predict_all(self) -> np.ndarray:
        if self.emb is None:
            raise RuntimeError("Model not fitted")
        return self.emb @ self.emb.T


class Node2VecPredictor(OpenNEPredictor):
    def fit(self, train_edges, n_nodes, mask_edges=None):
        from openne.node2vec import Node2vec

        G = self._prepare_graph(train_edges, n_nodes)

        # Defaults matching development/ppi/node_classification/run.py where possible
        embedding_dim = self.kwargs.get("rank", 64)
        if "embedding_dim" in self.kwargs:
            embedding_dim = self.kwargs["embedding_dim"]

        walk_length = self.kwargs.get("walk_length", 16)
        num_walks = self.kwargs.get("num_walks", 10)
        p = self.kwargs.get("p", 4.0)
        q = self.kwargs.get("q", 1.0)
        workers = self.kwargs.get("workers", 1)
        window_size = self.kwargs.get("window_size", 10)
        epochs = self.kwargs.get("epochs", 1)

        model = Node2vec(
            graph=G,
            path_length=walk_length,
            num_paths=num_walks,
            dim=embedding_dim,
            p=p,
            q=q,
            workers=workers,
            window=window_size,
            epochs=epochs,
        )

        self.emb = np.zeros((n_nodes, embedding_dim))
        for i in range(n_nodes):
            # OpenNE might use string keys if G was built from NX with default settings
            if i in model.vectors:
                self.emb[i] = model.vectors[i]
            elif str(i) in model.vectors:
                self.emb[i] = model.vectors[str(i)]


class DeepWalkPredictor(OpenNEPredictor):
    def fit(self, train_edges, n_nodes, mask_edges=None):
        from openne.node2vec import Node2vec

        G = self._prepare_graph(train_edges, n_nodes)

        # Defaults from run.py: path_length=40, num_paths=10, dim=rank, dw=True
        embedding_dim = self.kwargs.get("rank", 64)
        if "embedding_dim" in self.kwargs:
            embedding_dim = self.kwargs["embedding_dim"]

        walk_length = self.kwargs.get("walk_length", 40)
        num_walks = self.kwargs.get("num_walks", 10)
        workers = self.kwargs.get("workers", 1)
        window_size = self.kwargs.get("window_size", 10)
        epochs = self.kwargs.get("epochs", 1)

        model = Node2vec(
            graph=G,
            path_length=walk_length,
            num_paths=num_walks,
            dim=embedding_dim,
            dw=True,
            workers=workers,
            window=window_size,
            epochs=epochs,
        )

        self.emb = np.zeros((n_nodes, embedding_dim))
        for i in range(n_nodes):
            if i in model.vectors:
                self.emb[i] = model.vectors[i]
            elif str(i) in model.vectors:
                self.emb[i] = model.vectors[str(i)]


class LINEPredictor(OpenNEPredictor):
    def fit(self, train_edges, n_nodes, mask_edges=None):
        from openne.line import LINE
        import tensorflow as tf

        # Attempt to set global TF settings if not already set,
        # but respect existing environment.
        # Note: LINE in OpenNE uses TF v1 compat.
        # Assuming the environment or previous calls handled `tf.compat.v1.disable_v2_behavior()`
        # if this is running in a script that expects it.
        # If not, calling it here might be late if TF was already initialized, but worth a try
        # if it crashes. However, for library code, it is safer to assume the runner handles it
        # or catch the error.

        G = self._prepare_graph(train_edges, n_nodes)

        embedding_dim = self.kwargs.get("rank", 64)
        if "embedding_dim" in self.kwargs:
            embedding_dim = self.kwargs["embedding_dim"]

        order = self.kwargs.get("order", 3)
        epoch = self.kwargs.get("epochs", 20)
        batch_size = self.kwargs.get("batch_size", 500)
        negative_ratio = self.kwargs.get("negative_ratio", 5)

        model = LINE(
            G,
            rep_size=embedding_dim,
            epoch=epoch,
            batch_size=batch_size,
            order=order,
            negative_ratio=negative_ratio,
        )

        self.emb = np.zeros((n_nodes, embedding_dim))
        for i in range(n_nodes):
            if i in model.vectors:
                self.emb[i] = model.vectors[i]
            elif str(i) in model.vectors:
                self.emb[i] = model.vectors[str(i)]


class SkipGNNPredictor(BaseLinkPredictor):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed, **kwargs)
        self.model = None
        self.adj = None
        self.adj2 = None
        self.features = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters
        self.epochs = kwargs.get("epochs", 20)
        self.lr = kwargs.get("lr", 0.01)
        self.hidden1 = kwargs.get("hidden1", 64)
        self.hidden2 = kwargs.get("hidden2", 32)
        self.hidden_decode1 = kwargs.get("hidden_decode1", 16)
        self.dropout = kwargs.get("dropout", 0.5)
        self.batch_size = kwargs.get("batch_size", 128)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        import scipy.sparse as sp
        rowsum = np.array(adj.sum(1))
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def fit(self, train_edges, n_nodes, mask_edges=None):
        import scipy.sparse as sp
        import sys
        from pathlib import Path
        
        # SkipGNN's internal imports expect its directory to be in sys.path
        # this file is in experiments/ppi/models.py
        # third_party is in root/third_party
        root_path = Path(__file__).resolve().parent.parent.parent
        skipgnn_path = root_path / "third_party/SkipGNN/SkipGNN"
        
        if str(skipgnn_path) not in sys.path:
            sys.path.append(str(skipgnn_path))
            
        # Import from third_party
        try:
            from models import SkipGNN
        except ImportError as e:
            # If it fails, check if we are in a weird state. 
            # But with sys.path modified, it should find 'models.py' in SkipGNN folder.
            raise ImportError(f"Could not import SkipGNN models from {skipgnn_path}. Error: {e}")

        self.n_nodes = n_nodes

        # 1. Prepare Adjacency Matrices
        # Standard symmetric adjacency
        rows, cols = train_edges[:, 0], train_edges[:, 1]
        data_ones = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data_ones, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        
        # Make symmetric: A + A.T - diag
        # Note: train_edges usually has only one direction or mixed. 
        # The utils.py logic: adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # This assumes adj is not already symmetric. 
        # Let's just force symmetry safely.
        adj = adj + adj.T
        adj = adj.sign() # Binarize
        
        # Skip Graph: A^2
        adj2 = adj.dot(adj)
        adj2 = adj2.sign() # Binary skip graph
        
        # Add self-loops to original adj (as per utils.py)
        adj = adj + sp.eye(adj.shape[0])
        
        # Normalize
        adj_norm = self._normalize_adj(adj)
        adj2_norm = self._normalize_adj(adj2)
        
        self.adj = self._sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)
        self.adj2 = self._sparse_mx_to_torch_sparse_tensor(adj2_norm).to(self.device)
        
        # 2. Prepare Features (One-Hot)
        # Using dense identity matrix as per SkipGNN default for PPI/DDI
        self.features = torch.FloatTensor(np.eye(n_nodes)).to(self.device)
        
        # 3. Initialize Model
        self.model = SkipGNN(nfeat=n_nodes, 
                             nhid1=self.hidden1, 
                             nhid2=self.hidden2, 
                             nhid_decode1=self.hidden_decode1, 
                             dropout=self.dropout).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        loss_fct = torch.nn.BCELoss()
        m_sigmoid = torch.nn.Sigmoid()

        # 4. Training Loop
        print(f"Training SkipGNN on {self.device}...")
        self.model.train()
        
        for epoch in range(self.epochs):
            # Generate new balanced samples for each epoch
            pos_edges_arr, neg_edges_arr = get_balanced_samples(
                train_edges, n_nodes, seed=self.seed + epoch, neg_ratio=1.0
            )
            
            all_edges = np.concatenate([pos_edges_arr, neg_edges_arr], axis=0)
            all_labels = np.concatenate([np.ones(len(pos_edges_arr)), np.zeros(len(neg_edges_arr))])
            
            # Shuffle
            perm = np.random.permutation(len(all_labels))
            all_edges = all_edges[perm]
            all_labels = all_labels[perm]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(all_labels), self.batch_size):
                batch_edges = all_edges[i:i+self.batch_size]
                batch_labels = torch.FloatTensor(all_labels[i:i+self.batch_size]).to(self.device)
                
                # SkipGNN forward expects idx as tuple/list of two tensors/arrays?
                # utils.py: return y, (idx1, idx2)
                # train.py: output, _ = model(features, adj, adj2, inp) -> inp is (idx1, idx2)
                # So we need to pass a tuple (indices_source, indices_target)
                
                idx1 = torch.LongTensor(batch_edges[:, 0]).to(self.device)
                idx2 = torch.LongTensor(batch_edges[:, 1]).to(self.device)
                batch_idx = (idx1, idx2)
                
                optimizer.zero_grad()
                output, _ = self.model(self.features, self.adj, self.adj2, batch_idx)
                
                # output is raw logits?
                # models.py: o = self.decoder2(o) -> Linear. So yes, logits.
                # train.py: n = torch.squeeze(m(output)); loss = loss_fct(n, label)
                preds = torch.squeeze(m_sigmoid(output))
                
                loss = loss_fct(preds, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {total_loss / num_batches:.4f}")

    def predict_all(self) -> np.ndarray:
        self.model.eval()
        scores = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        m_sigmoid = torch.nn.Sigmoid()
        
        # Process in chunks
        rows, cols = np.triu_indices(self.n_nodes, k=1)
        
        batch_size = self.batch_size * 4
        num_pairs = len(rows)
        
        print(f"Predicting {num_pairs} pairs with SkipGNN...")
        
        with torch.no_grad():
            for i in range(0, num_pairs, batch_size):
                batch_rows = rows[i:i+batch_size]
                batch_cols = cols[i:i+batch_size]
                
                idx1 = torch.LongTensor(batch_rows).to(self.device)
                idx2 = torch.LongTensor(batch_cols).to(self.device)
                batch_idx = (idx1, idx2)
                
                output, _ = self.model(self.features, self.adj, self.adj2, batch_idx)
                preds = torch.squeeze(m_sigmoid(output)).cpu().numpy()
                
                if preds.ndim == 0:
                    preds = np.array([preds])
                
                scores[batch_rows, batch_cols] = preds
                scores[batch_cols, batch_rows] = preds
                
        return scores


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

    def __init__(self, edge_index, links, labels, num_nodes, num_hops=1, max_z=1000):
        super().__init__()
        self.edge_index = edge_index
        self.links = links  # List of (src, dst) pairs
        self.labels = labels  # List of labels (0 or 1)
        self.num_nodes = num_nodes
        self.num_hops = num_hops
        self.max_z = max_z

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        # 1. Extract k-hop subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], self.num_hops, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
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
            num_nodes=z.size(0)
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
            self.edge_index, train_links, train_labels, self.n_nodes, num_hops=self.num_hops
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
            self.edge_index, all_links, [0] * len(all_links), self.n_nodes, num_hops=self.num_hops
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
    rank = params.get("rank", 64)  # Common parameter

    if m in ["cn", "aa", "ra", "jc"]:
        return HeuristicPredictor(m, seed)
    elif m == "srf":
        return SRFPredictor(seed, rank=rank)
    elif m == "srf_ppmi":
        window = params.get("window", 10)
        return SRFPPMIPredictor(seed, rank=rank, window=window)
    elif m == "node2vec":
        return Node2VecPredictor(seed, rank=rank, **params.get("node2vec", {}))
    elif m == "deepwalk":
        return DeepWalkPredictor(seed, rank=rank, **params.get("deepwalk", {}))
    elif m == "line":
        return LINEPredictor(seed, rank=rank, **params.get("line", {}))
    elif m == "skipgnn":
        return SkipGNNPredictor(seed, **params.get("skipgnn", {}))
    elif m == "seal":
        return SEALPredictor(seed, **params.get("seal", {}))
    raise ValueError(f"Unknown: {method}")
