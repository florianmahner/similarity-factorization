"""Triplet-based iterative RSM optimization.

Optimize RSM entries directly using actual triplet cross-entropy loss.
Fast: just indexing into the matrix per mini-batch, no O(n^3).
Then feed corrected RSM to SRF.

Different from SPoSE: we optimize the full n x n matrix, not a low-rank embedding.
SRF then factorizes the optimized matrix.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def triplet_accuracy_embedding(w: np.ndarray, triplets: np.ndarray) -> float:
    ei, ej, ek = w[triplets[:, 0]], w[triplets[:, 1]], w[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.empty_cache()
    log.info(f"Device: {device}")

    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    # Initialize s from logit of count RSM
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)
    p_obs = np.divide(counts + 1, shown + 2, out=np.full((N, N), 0.5), where=shown > 0)
    np.fill_diagonal(p_obs, 1.0)
    p_clipped = np.clip(p_obs, 0.01, 0.99)
    s_init = np.log(p_clipped / (1 - p_clipped))
    np.fill_diagonal(s_init, np.max(s_init))

    # Save count RSM for comparison
    np.save(OUTPUT_DIR / "count_rsm.npy", p_obs)

    # s is the parameter we optimize
    s = torch.tensor(s_init, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([s], lr=args.lr)

    train_t = torch.tensor(train, dtype=torch.long, device=device)
    test_t = torch.tensor(test, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(42)

    log.info(f"\nAdam, {args.n_epochs} epochs, batch_size={args.batch_size}")
    log.info("=" * 60)

    for epoch in range(args.n_epochs):
        perm = torch.randperm(len(train_t), generator=rng, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_t), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            batch = train_t[batch_idx]
            bi, bj, bk = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()

            sij = s[bi, bj]
            sik = s[bi, bk]
            sjk = s[bj, bk]

            logits = torch.stack([sij, sik, sjk], dim=1)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -log_probs[:, 0].mean()

            loss.backward()

            # Symmetrize gradient
            if s.grad is not None:
                s.grad.data = (s.grad.data + s.grad.data.T) / 2

            optimizer.step()

            with torch.no_grad():
                s.data = (s.data + s.data.T) / 2

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate on test (using the matrix directly)
        with torch.no_grad():
            ti, tj, tk = test_t[:, 0], test_t[:, 1], test_t[:, 2]
            test_sij = s[ti, tj]
            test_sik = s[ti, tk]
            test_sjk = s[tj, tk]
            test_acc = float(((test_sij > test_sik) & (test_sij > test_sjk)).float().mean().item())

            tri, trj, trk = train_t[:200000, 0], train_t[:200000, 1], train_t[:200000, 2]
            train_sij = s[tri, trj]
            train_sik = s[tri, trk]
            train_sjk = s[trj, trk]
            train_acc = float(((train_sij > train_sik) & (train_sij > train_sjk)).float().mean().item())

        log.info(f"  epoch {epoch}: loss={avg_loss:.6f}, train={train_acc:.4f}, test={test_acc:.4f}")

    # Save corrected RSM normalized to [0, 1]
    with torch.no_grad():
        s.data.fill_diagonal_(s.data.max())
    s_np = s.detach().cpu().numpy()
    np.fill_diagonal(s_np, np.nan)
    lo, hi = np.nanmin(s_np), np.nanmax(s_np)
    s_rsm = (s_np - lo) / (hi - lo)
    np.fill_diagonal(s_rsm, 1.0)

    np.save(OUTPUT_DIR / "s_corrected_rsm.npy", s_rsm)
    log.info(f"\nSaved. Now run run_eval_srf.py")


if __name__ == "__main__":
    main()
