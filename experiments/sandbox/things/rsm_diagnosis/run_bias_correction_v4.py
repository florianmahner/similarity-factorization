"""Step 1: Optimize RSM with cross-entropy loss. Save corrected matrix."""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def build_counts_shown(triplets: np.ndarray, n: int):
    ii, jj, kk = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)
    return counts, shown


def _chunk_contrib(s, k_slice):
    s_ik = s[:, k_slice].unsqueeze(1)
    s_jk = s[:, k_slice].unsqueeze(0)
    s_ij = s.unsqueeze(2)
    d_ik = (s_ik - s_ij).clamp(-50, 50)
    d_jk = (s_jk - s_ij).clamp(-50, 50)
    return (1.0 / (1.0 + torch.exp(d_ik) + torch.exp(d_jk))).sum(dim=2)


def compute_p_hat(s: torch.Tensor, batch_size: int = 20) -> torch.Tensor:
    n = s.shape[0]
    chunks = []
    for k_start in range(0, n, batch_size):
        k_end = min(k_start + batch_size, n)
        chunks.append(checkpoint(
            _chunk_contrib, s, slice(k_start, k_end), use_reentrant=False
        ))
    p_hat = torch.stack(chunks).sum(dim=0)
    diag = s.diag()
    d_ii = (diag.unsqueeze(1) - s).clamp(-50, 50)
    p_hat = p_hat - 1.0 / (1.0 + torch.exp(d_ii) + 1.0)
    d_jj = (diag.unsqueeze(0) - s).clamp(-50, 50)
    p_hat = p_hat - 1.0 / (1.0 + 1.0 + torch.exp(d_jj))
    return p_hat / (n - 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.empty_cache()
    log.info(f"Device: {device}, lr={args.lr}, iters={args.n_iters}")

    train = load_triplets(DATA_DIR / "train_90.txt")
    counts, shown = build_counts_shown(train, N)
    observed = shown > 0

    p_obs_np = np.divide(counts + 1, shown + 2, out=np.full((N, N), 0.5), where=observed)
    np.fill_diagonal(p_obs_np, 1.0)

    counts_t = torch.tensor(counts, dtype=torch.float32, device=device)
    shown_t = torch.tensor(shown, dtype=torch.float32, device=device)
    obs_mask = torch.tensor(observed, dtype=torch.bool, device=device)

    p_clipped = np.clip(p_obs_np, 0.01, 0.99)
    s_init = np.log(p_clipped / (1 - p_clipped))
    np.fill_diagonal(s_init, np.max(s_init))

    s = torch.tensor(s_init, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([s], lr=args.lr)

    log.info("Optimizing RSM...")
    for it in range(args.n_iters):
        optimizer.zero_grad()
        p_hat = compute_p_hat(s)
        p_clamped = p_hat.clamp(1e-6, 1 - 1e-6)
        nll = -(counts_t * torch.log(p_clamped) +
                (shown_t - counts_t) * torch.log(1 - p_clamped))
        loss = nll[obs_mask].sum() / shown_t[obs_mask].sum()
        loss.backward()
        if s.grad is not None:
            s.grad.data = (s.grad.data + s.grad.data.T) / 2
        optimizer.step()
        with torch.no_grad():
            s.data = (s.data + s.data.T) / 2
            s.data.fill_diagonal_(s.data.max())

        if it % 10 == 0 or it < 5:
            log.info(f"  iter {it:3d}: nll={loss.item():.6f}")

        del p_hat, p_clamped, nll, loss
        torch.cuda.empty_cache()

    # Save corrected RSM normalized to [0, 1]
    s_np = s.detach().cpu().numpy()
    np.fill_diagonal(s_np, np.nan)
    lo, hi = np.nanmin(s_np), np.nanmax(s_np)
    s_rsm = (s_np - lo) / (hi - lo)
    np.fill_diagonal(s_rsm, 1.0)

    np.save(OUTPUT_DIR / "s_corrected_rsm.npy", s_rsm)
    np.save(OUTPUT_DIR / "count_rsm.npy", p_obs_np)
    log.info(f"Saved corrected RSM to {OUTPUT_DIR / 's_corrected_rsm.npy'}")
    log.info("Done. Now run run_eval_srf.py to evaluate.")


if __name__ == "__main__":
    main()
