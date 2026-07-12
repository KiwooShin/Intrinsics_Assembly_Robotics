"""Efficient ACT-style trainer: entire (small) dataset on GPU, bf16 + channels_last + compile.
Supports overfit (val==train) and proper held-out-episode validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclasses.dataclass
class NormStats:
    """Per-dimension normalization statistics for state and action tensors.

    Attributes:
        smean: State mean, shaped ``(state_dim,)``.
        sstd: State standard deviation (already ``+ eps``), shaped ``(state_dim,)``.
        amean: Action mean, shaped ``(action_dim,)``.
        astd: Action standard deviation (already ``+ eps``), shaped ``(action_dim,)``.
    """

    smean: torch.Tensor
    sstd: torch.Tensor
    amean: torch.Tensor
    astd: torch.Tensor


def build_action_chunks(actions: np.ndarray, k: int) -> np.ndarray:
    """Build per-frame action chunks, padding past the episode end with the last action.

    For an episode with ``n`` frames, frame ``t`` maps to the chunk
    ``[actions[t], actions[t + 1], ..., actions[t + k - 1]]`` where indices beyond
    ``n - 1`` are clamped to the last frame (ACT-style "repeat last action" padding).

    Args:
        actions: Action array shaped ``(n, action_dim)``.
        k: Chunk length (number of future actions per frame). Must be >= 1.

    Returns:
        An array shaped ``(n, k, action_dim)``.

    Raises:
        ValueError: If ``k`` < 1.
    """
    if k < 1:
        raise ValueError(f"chunk length k must be >= 1, got {k}")
    n = len(actions)
    idx = np.clip(np.arange(n)[:, None] + np.arange(k)[None, :], 0, n - 1)  # (n,k)
    return actions[idx]


def compute_norm_stats(
    state: torch.Tensor, act: torch.Tensor, eps: float = 1e-6
) -> NormStats:
    """Compute mean/std normalization statistics over a set of frames.

    Args:
        state: State tensor shaped ``(M, state_dim)``.
        act: Action tensor shaped ``(M, K, action_dim)``; flattened over the chunk
            axis before computing statistics.
        eps: Small constant added to every standard deviation to avoid divide-by-zero.

    Returns:
        A :class:`NormStats` with means and (eps-stabilised) standard deviations.
    """
    smean, sstd = state.mean(0), state.std(0) + eps
    a = act.reshape(-1, act.shape[-1])
    amean, astd = a.mean(0), a.std(0) + eps
    return NormStats(smean=smean, sstd=sstd, amean=amean, astd=astd)


def save_checkpoint(
    path: str, model_state: dict, stats: NormStats, k: int, img: int
) -> None:
    """Save model weights and normalization stats to ``path`` (CPU tensors).

    Args:
        path: Destination ``.pt`` file (parent directories are created).
        model_state: The model ``state_dict`` to save.
        stats: Normalization statistics to embed for inference-time de-normalization.
        k: Action chunk length used during training.
        img: Square input image size used during training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'model': model_state,
            'amean': stats.amean.cpu(), 'astd': stats.astd.cpu(),
            'smean': stats.smean.cpu(), 'sstd': stats.sstd.cpu(),
            'K': k, 'img': img,
        },
        path,
    )


def load_all(ep_dirs, img, K):
    """Return GPU tensors: imgs (N,3,3,img,img) fp16, state (N,7), act (N,K,6), and ep id per frame."""
    imgs, states, acts, epid = [], [], [], []
    for ei, d in enumerate(ep_dirs):
        c = np.load(f'{d}/center_images.npy'); l = np.load(f'{d}/left_images.npy'); r = np.load(f'{d}/right_images.npy')
        pose = np.load(f'{d}/tcp_poses.npy').astype(np.float32)
        vel = np.load(f'{d}/tcp_velocities.npy').astype(np.float32)
        n = len(c)
        # batch-resize on GPU per camera
        cam = []
        for arr in (l, c, r):
            t = torch.from_numpy(arr).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            t = F.interpolate(t, size=(img, img), mode='bilinear', align_corners=False)
            cam.append(((t - 0.5) / 0.5).half())
        imgs.append(torch.stack(cam, 1))                      # (n,3,3,img,img)
        states.append(torch.from_numpy(pose).to(DEV))
        acts.append(torch.from_numpy(build_action_chunks(vel, K)).to(DEV))  # (n,K,6)
        epid.append(torch.full((n,), ei, device=DEV))
    return (torch.cat(imgs), torch.cat(states), torch.cat(acts), torch.cat(epid))

class Encoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, out),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x): return self.net(x)

class Policy(nn.Module):
    def __init__(self, K, state_dim=7, feat=128):
        super().__init__(); self.K = K
        self.enc = Encoder(feat)
        self.head = nn.Sequential(nn.Linear(feat * 3 + state_dim, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, K * 6))
    def forward(self, imgs, state):
        B = imgs.shape[0]
        x = imgs.view(B * 3, *imgs.shape[2:]).to(memory_format=torch.channels_last)
        f = self.enc(x).view(B, -1)
        return self.head(torch.cat([f, state], 1)).view(B, self.K, 6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', default='/home/kiwoos/training/smoke/ep_*')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img', type=int, default=128)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--val_episodes', type=int, default=0, help='hold out last N episodes for validation')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--out', default='/home/kiwoos/training/ckpt/v2.pt')
    a = ap.parse_args()
    ep_dirs = sorted(d for part in a.eps.split(',') for d in glob.glob(part.strip()))
    imgs, state, act, epid = load_all(ep_dirs, a.img, a.k)
    N = len(imgs); nep = len(ep_dirs)
    gb = imgs.element_size() * imgs.nelement() / 1e9
    # split by episode
    if a.val_episodes > 0:
        val_eps = set(range(nep - a.val_episodes, nep))
        tr = (~torch.isin(epid, torch.tensor(list(val_eps), device=DEV))).nonzero(as_tuple=True)[0]
        va = torch.isin(epid, torch.tensor(list(val_eps), device=DEV)).nonzero(as_tuple=True)[0]
    else:
        tr = torch.arange(N, device=DEV); va = tr      # overfit: val == train
    # normalize on train
    ns = compute_norm_stats(state[tr], act[tr])
    smean, sstd, amean, astd = ns.smean, ns.sstd, ns.amean, ns.astd
    stn = (state - smean) / sstd
    actn = (act - amean) / astd
    print(f"frames={N} ({gb:.2f}GB on GPU) eps={nep} train={len(tr)} val={len(va)} "
          f"val_episodes={a.val_episodes} bs={a.bs} K={a.k}")

    m = Policy(a.k).to(DEV)
    if a.compile: m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), a.lr)

    def run_epoch(idx, train):
        m.train(train)
        if train: idx = idx[torch.randperm(len(idx), device=DEV)]
        tot = torch.zeros((), device=DEV)
        for i in range(0, len(idx), a.bs):
            b = idx[i:i + a.bs]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = m(imgs[b], stn[b]); loss = F.l1_loss(pred, actn[b])
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.detach() * len(b)           # accumulate on-GPU, no per-step sync
        return (tot / len(idx)).item()

    torch.cuda.synchronize(); t0 = time.time()
    for ep in range(a.epochs):
        trl = run_epoch(tr, True)
        if ep % 5 == 0 or ep == a.epochs - 1:
            with torch.no_grad(): val = run_epoch(va, False)
            print(f"  epoch {ep:3d}  train={trl:.4f}  val={val:.4f}")
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"\nthroughput: {a.epochs*len(tr)/dt:.0f} frames/s ({1000*dt/a.epochs:.1f} ms/epoch)")

    # un-normalized first-action error on val
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        err = 0.0
        for i in range(0, len(va), a.bs):
            b = va[i:i + a.bs]; pred = m(imgs[b], stn[b]).float()
            err += ((pred[:, 0] - actn[b][:, 0]).abs() * astd).mean(1).sum().item()
    print(f"val mean |err| first action: {err/len(va):.5f} m/s  (range ~0.05)")
    sd = m._orig_mod.state_dict() if hasattr(m, '_orig_mod') else m.state_dict()
    save_checkpoint(a.out, sd, ns, a.k, a.img)
    print(f"saved {a.out}")

if __name__ == '__main__':
    main()
