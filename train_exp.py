"""Flexible train/val experiment: explicit disjoint train & val episode sets.
Reuses model + loader from train_v2. Reports val L1 and first-action error (m/s).

The globbing / disjoint-split helpers are pure and importable without torch; the
heavy imports (torch, train_v2) are performed lazily inside :func:`main` so the
split logic can be unit tested on any interpreter.
"""
from __future__ import annotations

import argparse
import glob
import os


def expand(spec: str) -> list[str]:
    """Expand a comma-separated list of globs into a flat, per-glob-sorted path list.

    Args:
        spec: Comma-separated glob patterns, e.g. ``"~/ds/ep_*, ~/ds/extra_*"``.

    Returns:
        The concatenation of ``sorted(glob.glob(part))`` for each comma-separated part.
    """
    out: list[str] = []
    for part in spec.split(','):
        out += sorted(glob.glob(part.strip()))
    return out


def split_train_val(train_spec: str, val_spec: str) -> tuple[list[str], list[str]]:
    """Resolve train/val episode directories, excluding val episodes from train.

    Args:
        train_spec: Comma-separated globs for the training episodes.
        val_spec: Comma-separated globs for the validation episodes; any path that
            also matches ``train_spec`` is removed from the training set so the two
            sets are disjoint.

    Returns:
        A ``(train_dirs, val_dirs)`` tuple. ``train_dirs`` contains no element of
        ``val_dirs``.
    """
    val_dirs = expand(val_spec)
    val_set = set(val_dirs)
    train_dirs = [d for d in expand(train_spec) if d not in val_set]
    return train_dirs, val_dirs


def main() -> None:
    """Parse CLI args, train on the disjoint split, and report val metrics."""
    import torch
    import torch.nn.functional as F

    import train_v2 as TV

    dev = TV.DEV

    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='comma-separated globs')
    ap.add_argument('--val', required=True, help='comma-separated globs (excluded from train)')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img', type=int, default=128)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--tag', default='exp')
    a = ap.parse_args()

    train_dirs, val_dirs = split_train_val(a.train, a.val)
    print(f"[{a.tag}] train={len(train_dirs)} eps  val={len(val_dirs)} eps")
    print(f"    train: {[os.path.basename(d) for d in train_dirs]}")
    print(f"    val  : {[os.path.basename(d) for d in val_dirs]}")

    IMt, STt, ACt, _ = TV.load_all(train_dirs, a.img, a.k)
    IMv, STv, ACv, _ = TV.load_all(val_dirs, a.img, a.k)
    ns = TV.compute_norm_stats(STt, ACt)
    smean, sstd, amean, astd = ns.smean, ns.sstd, ns.amean, ns.astd
    STt = (STt - smean) / sstd; STv = (STv - smean) / sstd
    ACt = (ACt - amean) / astd; ACv = (ACv - amean) / astd

    m = TV.Policy(a.k).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr)

    def epoch(IM, ST, AC, train):
        m.train(train)
        idx = torch.randperm(len(IM), device=dev) if train else torch.arange(len(IM), device=dev)
        tot = torch.zeros((), device=dev)
        for i in range(0, len(idx), a.bs):
            b = idx[i:i + a.bs]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.l1_loss(m(IM[b], ST[b]), AC[b])
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.detach() * len(b)
        return (tot / len(idx)).item()

    def val_firststep():
        m.eval(); err = 0.0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(IMv), a.bs):
                b = slice(i, i + a.bs)
                pred = m(IMv[b], STv[b]).float()
                err += ((pred[:, 0] - ACv[b][:, 0]).abs() * astd).mean(1).sum().item()
        return err / len(IMv)

    best = 9e9
    for ep in range(a.epochs):
        trl = epoch(IMt, STt, ACt, True)
        if ep % 10 == 0 or ep == a.epochs - 1:
            with torch.no_grad(): vl = epoch(IMv, STv, ACv, False)
            fs = val_firststep(); best = min(best, fs)
            print(f"  ep {ep:3d}  train={trl:.4f}  val={vl:.4f}  val_first|err|={fs:.5f} m/s")
    print(f"[{a.tag}] BEST val first-action |err| = {best:.5f} m/s  (action range ~0.05)\n")


if __name__ == '__main__':
    main()
