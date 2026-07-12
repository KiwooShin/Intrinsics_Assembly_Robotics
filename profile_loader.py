"""Profile dataloader vs GPU compute to locate the bottleneck.

Heavy imports (torch, train_smoke) are performed inside :func:`main` so the module
can be imported without torch/GPU for smoke tests.
"""
from __future__ import annotations

import glob
import time


def main() -> None:
    """Benchmark DataLoader throughput against raw GPU compute throughput."""
    import torch
    from torch.utils.data import DataLoader

    import train_smoke as T

    ds = T.EpisodeData(sorted(glob.glob('/home/kiwoos/training/smoke/ep_*')))
    N = len(ds)

    def time_loader(bs, workers, pin, persist):
        kw = dict(batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)
        if workers > 0:
            kw.update(persistent_workers=persist, prefetch_factor=4)
        dl = DataLoader(ds, **kw)
        for _ in dl:
            break                      # warm up workers
        t0 = time.time(); n = 0
        for imgs, st, act in dl:
            n += len(imgs)
        dt = time.time() - t0
        return dt, n / dt

    print(f"\nframes={N}")
    for w in [4, 8, 12]:
        dt, fps = time_loader(64, w, True, True)
        print(f"  loader-only  workers={w:2d} pin=T persist=T : {dt:.2f}s  {fps:6.0f} frames/s")

    # pure GPU compute: fixed batch, repeated fwd+bwd
    m = T.Policy().to(T.DEV); opt = torch.optim.AdamW(m.parameters(), 1e-4)
    imgs = torch.randn(64, 3, 3, T.IMG, T.IMG, device=T.DEV)
    st = torch.randn(64, 7, device=T.DEV); act = torch.randn(64, T.K, 6, device=T.DEV)
    for _ in range(5):
        opt.zero_grad(); loss = torch.nn.functional.l1_loss(m(imgs, st), act); loss.backward(); opt.step()
    torch.cuda.synchronize(); t0 = time.time(); ITER = 200
    for _ in range(ITER):
        opt.zero_grad(); loss = torch.nn.functional.l1_loss(m(imgs, st), act); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"\n  GPU compute  : {ITER} steps bs64 in {dt:.2f}s = {ITER * 64 / dt:6.0f} frames/s "
          f"({1000 * dt / ITER:.2f} ms/step)")
    print(f"\n  => full epoch ~{N / 64:.0f} steps; if loader<<compute we are compute-bound, else loader-bound")


if __name__ == '__main__':
    main()
