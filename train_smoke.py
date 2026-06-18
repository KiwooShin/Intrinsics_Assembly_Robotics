"""Smoke-test ACT-style imitation policy on a few episodes.
Inputs : 3 RGB cams + TCP pose ; Output: action chunk of K TCP velocities (6D).
Goal   : confirm data loads + training drives loss down (overfit). val == train on purpose.
"""
import sys, os, glob, argparse, time
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IMG = 128          # resized square input
K = 16             # action chunk length
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class EpisodeData(Dataset):
    def __init__(self, ep_dirs):
        self.frames = []          # (ep_idx, t)
        self.eps = []
        for d in ep_dirs:
            c = np.load(f'{d}/center_images.npy'); l = np.load(f'{d}/left_images.npy'); r = np.load(f'{d}/right_images.npy')
            pose = np.load(f'{d}/tcp_poses.npy').astype(np.float32)         # (N,7) state
            vel = np.load(f'{d}/tcp_velocities.npy').astype(np.float32)     # (N,6) action
            n = len(c)
            self.eps.append(dict(c=c, l=l, r=r, pose=pose, vel=vel, n=n))
            for t in range(n):
                self.frames.append((len(self.eps) - 1, t))
        allvel = np.concatenate([e['vel'] for e in self.eps])
        allpose = np.concatenate([e['pose'] for e in self.eps])
        self.amean, self.astd = allvel.mean(0), allvel.std(0) + 1e-6
        self.smean, self.sstd = allpose.mean(0), allpose.std(0) + 1e-6
        print(f"  episodes={len(self.eps)} frames={len(self.frames)} "
              f"action|v|range=[{allvel.min():.3f},{allvel.max():.3f}]")

    def __len__(self): return len(self.frames)

    def _img(self, a):  # HWC uint8 RGB -> CHW float normalized, resized
        t = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t[None], size=(IMG, IMG), mode='bilinear', align_corners=False)[0]
        return (t - 0.5) / 0.5

    def __getitem__(self, i):
        ei, t = self.frames[i]; e = self.eps[ei]
        imgs = torch.stack([self._img(e['l'][t]), self._img(e['c'][t]), self._img(e['r'][t])])  # (3,3,H,W)
        state = torch.from_numpy((e['pose'][t] - self.smean) / self.sstd)
        # action chunk t..t+K (pad with last)
        idx = np.clip(np.arange(t, t + K), 0, e['n'] - 1)
        act = (e['vel'][idx] - self.amean) / self.astd                       # (K,6)
        return imgs, state, torch.from_numpy(act).float()

class Encoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, out),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x): return self.net(x)

class Policy(nn.Module):
    def __init__(self, state_dim=7, feat=128):
        super().__init__()
        self.enc = Encoder(feat)
        self.head = nn.Sequential(nn.Linear(feat * 3 + state_dim, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, K * 6))
    def forward(self, imgs, state):
        B = imgs.shape[0]
        f = self.enc(imgs.view(B * 3, *imgs.shape[2:])).view(B, -1)          # (B, 3*feat)
        out = self.head(torch.cat([f, state], 1))
        return out.view(B, K, 6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', default=os.path.expanduser('~/training/smoke/ep_*'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    a = ap.parse_args()
    ep_dirs = sorted(glob.glob(a.eps))
    print(f"Loading episodes: {ep_dirs}")
    ds = EpisodeData(ep_dirs)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4, drop_last=False)
    model = Policy().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"device={DEV} params={nparam/1e6:.2f}M  K={K} IMG={IMG}\nTraining...")
    astd = torch.tensor(ds.astd, device=DEV)
    for ep in range(a.epochs):
        model.train(); tot = 0; n = 0; t0 = time.time()
        for imgs, st, act in dl:
            imgs, st, act = imgs.to(DEV), st.to(DEV), act.to(DEV)
            pred = model(imgs, st)
            loss = F.l1_loss(pred, act)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(imgs); n += len(imgs)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  train L1(norm)={tot/n:.4f}  ({time.time()-t0:.1f}s)")
    # validate on SAME data (overfit confirmation) + un-normalized error
    model.eval(); err = 0; n = 0; firststep = 0
    with torch.no_grad():
        for imgs, st, act in DataLoader(ds, batch_size=a.bs):
            imgs, st, act = imgs.to(DEV), st.to(DEV), act.to(DEV)
            pred = model(imgs, st)
            err += (F.l1_loss(pred, act, reduction='none').mean((1, 2)) * 1).sum().item(); n += len(imgs)
            firststep += ((pred[:, 0] - act[:, 0]).abs() * astd).mean(1).sum().item()
    print(f"\n=== VALIDATION (same 5 episodes) ===")
    print(f"  mean L1 (normalized) over chunk : {err/n:.4f}")
    print(f"  mean |err| first action (m/s)   : {firststep/n:.5f}  (action range ~0.05 m/s)")
    os.makedirs(os.path.expanduser('~/training/ckpt'), exist_ok=True)
    torch.save({'model': model.state_dict(), 'amean': ds.amean, 'astd': ds.astd,
                'smean': ds.smean, 'sstd': ds.sstd}, os.path.expanduser('~/training/ckpt/smoke.pt'))
    print("  saved ~/training/ckpt/smoke.pt")

if __name__ == '__main__':
    main()
