"""Flexible train/val experiment: explicit disjoint train & val episode sets.
Reuses model + loader from train_v2. Reports val L1 and first-action error (m/s).
"""
import sys, os, glob, time, argparse
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import torch, torch.nn.functional as F
import train_v2 as TV
DEV = TV.DEV

def expand(spec):
    out = []
    for part in spec.split(','):
        out += sorted(glob.glob(part.strip()))
    return out

def main():
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

    val_dirs = expand(a.val)
    train_dirs = [d for d in expand(a.train) if d not in set(val_dirs)]
    print(f"[{a.tag}] train={len(train_dirs)} eps  val={len(val_dirs)} eps")
    print(f"    train: {[os.path.basename(d) for d in train_dirs]}")
    print(f"    val  : {[os.path.basename(d) for d in val_dirs]}")

    IMt, STt, ACt, _ = TV.load_all(train_dirs, a.img, a.k)
    IMv, STv, ACv, _ = TV.load_all(val_dirs, a.img, a.k)
    smean, sstd = STt.mean(0), STt.std(0) + 1e-6
    amean, astd = ACt.reshape(-1, 6).mean(0), ACt.reshape(-1, 6).std(0) + 1e-6
    STt = (STt - smean) / sstd; STv = (STv - smean) / sstd
    ACt = (ACt - amean) / astd; ACv = (ACv - amean) / astd

    m = TV.Policy(a.k).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), a.lr)

    def epoch(IM, ST, AC, train):
        m.train(train)
        idx = torch.randperm(len(IM), device=DEV) if train else torch.arange(len(IM), device=DEV)
        tot = torch.zeros((), device=DEV)
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
