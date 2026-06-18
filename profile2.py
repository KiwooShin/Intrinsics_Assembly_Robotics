"""Benchmark compute optimizations (batch, bf16 AMP, channels_last, compile) + cached loader."""
import sys, glob, time
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import torch, torch.nn.functional as F
import train_smoke as T
DEV = T.DEV

def bench(bs, amp=False, clast=False, compile_=False, iters=150):
    m = T.Policy().to(DEV)
    if clast: m = m.to(memory_format=torch.channels_last)
    if compile_: m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    imgs = torch.randn(bs, 3, 3, T.IMG, T.IMG, device=DEV)
    st = torch.randn(bs, 7, device=DEV); act = torch.randn(bs, T.K, 6, device=DEV)
    def step():
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.l1_loss(m(imgs, st), act)
        else:
            loss = F.l1_loss(m(imgs, st), act)
        loss.backward(); opt.step()
    for _ in range(10): step()              # warmup (compile builds here)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize(); dt = time.time() - t0
    return iters * bs / dt, 1000 * dt / iters

print("compute throughput (frames/s | ms/step):")
for name, kw in [("bs64 baseline", dict(bs=64)),
                 ("bs256", dict(bs=256)),
                 ("bs256 +bf16", dict(bs=256, amp=True)),
                 ("bs256 +bf16 +chlast", dict(bs=256, amp=True, clast=True)),
                 ("bs512 +bf16 +chlast", dict(bs=512, amp=True, clast=True)),
                 ("bs256 +bf16 +chlast +compile", dict(bs=256, amp=True, clast=True, compile_=True))]:
    try:
        fps, ms = bench(**kw); print(f"  {name:32s}: {fps:8.0f} fr/s  {ms:6.2f} ms/step")
    except Exception as e:
        print(f"  {name:32s}: ERR {e}")

# cached loader: pre-resize to IMG once, __getitem__ only normalizes
print("\ncached-loader test:")
from torch.utils.data import Dataset, DataLoader
import numpy as np
class Cached(Dataset):
    def __init__(self, ep_dirs):
        self.imgs=[]; self.st=[]; self.act=[]
        base = T.EpisodeData(ep_dirs)
        # pre-resize all frames once
        for i in range(len(base)):
            im, s, a = base[i]; self.imgs.append((im*0+im).half()); self.st.append(s); self.act.append(a)
        self.imgs=torch.stack(self.imgs); self.st=torch.stack(self.st); self.act=torch.stack(self.act)
        print(f"  cached tensor {tuple(self.imgs.shape)} {self.imgs.dtype} "
              f"({self.imgs.element_size()*self.imgs.nelement()/1e9:.2f} GB)")
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.imgs[i].float(), self.st[i], self.act[i]
c = Cached(sorted(glob.glob('/home/kiwoos/training/smoke/ep_*')))
dl = DataLoader(c, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
for _ in dl: break
t0=time.time(); n=0
for im,s,a in dl: n+=len(im)
print(f"  cached loader (in-RAM, nw=0): {n/(time.time()-t0):.0f} frames/s")
