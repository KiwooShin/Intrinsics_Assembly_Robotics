"""Verify a converted single-episode dataset is well-formed."""
import sys, os
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import numpy as np
import cv2

d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/training/one_episode')
print(f"=== Verifying {d} ===")
files = ['center_images', 'left_images', 'right_images', 'tcp_velocities', 'tcp_poses', 'timestamps']
arr = {}
for f in files:
    p = f"{d}/{f}.npy"
    if not os.path.exists(p):
        print(f"  MISSING: {f}.npy"); continue
    arr[f] = np.load(p)
    print(f"  {f:16s} shape={str(arr[f].shape):22s} dtype={arr[f].dtype}")

if 'center_images' not in arr:
    print("FAIL: no images"); sys.exit(1)

N = arr['center_images'].shape[0]
print(f"\nFrames: {N}")

# length consistency
lens = {k: v.shape[0] for k, v in arr.items()}
print(f"length match: {len(set(lens.values()))==1}  ({lens})")

# timestamps -> duration & rate
if 'timestamps' in arr and N > 1:
    t = arr['timestamps']; dur = t[-1]-t[0]
    print(f"duration: {dur:.2f}s  effective rate: {N/dur:.1f} Hz  monotonic: {bool(np.all(np.diff(t)>=0))}")

# image content: not blank/constant
for cam in ['center_images', 'left_images', 'right_images']:
    if cam in arr:
        im = arr[cam]
        print(f"{cam}: min={im.min()} max={im.max()} mean={im.mean():.1f} std={im.std():.1f}  "
              f"frame0!=frameLast: {not np.array_equal(im[0], im[-1])}")

# action (tcp_velocity) — should vary (robot moves) then settle
if 'tcp_velocities' in arr:
    v = arr['tcp_velocities']
    sp = np.linalg.norm(v[:, :3], axis=1)
    print(f"\ntcp_velocity linear speed: mean={sp.mean():.4f} max={sp.max():.4f} m/s  "
          f"nonzero frames: {(sp>1e-3).sum()}/{N}")
    print(f"  per-axis vel range: { {i: (round(float(v[:,i].min()),3), round(float(v[:,i].max()),3)) for i in range(6)} }")

# pose travel — did the TCP actually move across the episode?
if 'tcp_poses' in arr:
    p = arr['tcp_poses']
    travel = np.linalg.norm(p[-1, :3]-p[0, :3])
    print(f"TCP net travel: {travel*1000:.1f} mm  (start->end position)")

# dump a montage of a few frames for visual check
try:
    idxs = [0, N//4, N//2, 3*N//4, N-1]
    rows = []
    for cam in ['left_images', 'center_images', 'right_images']:
        if cam in arr:
            rows.append(np.hstack([arr[cam][i] for i in idxs]))
    if rows:
        montage = np.vstack(rows)
        out = f"{d}/_verify_montage.png"
        cv2.imwrite(out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"\nMontage (rows=L/C/R, cols=t0..tEnd) -> {out}")
except Exception as e:
    print(f"montage failed: {e}")
print("=== done ===")
