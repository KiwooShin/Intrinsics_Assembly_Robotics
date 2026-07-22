"""Offline action-probe: does the deterministic L1 head blend descend-vs-search?

For each checkpoint (m3c/m3e/m3f) and each staged lateral-offset bucket (0/1/2 mm),
run the policy on stored EARLY-descent frames (where the true offset is still present)
and report the predicted first-action twist decomposed into lateral (search) vs
vertical (descend) components. No ROS, no sim, no training — inference only.

Expected if the mechanism is "L1 = conditional median averages two modes":
  m3c commands little lateral at any offset (blind descend -> explains 2 mm 0/3);
  m3e/m3f command an intermediate lateral that grows with offset but under-corrects
  (the blend), never a clean straight-vs-search switch.

Run: ~/miniconda3/bin/python -m opt.action_probe
"""
from __future__ import annotations

import glob
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/kiwoos/work/Intrinsics_Assembly_Robotics")
import train_v2 as tv2  # noqa: E402  (repo-root Policy)

def decompose_first_action(chunk: np.ndarray) -> tuple[float, float]:
    """Split the first-action twist into lateral (search) vs vertical (descend) speed.

    Args:
        chunk: A ``(K, 6)`` twist chunk ``[vx, vy, vz, wx, wy, wz]`` per step.

    Returns:
        ``(lateral, vertical)`` where ``lateral = hypot(vx, vy)`` (>= 0) and
        ``vertical = vz`` (signed; negative = descending) of the FIRST step.

    Raises:
        ValueError: If ``chunk`` is not shaped ``(K, 6)`` with ``K >= 1``.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 6 or chunk.shape[0] < 1:
        raise ValueError(f"chunk must be (K,6) with K>=1, got {chunk.shape}")
    a0 = chunk[0]
    return float(np.hypot(a0[0], a0[1])), float(a0[2])


HOME = "/home/kiwoos"
CKPTS = {
    "m3c": f"{HOME}/training/ckpt/insert_m3c_wrench_k4.pt",
    "m3e": f"{HOME}/training/ckpt/insert_m3e_wrench_k4.pt",
    "m3f": f"{HOME}/training/ckpt/insert_m3f_wrench_k4.pt",
}
# Offset buckets -> episode-dir globs (staged lateral offset is ground-truth here).
BUCKETS = {
    "0.0mm": [f"{HOME}/training/ds_phase0/ep_*", f"{HOME}/training/ds_phase2/ep_*"],
    "1.0mm": [f"{HOME}/training/ds_curr/ep_curr_lat1_*"],
    "2.0mm": [f"{HOME}/training/ds_curr/ep_curr_lat2_*",
              f"{HOME}/training/ds_curr_late/ep_curr_lat2_*",
              f"{HOME}/training/ds_curr_late2/ep_curr_lat2_*"],
}
EPS_PER_BUCKET = 6      # episodes sampled per bucket
EARLY_FRAC = 0.30       # use frames in [0, EARLY_FRAC*seat] — offset still present
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    K, sd = int(ck["K"]), int(ck.get("state_dim", 7))
    img = int(ck.get("img", 128))
    m = tv2.Policy(K, state_dim=sd).to(DEV).eval()
    m.load_state_dict(ck["model"])
    stats = {k: ck[k].to(DEV) for k in ("smean", "sstd", "amean", "astd")}
    return m, K, sd, img, stats


def prep_img(arr_u8: np.ndarray, img: int) -> torch.Tensor:
    """Stored (H,W,3) uint8 -> normalized (1,3,img,img), mirroring DeployACT._img."""
    t = torch.from_numpy(arr_u8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEV)
    t = F.interpolate(t, size=(img, img), mode="bilinear", align_corners=False)
    return (t - 0.5) / 0.5


def episode_frames(ep: str):
    try:
        c = np.load(f"{ep}/center_images.npy"); l = np.load(f"{ep}/left_images.npy")
        r = np.load(f"{ep}/right_images.npy"); pose = np.load(f"{ep}/tcp_poses.npy")
        try:
            seat = int(np.asarray(np.load(f"{ep}/insertion_frame.npy")).ravel()[0])
        except FileNotFoundError:
            seat = max(2, len(c) - 5)  # pre-marker episodes: seat is near the end
        wr = None
        try:
            wr = np.load(f"{ep}/wrenches.npy")
        except FileNotFoundError:
            pass
        return c, l, r, pose, wr, seat
    except (FileNotFoundError, ValueError):
        return None


@torch.inference_mode()
def probe():
    results = {}
    for name, path in CKPTS.items():
        m, K, sd, img, st = load_model(path)
        results[name] = {}
        for bucket, globs in BUCKETS.items():
            eps = sorted({d for g in globs for d in glob.glob(g)})[:EPS_PER_BUCKET]
            lat_all, vz_all = [], []
            n_frames = 0
            for ep in eps:
                fr = episode_frames(ep)
                if fr is None:
                    continue
                c, l, r, pose, wr, seat = fr
                hi = max(2, int(EARLY_FRAC * max(seat, 1)))
                idxs = np.linspace(0, min(hi, len(c) - 1), 6).round().astype(int)
                for i in idxs:
                    imgs = torch.stack([prep_img(l[i], img), prep_img(c[i], img),
                                        prep_img(r[i], img)], 1)
                    state = pose[i].astype(np.float32)
                    if sd == 13 and wr is not None:
                        state = np.concatenate([state, wr[i].astype(np.float32)])
                    s = torch.from_numpy(state).to(DEV).unsqueeze(0)
                    s = (s - st["smean"]) / st["sstd"]
                    out = m(imgs, s)
                    act = out[0] if isinstance(out, tuple) else out
                    chunk = (act[0] * st["astd"] + st["amean"]).cpu().numpy()  # (K,6)
                    lat, vert = decompose_first_action(chunk)
                    lat_all.append(lat)
                    vz_all.append(vert)
                    n_frames += 1
            lat = np.array(lat_all); vz = np.array(vz_all)
            results[name][bucket] = {
                "n_frames": n_frames,
                "lateral_mean_mm_s": round(float(lat.mean()) * 1000, 3) if len(lat) else None,
                "vertical_mean_mm_s": round(float(vz.mean()) * 1000, 3) if len(vz) else None,
                "abs_vert_mean_mm_s": round(float(np.abs(vz).mean()) * 1000, 3) if len(vz) else None,
                "lateral_over_vertical": round(float(lat.mean() / (np.abs(vz).mean() + 1e-9)), 3)
                if len(lat) else None,
            }
    return results


def main() -> None:
    res = probe()
    print("\n=== ACTION PROBE: predicted first-action twist by staged offset ===")
    print(f"(early-descent frames, {DEV}; lateral=|vx,vy|, vertical=vz; mm/s)\n")
    hdr = f"{'ckpt':5} {'offset':7} {'lat mm/s':>9} {'vert mm/s':>10} {'lat/|vert|':>11} {'n':>4}"
    print(hdr); print("-" * len(hdr))
    for name in CKPTS:
        for b in BUCKETS:
            d = res[name][b]
            print(f"{name:5} {b:7} {str(d['lateral_mean_mm_s']):>9} "
                  f"{str(d['vertical_mean_mm_s']):>10} {str(d['lateral_over_vertical']):>11} "
                  f"{d['n_frames']:>4}")
        print()
    out = "/home/kiwoos/work/Intrinsics_Assembly_Robotics/results/action_probe.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
