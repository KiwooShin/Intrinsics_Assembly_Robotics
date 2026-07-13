"""GPU-native input regularizers for ACT-lite training (W1 experiments).

Two cheap, literature-backed regularizers screened by the W1 sub-agent:

  * :func:`random_shift` -- DrQ-style pad-by-``P`` + random crop-back image
    augmentation (Kostrikov et al., "Image Augmentation Is All You Need",
    arXiv:2004.13649). robomimic (Mandlekar et al., arXiv:2108.03298) reports it
    is the single most important augmentation for visuomotor BC (73.3% -> 26.7%
    success when removed). Implemented as replicate-pad then integer random crop
    so it is exact (no interpolation), fully vectorized on the GPU, and
    deterministic under a supplied ``torch.Generator``.
  * :func:`proprio_dropout` -- per-sample zeroing of the proprioceptive state
    with probability ``p`` during training only, to prevent the "state shortcut"
    and force visual reliance (Xie et al., arXiv:2509.18644). Because the state is
    fed to the model *after* mean/std normalization, zeroing is exactly
    "replace with the dataset mean" -- a neutral, information-free value. At
    eval/deploy the true state is always present; this train/test mismatch is
    intentional, matching the paper.

Both functions preserve the input dtype, shape, and device, apply no in-place
mutation to their argument, and are safe to call every mini-batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_shift(
    images: torch.Tensor,
    pad: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply DrQ-style pad + random-crop image augmentation.

    Each image is replicate-padded by ``pad`` pixels on every side and then
    cropped back to its original ``(H, W)`` at a random integer offset, so the
    content is shifted by up to ``+/-pad`` pixels. The shift is drawn
    independently per leading index (e.g. per frame and per camera), while the
    channel dimension immediately before ``(H, W)`` shares one shift so the RGB
    planes of a single image stay aligned. The crop always lies fully inside the
    padded image, so no out-of-bounds sampling can occur.

    Args:
        images: Tensor shaped ``(..., C, H, W)`` (at least 3 dims). The two
            trailing axes are spatial; the third-from-last is the channel axis
            that shares a shift. For this codebase the shape is
            ``(B, cameras, channels, H, W)`` -> one shift per ``(B, camera)``.
        pad: Non-negative padding/crop radius in pixels. ``0`` is a no-op that
            returns ``images`` unchanged.
        generator: Optional RNG for reproducible shifts. Must live on the same
            device as ``images``. When ``None`` the global default RNG is used.

    Returns:
        A new tensor with the same shape, dtype, and device as ``images``, with
        each image randomly shifted.

    Raises:
        ValueError: If ``pad`` is negative or ``images`` has fewer than 3 dims.
    """
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    if images.ndim < 3:
        raise ValueError(f"images must be (..., C, H, W) with >=3 dims, got {images.shape}")
    if pad == 0:
        return images

    lead = images.shape[:-3]
    c, h, w = images.shape[-3:]
    m = 1
    for s in lead:
        m *= s
    x = images.reshape(m, c, h, w)

    # Replicate-pad the spatial dims, then integer-crop back at a random offset.
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")  # (m, c, h+2p, w+2p)
    hp, wp = h + 2 * pad, w + 2 * pad
    high = 2 * pad + 1  # offsets in [0, 2*pad] inclusive
    dy = torch.randint(0, high, (m,), generator=generator, device=images.device)
    dx = torch.randint(0, high, (m,), generator=generator, device=images.device)

    ar_h = torch.arange(h, device=images.device)
    ar_w = torch.arange(w, device=images.device)
    rows = dy[:, None] + ar_h[None, :]  # (m, h), int64
    cols = dx[:, None] + ar_w[None, :]  # (m, w), int64

    # Two-step gather: first select rows (dim 2), then columns (dim 3).
    idx_r = rows[:, None, :, None].expand(m, c, h, wp)
    x_rows = torch.gather(x_pad, 2, idx_r)  # (m, c, h, w+2p)
    idx_c = cols[:, None, None, :].expand(m, c, h, w)
    out = torch.gather(x_rows, 3, idx_c)  # (m, c, h, w)

    return out.reshape(*lead, c, h, w)


def proprio_dropout(
    state: torch.Tensor,
    p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero the proprioceptive state per sample with probability ``p``.

    With probability ``p`` an entire sample's state vector is set to zero;
    otherwise it passes through unchanged. Because the state is normalized
    (mean 0, std 1) before this call, zeroing sets it to the dataset mean -- an
    information-free neutral value -- which is exactly the "drop proprioception"
    signal that forces the policy to rely on vision (arXiv:2509.18644). No
    inverted-dropout rescaling is applied: eval/deploy always supplies the true
    state, and that train/test mismatch is intentional.

    Args:
        state: Normalized state tensor shaped ``(B, D)``.
        p: Per-sample drop probability in ``[0, 1]``. ``0`` is a no-op.
        generator: Optional RNG for reproducible masks. Must live on the same
            device as ``state``. When ``None`` the global default RNG is used.

    Returns:
        A new tensor with the same shape, dtype, and device as ``state``, with a
        random subset of rows zeroed.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]`` or ``state`` is not 2-D.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if state.ndim != 2:
        raise ValueError(f"state must be 2-D (B, D), got shape {tuple(state.shape)}")
    if p == 0.0:
        return state

    b = state.shape[0]
    # keep with probability (1 - p); rand() is in [0, 1) so p=1 zeros everything.
    keep = torch.rand(b, generator=generator, device=state.device) >= p
    return state * keep.to(state.dtype)[:, None]
