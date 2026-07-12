"""Optimization sub-package for the AIC cable-insertion project.

Modules:
  config      -- frozen dataclasses shared by the trainer / benchmark / sweep.
  asha        -- pure successive-halving (ASHA-style) scheduler math + leaderboard.
  dataset     -- pure numpy data helpers (chunking, normalization, splits).
  train_v3    -- improved ACT-lite trainer (fused AdamW, max-autotune, EMA, fp8-opt).
  bench       -- reproducible train-throughput + inference-latency benchmark CLI.
  sweep       -- budgeted hyperparameter search harness (ASHA-style early stopping).

The pure modules (config, asha, dataset) import only the standard library and
numpy so their unit tests run CPU-only, without torch/ROS/GPU.
"""
