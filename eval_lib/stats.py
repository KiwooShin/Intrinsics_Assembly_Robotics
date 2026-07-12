"""Pure-numpy statistics for the matched-seed evaluation harness.

Implements the estimators recommended for small-sample policy comparison in
rliable (arXiv:2108.13264): the interquartile mean (IQM), percentile bootstrap
confidence intervals (plain, stratified, and paired), and the Wilson score
interval for binomial success rates. No third-party dependency beyond numpy;
in particular ``rliable`` and ``scipy`` are intentionally NOT required.

All estimators are deterministic given an explicit ``seed`` so that a reported
confidence interval can be reproduced byte-for-byte.
"""

from __future__ import annotations

import dataclasses

import numpy as np

DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05
WILSON_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclasses.dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate together with a two-sided confidence interval.

    Attributes:
        point: The statistic evaluated on the observed sample.
        lo: Lower confidence bound.
        hi: Upper confidence bound.
        level: Nominal coverage of the interval (e.g. ``0.95``).
    """

    point: float
    lo: float
    hi: float
    level: float = 0.95

    def excludes(self, value: float) -> bool:
        """Return whether ``value`` lies strictly outside the interval.

        Args:
            value: The reference value to test (typically ``0.0``).

        Returns:
            True if ``value`` is below ``lo`` or above ``hi``.
        """
        return value < self.lo or value > self.hi

    def as_tuple(self) -> tuple[float, float]:
        """Return the interval as a ``(lo, hi)`` tuple."""
        return (self.lo, self.hi)


def mean_stat(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Arithmetic mean, usable as a bootstrap statistic.

    Args:
        x: Sample values (1-D) or a stack of resamples (2-D).
        axis: Axis to reduce; ``None`` reduces the whole array.

    Returns:
        The mean along ``axis`` (scalar when ``axis is None``).
    """
    return np.mean(x, axis=axis)


def iqm(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Interquartile mean: the mean of the central 50% of the data.

    Matches ``scipy.stats.trim_mean(x, 0.25)``: exactly ``int(0.25 * n)``
    elements are dropped from each tail before averaging. This is the robust
    location estimator advocated by rliable (arXiv:2108.13264) because it is far
    less sensitive to a few catastrophic-failure or lucky-success rollouts than
    the mean while remaining unbiased under symmetric noise.

    Args:
        x: Sample values (1-D) or a stack of resamples (2-D). For 2-D input the
            statistic is computed independently along ``axis``.
        axis: Axis to reduce; ``None`` treats ``x`` as a flat 1-D sample.

    Returns:
        The IQM along ``axis`` (scalar when the result is 0-D). For fewer than
        four values there is nothing to trim, so the plain mean is returned
        (matching ``scipy.stats.trim_mean``).

    Raises:
        ValueError: If the reduced dimension is empty.
    """
    arr = np.asarray(x, dtype=float)
    if axis is None:
        arr = arr.ravel()
        work_axis = 0
    else:
        work_axis = axis
    n = arr.shape[work_axis]
    if n == 0:
        raise ValueError("IQM requires at least one sample")
    lowercut = int(0.25 * n)
    uppercut = n - lowercut
    ordered = np.sort(arr, axis=work_axis)
    kept = np.take(ordered, indices=range(lowercut, uppercut), axis=work_axis)
    result = np.mean(kept, axis=work_axis)
    return float(result) if np.ndim(result) == 0 else result


def wilson_ci(successes: int, n: int, z: float = WILSON_Z_95) -> ConfidenceInterval:
    """Wilson score interval for a binomial success proportion.

    The Wilson interval is preferred over the normal (Wald) interval for the
    small samples and extreme rates typical of insertion-success measurement:
    it never leaves [0, 1] and stays sensible at 0 and 100% success.

    Args:
        successes: Number of successes (``0 <= successes <= n``).
        n: Number of trials (``> 0``).
        z: Normal quantile for the desired coverage (default ~95%).

    Returns:
        A ``ConfidenceInterval`` whose ``point`` is the observed rate
        ``successes / n`` and whose bounds are the Wilson score bounds.

    Raises:
        ValueError: If ``n <= 0`` or ``successes`` is out of range.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return ConfidenceInterval(point=p, lo=float(lo), hi=float(hi))


def _percentile_bounds(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return the two-sided percentile bounds of a bootstrap distribution.

    Args:
        boot: 1-D array of bootstrap-replicated statistics.
        alpha: Total tail mass (e.g. ``0.05`` for a 95% interval).

    Returns:
        A ``(lo, hi)`` tuple of percentile bounds.
    """
    lo, hi = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def bootstrap_ci(
    values: np.ndarray,
    statistic=mean_stat,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for an arbitrary statistic.

    Args:
        values: 1-D sample of observations.
        statistic: Callable ``f(array, axis)`` returning the statistic; must
            accept an ``axis`` keyword so replicates can be vectorised. Defaults
            to the mean; pass :func:`iqm` for the interquartile mean.
        n_boot: Number of bootstrap resamples.
        alpha: Total tail mass for the interval (``0.05`` -> 95%).
        seed: Seed for the numpy random generator (determinism).

    Returns:
        A ``ConfidenceInterval`` with the observed point estimate and bounds.

    Raises:
        ValueError: If ``values`` is empty.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one observation")
    point = float(statistic(x, axis=None))
    if n == 1:
        return ConfidenceInterval(point=point, lo=point, hi=point, level=1.0 - alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(x[idx], axis=1)
    lo, hi = _percentile_bounds(np.asarray(boot, dtype=float), alpha)
    return ConfidenceInterval(point=point, lo=lo, hi=hi, level=1.0 - alpha)


def stratified_bootstrap_ci(
    values: np.ndarray,
    strata: np.ndarray,
    statistic=mean_stat,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> ConfidenceInterval:
    """Stratified percentile bootstrap: resample independently within strata.

    Resampling within each stratum (rather than over the pooled sample)
    preserves the design's fixed per-cell counts, giving tighter, less biased
    intervals when the evaluation suite is deliberately balanced across
    rail x plug x port cells.

    Args:
        values: 1-D sample of observations.
        strata: 1-D array of stratum labels aligned with ``values``.
        statistic: Callable ``f(array, axis)`` (see :func:`bootstrap_ci`).
        n_boot: Number of bootstrap resamples.
        alpha: Total tail mass for the interval.
        seed: Seed for the numpy random generator.

    Returns:
        A ``ConfidenceInterval`` for the statistic of the pooled resample.

    Raises:
        ValueError: If inputs are empty or mismatched in length.
    """
    x = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(strata).ravel()
    if x.size == 0:
        raise ValueError("stratified_bootstrap_ci requires at least one observation")
    if x.size != labels.size:
        raise ValueError(
            f"values ({x.size}) and strata ({labels.size}) must be the same length"
        )
    point = float(statistic(x, axis=None))
    rng = np.random.default_rng(seed)
    groups = [x[labels == lab] for lab in np.unique(labels)]
    columns = []
    for group in groups:
        gn = group.size
        gidx = rng.integers(0, gn, size=(n_boot, gn))
        columns.append(group[gidx])
    pooled = np.hstack(columns)
    boot = statistic(pooled, axis=1)
    lo, hi = _percentile_bounds(np.asarray(boot, dtype=float), alpha)
    return ConfidenceInterval(point=point, lo=lo, hi=hi, level=1.0 - alpha)


def paired_bootstrap_ci(
    diffs: np.ndarray,
    statistic=mean_stat,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> ConfidenceInterval:
    """Paired percentile bootstrap of per-config score differences.

    For the matched-seed protocol every checkpoint runs the identical configs,
    so comparisons are made on within-config differences ``score_A - score_B``.
    Resampling those differences (a paired design) removes between-config
    variance and separates policies at much smaller sample sizes than an
    unpaired comparison would.

    Args:
        diffs: 1-D array of paired differences (one per matched config).
        statistic: Callable ``f(array, axis)`` (mean or :func:`iqm`).
        n_boot: Number of bootstrap resamples.
        alpha: Total tail mass for the interval.
        seed: Seed for the numpy random generator.

    Returns:
        A ``ConfidenceInterval`` for the statistic of the differences. When its
        ``excludes(0.0)`` is True the comparison is significant at ``alpha``.
    """
    return bootstrap_ci(
        diffs, statistic=statistic, n_boot=n_boot, alpha=alpha, seed=seed
    )
