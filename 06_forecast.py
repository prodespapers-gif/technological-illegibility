"""
06_forecast.py — Diffusion fitting and conditional scenario projection (RQ4).

Stage 6 of the pipeline. Stages 4 and 5 establish that technology-mediated care
failures appear in the regulatory record and that a large share of them are
recorded under categories naming no technology. This module asks where that is
heading: it fits diffusion models to the observed series, subjects them to an
honest out-of-sample backtest, and projects the illegibility gap forward under
explicitly parameterised scenarios.

WHAT THIS MODULE CLAIMS, AND WHAT IT DOES NOT
---------------------------------------------
It does not predict the future. It performs *conditional scenario analysis*:
given a fitted diffusion process and a stated assumption about how legible those
failures are to the regulator, here is the implied trajectory. The scenario
parameters are assumptions, are named as such in every returned record, and are
reported in the manuscript as assumptions. A projection presented as a forecast
would overstate what a fitted curve on eleven years of citation data can carry.

Two disciplines enforce that honesty:

1. **Backtesting is mandatory and can fail.** :func:`backtest` fits on an early
   period, evaluates on a held-out later period, and returns a ``passed`` flag
   against a pre-registered error threshold. A model that fails its backtest is
   reported as having failed; the projection is not published on the strength of
   in-sample fit alone.

2. **Intervals, never point forecasts.** Every projected quantity carries a
   bootstrap prediction interval propagating parameter uncertainty. Interval
   coverage is itself verified.

THE MEASUREMENT CAVEAT THAT SHAPES THE WHOLE MODULE
---------------------------------------------------
The observed series counts *citations*, which is the product of two things: how
much technology-mediated failure occurs, and how reliably the regulator records
it as such. The paper's central claim is that the second term is well below one.
So a diffusion curve fitted to citations is a curve of **observed** failures,
not of underlying ones, and the scenario machinery models the two terms
separately — a diffusion process for underlying failures, and an explicit
legibility parameter for the share the record captures. Conflating them would
assume away the phenomenon the study exists to measure.

MODEL SPECIFICATIONS
--------------------
Bass:     F(t) = m * (1 - e^{-(p+q)t}) / (1 + (q/p) e^{-(p+q)t})
Logistic: F(t) = m / (1 + e^{-r(t - t0)})

Both are fitted to *period* counts, F(t+1) - F(t), rather than to the cumulative
series. Cumulative residuals are strongly autocorrelated by construction, and
fitting to them would understate parameter uncertainty and therefore produce
prediction intervals that are too narrow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

EPSILON = 1e-12


# --------------------------------------------------------------------------- #
# Model specifications
# --------------------------------------------------------------------------- #

def bass_cumulative(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    """Cumulative Bass diffusion at times ``t``.

    ``p`` is the coefficient of innovation (adoption independent of prior
    adopters), ``q`` the coefficient of imitation (adoption driven by them), and
    ``m`` the saturation level. ``F(0) = 0`` and ``F(t) -> m``.

    When ``q = 0`` this reduces to bounded exponential growth,
    ``F(t) = m(1 - e^{-pt})``; that identity is used as a correctness check.
    """
    t = np.asarray(t, dtype=float)
    p = max(float(p), EPSILON)
    q = float(q)
    total = p + q
    decay = np.exp(-total * t)
    return m * (1.0 - decay) / (1.0 + (q / p) * decay)


def bass_incremental(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    """Instantaneous Bass adoption rate, dF/dt.

    Provided for the continuous peak-time identity ``t* = ln(q/p)/(p+q)``, which
    is used to verify the parameterisation. Fitting uses
    :func:`periodic_counts`, since the data are period counts rather than an
    instantaneous rate.
    """
    t = np.asarray(t, dtype=float)
    p = max(float(p), EPSILON)
    q = float(q)
    total = p + q
    decay = np.exp(-total * t)
    return m * (total ** 2 / p) * decay / (1.0 + (q / p) * decay) ** 2


def logistic_cumulative(t: np.ndarray, m: float, r: float, t0: float) -> np.ndarray:
    """Cumulative logistic diffusion; ``F(t0) = m/2`` by construction."""
    t = np.asarray(t, dtype=float)
    return m / (1.0 + np.exp(-np.clip(r * (t - t0), -500, 500)))


CUMULATIVE = {"bass": bass_cumulative, "logistic": logistic_cumulative}
PARAM_NAMES = {"bass": ("p", "q", "m"), "logistic": ("m", "r", "t0")}


def periodic_counts(model: str, t: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """Expected count in each period as ``F(t+1) - F(t)``.

    The discrete difference, not the instantaneous derivative: a quarter's
    citations are an integral over the quarter, and using the derivative would
    bias the fitted parameters when growth within a period is steep.
    """
    function = CUMULATIVE[model]
    t = np.asarray(t, dtype=float)
    return function(t + 1.0, *params) - function(t, *params)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

@dataclass
class FitResult:
    """A fitted diffusion model and its in-sample diagnostics."""

    model: str
    params: dict[str, float]
    fitted: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    rmse: float = float("nan")
    mape: float | None = None
    aic: float = float("nan")
    bic: float = float("nan")
    n_obs: int = 0
    n_params: int = 0
    converged: bool = False
    n_starts_tried: int = 0
    message: str = ""
    # Mahajan, Muller & Bass: diffusion models estimated before the
    # inflection point yield an unreliable saturation parameter m. Attached
    # by :func:`identifiability_check`; ``project_scenarios`` refuses to
    # extrapolate when False unless the configuration explicitly overrides.
    m_identifiable: bool | None = None
    identifiability: dict[str, Any] = field(default_factory=dict)

    def param_vector(self) -> list[float]:
        return [self.params[name] for name in PARAM_NAMES[self.model]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounds_and_starts(
    model: str, y: np.ndarray, t: np.ndarray
) -> tuple[tuple[list[float], list[float]], list[list[float]]]:
    """Parameter bounds and a grid of starting points.

    Diffusion likelihoods are multi-modal and notoriously sensitive to
    initialisation, so a single start can converge to a local optimum that fits
    visibly badly. A small deterministic grid of starts, keeping the best
    solution, makes the fit reproducible and far more robust.
    """
    total = float(np.sum(y))
    span = float(t.max() - t.min()) + 1.0
    if model == "bass":
        ceiling = max(total, 1.0)
        bounds = ([1e-6, 0.0, ceiling * 0.5], [1.0, 1.0, ceiling * 20.0])
        starts = [
            [p0, q0, total * m0]
            for p0 in (0.001, 0.01, 0.05)
            for q0 in (0.05, 0.2, 0.5)
            for m0 in (1.1, 2.0, 5.0)
        ]
    elif model == "logistic":
        bounds = ([max(total, 1.0) * 0.5, 1e-4, t.min() - 5 * span],
                  [max(total, 1.0) * 20.0, 5.0, t.max() + 5 * span])
        starts = [
            [total * m0, r0, t.min() + span * frac]
            for m0 in (1.1, 2.0, 5.0)
            for r0 in (0.05, 0.2, 0.6)
            for frac in (0.3, 0.5, 0.8)
        ]
    else:
        raise ValueError(f"unknown model {model!r}; expected 'bass' or 'logistic'")
    clipped = [
        [min(max(value, low), high) for value, low, high in zip(start, *bounds)]
        for start in starts
    ]
    return bounds, clipped


def _validate_series(
    t: Sequence[float], y: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("t and y must be one-dimensional")
    if t_arr.size != y_arr.size:
        raise ValueError(f"t has {t_arr.size} points but y has {y_arr.size}")
    if t_arr.size < 6:
        raise ValueError(
            f"at least 6 periods are required to identify a three-parameter "
            f"diffusion model, got {t_arr.size}"
        )
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("t and y must be finite (no NaN or inf)")
    if np.any(y_arr < 0):
        raise ValueError("period counts must be non-negative")
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("t must be strictly increasing")
    return t_arr, y_arr


def fit_diffusion(
    t: Sequence[float], y: Sequence[float], model: str = "bass",
    max_nfev: int = 4000,
) -> FitResult:
    """Fit a diffusion model to period counts by nonlinear least squares.

    Parameters
    ----------
    t
        Period indices, strictly increasing (typically ``0, 1, 2, ...``).
    y
        Count observed in each period — not the cumulative series.

    Returns
    -------
    FitResult
        Including ``converged``. A non-converged fit is returned rather than
        raised, so the orchestrator can record the failure in the results file
        instead of losing the run; downstream consumers must check the flag.
    """
    t_arr, y_arr = _validate_series(t, y)
    bounds, starts = _bounds_and_starts(model, y_arr, t_arr)

    def residual(params: np.ndarray) -> np.ndarray:
        return periodic_counts(model, t_arr, params) - y_arr

    best = None
    tried = 0
    for start in starts:
        tried += 1
        try:
            solution = least_squares(
                residual, start, bounds=bounds, method="trf", max_nfev=max_nfev,
            )
        except (ValueError, RuntimeError):
            continue
        if best is None or solution.cost < best.cost:
            best = solution

    names = PARAM_NAMES[model]
    if best is None:
        return FitResult(model=model, params=dict.fromkeys(names, float("nan")),
                         n_obs=int(y_arr.size), n_params=len(names),
                         converged=False, n_starts_tried=tried,
                         message="no starting point converged")

    params = dict(zip(names, (float(value) for value in best.x)))
    fitted = periodic_counts(model, t_arr, best.x)
    residuals = y_arr - fitted
    rss = float(np.sum(residuals ** 2))
    n = int(y_arr.size)
    k = len(names)

    # Gaussian least-squares information criteria. A guard on RSS keeps a
    # perfect noiseless fit from producing -inf and poisoning model comparison.
    rss_safe = max(rss, EPSILON)
    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)

    nonzero = y_arr > 0
    mape = (float(np.mean(np.abs(residuals[nonzero] / y_arr[nonzero])))
            if np.any(nonzero) else None)

    return attach_identifiability(FitResult(
        model=model, params=params, fitted=fitted.tolist(),
        residuals=residuals.tolist(), rmse=float(np.sqrt(rss / n)), mape=mape,
        aic=float(aic), bic=float(bic), n_obs=n, n_params=k,
        converged=bool(best.success), n_starts_tried=tried,
        message=str(best.message),
    ), t_arr, y_arr)


def compare_models(
    t: Sequence[float], y: Sequence[float],
    models: Sequence[str] = ("bass", "logistic"),
) -> dict[str, Any]:
    """Fit several specifications and rank them.

    Ranking is by AIC, but the backtest — not the information criterion — is
    what licenses a projection. In-sample fit rewards flexibility; only
    out-of-sample error speaks to whether the curve extrapolates.

    A caveat that must be reported alongside any ranking: **the Bass model nests
    the logistic**. As the innovation coefficient ``p`` approaches zero, Bass
    reduces to logistic growth, so Bass can fit logistic-generated data at least
    as well as the logistic itself and will tend to win on AIC at equal
    parameter count. A Bass victory is therefore weak evidence about the true
    generating process. The informative quantity is the fitted ``p``: a value
    near zero indicates the data carry no innovation signal and the process is
    effectively logistic, which is reported in ``bass_p_near_zero``.
    """
    fits = {name: fit_diffusion(t, y, name) for name in models}
    usable = {name: fit for name, fit in fits.items()
              if fit.converged and np.isfinite(fit.aic)}
    best = min(usable, key=lambda name: usable[name].aic) if usable else None
    bass_fit = fits.get("bass")
    bass_p = bass_fit.params.get("p") if bass_fit and bass_fit.converged else None
    return {
        "fits": {name: fit.to_dict() for name, fit in fits.items()},
        "ranking_by_aic": sorted(usable, key=lambda name: usable[name].aic),
        "best_model": best,
        "bass_p": bass_p,
        "bass_p_near_zero": (None if bass_p is None else bool(bass_p < 1e-3)),
        "nesting_note": (
            "Bass nests the logistic as p -> 0; an AIC win for Bass is weak "
            "evidence about the generating process. Read bass_p alongside it."
        ),
    }


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #

def backtest(
    t: Sequence[float], y: Sequence[float], cutoff_index: int,
    model: str = "bass", mape_max: float = 0.30,
) -> dict[str, Any]:
    """Fit on an early period and evaluate on the held-out remainder.

    Parameters
    ----------
    cutoff_index
        Number of leading periods used for training. The remainder is the test
        window and is never seen by the fit.
    mape_max
        Pre-registered ceiling on held-out mean absolute percentage error.

    Returns
    -------
    dict
        With ``passed``. A model that fails here has not earned a projection,
        and the manuscript reports the failure rather than quietly falling back
        to in-sample fit.
    """
    t_arr, y_arr = _validate_series(t, y)
    if not 6 <= cutoff_index < t_arr.size:
        raise ValueError(
            f"cutoff_index must leave at least 6 training periods and one test "
            f"period; got {cutoff_index} for {t_arr.size} periods"
        )

    fit = fit_diffusion(t_arr[:cutoff_index], y_arr[:cutoff_index], model)
    if not fit.converged:
        return {"model": model, "passed": False, "converged": False,
                "reason": "training fit did not converge",
                "cutoff_index": int(cutoff_index)}

    t_test, y_test = t_arr[cutoff_index:], y_arr[cutoff_index:]
    predicted = periodic_counts(model, t_test, fit.param_vector())
    error = y_test - predicted
    nonzero = y_test > 0
    mape = (float(np.mean(np.abs(error[nonzero] / y_test[nonzero])))
            if np.any(nonzero) else None)
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # A naive persistence benchmark: repeat the last training observation.
    naive = np.full(y_test.shape, y_arr[cutoff_index - 1], dtype=float)
    naive_rmse = float(np.sqrt(np.mean((y_test - naive) ** 2)))

    return {
        "model": model,
        "converged": True,
        "cutoff_index": int(cutoff_index),
        "n_train": int(cutoff_index),
        "n_test": int(y_test.size),
        "train_params": fit.params,
        "test_mape": mape,
        "test_rmse": rmse,
        "naive_rmse": naive_rmse,
        "beats_naive": bool(rmse < naive_rmse),
        "mape_max": float(mape_max),
        "passed": bool(mape is not None and mape <= mape_max and rmse < naive_rmse),
        "predicted": predicted.tolist(),
        "observed": y_test.tolist(),
    }


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #

def _bootstrap_parameters(
    t: np.ndarray, y: np.ndarray, fit: FitResult, n_boot: int, seed: int,
) -> np.ndarray:
    """Residual bootstrap over the fitted parameters.

    Residuals are resampled on the *period-count* scale, where they are close to
    independent, then added back to the fitted curve and the model refitted.
    Resampling cumulative residuals instead would inherit their autocorrelation
    and yield intervals that are far too narrow.
    """
    residuals = np.asarray(fit.residuals, dtype=float)
    fitted = np.asarray(fit.fitted, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        resampled = rng.choice(residuals, size=residuals.size)
        synthetic = np.clip(fitted + resampled, 0, None)
        candidate = fit_diffusion(t, synthetic, fit.model)
        if candidate.converged:
            draws.append(candidate.param_vector())
    return np.asarray(draws, dtype=float)


def prediction_intervals(
    t: Sequence[float], y: Sequence[float], fit: FitResult, horizon: int,
    n_boot: int = 200, seed: int = 0, alpha: float = 0.05,
) -> dict[str, Any]:
    """Projected period counts with bootstrap prediction intervals.

    Intervals propagate parameter uncertainty from the residual bootstrap. They
    do not include model uncertainty — the risk that the diffusion family itself
    is wrong — which no interval of this kind can capture and which the
    manuscript therefore states as a limitation rather than papering over.
    """
    t_arr, y_arr = _validate_series(t, y)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if not fit.converged:
        raise ValueError("cannot project from a fit that did not converge")

    future = np.arange(t_arr[-1] + 1, t_arr[-1] + 1 + horizon, dtype=float)
    central = periodic_counts(fit.model, future, fit.param_vector())

    draws = _bootstrap_parameters(t_arr, y_arr, fit, n_boot, seed)
    if draws.size == 0:
        return {"t": future.tolist(), "central": central.tolist(),
                "lo": None, "hi": None, "n_bootstrap_kept": 0}

    paths = np.vstack([periodic_counts(fit.model, future, row) for row in draws])
    lo = np.percentile(paths, 100 * alpha / 2, axis=0)
    hi = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)
    return {
        "t": future.tolist(),
        "central": central.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "n_bootstrap_kept": int(draws.shape[0]),
        "alpha": float(alpha),
    }


# --------------------------------------------------------------------------- #
# Scenario projection
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Scenario:
    """A named set of assumptions about diffusion and regulatory legibility.

    ``p_mult``, ``q_mult``, and ``m_mult`` scale the fitted diffusion
    parameters. ``legibility_start`` and ``legibility_end`` give the share of
    technology-mediated failures the regulatory record captures as such, ramped
    linearly across the horizon; the illegibility gap is one minus that share.

    Separating diffusion from legibility is what lets a taxonomy reform be
    represented as what it is: a change in what the record *sees*, not in what
    happens in the homes.
    """

    name: str
    p_mult: float = 1.0
    q_mult: float = 1.0
    m_mult: float = 1.0
    legibility_start: float = 0.35
    legibility_end: float = 0.35
    description: str = ""

    def __post_init__(self) -> None:
        for value, label in ((self.p_mult, "p_mult"), (self.q_mult, "q_mult"),
                             (self.m_mult, "m_mult")):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")
        for value, label in ((self.legibility_start, "legibility_start"),
                             (self.legibility_end, "legibility_end")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("status_quo", description=(
        "Diffusion continues on the fitted trend; the citation taxonomy is "
        "unchanged, so the share of technology-mediated failures recorded as "
        "such stays at its estimated level.")),
    Scenario("taxonomy_reform", legibility_end=0.85, description=(
        "Technology-naming categories are introduced and surveyors are trained "
        "to use them. Underlying diffusion is unchanged; what changes is how "
        "much of it the record captures.")),
    Scenario("accelerated_adoption", q_mult=1.5, m_mult=1.3, description=(
        "Care technology diffuses faster and further than the fitted trend "
        "while the taxonomy stands still, so the gap widens in absolute terms.")),
)


def _saturation(model: str, params: Sequence[float]) -> float:
    """Ceiling parameter for a fitted model (``m`` in both specifications)."""
    return float(params[2] if model == "bass" else params[0])


def _anchor_point(model: str, params: Sequence[float], target: float) -> float:
    """Find the time ``a`` at which a scenario curve has reached ``target``.

    Scenarios must not rewrite the past. Rescaling a diffusion parameter and
    re-running the curve from time zero would assert that imitation was *always*
    stronger, which pushes adoption mass earlier and can leave a saturated
    process with *smaller* future increments than the status quo — the opposite
    of the assumption being modelled.

    Anchoring instead solves ``F_scenario(a) = target``, where ``target`` is the
    cumulative count actually observed, and projects forward from ``a``. The
    scenario then reproduces history exactly and diverges only in the future,
    which is what a forward-looking assumption means.

    ``F`` is strictly increasing in time, so the root is unique; the bracket is
    expanded geometrically until it straddles the target.
    """
    from scipy.optimize import brentq

    function = CUMULATIVE[model]

    def offset(a: float) -> float:
        return float(function(np.array([a], dtype=float), *params)[0]) - target

    lo, hi = -1.0, 1.0
    for _ in range(200):
        if offset(lo) > 0:
            lo *= 2.0
        elif offset(hi) < 0:
            hi *= 2.0
        else:
            break
    else:  # pragma: no cover - unreachable while F is monotone and bounded
        raise ValueError("could not bracket the scenario anchor point")
    return float(brentq(offset, lo, hi, xtol=1e-10, maxiter=200))


def identifiability_check(
    t: Sequence[float], y: Sequence[float], min_periods: int = 12,
    decline_fraction: float = 0.90,
) -> dict[str, Any]:
    """Whether the saturation parameter is identifiable from this window.

    Mahajan, Muller & Bass's review documents that pre-inflection series
    leave m effectively unidentifiable — the curve's ceiling is set by data
    the window has not seen. Three observable conditions proxy "the
    inflection is inside the window": enough periods; the peak period count
    is interior (not the last observation); and the tail has genuinely
    declined below ``decline_fraction`` of the peak. The check DESCRIBES the
    window; it never modifies the fit.
    """
    t_arr, y_arr = _validate_series(t, y)
    n = y_arr.size
    peak_index = int(np.argmax(y_arr))
    peak_value = float(y_arr[peak_index])
    tail = float(np.mean(y_arr[-2:])) if n >= 2 else float(y_arr[-1])
    conditions = {
        "enough_periods": bool(n >= min_periods),
        "peak_interior": bool(peak_index < n - 2),
        "tail_declined": bool(peak_value > 0
                              and tail < decline_fraction * peak_value),
    }
    reasons = [name for name, ok in conditions.items() if not ok]
    return {
        "m_identifiable": bool(not reasons),
        "n_periods": int(n), "peak_index": peak_index,
        "peak_value": peak_value, "tail_mean": tail,
        "min_periods": int(min_periods),
        "decline_fraction": float(decline_fraction),
        "conditions": conditions,
        "reason": ("" if not reasons else
                   "pre-inflection window: " + ", ".join(reasons)),
    }


def attach_identifiability(
    fit: "FitResult", t: Sequence[float], y: Sequence[float],
    min_periods: int = 12,
) -> "FitResult":
    """Attach the window diagnostic to a fit (does not alter estimates)."""
    check = identifiability_check(t, y, min_periods=min_periods)
    fit.m_identifiable = check["m_identifiable"]
    fit.identifiability = check
    return fit


def project_scenarios(
    t: Sequence[float], y: Sequence[float], fit: FitResult, horizon: int,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    n_boot: int = 200, seed: int = 0, alpha: float = 0.05,
    require_identifiable: bool = True,
) -> dict[str, Any]:
    """Project underlying failures, recorded failures, and the illegibility gap.

    Each scenario rescales the fitted diffusion parameters, anchors the rescaled
    curve to the cumulative count actually observed (see :func:`_anchor_point`,
    so no scenario rewrites history), projects the implied period counts with
    bootstrap intervals, and applies the legibility ramp to split those counts
    into the part the regulatory record captures and the part it does not.

    A scenario whose assumed ceiling lies below the observed cumulative is
    internally inconsistent with history. It is marked ``infeasible`` with a
    stated reason rather than silently returning zeros, and the remaining
    scenarios still project.

    Every returned record carries its assumptions verbatim, so a figure derived
    from it cannot be reproduced without them and no trajectory can be quoted as
    though it were unconditional.
    """
    t_arr, y_arr = _validate_series(t, y)
    if not fit.converged:
        raise ValueError("cannot project from a fit that did not converge")
    if require_identifiable and fit.m_identifiable is False:
        raise ValueError(
            "cannot project: the observation window is pre-inflection, so "
            "the saturation parameter is not identifiable "
            f"({fit.identifiability.get('reason', 'unspecified')}). "
            "Extend the series (archived CMS files) or set "
            "forecast.require_identifiable_m to false to override "
            "explicitly."
        )
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise ValueError(f"scenario names must be unique, got {names}")

    future = np.arange(t_arr[-1] + 1, t_arr[-1] + 1 + horizon, dtype=float)
    observed_cumulative = float(np.sum(y_arr))
    draws = _bootstrap_parameters(t_arr, y_arr, fit, n_boot, seed)
    base = fit.param_vector()
    steps = np.arange(horizon, dtype=float)

    def project(params: Sequence[float]) -> tuple[np.ndarray, str]:
        """Return projected period counts and a status for one parameter set.

        Three cases, all returning arrays of the same shape so that every
        scenario record has identical keys. A consumer must never have to guess
        whether a field exists; the ``status`` field carries the meaning.
        """
        ceiling = _saturation(fit.model, params)
        remaining = ceiling - observed_cumulative
        tolerance = 1e-6 * max(abs(ceiling), 1.0)
        if remaining < -tolerance:
            return np.zeros(horizon), "ceiling_below_observed"
        if remaining <= tolerance:
            return np.zeros(horizon), "saturated"
        anchor = _anchor_point(fit.model, params, observed_cumulative)
        function = CUMULATIVE[fit.model]
        counts = (function(anchor + steps + 1.0, *params)
                  - function(anchor + steps, *params))
        return np.clip(counts, 0.0, None), "projected"

    results: dict[str, Any] = {}
    for scenario in scenarios:
        scaled = _apply_scenario(fit.model, base, scenario)
        central, status = project(scaled)
        legibility = np.linspace(scenario.legibility_start,
                                 scenario.legibility_end, horizon)

        paths = [
            path for path, path_status in
            (project(_apply_scenario(fit.model, row, scenario)) for row in draws)
            if path_status == "projected"
        ]
        if paths:
            stacked = np.vstack(paths)
            lo = np.percentile(stacked, 100 * alpha / 2, axis=0).tolist()
            hi = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0).tolist()
        else:
            lo = hi = None

        note = {
            "projected": "",
            "saturated": (
                "The fitted process has reached its estimated ceiling within the "
                "observed window, so this scenario implies no further growth. "
                "Zero future increments is the projection, not a failure."
            ),
            "ceiling_below_observed": (
                f"The assumed ceiling ({_saturation(fit.model, scaled):.1f}) lies "
                f"below the observed cumulative count ({observed_cumulative:.1f}), "
                f"so this scenario contradicts the observed history and its "
                f"trajectory should not be interpreted."
            ),
        }[status]

        results[scenario.name] = {
            "assumptions": asdict(scenario),
            "status": status,
            "interpretable": status == "projected",
            "note": note,
            "t": future.tolist(),
            "failures_total": central.tolist(),
            "failures_recorded": (central * legibility).tolist(),
            "failures_illegible": (central * (1.0 - legibility)).tolist(),
            "legibility": legibility.tolist(),
            "illegibility_gap": (1.0 - legibility).tolist(),
            "failures_total_lo": lo,
            "failures_total_hi": hi,
            "cumulative_illegible": float(np.sum(central * (1.0 - legibility))),
            "n_bootstrap_paths": len(paths),
        }

    return {
        "model": fit.model,
        "fitted_params": fit.params,
        "horizon": int(horizon),
        "observed_cumulative": observed_cumulative,
        "n_bootstrap_kept": int(draws.shape[0]) if draws.size else 0,
        "interpretation": (
            "Conditional scenario analysis, not prediction. Each trajectory "
            "holds only under the assumptions recorded alongside it, and is "
            "anchored to the observed cumulative count so that scenarios differ "
            "only in the future."
        ),
        "scenarios": results,
    }


def _apply_scenario(
    model: str, params: Sequence[float], scenario: Scenario
) -> list[float]:
    """Rescale fitted diffusion parameters according to a scenario."""
    values = list(params)
    if model == "bass":
        p, q, m = values
        return [p * scenario.p_mult, q * scenario.q_mult, m * scenario.m_mult]
    if model == "logistic":
        m, r, t0 = values
        # The logistic has no separate innovation term; growth rate absorbs the
        # imitation multiplier so the two families respond comparably.
        return [m * scenario.m_mult, r * scenario.q_mult, t0]
    raise ValueError(f"unknown model {model!r}")


def scenarios_from_config(spec: Mapping[str, Any] | Sequence[str]) -> list[Scenario]:
    """Build scenarios from configuration.

    Accepts either a list of names drawn from :data:`DEFAULT_SCENARIOS` or a
    mapping of name to overrides, so the defaults can be used as-is or tuned
    without editing code.
    """
    defaults = {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}
    if isinstance(spec, Mapping):
        out = []
        for name, overrides in spec.items():
            base = defaults.get(name)
            fields = asdict(base) if base else {"name": name}
            fields.update(overrides or {})
            fields["name"] = name
            out.append(Scenario(**fields))
        return out
    missing = [name for name in spec if name not in defaults]
    if missing:
        raise ValueError(f"unknown scenario name(s): {missing}")
    return [defaults[name] for name in spec]


__all__ = [
    "bass_cumulative", "bass_incremental", "logistic_cumulative",
    "periodic_counts", "FitResult", "fit_diffusion", "compare_models",
    "backtest", "prediction_intervals", "Scenario", "DEFAULT_SCENARIOS",
    "project_scenarios", "scenarios_from_config",
]
