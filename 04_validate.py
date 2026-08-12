"""
04_validate.py — Validating the measurement instrument.

Stage 4 of the pipeline, and the module that carries this study's methodological
contribution. Stage 3 produces a variable — "was a care technology implicated in
this cited deficiency, and in what role?" — by having a language model read
regulatory prose. That variable is not admissible as social-science evidence
until its error properties are characterised. This module characterises them,
and then *gates* the substantive analyses on the result.

The gate is the point. A validation section that reports diagnostics and then
proceeds regardless is decorative. Here the thresholds are declared in the
configuration before the extraction is run, ``evaluate_gate`` compares the
measured diagnostics against them, and ``07_experiments.py`` withholds the
dependent analyses when the gate fails. Deciding the rule in advance is what
separates validation from post-hoc justification.

THE FIVE CHECKS
---------------
1. HUMAN RELIABILITY (:func:`krippendorff_alpha`). Two or more annotators code a
   stratified sample using the same frame given to the model. Reported *first*,
   because if trained humans cannot agree on whether a narrative implicates a
   technology, the construct is not codeable and no model can rescue it. This
   check can fail the study rather than the model.

2. MODEL-VERSUS-HUMAN AGREEMENT (:func:`agreement_binary`,
   :func:`agreement_multiclass`). Accuracy, precision, recall, F1, and Cohen's
   kappa against the human gold standard, with bootstrap confidence intervals
   over documents. Kappa matters more than accuracy here: technology-mediated
   failures are a minority of citations, so a model that always answers "no"
   scores well on accuracy and is worthless.

3. PROMPT SENSITIVITY (:func:`prompt_sensitivity`). The same corpus re-coded
   under paraphrased prompts. A prevalence estimate that moves materially with
   phrasing is a finding about the prompt, not about nursing homes.

4. MODEL SENSITIVITY (:func:`model_sensitivity`). Re-coding under a second
   open-weight model from a different family, compared at document level and at
   facility level. This separates the substantive signal from one model's
   idiosyncrasies.

5. DIFFERENTIAL ERROR (:func:`differential_error_audit`). Whether extraction
   error correlates with facility characteristics — ownership, size, state,
   community composition. This is the check that governs the equity analysis.
   Non-differential error attenuates estimates toward the null and is tolerable;
   error that is *itself* patterned by the covariates used in the equity model
   would manufacture the very disparity the paper claims to find.

STATISTICAL NOTES
-----------------
All estimators are implemented here against numpy rather than delegated, so the
exact definition behind every published number is inspectable in one file, and
so that this stage runs on a machine with no modelling stack. Each is verified
against published reference values or an independent implementation.

Bootstrap resampling is over *documents*, never over predictions independently:
the unit of sampling must match the unit of measurement, or intervals will be
far too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

MISSING = -1  # sentinel for an uncoded cell in a reliability matrix


# --------------------------------------------------------------------------- #
# Thresholds (pre-registered)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Thresholds:
    """Pre-registered acceptance criteria, read from the configuration.

    Declared before extraction is run and recorded in the results file, so a
    reader can see that the bar was set in advance rather than fitted to the
    outcome.
    """

    # Krippendorff (2004, Hum. Commun. Res. 30(3)): "it is customary to
    # require alpha >= .800. Where tentative conclusions are still acceptable,
    # alpha >= .667 is the lowest conceivable limit" — and "when human lives
    # hang on the results of a content analysis, decision criteria have to be
    # set far higher." This study's subject is resident harm, so the customary
    # standard, not the tentative floor, is pre-registered.
    alpha_min: float = 0.80
    # Hayes & Krippendorff (2007): the decision statistic is q = P(alpha_true
    # < alpha_min), estimated from the pair-level bootstrap. Their worked
    # example shows why a point test misleads: the same data give q = 0.0125
    # at alpha_min 0.70 but q = 0.9473 at 0.80. Accept only when the risk of
    # the data being unreliable is at most this.
    alpha_q_max: float = 0.05
    # Deliberately above the zero-shot state of the art: Ziems et al. (2024,
    # Comput. Linguist. 50(1)) report best zero-shot kappa ~ 0.55 on
    # comparable taxonomic tasks (human baseline ~ 0.51). A validated
    # measurement must clear a higher bar than the field's zero-shot ceiling.
    f1_min: float = 0.80             # binary technology_present
    # MODEL-vs-GOLD accuracy ONLY. Krippendorff (2004) discourages kappa for
    # coder reliability (it corrects chance by the coders' marginals, not the
    # data's); human-human reliability is measured exclusively by alpha here.
    # kappa is retained solely as chance-corrected model accuracy against the
    # human gold standard, where "agreement with a standard" is the question.
    kappa_min: float = 0.60
    prompt_max_spread: float = 0.05  # absolute prevalence spread across prompts
    cross_model_rho_min: float = 0.70
    differential_error_max_flags: int = 0  # any flagged covariate blocks equity

    @classmethod
    def from_config(cls, validation: Mapping[str, Any]) -> "Thresholds":
        known = {f: validation[f] for f in cls.__dataclass_fields__ if f in validation}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _as_1d(array: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(array)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} is empty")
    return out


def _check_aligned(a: np.ndarray, b: np.ndarray, na: str, nb: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{na} and {nb} must be the same length, got {a.shape[0]} and {b.shape[0]}"
        )


def bootstrap_ci(
    statistic,
    *arrays: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval for a statistic over documents.

    ``statistic`` is called with resampled copies of ``arrays`` (resampled with
    a common index, so paired observations stay paired). Resamples on which the
    statistic is undefined — a draw containing only one class, for instance —
    are skipped and counted, rather than silently coerced to zero, which would
    drag the interval toward an unearned certainty.
    """
    if not arrays:
        raise ValueError("bootstrap_ci requires at least one array")
    n = arrays[0].shape[0]
    for other in arrays[1:]:
        if other.shape[0] != n:
            raise ValueError("all arrays must share the first dimension")

    point = _safe_statistic(statistic, arrays)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        value = _safe_statistic(statistic, tuple(array[index] for array in arrays))
        if value is None:
            skipped += 1
            continue
        draws.append(value)

    if not draws:
        return {"point": point, "lo": None, "hi": None,
                "n_boot": n_boot, "n_skipped": skipped}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "n_skipped": skipped}


def _safe_statistic(statistic, arrays: tuple[np.ndarray, ...]) -> float | None:
    """Evaluate a statistic, returning ``None`` where it is undefined.

    Applied to the point estimate as well as to every resample. A gold-standard
    stratum can legitimately contain a single class, which leaves kappa
    undefined; that must be reported as undefined rather than crashing the whole
    validation run, and it must never be silently coerced to zero, which would
    read as "no agreement beyond chance" when the truth is "not estimable".
    """
    try:
        value = statistic(*arrays)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    return _finite_or_none(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, with midranks for ties.

    Implemented directly (rank, then Pearson) so the tie handling behind a
    published coefficient is visible rather than delegated.
    """
    a, b = _as_1d(x, "x").astype(float), _as_1d(y, "y").astype(float)
    _check_aligned(a, b, "x", "y")
    if a.size < 2:
        raise ValueError("spearman_rho requires at least two observations")
    ra, rb = _midrank(a), _midrank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        raise ValueError("spearman_rho undefined when a variable is constant")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _midrank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties assigned their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        mask = inverse == index
        ranks[mask] = ranks[mask].mean()
    return ranks


# --------------------------------------------------------------------------- #
# 1. Human reliability
# --------------------------------------------------------------------------- #

def krippendorff_alpha(
    matrix: Sequence[Sequence[Any]] | np.ndarray, level: str = "nominal"
) -> float:
    """Krippendorff's alpha for any number of coders, with missing values.

    Parameters
    ----------
    matrix
        Reliability data as coders (rows) x units (columns). Uncoded cells are
        ``MISSING`` (-1) or ``nan``. Units rated by fewer than two coders carry
        no information about agreement and are excluded, as the coefficient
        requires.
    level
        Difference function: ``nominal`` (categories unordered), ``ordinal``
        (ranked), ``interval``, or ``ratio``. The coding frame in ``03_extract``
        is categorical, so ``nominal`` is the operative choice; the others are
        provided for the confidence field, which is continuous.

    Returns
    -------
    float
        1.0 is perfect agreement; 0.0 is agreement at chance; negative values
        indicate systematic disagreement — worse than coders answering at
        random, which usually means the coders interpreted the frame
        differently rather than applying it noisily.

    Notes
    -----
    Alpha is preferred to percentage agreement and to Cohen's kappa because it
    handles more than two coders, tolerates missing cells (annotators need not
    code every document), and corrects for chance under the observed marginal
    distribution — all three of which apply to this study's annotation design.

    Verified against the published worked example for this coefficient.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {data.shape}")
    if data.shape[0] < 2:
        raise ValueError("at least two coders are required")
    data = np.where(data == MISSING, np.nan, data)

    columns = [column[~np.isnan(column)] for column in data.T]
    usable = [column for column in columns if column.size >= 2]
    if not usable:
        raise ValueError("no unit was coded by two or more coders")

    values = np.unique(np.concatenate(usable))
    if values.size == 1:
        return 1.0  # every coder gave the same value everywhere
    index_of = {value: position for position, value in enumerate(values)}
    size = values.size

    # Coincidence matrix: ordered pairs of distinct coders within each unit,
    # weighted by 1/(m_u - 1) so units with more coders are not over-counted.
    coincidence = np.zeros((size, size), dtype=float)
    for unit in usable:
        m = unit.size
        positions = [index_of[value] for value in unit]
        weight = 1.0 / (m - 1)
        for i, first in enumerate(positions):
            for j, second in enumerate(positions):
                if i != j:
                    coincidence[first, second] += weight

    marginals = coincidence.sum(axis=1)
    total = marginals.sum()
    if total <= 1:
        raise ValueError("insufficient pairable values to compute alpha")

    delta = _difference_matrix(values, marginals, level)
    observed = float((coincidence * delta).sum())
    expected_pairs = np.outer(marginals, marginals) - np.diag(marginals)
    expected = float((expected_pairs * delta).sum() / (total - 1))

    if expected == 0:
        return 1.0 if observed == 0 else float("-inf")
    return float(1.0 - observed / expected)


def _difference_matrix(
    values: np.ndarray, marginals: np.ndarray, level: str
) -> np.ndarray:
    """Squared-difference (delta) matrix for the requested measurement level."""
    size = values.size
    delta = np.zeros((size, size), dtype=float)
    level = level.lower()

    if level == "nominal":
        delta = 1.0 - np.eye(size)
    elif level in {"interval", "ratio"}:
        for i in range(size):
            for j in range(size):
                if level == "interval":
                    delta[i, j] = (values[i] - values[j]) ** 2
                else:
                    denominator = values[i] + values[j]
                    delta[i, j] = 0.0 if denominator == 0 else (
                        (values[i] - values[j]) / denominator
                    ) ** 2
    elif level == "ordinal":
        # Ordinal distance depends on the observed marginals, not on the numeric
        # labels: the gap between adjacent categories is the mass lying between
        # them.
        for i in range(size):
            for j in range(size):
                lo, hi = (i, j) if i <= j else (j, i)
                cumulative = marginals[lo:hi + 1].sum()
                delta[i, j] = (
                    cumulative - (marginals[i] + marginals[j]) / 2.0
                ) ** 2
    else:
        raise ValueError(
            f"unknown level {level!r}; expected nominal, ordinal, interval, or ratio"
        )
    return delta


# --------------------------------------------------------------------------- #
# 2. Model versus human agreement
# --------------------------------------------------------------------------- #

def _alpha_pairs(
    matrix: Sequence[Sequence[Any]] | np.ndarray, level: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decompose reliability data into weighted judgement pairs.

    Returns ``(weights, deltas, expected)`` where each unordered within-unit
    coder pair contributes one entry: its coincidence weight ``1/(m_u - 1)``
    and its disagreement ``delta``. The expected disagreement is computed once
    from the full data and held fixed, matching the published procedure. The
    identity ``alpha = 1 - 2 * sum(w * d) / expected`` reproduces
    :func:`krippendorff_alpha` exactly and is asserted by the harness.
    """
    data = np.asarray(matrix, dtype=float)
    data = np.where(data == MISSING, np.nan, data)
    columns = [column[~np.isnan(column)] for column in data.T]
    usable = [column for column in columns if column.size >= 2]
    if not usable:
        raise ValueError("no unit was coded by two or more coders")
    values = np.unique(np.concatenate(usable))
    index_of = {value: position for position, value in enumerate(values)}
    # marginals as in the coincidence matrix: each judgement counts once
    marginals = np.zeros(values.size, dtype=float)
    for unit in usable:
        for value in unit:
            marginals[index_of[value]] += 1.0
    total = marginals.sum()
    delta = _difference_matrix(values, marginals, level)
    expected_pairs = np.outer(marginals, marginals) - np.diag(marginals)
    expected = float((expected_pairs * delta).sum() / (total - 1))

    weights: list[float] = []
    deltas: list[float] = []
    for unit in usable:
        m = unit.size
        weight = 1.0 / (m - 1)
        positions = [index_of[value] for value in unit]
        for i in range(m):
            for j in range(i + 1, m):
                weights.append(weight)
                deltas.append(float(delta[positions[i], positions[j]]))
    return np.asarray(weights), np.asarray(deltas), expected


def alpha_bootstrap(
    matrix: Sequence[Sequence[Any]] | np.ndarray, level: str = "nominal",
    n_boot: int = 10000, seed: int = 0,
    alpha_mins: Sequence[float] = (0.9, 0.8, 0.7, 0.667),
    ci_alpha: float = 0.05,
) -> dict[str, Any]:
    """Pair-level bootstrap for Krippendorff's alpha (Hayes & Krippendorff).

    The resampling unit is the PAIR OF JUDGMENTS, not the document: "the units
    of bootstrapping for reliability are the pairs of judgments associated
    with particular units", weighted by how many observers judged each unit
    (their example: 40 units x 5 coders -> 159 judgments -> 239 pairs).
    Resampling documents instead understates the sampling variability of
    alpha, because within-document pairs are correlated through their unit.

    Expected disagreement is held fixed at its full-sample estimate; observed
    disagreement is recomputed on each resample of the pairs. Alongside the
    percentile interval, returns the decision statistic
    ``q(alpha_min) = P(alpha_true < alpha_min)`` at each requested minimum —
    the quantity the gate tests, per the source. ``n_boot`` defaults to
    10,000, past which the source reports little additional precision.
    """
    weights, deltas, expected = _alpha_pairs(matrix, level)
    n_pairs = weights.size
    if expected == 0:
        point = 1.0 if float((weights * deltas).sum()) == 0 else float("-inf")
        return {"point": point, "lo": point, "hi": point, "n_pairs": int(n_pairs),
                "n_boot": 0, "q": {f"{m:g}": None for m in alpha_mins},
                "level": level}
    point = float(1.0 - 2.0 * float((weights * deltas).sum()) / expected)

    rng = np.random.default_rng(seed)
    total_weighted = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        index = rng.integers(0, n_pairs, n_pairs)
        total_weighted[b] = float((weights[index] * deltas[index]).sum())
    draws = 1.0 - 2.0 * total_weighted / expected
    lo, hi = np.percentile(draws, [100 * ci_alpha / 2,
                                   100 * (1 - ci_alpha / 2)])
    return {
        "point": point,
        "lo": float(lo), "hi": float(hi),
        "n_pairs": int(n_pairs), "n_boot": int(n_boot), "level": level,
        "q": {f"{minimum:g}": float(np.mean(draws < minimum))
              for minimum in alpha_mins},
    }


def krippendorff_alpha_by_category(
    matrix: Sequence[Sequence[Any]] | np.ndarray,
) -> dict[str, Any]:
    """Per-category alpha, with the minimum as the variable's reliability.

    Krippendorff (2004): when it is individual categories that matter — as
    when reporting differences in their frequencies, which the per-role gap
    breakdown does — "the reliability of each distinction within a variable
    should be tested and the lowest among them" taken as the reliability of
    that variable. Each category is tested as itself-vs-rest at the nominal
    level; a category no coder ever used carries no information and is
    reported as ``None`` rather than silently inflating the minimum.
    """
    data = np.asarray(matrix, dtype=float)
    data = np.where(data == MISSING, np.nan, data)
    observed = np.unique(data[~np.isnan(data)])
    by_category: dict[str, float | None] = {}
    for value in observed:
        binary = np.where(np.isnan(data), np.nan,
                          (data == value).astype(float))
        try:
            by_category[f"{value:g}"] = float(
                krippendorff_alpha(binary, "nominal")
            )
        except ValueError:
            by_category[f"{value:g}"] = None
    finite = {k: v for k, v in by_category.items()
              if v is not None and np.isfinite(v)}
    min_category = min(finite, key=finite.get) if finite else None
    return {
        "by_category": by_category,
        "min_alpha": finite[min_category] if min_category else None,
        "min_category": min_category,
        "rule": "minimum per-category alpha is the variable's reliability "
                "for category-level claims (Krippendorff 2004)",
    }


def cohens_kappa(pred: np.ndarray, gold: np.ndarray) -> float:
    """Chance-corrected agreement between two labellings of the same units."""
    pred, gold = _as_1d(pred, "pred"), _as_1d(gold, "gold")
    _check_aligned(pred, gold, "pred", "gold")
    labels = np.unique(np.concatenate([pred, gold]))
    observed = float((pred == gold).mean())
    expected = sum(
        (pred == label).mean() * (gold == label).mean() for label in labels
    )
    if expected >= 1.0:
        # Both labellings are constant and identical: agreement is total, but
        # chance-corrected agreement is undefined rather than perfect.
        raise ValueError("kappa undefined: expected agreement is 1")
    return float((observed - expected) / (1.0 - expected))


def _prf(
    pred: np.ndarray, gold: np.ndarray, positive: Any = 1
) -> tuple[float, float, float]:
    tp = float(((pred == positive) & (gold == positive)).sum())
    fp = float(((pred == positive) & (gold != positive)).sum())
    fn = float(((pred != positive) & (gold == positive)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def agreement_binary(
    pred: Sequence[Any], gold: Sequence[Any], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Model-versus-human agreement on a binary field, with bootstrap intervals.

    Reports kappa alongside accuracy deliberately. Technology-mediated failures
    are a minority of cited deficiencies, so a model that never fires attains
    high accuracy and near-zero kappa; reporting accuracy alone would present
    that model as successful.
    """
    p, g = _as_1d(pred, "pred").astype(int), _as_1d(gold, "gold").astype(int)
    _check_aligned(p, g, "pred", "gold")

    precision, recall, f1 = _prf(p, g)
    return {
        "n": int(p.size),
        "prevalence_gold": float((g == 1).mean()),
        "prevalence_pred": float((p == 1).mean()),
        "accuracy": bootstrap_ci(lambda a, b: (a == b).mean(), p, g,
                                 n_boot=n_boot, seed=seed),
        "precision": bootstrap_ci(lambda a, b: _prf(a, b)[0], p, g,
                                  n_boot=n_boot, seed=seed + 1),
        "recall": bootstrap_ci(lambda a, b: _prf(a, b)[1], p, g,
                               n_boot=n_boot, seed=seed + 2),
        "f1": bootstrap_ci(lambda a, b: _prf(a, b)[2], p, g,
                           n_boot=n_boot, seed=seed + 3),
        "kappa": bootstrap_ci(cohens_kappa, p, g, n_boot=n_boot, seed=seed + 4),
        "point_estimates": {"precision": precision, "recall": recall, "f1": f1},
        "confusion": {
            "tp": int(((p == 1) & (g == 1)).sum()),
            "fp": int(((p == 1) & (g == 0)).sum()),
            "fn": int(((p == 0) & (g == 1)).sum()),
            "tn": int(((p == 0) & (g == 0)).sum()),
        },
    }


def confusion_matrix(
    pred: Sequence[Any], gold: Sequence[Any], labels: Sequence[Any] | None = None
) -> tuple[list[Any], list[list[int]]]:
    """Confusion matrix with rows = gold, columns = predicted."""
    p, g = _as_1d(pred, "pred"), _as_1d(gold, "gold")
    _check_aligned(p, g, "pred", "gold")
    if labels is None:
        labels = sorted(set(np.concatenate([p, g]).tolist()))
    position = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(g.tolist(), p.tolist()):
        if actual in position and predicted in position:
            matrix[position[actual], position[predicted]] += 1
    return list(labels), matrix.tolist()


def macro_f1(pred: np.ndarray, gold: np.ndarray, labels: Sequence[Any]) -> float:
    """Unweighted mean F1 across classes.

    Unweighted on purpose: the rare failure roles — over-reliance on a device,
    an alert that went unanswered — are the theoretically decisive categories,
    and a frequency-weighted average would let performance on the common
    categories conceal failure on them.
    """
    scores = [_prf(pred, gold, positive=label)[2] for label in labels]
    return float(np.mean(scores)) if scores else 0.0


def agreement_multiclass(
    pred: Sequence[Any], gold: Sequence[Any], labels: Sequence[Any] | None = None,
    n_boot: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Agreement on a categorical field (``technology_type``, ``failure_role``)."""
    p, g = _as_1d(pred, "pred"), _as_1d(gold, "gold")
    _check_aligned(p, g, "pred", "gold")
    if labels is None:
        labels = sorted(set(np.concatenate([p, g]).tolist()))

    per_class = {}
    for label in labels:
        precision, recall, f1 = _prf(p, g, positive=label)
        per_class[str(label)] = {
            "support": int((g == label).sum()),
            "precision": precision, "recall": recall, "f1": f1,
        }
    label_list, matrix = confusion_matrix(p, g, labels)
    return {
        "n": int(p.size),
        "labels": [str(label) for label in label_list],
        "accuracy": bootstrap_ci(lambda a, b: (a == b).mean(), p, g,
                                 n_boot=n_boot, seed=seed),
        "macro_f1": bootstrap_ci(lambda a, b: macro_f1(a, b, labels), p, g,
                                 n_boot=n_boot, seed=seed + 1),
        "kappa": bootstrap_ci(cohens_kappa, p, g, n_boot=n_boot, seed=seed + 2),
        "per_class": per_class,
        "confusion_gold_by_pred": matrix,
    }


# --------------------------------------------------------------------------- #
# 3. Prompt sensitivity
# --------------------------------------------------------------------------- #

def prompt_sensitivity(
    predictions_by_variant: Mapping[str, Sequence[Any]]
) -> dict[str, Any]:
    """Stability of the headline estimate across paraphrased prompts.

    Reports the prevalence implied by each prompt variant, the absolute spread
    between the extremes, and mean pairwise document-level agreement. The spread
    is the quantity that matters for the paper: it bounds how much of the
    reported prevalence is an artefact of one particular phrasing, and it is
    reported in the manuscript whether or not it is flattering.
    """
    if len(predictions_by_variant) < 2:
        raise ValueError("at least two prompt variants are required")
    names = sorted(predictions_by_variant)
    arrays = {name: _as_1d(predictions_by_variant[name], name).astype(int)
              for name in names}
    sizes = {array.shape[0] for array in arrays.values()}
    if len(sizes) != 1:
        raise ValueError(f"variants cover different numbers of documents: {sizes}")

    prevalence = {name: float((array == 1).mean()) for name, array in arrays.items()}
    values = list(prevalence.values())

    pairwise = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            agree = float((arrays[first] == arrays[second]).mean())
            try:
                kappa = cohens_kappa(arrays[first], arrays[second])
            except ValueError:
                kappa = None
            pairwise[f"{first}|{second}"] = {"agreement": agree, "kappa": kappa}

    return {
        "n_variants": len(names),
        "n_documents": int(next(iter(arrays.values())).shape[0]),
        "prevalence_by_variant": prevalence,
        "prevalence_spread": float(max(values) - min(values)),
        "prevalence_min": float(min(values)),
        "prevalence_max": float(max(values)),
        "pairwise": pairwise,
        "mean_pairwise_agreement": float(
            np.mean([entry["agreement"] for entry in pairwise.values()])
        ),
    }


# --------------------------------------------------------------------------- #
# 4. Model sensitivity
# --------------------------------------------------------------------------- #

def model_sensitivity(
    primary: Sequence[Any], secondary: Sequence[Any],
    facility_ids: Sequence[Any] | None = None,
    min_facility_docs: int = 5,
) -> dict[str, Any]:
    """Concordance between two independently prompted models.

    Document-level agreement answers whether the models label the same cases;
    facility-level rank correlation answers the question the paper actually
    depends on — whether they order facilities the same way. The second can hold
    while the first is mediocre, and it is the ordering that the trend and
    equity analyses consume.
    """
    p = _as_1d(primary, "primary").astype(int)
    s = _as_1d(secondary, "secondary").astype(int)
    _check_aligned(p, s, "primary", "secondary")

    try:
        kappa = cohens_kappa(p, s)
    except ValueError:
        kappa = None

    result: dict[str, Any] = {
        "n_documents": int(p.size),
        "document_agreement": float((p == s).mean()),
        "document_kappa": kappa,
        "prevalence_primary": float((p == 1).mean()),
        "prevalence_secondary": float((s == 1).mean()),
        "facility_rho": None,
        "n_facilities_compared": 0,
    }

    if facility_ids is None:
        return result

    ids = _as_1d(facility_ids, "facility_ids")
    _check_aligned(ids, p, "facility_ids", "primary")
    rates_primary, rates_secondary = [], []
    for facility in np.unique(ids):
        mask = ids == facility
        if int(mask.sum()) < min_facility_docs:
            continue
        rates_primary.append(float(p[mask].mean()))
        rates_secondary.append(float(s[mask].mean()))

    result["n_facilities_compared"] = len(rates_primary)
    if len(rates_primary) >= 2:
        try:
            result["facility_rho"] = spearman_rho(rates_primary, rates_secondary)
        except ValueError:
            result["facility_rho"] = None
    return result


# --------------------------------------------------------------------------- #
# 5. Differential error
# --------------------------------------------------------------------------- #

def _logistic_irls(
    X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Logistic regression by iteratively reweighted least squares.

    Implemented here rather than delegated so the estimator behind a published
    coefficient is inspectable, and so this stage runs without a modelling
    stack. A small ridge term keeps the Hessian invertible under separation,
    which is common when an error indicator is rare.

    ``X`` must already include an intercept column.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        gradient = X.T @ (y - mu) - ridge * beta
        hessian = (X.T * w) @ X + ridge * np.eye(p)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def differential_error_audit(
    errors: Sequence[int], covariates: Sequence[Sequence[float]] | np.ndarray,
    names: Sequence[str], n_boot: int = 1000, seed: int = 0,
    standardise: bool = True,
) -> dict[str, Any]:
    """Test whether extraction error is patterned by facility characteristics.

    Fits ``error ~ covariates`` by logistic regression and bootstraps each
    coefficient over documents. A covariate whose interval excludes zero is
    flagged.

    Why this gates the equity analysis: random misclassification attenuates
    associations toward the null, so a finding that survives it is conservative.
    But error that is *itself* correlated with community composition or
    ownership does not attenuate — it can create an apparent disparity where
    none exists, in precisely the analysis this paper treats as its equity
    contribution. The decision rule is therefore committed here in advance: any
    flagged covariate blocks the corresponding equity claim unless the estimate
    is reported with an explicit attenuation correction and the flag disclosed.

    Covariates are standardised by default so coefficients are comparable across
    variables measured on different scales.
    """
    y = _as_1d(errors, "errors").astype(float)
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"covariates has {X.shape[0]} rows but errors has {y.shape[0]}"
        )
    if X.shape[1] != len(names):
        raise ValueError(
            f"names has {len(names)} entries for {X.shape[1]} covariates"
        )
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("errors must be a binary indicator (0/1)")

    if standardise:
        centre, scale = X.mean(axis=0), X.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        X = (X - centre) / scale

    design = np.column_stack([np.ones(X.shape[0]), X])
    beta = _logistic_irls(design, y)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, design.shape[1]), dtype=float)
    n = design.shape[0]
    kept = 0
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        y_boot = y[index]
        if y_boot.min() == y_boot.max():
            continue  # degenerate resample: no variation in the outcome
        draws[kept] = _logistic_irls(design[index], y_boot)
        kept += 1
    draws = draws[:kept]

    per_covariate = {}
    flags: list[str] = []
    for position, name in enumerate(names, start=1):
        if kept >= 20:
            lo, hi = np.percentile(draws[:, position], [2.5, 97.5])
            flagged = bool(lo > 0 or hi < 0)
        else:
            lo = hi = None
            flagged = False
        per_covariate[name] = {
            "coefficient": float(beta[position]),
            "ci_lo": _finite_or_none(lo) if lo is not None else None,
            "ci_hi": _finite_or_none(hi) if hi is not None else None,
            "flagged": flagged,
        }
        if flagged:
            flags.append(name)

    return {
        "n": int(n),
        "error_rate": float(y.mean()),
        "intercept": float(beta[0]),
        "standardised": bool(standardise),
        "n_bootstrap_kept": int(kept),
        "covariates": per_covariate,
        "flagged": flags,
        "differential_error_detected": bool(flags),
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def evaluate_gate(
    results: Mapping[str, Any], thresholds: Thresholds
) -> dict[str, Any]:
    """Compare measured diagnostics against the pre-registered thresholds.

    Returns a record with an overall ``passed`` flag, per-criterion outcomes, and
    human-readable reasons. ``07_experiments.py`` consumes this and withholds the
    dependent analyses when it fails; ``equity_permitted`` is reported separately
    because differential error blocks the equity claim specifically rather than
    the whole study.

    A criterion whose input is absent is recorded as ``None`` and counted as not
    satisfied: a check that was never run cannot license the analysis it was
    meant to protect.
    """
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, observed: float | None, ok: bool | None, rule: str) -> None:
        checks[name] = {"observed": _finite_or_none(observed) if observed is not None
                        else None, "passed": ok, "rule": rule}

    alpha = results.get("human_alpha")
    q = _nested(results, "human_alpha_bootstrap", "q",
                f"{thresholds.alpha_min:g}")
    record("human_reliability", q,
           None if q is None else bool(q <= thresholds.alpha_q_max),
           f"q = P(alpha < {thresholds.alpha_min}) <= "
           f"{thresholds.alpha_q_max} (pair-level bootstrap; "
           f"point alpha = {alpha})")

    f1 = _nested(results, "model_vs_human", "f1", "point")
    record("model_f1", f1, None if f1 is None else bool(f1 >= thresholds.f1_min),
           f"binary F1 >= {thresholds.f1_min}")

    kappa = _nested(results, "model_vs_human", "kappa", "point")
    record("model_kappa", kappa,
           None if kappa is None else bool(kappa >= thresholds.kappa_min),
           f"Cohen kappa >= {thresholds.kappa_min}")

    spread = (results.get("prompt_sensitivity") or {}).get("prevalence_spread")
    record("prompt_stability", spread,
           None if spread is None else bool(spread <= thresholds.prompt_max_spread),
           f"prevalence spread <= {thresholds.prompt_max_spread}")

    rho = (results.get("model_sensitivity") or {}).get("facility_rho")
    record("cross_model", rho,
           None if rho is None else bool(rho >= thresholds.cross_model_rho_min),
           f"facility-level Spearman rho >= {thresholds.cross_model_rho_min}")

    audit = results.get("differential_error") or {}
    n_flags = len(audit.get("flagged", [])) if audit else None
    record("differential_error", n_flags,
           None if not audit else
           bool(n_flags <= thresholds.differential_error_max_flags),
           f"flagged covariates <= {thresholds.differential_error_max_flags}")

    core = ["human_reliability", "model_f1", "model_kappa",
            "prompt_stability", "cross_model"]
    passed = all(checks[name]["passed"] is True for name in core)
    reasons = [
        f"{name}: observed {checks[name]['observed']} "
        f"fails rule ({checks[name]['rule']})"
        if checks[name]["passed"] is False else
        f"{name}: not evaluated ({checks[name]['rule']})"
        for name in checks
        if checks[name]["passed"] is not True
    ]

    minimum = _nested(results, "per_category_alpha", "min_alpha")
    record("per_category_alpha_min", minimum,
           None if minimum is None else bool(minimum >= thresholds.alpha_min),
           f"minimum per-category alpha >= {thresholds.alpha_min} "
           f"(gates per-role claims only, not the study)")

    return {
        "passed": bool(passed),
        "equity_permitted": bool(
            passed and checks["differential_error"]["passed"] is True
        ),
        # Krippendorff's minimum rule licenses claims about individual
        # categories; failing it withholds the per-role breakdown, not the
        # study, mirroring how equity_permitted narrows rather than vetoes.
        "per_role_claims_permitted": bool(
            passed and checks["per_category_alpha_min"]["passed"] is True
        ),
        "checks": checks,
        "reasons": reasons,
        "thresholds": {field: getattr(thresholds, field)
                       for field in Thresholds.__dataclass_fields__},
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


__all__ = [
    "MISSING", "Thresholds", "bootstrap_ci", "spearman_rho",
    "krippendorff_alpha", "cohens_kappa", "agreement_binary",
    "agreement_multiclass", "confusion_matrix", "macro_f1",
    "prompt_sensitivity", "model_sensitivity", "differential_error_audit",
    "evaluate_gate",
]
