"""
07_experiments.py — The orchestrator ("run everything").

Stage 7 of the pipeline and the entry point a replicator invokes. It composes
the other modules, enforces the ordering the study's design requires, and writes
one machine-readable results file that ``08_figures.py`` turns into every
exhibit in the manuscript.

No primitive is computed here. Every number comes from ``02``–``06``; this file
decides *what runs, in what order, under which seed, and under what conditions*.
Concentrating that in one place means the experimental logic can be audited
without reading five statistical modules.

THE GATE IS STRUCTURAL, NOT ADVISORY
------------------------------------
Validation runs first and its outcome is a precondition, declared in the stage
registry rather than left to the reader's discipline. A stage that requires the
gate does not run when the gate fails: it is recorded as ``withheld`` with the
failing criteria attached. The equity analysis carries an additional
precondition — the differential-error audit — because patterned extraction error
can manufacture the very disparity that analysis would report, whereas the other
analyses only suffer attenuation. Encoding those conditions as data, and
checking them in one loop, is what makes "validation gates the analysis" a
property of the program rather than a claim in the methods section.

PER-STAGE SEEDING
-----------------
Each stage is seeded from a deterministic function of the base seed and the
stage's own name, never from a single global stream consumed in order. This
matters for reproducibility in a way that is easy to miss: with one shared
stream, running ``--stage rq4`` alone would consume different random draws than
running it after ``rq1``, and the two would disagree. Deriving the seed from the
stage name makes each stage's result identical whether it is run alone, rerun
after a failure, or run as part of the full pipeline.

FAILURE ISOLATION
-----------------
A stage that raises does not abort the run. It is recorded with its exception
and traceback, stages that depend on it are marked ``skipped``, and independent
stages still execute. A long extraction run must not be lost because a
downstream plot-table stage had a bug. Results are written after every stage, so
an interrupted run retains everything completed up to that point.

Usage
-----
    python src/07_experiments.py --list
    python src/07_experiments.py --dry-run
    python src/07_experiments.py --stage all
    python src/07_experiments.py --stage rq4
    python src/07_experiments.py --stage all --strict

Exit codes
----------
    0  every requested stage completed (or is pending implementation)
    1  at least one stage failed unexpectedly
    2  the validation gate did not pass, so analyses were withheld
    3  a required input is missing; run the earlier stage named in the log
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from utils import (RESULTS, SRC, Config, DATA_PROCESSED, ensure_dirs,
                   get_logger, provenance, save_json, set_seed, timed)

LOG = get_logger("experiments")

# Stage outcome vocabulary. Recorded verbatim in the results file.
COMPLETED = "completed"          # ran and returned a result
FAILED = "failed"                # raised an unexpected exception
WITHHELD = "withheld"            # a precondition (the gate) was not satisfied
SKIPPED = "skipped"              # a stage it depends on did not complete
PENDING = "pending"              # module not yet implemented (NotImplementedError)
MISSING_INPUT = "missing_input"  # a required upstream artefact is absent
NOT_REQUESTED = "not_requested"  # excluded by --stage


# --------------------------------------------------------------------------- #
# Sibling module loading
# --------------------------------------------------------------------------- #

_MODULE_CACHE: dict[str, Any] = {}


def load_stage_module(filename: str) -> Any:
    """Import a numbered sibling module by filename.

    ``02_corpus.py`` and its siblings are not importable as normal packages —
    a module name cannot begin with a digit — so they are loaded from an
    explicit file location. The module is registered in ``sys.modules`` *before*
    execution because ``dataclasses`` resolves a class's module during
    decoration and fails with an opaque ``AttributeError`` if it is absent.

    Loading is lazy and cached, so a machine without the inference stack can
    still run the analysis stages: modules are imported only when a stage that
    needs them actually runs.
    """
    if filename in _MODULE_CACHE:
        return _MODULE_CACHE[filename]
    path = SRC / filename
    if not path.is_file():
        raise FileNotFoundError(f"pipeline module not found: {path}")
    name = "pipeline_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # must precede exec_module
    spec.loader.exec_module(module)
    _MODULE_CACHE[filename] = module
    return module


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def stage_seed(base_seed: int, stage_name: str) -> int:
    """Derive a stage's seed from the base seed and the stage's name.

    Deterministic, order-independent, and stable across machines and Python
    versions (BLAKE2b rather than the randomised built-in ``hash``). Two stages
    receive different seeds; the same stage receives the same seed no matter
    what ran before it.
    """
    digest = hashlib.blake2b(
        f"{base_seed}|{stage_name}".encode("utf-8"), digest_size=4
    ).digest()
    return int.from_bytes(digest, "big")


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Stage:
    """One unit of work, with its preconditions declared as data.

    Attributes
    ----------
    requires_gate
        The stage runs only if the validation gate passed.
    requires_equity_clearance
        Additionally requires that the differential-error audit found no
        covariate patterning the extraction error. Only the equity analysis sets
        this: non-differential error attenuates estimates toward the null and is
        survivable, whereas differential error can create an apparent disparity
        where none exists.
    depends_on
        Stages whose results this one consumes.
    """

    name: str
    run: Callable[["RunContext"], dict[str, Any]]
    description: str
    requires_gate: bool = False
    requires_equity_clearance: bool = False
    depends_on: tuple[str, ...] = ()
    exhibits: tuple[str, ...] = ()


@dataclass
class RunContext:
    """Everything a stage function needs, plus results already computed."""

    cfg: Config
    seed: int
    results: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)

    def result_of(self, stage_name: str) -> dict[str, Any]:
        """Return a completed stage's result, or raise if it is unavailable."""
        record = self.results.get(stage_name)
        if not record or record.get("status") != COMPLETED:
            raise RuntimeError(
                f"stage {stage_name!r} has no completed result to consume"
            )
        return record["result"]


# --- stage implementations -------------------------------------------------- #

def _run_study1(ctx: RunContext) -> dict[str, Any]:
    """Study 1 — document analysis of Appendix PP. No LLM, no gate.

    Runs the vocabulary census and reports the shipped taxonomy's level
    counts and provenance. Requires only the Appendix PP text extraction; a
    missing file surfaces as ``missing_input`` with the acquisition
    instruction in the exception message, not as a failure.
    """
    analysis = load_stage_module("05_analysis.py")
    cfg = ctx.cfg
    from utils import ROOT
    appendix = Path(cfg.get("analysis.appendix_pp_file",
                            "data/raw/appendix-pp.txt"))
    if not appendix.is_absolute():
        appendix = ROOT / appendix
    if not appendix.is_file():
        # Checked before the term-list configuration on purpose: without the
        # source document nothing can run regardless of configuration, and
        # ``missing_input`` (with the acquisition instruction) is the
        # actionable classification, not ``failed``.
        raise FileNotFoundError(
            f"Appendix PP text not found at {appendix}. Download the current "
            f"'State Operations Manual, Appendix PP' PDF from cms.gov and "
            f"convert it with `pdftotext -layout <pdf> {appendix}`."
        )
    census = analysis.appendix_pp_census(
        appendix, cfg.get("analysis.census_terms", {})
    )
    taxonomy = analysis.load_tag_taxonomy(cfg)
    levels = list(taxonomy.values())
    return {
        "census": census,
        "taxonomy_level_counts": {
            level: levels.count(level)
            for level in analysis.TAXONOMY_LEVELS
        },
        "n_tags": len(taxonomy),
    }


def _optional_check(label: str, function, *args, **kwargs) -> tuple[Any, str | None]:
    """Run a diagnostic that may be impossible under the current configuration.

    Returns ``(result, unavailable_reason)``. A check that cannot be run — only
    one prompt variant configured, no second model, too few facilities — is a
    check that was *not performed*, which is different from one that failed. It
    must not abort the whole validation stage and discard the diagnostics that
    did run.

    The gate treats an absent observation as not satisfied, so degrading here
    cannot let an unvalidated study proceed: it only preserves the other
    numbers, and records plainly why the missing one is missing.
    """
    try:
        return function(*args, **kwargs), None
    except (ValueError, KeyError) as exc:
        LOG.warning("validation check %r not evaluated: %s", label, exc)
        return None, str(exc)


def _run_validate(ctx: RunContext) -> dict[str, Any]:
    """Measurement validation: reliability, agreement, sensitivity, error audit.

    Consumes the corpus and extraction stages, then applies every check in
    ``04_validate``. Returns the raw diagnostics; the gate itself is evaluated by
    the runner, so the pass/fail decision is visible at the orchestration level
    rather than buried inside a stage.
    """
    corpus_mod = load_stage_module("02_corpus.py")
    extract_mod = load_stage_module("03_extract.py")
    validate = load_stage_module("04_validate.py")
    cfg = ctx.cfg
    n_boot = int(cfg.get("validation.n_bootstrap", 1000))

    corpus = corpus_mod.build_corpus(cfg)
    sample = corpus_mod.stratified_sample(
        corpus, int(cfg.get("corpus.validation_sample_n", 1200)), ctx.seed,
        severity_floors=cfg.get("corpus.sampling.min_per_severity_level", {}),
    )
    extracted = extract_mod.extract_corpus(sample, cfg)

    unavailable: dict[str, str] = {}
    human, reason = _optional_check(
        "human_alpha", validate.krippendorff_alpha,
        extracted["annotation_matrix"], "nominal",
    )
    if reason:
        unavailable["human_alpha"] = reason

    # Always include the configured alpha_min in the q-threshold set so that
    # evaluate_gate can always find its key; keep the standard ladder too so
    # that the table can show the full picture.
    _configured_min = float(cfg.get("validation.alpha_min", 0.80))
    _standard_mins: tuple[float, ...] = (0.9, 0.8, 0.7, 0.667)
    _alpha_mins = tuple(sorted(
        set(_standard_mins) | {_configured_min}, reverse=True
    ))
    alpha_boot, reason = _optional_check(
        "human_alpha_bootstrap", validate.alpha_bootstrap,
        extracted["annotation_matrix"], "nominal",
        n_boot=int(cfg.get("validation.alpha_bootstrap", 10000)),
        seed=ctx.seed,
        alpha_mins=_alpha_mins,
    )
    if reason:
        unavailable["human_alpha_bootstrap"] = reason

    per_category, reason = _optional_check(
        "per_category_alpha", validate.krippendorff_alpha_by_category,
        extracted["annotation_matrix"],
    )
    if reason:
        unavailable["per_category_alpha"] = reason

    binary = validate.agreement_binary(
        extracted["pred_technology_present"], extracted["gold_technology_present"],
        n_boot=n_boot, seed=ctx.seed,
    )
    roles, reason = _optional_check(
        "failure_role_agreement", validate.agreement_multiclass,
        extracted["pred_failure_role"], extracted["gold_failure_role"],
        n_boot=n_boot, seed=ctx.seed + 1,
    )
    if reason:
        unavailable["failure_role_agreement"] = reason

    prompts, reason = _optional_check(
        "prompt_sensitivity", validate.prompt_sensitivity,
        extracted["predictions_by_prompt"],
    )
    if reason:
        unavailable["prompt_sensitivity"] = reason

    models, reason = _optional_check(
        "model_sensitivity", validate.model_sensitivity,
        extracted["pred_primary"], extracted["pred_secondary"],
        extracted["facility_ids"],
        min_facility_docs=int(cfg.get("validation.min_facility_docs", 5)),
    )
    if reason:
        unavailable["model_sensitivity"] = reason

    audit, reason = _optional_check(
        "differential_error", validate.differential_error_audit,
        extracted["errors"], extracted["covariates"], extracted["covariate_names"],
        n_boot=n_boot, seed=ctx.seed + 2,
    )
    if reason:
        unavailable["differential_error"] = reason

    if unavailable:
        LOG.warning("%d validation check(s) not evaluated; the gate treats an "
                    "unevaluated check as not satisfied.", len(unavailable))
    gold_pred = extracted["pred_technology_present"]
    gold_true = extracted["gold_technology_present"]
    positives = gold_true == 1
    negatives = gold_true == 0
    error_rates = {
        "sensitivity": (float(gold_pred[positives].mean())
                        if positives.any() else None),
        "specificity": (float(1.0 - gold_pred[negatives].mean())
                        if negatives.any() else None),
        "n_gold": int(len(gold_true)),
    }

    return {
        "human_alpha": human,
        "human_alpha_bootstrap": alpha_boot,
        "per_category_alpha": per_category,
        "error_rates": error_rates,
        "gold_pred": [int(v) for v in gold_pred],
        "gold_true": [int(v) for v in gold_true],
        "self_consistency": extracted.get("self_consistency"),
        "n_refusals": int(extracted.get("n_refusals", 0)),
        "model_vs_human": binary,
        "failure_role_agreement": roles,
        "prompt_sensitivity": prompts,
        "model_sensitivity": models,
        "differential_error": audit,
        "checks_not_evaluated": unavailable,
        "stats": extracted.get("stats", {}),
    }


def _load_analysis_frame(ctx: RunContext) -> Any:
    """Load the full-corpus annotated frame saved by 03_extract.py.

    The analysis RQs require the full corpus with extraction predictions
    joined; the validation stage only annotates a sample. In a normal run
    03_extract.py writes ``data/processed/extracted_<fingerprint>.parquet``
    before the orchestrator is invoked. Here we load the most recent file
    matching the current config fingerprint, falling back to the newest
    extracted file regardless of fingerprint (with a warning) so that a run
    with a stale fingerprint still produces output rather than silently
    missing data.
    """
    fingerprint = ctx.cfg.fingerprint()
    # Prefer fingerprint-matched file; fall back to any extracted file.
    candidates = sorted(DATA_PROCESSED.glob(f"extracted_{fingerprint}*.parquet"))
    if not candidates:
        candidates = sorted(DATA_PROCESSED.glob("extracted_*.parquet"))
        if candidates:
            LOG.warning(
                "no extracted file for config fingerprint %s; "
                "using %s — re-run 03_extract.py to remove this warning",
                fingerprint, candidates[-1].name,
            )
    if not candidates:
        raise RuntimeError(
            "no extraction output found in data/processed/. "
            "Run  python src/03_extract.py --allow-stub  (or the full "
            "vLLM extraction) before the orchestrator."
        )
    import pandas as pd
    frame = pd.read_parquet(candidates[-1])
    LOG.info("analysis frame: %d rows from %s", len(frame), candidates[-1].name)
    return frame


def _run_rq1(ctx: RunContext) -> dict[str, Any]:
    """RQ1 — prevalence of technology-mediated failures and its trend."""
    analysis = load_stage_module("05_analysis.py")
    frame = _load_analysis_frame(ctx)
    trend = analysis.prevalence_trend(
        frame, by=("year",),
        denominator=ctx.cfg.get("analysis.exposure_denominator", "surveys"),
    )
    # Misclassification-corrected headline prevalence (Rogan-Gladen; the DSL
    # logic Ziems et al. point to): naive rate from the full corpus, error
    # rates from the validation stage's gold subset. Reported alongside the
    # naive series, never silently replacing it.
    validate_entry = (ctx.results or {}).get("validate") or {}
    validate_result = validate_entry.get("result") or {}
    gold_pred = validate_result.get("gold_pred")
    gold_true = validate_result.get("gold_true")
    corrected = None
    reason = None
    if gold_pred and gold_true:
        naive = float(
            frame["technology_present"].fillna(False).astype(bool).mean()
        )
        try:
            corrected = analysis.bias_corrected_prevalence(
                naive, gold_pred, gold_true, seed=ctx.seed
            )
        except analysis.AnalysisError as exc:
            reason = str(exc)
    else:
        reason = "validation stage results unavailable in this run"
    trend["corrected_prevalence"] = corrected
    if reason:
        trend["correction_unavailable"] = reason
    return trend


def _run_rq2(ctx: RunContext) -> dict[str, Any]:
    """RQ2 — the illegibility gap, with its rival explanations tested."""
    analysis = load_stage_module("05_analysis.py")
    frame = _load_analysis_frame(ctx)
    taxonomy = analysis.load_tag_taxonomy(ctx.cfg)
    return {
        "gap": analysis.illegibility_gap(frame, taxonomy),
        "tag_adequacy": analysis.tag_adequacy_check(frame, taxonomy),
    }


def _run_rq3(ctx: RunContext) -> dict[str, Any]:
    """RQ3 — distribution of the gap across communities and ownership forms."""
    analysis = load_stage_module("05_analysis.py")
    frame = _load_analysis_frame(ctx)
    taxonomy = analysis.load_tag_taxonomy(ctx.cfg)
    # hierarchical_equity_model requires the `illegible` indicator; attach it
    # from the taxonomy before the equity model consumes the frame.
    frame = analysis.attach_illegible(frame, taxonomy)
    return analysis.hierarchical_equity_model(
        frame,
        covariates=ctx.cfg.get("analysis.equity_attrs", []),
        gate=ctx.gate,
    )


def _run_rq4(ctx: RunContext) -> dict[str, Any]:
    """RQ4 — diffusion fit, mandatory backtest, and scenario projection.

    The backtest can fail. When it does the projection is still computed and
    returned, but flagged ``backtest_passed = False``, and the manuscript
    reports the failure rather than presenting the trajectory as earned.
    """
    forecast = load_stage_module("06_forecast.py")
    series = ctx.result_of("rq1")["series"]
    t, y = series["t"], series["counts"]
    spec = ctx.cfg.forecast

    comparison = forecast.compare_models(t, y, spec.get("models", ("bass", "logistic")))
    model = comparison["best_model"] or "bass"
    fit = forecast.fit_diffusion(t, y, model)
    # Backtest can fail when the series is too short (Mahajan, Muller & Bass:
    # even 20 periods may be insufficient; with fewer the backtest itself is
    # uninformative and the manuscript notes the limitation rather than crashing.)
    try:
        back = forecast.backtest(
            t, y, int(spec.get("backtest_cutoff_index", 40)), model,
            float(spec.get("backtest_mape_max", 0.30)),
        )
        backtest_note = None
    except ValueError as exc:
        LOG.warning("RQ4 backtest skipped: %s", exc)
        back = {"passed": False, "note": str(exc)}
        backtest_note = str(exc)
    scenarios = forecast.scenarios_from_config(spec.get("scenarios", {}))
    projection = forecast.project_scenarios(
        t, y, fit, int(spec.get("horizon_periods", 20)), scenarios,
        n_boot=int(spec.get("n_bootstrap", 200)), seed=ctx.seed,
        alpha=float(spec.get("interval_alpha", 0.05)),
        require_identifiable=bool(spec.get("require_identifiable_m", True)),
    )
    return {
        "comparison": comparison,
        "selected_model": model,
        "m_identifiable": fit.m_identifiable,
        "identifiability": fit.identifiability,
        "fit": fit.to_dict(),
        "backtest": back,
        "backtest_passed": bool(back.get("passed")),
        "backtest_note": backtest_note,
        "projection": projection,
    }


REGISTRY: tuple[Stage, ...] = (
    Stage("study1", _run_study1,
          "Study 1: Appendix PP vocabulary census and taxonomy provenance.",
          exhibits=("Table 1",)),
    Stage("validate", _run_validate,
          "Measurement validation; produces the gate.",
          exhibits=("Table 2", "Table A1")),
    Stage("rq1", _run_rq1,
          "Prevalence of technology-mediated failures and its trend.",
          requires_gate=True, exhibits=("Figure 2",)),
    Stage("rq2", _run_rq2,
          "The illegibility gap and its rival explanations.",
          requires_gate=True, exhibits=("Figure 3", "Table 3")),
    Stage("rq3", _run_rq3,
          "Distribution of the gap by community and ownership.",
          requires_gate=True, requires_equity_clearance=True,
          exhibits=("Table 4",)),
    Stage("rq4", _run_rq4,
          "Diffusion fit, backtest, and scenario projection.",
          requires_gate=True, depends_on=("rq1",), exhibits=("Figure 4",)),
)

STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in REGISTRY)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def _blocked_reason(
    stage: Stage, gate: dict[str, Any], results: dict[str, Any]
) -> tuple[str, str] | None:
    """Return ``(status, reason)`` if a stage must not run, else ``None``."""
    for dependency in stage.depends_on:
        record = results.get(dependency)
        if not record or record.get("status") != COMPLETED:
            status = record.get("status") if record else "never run"
            return SKIPPED, f"depends on stage {dependency!r}, which is {status}"

    if stage.requires_gate and not gate.get("passed", False):
        reasons = "; ".join(gate.get("reasons", [])) or "gate not evaluated"
        return WITHHELD, f"validation gate did not pass: {reasons}"

    if stage.requires_equity_clearance and not gate.get("equity_permitted", False):
        flagged = ((results.get("validate") or {}).get("result", {})
                   .get("differential_error", {}).get("flagged", []))
        return WITHHELD, (
            "differential extraction error was detected"
            + (f" for {flagged}" if flagged else "")
            + "; patterned error can manufacture an apparent disparity, so this "
              "analysis is withheld rather than reported with a caveat"
        )
    return None


def run_pipeline(
    cfg: Config, requested: Sequence[str], results_path: Path,
    force_gate: bool = False,
) -> dict[str, Any]:
    """Execute the requested stages in registry order and return the results."""
    validate_mod = load_stage_module("04_validate.py")
    thresholds = validate_mod.Thresholds.from_config(cfg.validation)

    from utils import DATA_RAW
    manifest = DATA_RAW / "_manifest.json"
    if not manifest.is_file():
        LOG.warning(
            "No acquisition manifest at %s. Stages that read raw data will "
            "report '%s'. Run `make acquire` first.", manifest, MISSING_INPUT
        )

    record: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_utc": None,
        "config_fingerprint": cfg.fingerprint(),
        "base_seed": cfg.seed,
        "requested_stages": list(requested),
        "provenance": provenance(cfg),
        "gate": {},
        "stages": {},
    }
    ctx = RunContext(cfg=cfg, seed=cfg.seed)

    def flush() -> None:
        """Persist after every stage so an interrupted run keeps its work."""
        save_json(record, results_path)

    for stage in REGISTRY:
        if stage.name not in requested:
            record["stages"][stage.name] = {
                "status": NOT_REQUESTED, "description": stage.description
            }
            continue

        blocked = _blocked_reason(stage, record["gate"], record["stages"])
        if blocked and not (force_gate and blocked[0] == WITHHELD):
            status, reason = blocked
            LOG.warning("%-9s %s: %s", stage.name, status.upper(), reason)
            record["stages"][stage.name] = {
                "status": status, "reason": reason,
                "description": stage.description, "exhibits": list(stage.exhibits),
            }
            flush()
            continue
        if blocked and force_gate:
            LOG.warning("%-9s gate OVERRIDDEN (--force-gate): %s",
                        stage.name, blocked[1])

        seed = stage_seed(cfg.seed, stage.name)
        ctx.seed = seed
        ctx.results = record["stages"]
        ctx.gate = record["gate"]
        set_seed(seed)

        entry: dict[str, Any] = {
            "description": stage.description,
            "exhibits": list(stage.exhibits),
            "seed": seed,
        }
        started = time.perf_counter()
        try:
            with timed(f"stage {stage.name}", LOG):
                entry["result"] = stage.run(ctx)
            entry["status"] = COMPLETED
        except NotImplementedError as exc:
            # An honest, expected state during development: the stage's
            # dependencies exist but are not yet written. Distinguished from a
            # real failure so a partial pipeline reports accurately.
            entry["status"] = PENDING
            entry["reason"] = f"not yet implemented: {exc}"
            LOG.warning("%-9s PENDING: %s", stage.name, exc)
        except FileNotFoundError as exc:
            # A required upstream artefact is absent. This is an operator
            # instruction, not a defect, and burying it in a traceback would
            # make a replicator debug code when they simply need to run an
            # earlier stage.
            entry["status"] = MISSING_INPUT
            entry["reason"] = str(exc)
            entry["hint"] = (
                "Run the acquisition stage first: `python src/01_acquire.py` "
                "(or `make acquire`), then re-run this stage."
            )
            LOG.error("%-9s MISSING INPUT: %s", stage.name, exc)
            LOG.error("           %s", entry["hint"])
        except Exception as exc:  # noqa: BLE001 - isolate stage failures
            entry["status"] = FAILED
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            entry["traceback"] = traceback.format_exc()
            LOG.error("%-9s FAILED: %s: %s", stage.name, type(exc).__name__, exc)
        entry["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        record["stages"][stage.name] = entry

        if stage.name == "validate" and entry["status"] == COMPLETED:
            record["gate"] = validate_mod.evaluate_gate(entry["result"], thresholds)
            outcome = "PASSED" if record["gate"]["passed"] else "NOT PASSED"
            LOG.info("Validation gate %s (equity permitted: %s)",
                     outcome, record["gate"]["equity_permitted"])
            for reason in record["gate"]["reasons"]:
                LOG.warning("  gate: %s", reason)
        flush()

    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["summary"] = summarise(record)
    flush()
    return record


def summarise(record: dict[str, Any]) -> dict[str, Any]:
    """Counts by status, plus the exhibits that will and will not be produced."""
    counts: dict[str, int] = {}
    produced: list[str] = []
    missing: list[str] = []
    for name, entry in record["stages"].items():
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        target = produced if status == COMPLETED else missing
        target.extend(entry.get("exhibits", []))
    return {
        "counts": counts,
        "missing_inputs": [
            name for name, entry in record["stages"].items()
            if entry.get("status") == MISSING_INPUT
        ],
        "gate_passed": bool(record.get("gate", {}).get("passed", False)),
        "equity_permitted": bool(record.get("gate", {}).get("equity_permitted", False)),
        "exhibits_available": sorted(set(produced)),
        "exhibits_unavailable": sorted(set(missing) - set(produced)),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _resolve_stages(selection: str) -> list[str]:
    if selection == "all":
        return list(STAGE_NAMES)
    wanted = [name.strip() for name in selection.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in STAGE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown stage(s) {unknown}; available: {list(STAGE_NAMES)}"
        )
    # Preserve registry order regardless of the order given on the command line,
    # because the gate must be evaluated before the stages it governs.
    return [name for name in STAGE_NAMES if name in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the study end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", default="all",
                        help="'all' or a comma-separated subset")
    parser.add_argument("--list", action="store_true",
                        help="print the stage registry and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the execution plan without running it")
    parser.add_argument("--strict", action="store_true",
                        help="treat pending (unimplemented) stages as failures")
    parser.add_argument("--force-gate", action="store_true",
                        help="run gated stages even if validation failed; for "
                             "debugging only, and recorded in the results file")
    args = parser.parse_args(argv)

    if args.list:
        for stage in REGISTRY:
            flags = []
            if stage.requires_gate:
                flags.append("gated")
            if stage.requires_equity_clearance:
                flags.append("equity-cleared")
            if stage.depends_on:
                flags.append("after " + ",".join(stage.depends_on))
            print(f"{stage.name:10s} {stage.description}"
                  f"{'  [' + '; '.join(flags) + ']' if flags else ''}")
        return 0

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path
    cfg = Config.load(config_path)
    ensure_dirs()

    try:
        stages = _resolve_stages(args.stage)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 1

    if args.dry_run:
        LOG.info("Plan (config fingerprint %s, base seed %d):",
                 cfg.fingerprint(), cfg.seed)
        for name in stages:
            LOG.info("  %-10s seed=%d", name, stage_seed(cfg.seed, name))
        return 0

    results_path = RESULTS / f"results_{cfg.fingerprint()}.json"
    if args.force_gate:
        LOG.warning("--force-gate is set: gated stages will run regardless of "
                    "validation outcome. This is recorded in the results file.")

    record = run_pipeline(cfg, stages, results_path, force_gate=args.force_gate)
    record["force_gate"] = bool(args.force_gate)
    save_json(record, results_path)

    summary = record["summary"]
    LOG.info("Finished: %s", summary["counts"])
    LOG.info("Results: %s", results_path)

    if summary["counts"].get(FAILED):
        return 1
    if args.strict and summary["counts"].get(PENDING):
        return 1
    if summary["counts"].get(MISSING_INPUT):
        return 3
    if summary["counts"].get(WITHHELD):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
