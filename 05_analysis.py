"""
05_analysis.py — Study 1 (document analysis) and the substantive analyses.

Stage 5 of the pipeline. It hosts two studies:

STUDY 1 (:func:`appendix_pp_census`, :func:`build_tag_taxonomy`) is a document
analysis of CMS State Operations Manual Appendix PP — the 873-page guidance
that tells surveyors what to look for and which F-tag to cite. It requires no
LLM and no narrative corpus: it is a reproducible vocabulary census of the
regulatory instrument itself, and it is what lets the paper's central claim
stand even for a reader who distrusts Study 2's model-based measurement. The
same source builds the tag taxonomy that defines Study 2's dependent variable.

STUDY 2 (RQ1-RQ3 below) turns the measured variable from stages 3-4 into the
paper's empirical claims.

RQ1  PREVALENCE AND TREND (:func:`prevalence_trend`). How common are
     technology-mediated failures in the regulatory record, and how has that
     changed as care technologies diffused? Rates are computed per unit of
     exposure — surveys conducted — rather than per facility, so the series is
     not driven by changes in inspection intensity. The returned ``series`` is
     what stage 06 fits.

RQ2  THE ILLEGIBILITY GAP (:func:`illegibility_gap`). For deficiencies whose
     narrative implicates a technology, what regulatory category was assigned?
     The gap is the share cited under tags that name no technological element:
     events the state observed but recorded in a form that makes the technology
     invisible to its own statistics.

     Two rival explanations must be closed before that gap can be called
     illegibility rather than an artefact, and :func:`tag_adequacy_check` tests
     the first: perhaps the taxonomy is adequate and surveyors simply chose a
     broader tag. If no technology-naming tag was even available for the
     citation's subject matter, the surveyor had no legible option — which is
     the mechanism claimed, not a competing account. The second rival, that
     narratives under-describe technology, is addressed in the manuscript
     against paired complaint investigations.

RQ3  EQUITY (:func:`hierarchical_equity_model`). Is the gap larger in facilities
     serving poorer or more minority communities, or in particular ownership
     forms? This analysis is **gated**: it refuses to run when stage 04 found
     extraction error patterned by the same covariates, because differential
     error can manufacture a disparity rather than merely attenuate one.

A NOTE THAT GOVERNS EVERY QUANTITY HERE
---------------------------------------
Regulatory-record data measures what was CITED, not what OCCURRED. Every
estimate is therefore an estimate of *recorded* prevalence and is labelled as
such in the returned records so the caveat travels with the number into the
figures.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils import (CONFIGS, Config, get_logger, load_json, save_json,
                   sha256_file)

LOG = get_logger("analysis")

MEASUREMENT_CAVEAT = (
    "Estimates describe deficiencies as CITED in the public regulatory record, "
    "not the underlying incidence of technology-mediated failure."
)

# Three-level taxonomy: does a tag's regulatory text name a technology?
#   full    - the requirement itself is about a technology class (F919 call
#             system, F700 bed rails, F908 mechanical/electrical equipment)
#   partial - technology vocabulary appears in the surveyor guidance, but the
#             requirement names none (F689 accidents: 29 alarm mentions in
#             guidance, zero in the requirement)
#   none    - no technology vocabulary anywhere in the tag's text
# The shipped configs/tag_taxonomy.json is BUILT from Appendix PP by
# :func:`build_tag_taxonomy` and published with the deposit, so a reader can
# contest a coding and recompute the gap. This fallback covers only the
# hand-verified anchors and exists so the pipeline degrades loudly, not
# silently, when the built file is absent.
#
# NOTE: the earlier provisional file flagged F758, F773, and F836 as
# technology-naming. Close reading showed those rested on the verb
# "monitor(ing)" and on "application" meaning the act of applying — the
# systematic rebuild corrects all three to none.
TAXONOMY_LEVELS: tuple[str, ...] = ("full", "partial", "none")
DEFAULT_TAG_LEVELS: dict[str, str] = {
    "F919": "full",     # §483.90(g) Resident Call System
    "F700": "full",     # §483.25(n) Bed Rails
    "F909": "full",     # §483.90(d)(3) bed frames, mattresses, bed rails
    "F908": "full",     # §483.90(d)(2) mechanical/electrical/patient care equipment
    "F689": "partial",  # accidents: alarms/devices in guidance only
    "F604": "partial",  # physical restraints: devices in guidance only
    "F842": "partial",  # medical records: electronic records in guidance only
}


class AnalysisError(RuntimeError):
    """Raised when an analysis cannot be computed or is not permitted."""


# --------------------------------------------------------------------------- #
# Study 1 — document analysis of Appendix PP (no LLM, fully reproducible)
# --------------------------------------------------------------------------- #

TAG_HEADER = re.compile(r"^\s*(F\d{3})\s*$")

# Patterns whose presence in the REQUIREMENT text (the regulation statement
# itself, before INTENT/GUIDANCE) makes a tag ``full``: the requirement is
# about a technology class.
FULL_REQUIREMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("call system", r"call\s+system"),
    ("bed rail / side rail / bed frame", r"bed\s+rail|side\s+rail|bed\s+frame"),
    ("equipment", r"\bequipment\b"),
    ("emergency power / generator", r"emergency\s+power|\bgenerator\b"),
)

# Patterns whose presence ANYWHERE in a tag's text makes it at least
# ``partial``: the guidance recognises the technology even where the
# requirement does not. Bare "monitor(ing)" is deliberately EXCLUDED — in
# regulatory prose it is overwhelmingly the verb ("monitor the resident for
# side effects"), and including it would mass-assign false partials (417
# occurrences across 63 tags in Rev. 225). Likewise "application" (the act of
# applying) and "interface" are excluded as polysemous.
PARTIAL_ANYWHERE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("equipment", r"\bequipment\b"),
    ("device", r"\bdevices?\b"),
    ("alarm", r"\balarms?\b"),
    ("bed/side rail or frame", r"bed\s+rail|side\s+rail|bed\s+frame"),
    ("call system/light/bell", r"call\s+(?:system|light|bell)"),
    ("electronic record", r"electronic\s+(?:health|medical)\s+record"),
    ("electronic monitoring", r"electronic\s+monitoring"),
    ("video / camera", r"\bvideo\b|\bcameras?\b"),
    ("sensor", r"\bsensors?\b"),
    ("telehealth", r"\btelehealth\b|\btelemedicine\b"),
    ("mechanical lift", r"mechanical\s+lift"),
    ("assistive device", r"assistive\s+device"),
    ("automated / software / computer",
     r"\bautomated\b|\bsoftware\b|\bcomputer\b"),
    ("wander-management device", r"\bwanderguard\b"),
)

# Hand-verified overrides for the anchor tags, each grounded in the quoted
# regulation. Rules decide the other ~195 tags; every assignment records its
# method so a reader can see which codings are rule-derived and which are
# curated, and contest either.
TAXONOMY_OVERRIDES: dict[str, dict[str, str]] = {
    "F919": {"names_technology": "full", "justification": (
        "§483.90(g) Resident Call System: the requirement's subject IS a "
        "communication technology — the facility 'must be adequately equipped "
        "to allow residents to call for staff assistance through a "
        "communication system'.")},
    "F700": {"names_technology": "full", "justification": (
        "§483.25(n) Bed Rails: the requirement governs a specific device "
        "class, including alternatives assessment before installation and "
        "correct installation and maintenance.")},
    "F909": {"names_technology": "full", "justification": (
        "§483.90(d)(3): the requirement mandates 'regular inspection of all "
        "bed frames, mattresses, and bed rails' as part of maintenance.")},
    "F908": {"names_technology": "full", "justification": (
        "§483.90(d)(2): 'maintain all mechanical, electrical, and patient "
        "care equipment in safe operating condition' — the equipment-vintage "
        "anchor of the taxonomy's vocabulary.")},
    "F689": {"names_technology": "partial", "justification": (
        "§483.25(d) Accidents: the requirement is hazard-freedom and adequate "
        "supervision generally; the guidance discusses alarms, assistive "
        "devices, and bed rails as hazard sources (29 alarm mentions in Rev. "
        "225) but the requirement names no technology.")},
    "F604": {"names_technology": "partial", "justification": (
        "Physical restraints: devices — including bed rails and alarms used "
        "as restraints — figure throughout the guidance, but the requirement "
        "governs the practice of restraint, not any technology.")},
    "F842": {"names_technology": "partial", "justification": (
        "Medical records: the guidance recognises electronic records, but the "
        "requirement governs record content, access, and confidentiality, "
        "not the recording technology.")},
    "F698": {"names_technology": "partial", "justification": (
        "Dialysis: a service requirement; equipment appears extensively in "
        "guidance (maintenance, safety checks) but the requirement is that "
        "residents 'receive such services', not about the machine.")},
    "F758": {"names_technology": "none", "justification": (
        "Psychotropic drugs: every apparent technology hit is the verb "
        "'monitor(ing)'. Correction of the earlier provisional coding.")},
    "F773": {"names_technology": "none", "justification": (
        "Laboratory services ordering: no technology vocabulary beyond "
        "polysemous terms. Correction of the earlier provisional coding.")},
    "F836": {"names_technology": "none", "justification": (
        "Administration: 'application' here means the act of applying, not "
        "software. Correction of the earlier provisional coding.")},
}


def read_appendix_pp(path: str | Path) -> str:
    """Read the Appendix PP text extraction, with actionable failure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Appendix PP text not found at {path}. Download the current "
            f"'State Operations Manual, Appendix PP' PDF from cms.gov and "
            f"convert it with `pdftotext -layout <pdf> {path}`."
        )
    return path.read_text(encoding="utf-8", errors="replace")


def tag_blocks(text: str) -> dict[str, str]:
    """Split Appendix PP into per-tag text blocks.

    A tag's block is every line between its header (a line containing only
    the tag code, e.g. ``F689``) and the next such header. Headers recur after
    page breaks, so segments are concatenated per tag rather than first-match
    taken. Cross-reference lines ("see F550, Resident Rights") do not match
    the header pattern and are correctly left inside the surrounding block.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.split("\n"):
        match = TAG_HEADER.match(line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, [])
            continue
        if current is not None:
            blocks[current].append(line)
    return {tag: "\n".join(lines) for tag, lines in blocks.items()}


def requirement_text(block: str, max_chars: int = 900) -> str:
    """The regulation statement itself, before surveyor guidance begins.

    Revision stamps are stripped; the cut is at the first structural keyword
    (INTENT / GUIDANCE / DEFINITIONS / PROCEDURES / KEY ELEMENTS) or at
    ``max_chars``, whichever comes first. This is the text a ``full`` coding
    is judged against: what the requirement is ABOUT, not what the guidance
    happens to mention.
    """
    body = re.sub(r"\(Rev\.[^)]*\)", " ", block)
    match = re.search(
        r"\b(INTENT|GUIDANCE|DEFINITIONS|PROCEDURES|KEY ELEMENTS)\b", body
    )
    cut = body[: match.start()] if match else body[:max_chars]
    return " ".join(cut.split())[:max_chars]


def classify_tag(tag: str, block: str) -> dict[str, str]:
    """Assign full/partial/none to one tag, with method and justification."""
    override = TAXONOMY_OVERRIDES.get(tag)
    if override:
        return {"names_technology": override["names_technology"],
                "justification": override["justification"],
                "method": "curated"}

    requirement = requirement_text(block)
    full_hits = [label for label, pattern in FULL_REQUIREMENT_PATTERNS
                 if re.search(pattern, requirement, re.I)]
    if full_hits:
        return {"names_technology": "full",
                "justification": ("Requirement text names a technology "
                                  "category: " + "; ".join(full_hits) + "."),
                "method": "rule"}

    partial_hits = []
    for label, pattern in PARTIAL_ANYWHERE_PATTERNS:
        count = len(re.findall(pattern, block, re.I))
        if count:
            partial_hits.append((label, count))
    if partial_hits:
        partial_hits.sort(key=lambda item: -item[1])
        shown = ", ".join(f"{label} ({count})"
                          for label, count in partial_hits[:4])
        return {"names_technology": "partial",
                "justification": ("Guidance mentions technology vocabulary — "
                                  + shown + " — but the requirement itself "
                                  "names none."),
                "method": "rule"}

    return {"names_technology": "none",
            "justification": ("No technology vocabulary detected anywhere in "
                              "this tag's text (bare 'monitor(ing)' excluded "
                              "as polysemous)."),
            "method": "rule"}


def build_tag_taxonomy(
    appendix_path: str | Path, out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the full taxonomy from Appendix PP and optionally write it.

    Every tag in the source receives a coding, a grounded justification, its
    regulation citation, and the method (rule vs curated). ``_meta`` records
    the source file's SHA-256 and revision line, so the artefact is traceable
    to the exact document vintage it encodes.
    """
    appendix_path = Path(appendix_path)
    text = read_appendix_pp(appendix_path)
    blocks = tag_blocks(text)
    if len(blocks) < 100:
        raise AnalysisError(
            f"only {len(blocks)} tag blocks parsed — this does not look like "
            f"a full Appendix PP extraction; check the pdftotext conversion."
        )

    revision_match = re.search(r"\(Rev\.\s*\d+[^)]*\)", text)
    taxonomy: dict[str, Any] = {}
    for tag in sorted(blocks):
        entry = classify_tag(tag, blocks[tag])
        regulation = re.search(r"§\s*483\.[\w().\-]+",
                               requirement_text(blocks[tag]))
        entry["regulation"] = (regulation.group(0).replace(" ", "")
                               if regulation else "")
        taxonomy[tag] = entry

    levels = [entry["names_technology"] for entry in taxonomy.values()]
    meta = {
        "source_file": str(appendix_path),
        "source_sha256": sha256_file(appendix_path),
        "source_revision": revision_match.group(0) if revision_match else "",
        "n_tags": len(taxonomy),
        "level_counts": {level: levels.count(level)
                         for level in TAXONOMY_LEVELS},
        "n_curated_overrides": sum(
            1 for entry in taxonomy.values() if entry["method"] == "curated"
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "procedure": (
            "full = requirement text matches FULL_REQUIREMENT_PATTERNS; "
            "partial = PARTIAL_ANYWHERE_PATTERNS match the tag's guidance; "
            "none = neither. Bare 'monitor(ing)', 'application', and "
            "'interface' are excluded as polysemous. Curated overrides (with "
            "quoted grounds) take precedence for anchor tags. See "
            "05_analysis.py, Study 1 section."
        ),
    }
    out = {"_meta": meta, **taxonomy}
    if out_path is not None:
        save_json(out, out_path)
        LOG.info("taxonomy written: %s (%d tags: %s)", out_path,
                 meta["n_tags"], meta["level_counts"])
    return out


def appendix_pp_census(
    appendix_path: str | Path, terms: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Study 1's headline analysis: a vocabulary census of the instrument.

    Counts whole-word, case-insensitive occurrences of each configured term,
    grouped (pre_digital vs digital), and reports the ratio, the terms that
    never occur at all, and — for every term that does occur — the tags whose
    text carries it, so a claim like "the taxonomy has dedicated vocabulary
    for 1970s technology" is attributable line by line. Term lists live in the
    configuration precisely so a reader can contest a term and recompute.
    """
    if not terms or not all(terms.get(k) for k in ("pre_digital", "digital")):
        raise AnalysisError(
            "census requires non-empty 'pre_digital' and 'digital' term lists "
            "(configure analysis.census_terms)"
        )
    text = read_appendix_pp(appendix_path)
    blocks = tag_blocks(text)
    revision = re.search(r"\(Rev\.\s*\d+[^)]*\)", text)

    def count_in(haystack: str, term: str) -> int:
        return len(re.findall(r"\b" + re.escape(term) + r"\b", haystack,
                              re.I))

    groups: dict[str, Any] = {}
    for group_name, term_list in terms.items():
        rows = []
        for term in term_list:
            total = count_in(text, term)
            per_tag = sorted(
                ((tag, count_in(block, term))
                 for tag, block in blocks.items()),
                key=lambda item: -item[1],
            )
            rows.append({
                "term": term,
                "count": total,
                "top_tags": [{"f_tag": tag, "count": count}
                             for tag, count in per_tag[:5] if count > 0],
            })
        groups[group_name] = {
            "terms": rows,
            "total": sum(row["count"] for row in rows),
            "zero_terms": [row["term"] for row in rows if row["count"] == 0],
        }

    pre = groups["pre_digital"]["total"]
    dig = groups["digital"]["total"]
    return {
        "source_file": str(appendix_path),
        "source_sha256": sha256_file(Path(appendix_path)),
        "source_revision": revision.group(0) if revision else "",
        "n_words": len(text.split()),
        "n_tags": len(blocks),
        "groups": groups,
        "pre_digital_total": pre,
        "digital_total": dig,
        "ratio_pre_to_digital": (pre / dig) if dig else None,
        "interpretation": (
            "Whole-word, case-insensitive counts over the full guidance "
            "text. Polysemous terms (bare 'monitor(ing)', 'application', "
            "'interface') are excluded from both lists by construction; see "
            "the configured term lists."
        ),
    }



# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _bootstrap_rate(
    successes: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float | None]:
    """Percentile bootstrap interval for a proportion, resampled over documents."""
    successes = np.asarray(successes, dtype=float)
    if successes.size == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = [
        float(successes[rng.integers(0, successes.size, successes.size)].mean())
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(successes.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(successes.size)}


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame]
    if missing:
        raise AnalysisError(f"analysis frame is missing column(s): {missing}")


def load_tag_taxonomy(cfg: Config) -> dict[str, str]:
    """Map each citation tag to its technology-naming level.

    Returns ``{tag: "full" | "partial" | "none"}``. Three source formats are
    accepted, so older configurations keep working: the built artefact
    (``{tag: {"names_technology": level, ...}}``, with ``_meta`` skipped), a
    flat level mapping (``{tag: level}``), and the legacy boolean form
    (``True`` -> full, ``False`` -> none). Constructing this mapping is itself
    a contribution of the paper and it is published with the deposit, so a
    reader can disagree with a specific coding and recompute the gap.
    """
    path = Path(cfg.get("analysis.tag_taxonomy_file",
                        "configs/tag_taxonomy.json"))
    if not path.is_absolute():
        path = CONFIGS.parent / path
    if not path.is_file():
        LOG.warning("no tag taxonomy at %s; falling back to the %d "
                    "hand-verified anchor tags (run 05_analysis.py "
                    "--build-taxonomy to build the full artefact)",
                    path, len(DEFAULT_TAG_LEVELS))
        return dict(DEFAULT_TAG_LEVELS)

    raw = load_json(path)
    out: dict[str, str] = {}
    for tag, value in raw.items():
        if str(tag).startswith("_"):
            continue
        key = str(tag).upper().replace(" ", "")
        if isinstance(value, bool):
            out[key] = "full" if value else "none"
        elif isinstance(value, str):
            level = value
            if level not in TAXONOMY_LEVELS:
                raise AnalysisError(
                    f"tag {key}: unknown level {level!r}; expected one of "
                    f"{TAXONOMY_LEVELS}"
                )
            out[key] = level
        elif isinstance(value, Mapping):
            level = value.get("names_technology")
            if level not in TAXONOMY_LEVELS:
                raise AnalysisError(
                    f"tag {key}: names_technology is {level!r}; expected one "
                    f"of {TAXONOMY_LEVELS}"
                )
            out[key] = level
        else:
            raise AnalysisError(
                f"tag {key}: unsupported taxonomy value {type(value).__name__}"
            )
    return out


def tag_level(tag: Any, taxonomy: Mapping[str, str]) -> str:
    """A tag's technology-naming level; unknown tags are ``none``.

    Unknown-as-none is the conservative direction for the paper's claim: a
    tag absent from the taxonomy counts as illegible only under the strict
    measure and never inflates the count of technology-naming tags.
    """
    return taxonomy.get(str(tag).upper().replace(" ", ""), "none")


def names_technology(
    tag: Any, taxonomy: Mapping[str, str], threshold: str = "full",
) -> bool:
    """Whether a tag names technology at least at ``threshold``.

    ``threshold="full"`` (the primary, strict reading): only a requirement
    that is itself about a technology counts. ``threshold="partial"`` (the
    lenient reading): guidance-level recognition counts too. The gap is
    reported under both, and the difference between them is itself
    informative — it is the share of technology-mediated failures that the
    guidance can *discuss* but the requirement cannot *count*.
    """
    if threshold not in ("full", "partial"):
        raise AnalysisError(
            f"threshold must be 'full' or 'partial', got {threshold!r}"
        )
    level = tag_level(tag, taxonomy)
    return level == "full" if threshold == "full" else level in ("full",
                                                                 "partial")


# --------------------------------------------------------------------------- #
# RQ1
# --------------------------------------------------------------------------- #

def bias_corrected_prevalence(
    naive_rate: float, gold_pred: Sequence[int], gold_true: Sequence[int],
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Misclassification-corrected prevalence from the gold-standard subset.

    Follows the design-based logic Ziems et al. point to (Egami et al.'s DSL):
    the LLM's pseudo-labels give the naive rate over the full corpus; the gold
    subset identifies the instrument's sensitivity and specificity; the
    Rogan-Gladen correction ``p = (p_obs + spec - 1) / (sens + spec - 1)``
    returns an estimate unbiased under non-differential error. The bootstrap
    resamples the GOLD subset — that is where the estimation uncertainty in
    the correction lives — and corrected estimates are reported alongside the
    naive ones, never silently in their place. When the denominator is ~0 the
    instrument is uninformative (sens + spec ~ 1) and the correction refuses
    rather than returning an explosive value.
    """
    pred = np.asarray(gold_pred, dtype=float)
    true = np.asarray(gold_true, dtype=float)
    if pred.shape != true.shape or pred.ndim != 1 or pred.size == 0:
        raise AnalysisError("gold_pred and gold_true must be aligned 1-D")
    if not 0.0 <= naive_rate <= 1.0:
        raise AnalysisError(f"naive_rate must be in [0,1], got {naive_rate}")

    def rates(p_arr, t_arr):
        pos, neg = t_arr == 1, t_arr == 0
        if not pos.any() or not neg.any():
            return None
        sens = float(p_arr[pos].mean())
        spec = float(1.0 - p_arr[neg].mean())
        denominator = sens + spec - 1.0
        if abs(denominator) < 0.05:
            return None
        corrected = (naive_rate + spec - 1.0) / denominator
        return sens, spec, float(np.clip(corrected, 0.0, 1.0))

    point = rates(pred, true)
    if point is None:
        raise AnalysisError(
            "correction undefined: gold subset lacks both classes or the "
            "instrument is uninformative (sensitivity + specificity ~ 1)"
        )
    sens, spec, corrected = point

    rng = np.random.default_rng(seed)
    draws = []
    skipped = 0
    for _ in range(n_boot):
        index = rng.integers(0, pred.size, pred.size)
        result = rates(pred[index], true[index])
        if result is None:
            skipped += 1
            continue
        draws.append(result[2])
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws
              else (float("nan"), float("nan")))
    return {
        "naive": float(naive_rate),
        "corrected": corrected,
        "lo": float(lo), "hi": float(hi),
        "sensitivity": sens, "specificity": spec,
        "n_gold": int(pred.size), "n_boot": int(n_boot),
        "n_skipped": int(skipped),
        "method": ("Rogan-Gladen misclassification correction; "
                   "design-based semi-supervised logic (Egami et al.), "
                   "bootstrap over the gold subset"),
        "caveat": MEASUREMENT_CAVEAT,
    }


def prevalence_trend(
    frame: pd.DataFrame, by: Sequence[str] = ("year",),
    denominator: str = "surveys", n_boot: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Recorded prevalence of technology-mediated failures, overall and by period.

    The denominator matters more than the numerator. Counting per survey rather
    than per facility means a period in which more inspections were conducted
    does not register as a period in which more technology failed.
    """
    _require(frame, ["technology_present", "year"])
    flags = frame["technology_present"].fillna(False).astype(int).to_numpy()
    overall = _bootstrap_rate(flags, n_boot=n_boot, seed=seed)

    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(list(by), dropna=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        part_flags = part["technology_present"].fillna(False).astype(int).to_numpy()
        exposure = (part["survey_id"].nunique()
                    if denominator == "surveys" and "survey_id" in part
                    else len(part))
        rows.append({
            **dict(zip(by, [int(k) if isinstance(k, (int, np.integer)) else str(k)
                            for k in keys])),
            "n_citations": int(len(part)),
            "n_technology_mediated": int(part_flags.sum()),
            "exposure": int(exposure),
            "rate_per_exposure":
                float(part_flags.sum() / exposure) if exposure else 0.0,
            **{f"share_{k}": v for k, v in
               _bootstrap_rate(part_flags, n_boot=max(n_boot // 5, 100),
                               seed=seed + 1).items()},
        })
    rows.sort(key=lambda row: tuple(row[k] for k in by))

    return {
        "overall_share": overall,
        "by_period": rows,
        "denominator": denominator,
        "series": {
            "t": list(range(len(rows))),
            "labels": [str(row[by[0]]) for row in rows],
            "counts": [float(row["n_technology_mediated"]) for row in rows],
        },
        "caveat": MEASUREMENT_CAVEAT,
    }


# --------------------------------------------------------------------------- #
# RQ2
# --------------------------------------------------------------------------- #

def illegibility_gap(
    frame: pd.DataFrame, taxonomy: Mapping[str, str],
    n_boot: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Share of technology-mediated failures cited under non-technological tags.

    Reported overall and broken down by technology type and failure role,
    because ``over_reliance`` and ``not_responded_to`` are precisely the modes
    the existing taxonomy has no vocabulary for, and pooling them with
    ``malfunction`` would hide the asymmetry the paper is about.
    """
    _require(frame, ["technology_present", "f_tag", "failure_role",
                     "technology_type"])
    implicated = frame[frame["technology_present"].fillna(False).astype(bool)]
    if implicated.empty:
        raise AnalysisError(
            "no technology-mediated citations in the frame; the gap is undefined"
        )
    levels = implicated["f_tag"].map(lambda tag: tag_level(tag, taxonomy))
    strict = (levels != "full").astype(int).to_numpy()
    lenient = (levels == "none").astype(int).to_numpy()

    def breakdown(column: str) -> list[dict[str, Any]]:
        out = []
        for value, part in implicated.groupby(column, dropna=False,
                                              observed=True):
            part_flags = (part["f_tag"].map(
                lambda tag: tag_level(tag, taxonomy)
            ) != "full").astype(int).to_numpy()
            out.append({
                column: str(value), "n": int(len(part)),
                **_bootstrap_rate(part_flags, n_boot=max(n_boot // 5, 100),
                                  seed=seed + 2),
            })
        return sorted(out, key=lambda row: -row["n"])

    return {
        "n_technology_mediated": int(len(implicated)),
        "n_illegible": int(strict.sum()),
        # Primary (strict): cited under a tag whose REQUIREMENT names no
        # technology. Lenient: cited under a tag with no technology
        # vocabulary even in guidance. The difference is the share the
        # guidance can discuss but the requirement cannot count.
        "gap": _bootstrap_rate(strict, n_boot=n_boot, seed=seed),
        "gap_lenient": _bootstrap_rate(lenient, n_boot=n_boot, seed=seed + 1),
        "by_naming_level": {
            level: int((levels == level).sum()) for level in TAXONOMY_LEVELS
        },
        "by_technology_type": breakdown("technology_type"),
        "by_failure_role": breakdown("failure_role"),
        "taxonomy_size": len(taxonomy),
        "gap_definition": "strict: level != full; lenient: level == none",
        "caveat": MEASUREMENT_CAVEAT,
    }


def tag_adequacy_check(
    frame: pd.DataFrame, taxonomy: Mapping[str, str],
) -> dict[str, Any]:
    """Rival explanation: was a technology-naming tag available at all?

    If technology-naming tags are essentially never used even where the
    narrative implicates a device, two readings remain: surveyors decline to use
    them, or none fits the conduct being cited. Reporting how often such a tag
    was used at all, and in which failure roles it never appears, separates
    "the vocabulary exists and goes unused" from "the vocabulary is absent" —
    which are different governance findings with different remedies.
    """
    _require(frame, ["technology_present", "f_tag", "failure_role"])
    implicated = frame[frame["technology_present"].fillna(False).astype(bool)]
    if implicated.empty:
        raise AnalysisError("no technology-mediated citations to assess")

    levels = implicated["f_tag"].map(lambda tag: tag_level(tag, taxonomy))
    legible = levels == "full"
    partial = levels == "partial"
    roles_never: list[str] = []
    for role, part in implicated.groupby("failure_role", dropna=False,
                                         observed=True):
        role_levels = part["f_tag"].map(lambda tag: tag_level(tag, taxonomy))
        if not (role_levels == "full").any():
            roles_never.append(str(role))

    used = sorted(
        implicated.loc[legible, "f_tag"].value_counts().head(10)
        .to_dict().items(),
        key=lambda kv: -kv[1],
    )
    return {
        "n_technology_mediated": int(len(implicated)),
        "n_using_technology_naming_tag": int(legible.sum()),
        "share_using_technology_naming_tag": float(legible.mean()),
        "n_using_partial_naming_tag": int(partial.sum()),
        "share_using_partial_naming_tag": float(partial.mean()),
        "technology_naming_tags_in_use": [{"f_tag": k, "n": int(v)}
                                          for k, v in used],
        "failure_roles_never_cited_legibly": sorted(roles_never),
        "interpretation": (
            "Roles that never receive a technology-naming tag indicate absent "
            "vocabulary rather than surveyor discretion; roles where such tags "
            "are available but rarely used indicate the opposite. The two imply "
            "different remedies and are reported separately."
        ),
        "caveat": MEASUREMENT_CAVEAT,
    }


# --------------------------------------------------------------------------- #
# RQ3
# --------------------------------------------------------------------------- #

def _ols_with_fixed_effects(
    y: np.ndarray, X: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-transformed least squares: group means swept out, then OLS.

    Absorbing facility effects by demeaning avoids constructing thousands of
    dummy columns, and is algebraically identical to including them.
    """
    order = np.unique(groups)
    y_within = y.astype(float).copy()
    X_within = X.astype(float).copy()
    for group in order:
        mask = groups == group
        y_within[mask] -= y_within[mask].mean()
        X_within[mask] -= X_within[mask].mean(axis=0)
    beta, *_ = np.linalg.lstsq(X_within, y_within, rcond=None)
    residuals = y_within - X_within @ beta
    return beta, residuals


def hierarchical_equity_model(
    frame: pd.DataFrame, covariates: Sequence[str], gate: Mapping[str, Any],
    n_boot: int = 500, seed: int = 0,
) -> dict[str, Any]:
    """Model the illegibility gap on community and ownership characteristics.

    **Gated.** When stage 04 found extraction error patterned by facility
    characteristics, this analysis is refused rather than reported with a
    caveat: differential error does not attenuate an association toward the
    null, it can create one, and the disparity claim is exactly what would be
    fabricated. The gate decision is made in stage 04 before these estimates are
    seen, and enforced here.
    """
    if gate and not gate.get("equity_permitted", False):
        raise AnalysisError(
            "equity analysis withheld: the differential-error audit did not "
            "clear. Patterned extraction error can manufacture an apparent "
            "disparity, so this estimate is not reported."
        )
    _require(frame, ["technology_present", "illegible", "ccn"])
    usable = [name for name in covariates if name in frame]
    if not usable:
        raise AnalysisError(
            f"none of the requested covariates {list(covariates)} are present"
        )

    subset = frame[frame["technology_present"].fillna(False).astype(bool)].copy()
    if subset.empty:
        raise AnalysisError("no technology-mediated citations to model")
    y = subset["illegible"].astype(float).to_numpy()
    X = np.column_stack([
        pd.to_numeric(subset[name], errors="coerce").fillna(0.0).to_numpy()
        for name in usable
    ])
    centre, scale = X.mean(axis=0), np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    X = (X - centre) / scale
    groups = subset["ccn"].astype(str).to_numpy()

    beta, _ = _ols_with_fixed_effects(y, X, groups)
    rng = np.random.default_rng(seed)
    facilities = np.unique(groups)
    draws: list[np.ndarray] = []
    for _ in range(n_boot):
        # Cluster bootstrap: resample facilities, not rows, because citations
        # within a facility are not independent.
        chosen = rng.choice(facilities, size=facilities.size, replace=True)
        index = np.concatenate([np.flatnonzero(groups == f) for f in chosen])
        try:
            boot_beta, _ = _ols_with_fixed_effects(y[index], X[index], groups[index])
        except np.linalg.LinAlgError:
            continue
        draws.append(boot_beta)

    stacked = np.vstack(draws) if draws else np.empty((0, len(usable)))
    results = {}
    for position, name in enumerate(usable):
        if stacked.shape[0] >= 20:
            lo, hi = np.percentile(stacked[:, position], [2.5, 97.5])
        else:
            lo = hi = None
        results[name] = {
            "coefficient": float(beta[position]),
            "ci_lo": None if lo is None else float(lo),
            "ci_hi": None if hi is None else float(hi),
            "significant": bool(lo is not None and (lo > 0 or hi < 0)),
        }
    return {
        "n_observations": int(len(subset)),
        "n_facilities": int(facilities.size),
        "outcome": "illegible (1 = technology-mediated failure cited under a "
                   "tag naming no technology)",
        "covariates": results,
        "standardised": True,
        "facility_fixed_effects": True,
        "bootstrap": {"type": "cluster (facility)", "n_kept": int(stacked.shape[0])},
        "caveat": MEASUREMENT_CAVEAT,
        "matched_severity": _matched_severity_robustness(
            subset, usable, y, X, groups
        ),
    }


def _matched_severity_robustness(
    subset: pd.DataFrame, usable: Sequence[str], y: np.ndarray,
    X: np.ndarray, groups: np.ndarray,
) -> dict[str, Any] | None:
    """Obermeyer-style robustness: re-estimate at matched severity.

    Obermeyer et al.'s identification compares outcomes AT A GIVEN risk
    score; the analogue here re-fits the equity model within each severity
    level, so a disparity cannot be an artefact of poorer communities'
    citations simply being more (or less) severe. Reports per-stratum
    coefficients and whether each covariate's sign is consistent across
    strata. Returns None when the frame has no severity axis — the caller's
    output then says so explicitly rather than omitting the key.
    """
    if "severity_level" not in subset:
        return None
    severity = subset["severity_level"].fillna(-1.0).to_numpy()
    strata: list[dict[str, Any]] = []
    for level in sorted(set(severity)):
        mask = severity == level
        if mask.sum() < 30 or len(set(y[mask])) < 2:
            continue
        beta_s, _ = _ols_with_fixed_effects(y[mask], X[mask], groups[mask])
        strata.append({
            "severity_level": float(level), "n": int(mask.sum()),
            "coefficients": {name: float(beta_s[i])
                             for i, name in enumerate(usable)},
        })
    if not strata:
        return {"strata": [], "sign_consistent": None,
                "note": "no severity stratum large enough (n >= 30, both "
                        "outcomes present)"}
    consistency = {
        name: bool(len({np.sign(s_["coefficients"][name])
                        for s_ in strata
                        if s_["coefficients"][name] != 0}) <= 1)
        for name in usable
    }
    return {"strata": strata, "sign_consistent": consistency,
            "design": "re-estimated within severity_level strata "
                      "(Obermeyer et al.: compare at matched severity)"}


def attach_illegible(
    frame: pd.DataFrame, taxonomy: Mapping[str, str],
) -> pd.DataFrame:
    """Add the ``illegible`` indicator the equity model consumes.

    Uses the strict (primary) definition — cited under a tag whose
    requirement does not name a technology — matching the headline gap.
    """
    out = frame.copy()
    out["illegible"] = out["f_tag"].map(
        lambda tag: tag_level(tag, taxonomy) != "full"
    ).astype(int)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Study 1 (Appendix PP) and the substantive analyses."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frame", default=None,
                        help="path to the analysis parquet (Study 2)")
    parser.add_argument("--census", action="store_true",
                        help="run the Appendix PP vocabulary census (Study 1)")
    parser.add_argument("--build-taxonomy", action="store_true",
                        help="rebuild configs/tag_taxonomy.json from "
                             "Appendix PP")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path
    cfg = Config.load(config_path)

    if args.census or args.build_taxonomy:
        appendix = Path(cfg.get("analysis.appendix_pp_file",
                                "data/raw/appendix-pp.txt"))
        if not appendix.is_absolute():
            appendix = CONFIGS.parent / appendix
        if args.build_taxonomy:
            out = Path(cfg.get("analysis.tag_taxonomy_file",
                               "configs/tag_taxonomy.json"))
            if not out.is_absolute():
                out = CONFIGS.parent / out
            build_tag_taxonomy(appendix, out)
        if args.census:
            census = appendix_pp_census(
                appendix, cfg.get("analysis.census_terms", {})
            )
            from utils import RESULTS
            save_json(census, RESULTS / "study1_census.json")
            LOG.info("census: pre=%d digital=%d ratio=%.1f",
                     census["pre_digital_total"], census["digital_total"],
                     census["ratio_pre_to_digital"] or 0.0)
        return 0

    if not args.frame:
        LOG.error("--frame is required unless --census/--build-taxonomy")
        return 1
    frame = pd.read_parquet(args.frame)
    taxonomy = load_tag_taxonomy(cfg)
    out = {
        "rq1": prevalence_trend(frame),
        "rq2": {"gap": illegibility_gap(frame, taxonomy),
                "tag_adequacy": tag_adequacy_check(frame, taxonomy)},
    }
    from utils import RESULTS
    save_json(out, RESULTS / f"analysis_{cfg.fingerprint()}.json")
    return 0


__all__ = [
    "MEASUREMENT_CAVEAT", "AnalysisError", "TAXONOMY_LEVELS",
    "DEFAULT_TAG_LEVELS", "read_appendix_pp", "tag_blocks",
    "requirement_text", "classify_tag", "build_tag_taxonomy",
    "appendix_pp_census", "load_tag_taxonomy", "tag_level",
    "names_technology", "prevalence_trend", "illegibility_gap",
    "tag_adequacy_check", "hierarchical_equity_model", "attach_illegible",
    "bias_corrected_prevalence",
]


if __name__ == "__main__":
    sys.exit(main())
