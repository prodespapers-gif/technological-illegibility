"""
02_corpus.py — From raw archives to the analysis corpus. (The dataset module.)

Stage 2 of the pipeline. Defines the single object every downstream stage
consumes: a deficiency-level corpus in which each record is one cited
deficiency carrying the surveyor's narrative, the structured citation metadata
published alongside it, and facility covariates.

WHY DEFICIENCY-LEVEL
--------------------
The unit of regulatory action is the cited deficiency, not the inspection. The
study's claim concerns how the regulatory instrument *classifies* what surveyors
*observed*, so the narrative and the assigned tag must sit in the same record
for the attribution analysis in stage 05 to mean anything.

THE FORMAT BOUNDARY — READ THIS BEFORE RUNNING ON REAL DATA
------------------------------------------------------------
Everything in this module is format-independent except the parsers registered in
:data:`NARRATIVE_PARSERS` and the block splitter :func:`split_tag_blocks`. That
boundary is deliberate and is the whole design of the file: the schema below is
a contract, stages 03-08 are written against the contract, and adapting to a new
archive layout means writing one function, not touching the pipeline.

Reference parsers are provided for tabular (CSV) and line-delimited JSON
archives, plus a splitter for concatenated plain-text reports. They are
*reference implementations*, not calibrated ones. Before any published run:

  1. Inspect a real archive and confirm which parser applies.
  2. Run ``python src/02_corpus.py --inspect <path>`` to see what the parser
     extracts from the first records.
  3. Check the coverage audit (:func:`coverage_audit`) — an unmatched-narrative
     rate above a few percent means the join keys are wrong, and a systematic
     non-match would bias every prevalence estimate downstream.

The audit is what turns "the parser probably works" into evidence, and its
output belongs in the paper's data appendix.

PRIVACY POSTURE
---------------
Narratives are published by the regulator with resident identifiers redacted,
but still describe individual care episodes in detail. This pipeline uses the
redacted public versions only, never transmits text to a third-party API (see
stage 03), and deposits only derived, non-reconstructive annotations (stage 09).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from utils import (DATA_PROCESSED, DATA_RAW, Config, ensure_dirs, get_logger,
                   save_json, set_seed, stable_doc_id, timed)

LOG = get_logger("corpus")


# --------------------------------------------------------------------------- #
# The schema contract
# --------------------------------------------------------------------------- #

# The scope/severity letter is a 2-D grid: severity (1 no-harm-minimal,
# 2 no-harm-potential, 3 ACTUAL HARM, 4 IMMEDIATE JEOPARDY) x scope
# (1 isolated, 2 pattern, 3 widespread). The letter alone hides both axes;
# decomposing them matters twice over: severity is ORDINAL (Krippendorff's
# worked example shows nominal alpha understating reliability on ordered data,
# 0.4765 vs 0.7598 on identical judgements), and the empirical distribution is
# so skewed (D alone ~63%; actual harm G-L ~5.5%) that sampling must see the
# axis explicitly to protect the strata where the theory lives.
SEVERITY_GRID: dict[str, tuple[int, int]] = {
    "A": (1, 1), "B": (1, 2), "C": (1, 3),
    "D": (2, 1), "E": (2, 2), "F": (2, 3),
    "G": (3, 1), "H": (3, 2), "I": (3, 3),
    "J": (4, 1), "K": (4, 2), "L": (4, 3),
}


def decompose_scope_severity(codes: pd.Series) -> pd.DataFrame:
    """Severity level, scope level, and the two harm flags from the letter.

    Unknown or missing letters yield NaN levels and False flags — never a
    guessed severity.
    """
    letters = codes.astype("string").str.strip().str.upper().str[:1]
    severity = letters.map({k: v[0] for k, v in SEVERITY_GRID.items()})
    scope = letters.map({k: v[1] for k, v in SEVERITY_GRID.items()})
    severity = pd.to_numeric(severity, errors="coerce")
    scope = pd.to_numeric(scope, errors="coerce")
    return pd.DataFrame({
        "severity_level": severity.astype("float64"),
        "scope_level": scope.astype("float64"),
        "actual_harm": (severity >= 3).fillna(False).astype(bool),
        "immediate_jeopardy": (severity >= 4).fillna(False).astype(bool),
    }, index=codes.index)


CORPUS_SCHEMA: dict[str, str] = {
    "doc_id": "string",          # stable identifier (utils.stable_doc_id)
    "ccn": "string",             # facility certification number
    "survey_date": "string",     # ISO date of the survey
    "year": "Int64",
    "quarter": "Int64",
    "f_tag": "string",           # citation tag assigned by the surveyor
    "scope_severity": "string",  # scope/severity letter (A-L)
    "severity_level": "float64",   # 1-4 ordinal, NaN when letter unknown
    "scope_level": "float64",      # 1-3 ordinal, NaN when letter unknown
    "actual_harm": "bool",         # severity >= 3 (G-L)
    "immediate_jeopardy": "bool",  # severity 4 (J-L)
    "is_complaint": "boolean",   # complaint investigation vs standard survey
    "state": "string",
    "narrative": "string",       # surveyor's findings text
    "n_tokens": "Int64",
    # facility covariates, joined from the public provider file
    "ownership_type": "string",
    "certified_beds": "Int64",
}

REQUIRED_FROM_PARSER: tuple[str, ...] = (
    "ccn", "survey_date", "f_tag", "narrative",
)

# Tag blocks in plain-text reports open with a tag code at the start of a line.
TAG_BLOCK = re.compile(r"^\s*(?P<tag>[FK]\s?\d{3,4})\b", re.MULTILINE)

# Statutory boilerplate that precedes the surveyor's own findings. Left in
# place, the extractor in stage 03 would match on regulation language rather
# than on what happened in the home, inflating apparent prevalence.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "based on observation",
    "based on interview",
    "based on record review",
    "this requirement is not met as evidenced by",
)


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be built or fails its integrity checks."""


# --------------------------------------------------------------------------- #
# The format boundary: parsers
# --------------------------------------------------------------------------- #

def split_tag_blocks(text: str) -> Iterator[tuple[str, str]]:
    """Split one plain-text inspection report into ``(tag, findings)`` blocks.

    FORMAT-DEPENDENT. Calibrate against a real archive before publishing: the
    assumption is that each cited deficiency opens with its tag code at the
    start of a line and runs until the next such code.
    """
    matches = list(TAG_BLOCK.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tag = match.group("tag").replace(" ", "").upper()
        yield tag, text[start:end].strip()


def strip_boilerplate(text: str) -> str:
    """Drop statutory preamble, keeping the surveyor's observed findings.

    Cuts at the last marker found, because reports sometimes repeat the
    preamble; keeping everything after the final marker is the conservative
    choice. Matching is case-insensitive — archives vary between "This
    REQUIREMENT", "This Requirement", and lower case, and a case-sensitive match
    would silently leave statutory language in the text the extractor reads,
    inflating apparent prevalence by matching on regulation wording rather than
    on what happened in the home.

    When no marker is present the text is returned unchanged rather than
    guessed at.
    """
    haystack = text.lower()
    cut = 0
    for marker in BOILERPLATE_MARKERS:
        position = haystack.rfind(marker)
        if position >= 0:
            cut = max(cut, position + len(marker))
    return text[cut:].strip(" :,;-\n\t") if cut else text.strip()


def parse_csv_archive(path: Path, cfg: Config) -> list[dict[str, Any]]:
    """Reference parser for tabular archives with one row per deficiency.

    Column names are read from ``corpus.column_map`` so a renamed source column
    is a configuration change, not a code change.
    """
    mapping: Mapping[str, str] = cfg.get("corpus.column_map", {}) or {}
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    records: list[dict[str, Any]] = []
    for position, raw in enumerate(frame.to_dict("records")):
        record = {
            field: raw.get(mapping.get(field, field))
            for field in ("ccn", "survey_date", "f_tag", "narrative",
                          "scope_severity", "is_complaint", "state")
        }
        record["offset"] = position
        records.append(record)
    return records


def parse_jsonl_archive(path: Path, cfg: Config) -> list[dict[str, Any]]:
    """Reference parser for line-delimited JSON archives."""
    mapping: Mapping[str, str] = cfg.get("corpus.column_map", {}) or {}
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            record = {
                field: raw.get(mapping.get(field, field))
                for field in ("ccn", "survey_date", "f_tag", "narrative",
                              "scope_severity", "is_complaint", "state")
            }
            record["offset"] = position
            records.append(record)
    return records


def parse_text_reports(path: Path, cfg: Config) -> list[dict[str, Any]]:
    """Reference parser for whole-report text, one report per file or block.

    Expects a header line carrying the facility identifier and survey date; the
    header pattern is configurable because it is the part most likely to differ
    between archives.
    """
    header = re.compile(cfg.get(
        "corpus.report_header_pattern",
        r"CCN[:\s]+(?P<ccn>\d{6}).*?DATE[:\s]+(?P<date>\d{4}-\d{2}-\d{2})",
    ), re.IGNORECASE | re.DOTALL)
    text = path.read_text(encoding="utf-8", errors="replace")
    match = header.search(text)
    if not match:
        raise CorpusError(
            f"no report header matched in {path.name}. Set "
            f"corpus.report_header_pattern to the archive's actual header form."
        )
    ccn, date = match.group("ccn"), match.group("date")
    return [
        {"ccn": ccn, "survey_date": date, "f_tag": tag, "narrative": body,
         "scope_severity": None, "is_complaint": None, "state": None,
         "offset": position}
        for position, (tag, body) in enumerate(split_tag_blocks(text[match.end():]))
    ]


NARRATIVE_PARSERS: dict[str, Callable[[Path, Config], list[dict[str, Any]]]] = {
    "csv": parse_csv_archive,
    "jsonl": parse_jsonl_archive,
    "text": parse_text_reports,
}


# --------------------------------------------------------------------------- #
# Assembly (format-independent from here down)
# --------------------------------------------------------------------------- #

def _quarter(date: str) -> tuple[int | None, int | None]:
    try:
        stamp = pd.Timestamp(date)
    except (ValueError, TypeError):
        return None, None
    return int(stamp.year), int((stamp.month - 1) // 3 + 1)


def parse_narratives(cfg: Config) -> pd.DataFrame:
    """Parse every configured archive into one row per cited deficiency."""
    fmt = cfg.get("corpus.narrative_format", "csv")
    if fmt not in NARRATIVE_PARSERS:
        raise CorpusError(
            f"unknown narrative_format {fmt!r}; available: "
            f"{sorted(NARRATIVE_PARSERS)}"
        )
    directory = DATA_RAW / "narratives"
    paths = sorted(directory.glob(cfg.get("corpus.narrative_glob", "*")))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"no narrative archives found in {directory}. Run 01_acquire.py first."
        )

    parser = NARRATIVE_PARSERS[fmt]
    rows: list[dict[str, Any]] = []
    for path in paths:
        LOG.info("parsing %s with the %s parser", path.name, fmt)
        rows.extend(parser(path, cfg))

    minimum = int(cfg.get("corpus.min_narrative_tokens", 40))
    strip = bool(cfg.get("corpus.drop_regulation_boilerplate", True))
    parsed: list[dict[str, Any]] = []
    dropped_short = 0
    for raw in rows:
        missing = [f for f in REQUIRED_FROM_PARSER if not raw.get(f)]
        if missing:
            continue
        narrative = str(raw["narrative"])
        if strip:
            narrative = strip_boilerplate(narrative)
        tokens = len(narrative.split())
        if tokens < minimum:
            dropped_short += 1
            continue
        ccn = str(raw["ccn"]).strip()
        date = str(raw["survey_date"]).strip()
        tag = str(raw["f_tag"]).strip().upper().replace(" ", "")
        year, quarter = _quarter(date)
        parsed.append({
            "doc_id": stable_doc_id(ccn, date, tag, int(raw.get("offset", 0))),
            "ccn": ccn, "survey_date": date, "year": year, "quarter": quarter,
            "f_tag": tag, "narrative": narrative, "n_tokens": tokens,
            "scope_severity": raw.get("scope_severity"),
            "is_complaint": raw.get("is_complaint"),
            "state": raw.get("state"),
        })

    LOG.info("parsed %d deficiency records (%d dropped as too short)",
             len(parsed), dropped_short)
    if not parsed:
        raise CorpusError(
            "no usable deficiency records were parsed. Inspect the archive with "
            "`python src/02_corpus.py --inspect <path>` and check "
            "corpus.narrative_format and corpus.column_map."
        )
    return pd.DataFrame(parsed)


def load_structured() -> dict[str, pd.DataFrame]:
    """Read the public structured files acquired by stage 01."""
    frames: dict[str, pd.DataFrame] = {}
    for name in ("provider_information", "health_deficiencies"):
        path = DATA_RAW / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run 01_acquire.py first.")
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        frame.columns = [str(c).strip().lower().replace(" ", "_")
                         for c in frame.columns]
        frames[name] = frame
    return frames


def coverage_audit(
    narratives: pd.DataFrame, joined: pd.DataFrame, structured_rows: int,
) -> dict[str, Any]:
    """Quantify how much of each source survived the join.

    A low match rate means the join keys are wrong. A match rate that varies
    systematically — by state, by year — is worse than a uniformly low one,
    because it biases prevalence estimates rather than merely shrinking the
    sample. Both are reported; the second is why the by-year breakdown exists.
    """
    matched = int(joined["scope_severity"].notna().sum()) if len(joined) else 0
    by_year = (
        joined.assign(_m=joined["scope_severity"].notna())
        .groupby("year", dropna=True)["_m"].mean().round(4).to_dict()
        if len(joined) else {}
    )
    rate = matched / len(narratives) if len(narratives) else 0.0
    return {
        "n_narratives": int(len(narratives)),
        "n_structured_rows": int(structured_rows),
        "n_joined": int(len(joined)),
        "n_matched_to_structured": matched,
        "match_rate": round(rate, 4),
        "match_rate_by_year": {str(k): float(v) for k, v in by_year.items()},
        "warning": (
            "" if rate >= 0.95 else
            "Match rate below 95%: verify the join keys (ccn, survey_date, "
            "f_tag) before interpreting any prevalence estimate."
        ),
    }


def build_corpus(cfg: Config) -> pd.DataFrame:
    """Join narratives to structured citations and facility covariates."""
    narratives = parse_narratives(cfg)
    frames = load_structured()

    deficiencies = frames["health_deficiencies"].rename(columns={
        cfg.get("corpus.deficiency_key_ccn", "cms_certification_number_ccn"): "ccn",
        cfg.get("corpus.deficiency_key_date", "survey_date"): "survey_date",
        cfg.get("corpus.deficiency_key_tag", "deficiency_tag_number"): "f_tag",
    })
    for column in ("ccn", "survey_date", "f_tag"):
        if column in deficiencies:
            deficiencies[column] = (
                deficiencies[column].astype("string").str.strip().str.upper()
            )
    keep = [c for c in ("ccn", "survey_date", "f_tag", "scope_severity",
                        "deficiency_category", "standard_deficiency")
            if c in deficiencies]
    deficiencies = deficiencies[keep].drop_duplicates(
        subset=[c for c in ("ccn", "survey_date", "f_tag") if c in deficiencies]
    )

    narratives = narratives.copy()
    narratives["ccn"] = narratives["ccn"].astype("string").str.upper()
    narratives["f_tag"] = narratives["f_tag"].astype("string").str.upper()
    structured_cols = [c for c in deficiencies.columns
                       if c not in ("ccn", "survey_date", "f_tag")]
    joined = narratives.merge(
        deficiencies, on=[c for c in ("ccn", "survey_date", "f_tag")
                          if c in deficiencies],
        how="left", suffixes=("", "_structured"),
    )
    for column in structured_cols:
        target = column if column in ("scope_severity",) else column
        if target in joined and f"{target}_structured" in joined:
            joined[target] = joined[target].fillna(joined[f"{target}_structured"])

    providers = frames["provider_information"].rename(columns={
        cfg.get("corpus.provider_key_ccn", "cms_certification_number_ccn"): "ccn",
        cfg.get("corpus.provider_ownership", "ownership_type"): "ownership_type",
        cfg.get("corpus.provider_beds", "number_of_certified_beds"): "certified_beds",
        cfg.get("corpus.provider_state", "state"): "state_provider",
    })
    provider_cols = [c for c in ("ccn", "ownership_type", "certified_beds",
                                 "state_provider") if c in providers]
    providers = providers[provider_cols].drop_duplicates(subset=["ccn"])
    providers["ccn"] = providers["ccn"].astype("string").str.upper()
    joined = joined.merge(providers, on="ccn", how="left")

    if "state_provider" in joined:
        joined["state"] = joined["state"].fillna(joined["state_provider"])
        joined = joined.drop(columns=["state_provider"])
    joined = joined.join(decompose_scope_severity(joined["scope_severity"]))
    for column, dtype in CORPUS_SCHEMA.items():
        if column not in joined:
            joined[column] = pd.NA
        try:
            joined[column] = joined[column].astype(dtype)
        except (ValueError, TypeError):
            LOG.warning("could not cast %s to %s; left as-is", column, dtype)
    joined = joined[list(CORPUS_SCHEMA)]

    audit = coverage_audit(narratives, joined, len(deficiencies))
    if audit["warning"]:
        LOG.warning(audit["warning"])
    save_json(audit, DATA_PROCESSED / "coverage_audit.json")

    duplicates = int(joined["doc_id"].duplicated().sum())
    if duplicates:
        raise CorpusError(
            f"{duplicates} duplicate doc_id values. The identifier must be "
            f"unique; check the parser's offset assignment."
        )
    LOG.info("corpus built: %d records, match rate %.1f%%",
             len(joined), 100 * audit["match_rate"])
    return joined


def stratified_sample(
    corpus: pd.DataFrame, n: int, seed: int,
    severity_floors: Mapping[Any, int] | None = None,
) -> pd.DataFrame:
    """Draw the human-annotation sample used by stage 04, with design weights.

    Proportional allocation is statistically wrong for this corpus: with D
    alone ~63% of deficiencies and actual harm (G-L) ~5.5%, a proportional
    n=1,200 draw contains roughly one severity-H document — the validation
    would be silent exactly where the paper's claims live. Allocation is
    therefore disproportionate: per-severity floors (``severity_floors``,
    keyed by severity level) are applied first, the remainder is shared
    proportionally, and every row carries ``sampling_weight`` =
    N_stratum / n_stratum at severity-stratum granularity, so any
    population-level statistic computed from the sample can undo the
    oversampling. Within a severity level, allocation across years stays
    proportional, which keeps weights constant within the stratum.
    """
    if n < 1:
        raise ValueError(f"sample size must be positive, got {n}")
    if corpus.empty:
        raise CorpusError("cannot sample from an empty corpus")
    n = min(n, len(corpus))
    rng = np.random.default_rng(seed)

    if "severity_level" not in corpus:
        index = rng.choice(len(corpus), size=n, replace=False)
        sample = corpus.iloc[np.sort(index)].reset_index(drop=True)
        sample["sampling_weight"] = float(len(corpus)) / len(sample)
        return sample

    severity = corpus["severity_level"].fillna(-1.0)
    floors = {float(k): int(v) for k, v in (severity_floors or {}).items()}
    pop_counts = severity.value_counts().to_dict()

    # Stage 1: quota per severity level — proportional, then floors, then
    # scale the non-floored levels down so the total returns to n.
    quota: dict[float, int] = {}
    for level, count in pop_counts.items():
        proportional = n * count / len(corpus)
        quota[level] = min(count, max(int(round(proportional)),
                                      floors.get(level, 0)))
    floored = {level for level in quota
               if floors.get(level, 0) > 0 and quota[level] == min(
                   pop_counts[level], floors[level])}
    excess = sum(quota.values()) - n
    if excess > 0:
        adjustable = [lvl for lvl in quota if lvl not in floored]
        weight_total = sum(quota[lvl] for lvl in adjustable)
        for lvl in adjustable:
            share = quota[lvl] / weight_total if weight_total else 0.0
            quota[lvl] = max(1, quota[lvl] - int(round(excess * share)))

    # Stage 2: within a severity level, proportional across years.
    picked: list[pd.DataFrame] = []
    for level, level_quota in quota.items():
        pool = corpus[severity == level]
        if pool.empty or level_quota < 1:
            continue
        take_total = min(level_quota, len(pool))
        if "year" in pool:
            parts = list(pool.groupby("year", dropna=False, observed=True))
            takes = {key: max(1, round(take_total * len(part) / len(pool)))
                     for key, part in parts}
            for key, part in parts:
                take = min(takes[key], len(part))
                index = rng.choice(len(part), size=take, replace=False)
                picked.append(part.iloc[np.sort(index)])
        else:
            index = rng.choice(len(pool), size=take_total, replace=False)
            picked.append(pool.iloc[np.sort(index)])
    sample = pd.concat(picked).drop_duplicates(subset="doc_id")

    if len(sample) > n:
        keep = rng.choice(len(sample), size=n, replace=False)
        sample = sample.iloc[np.sort(keep)]
    elif len(sample) < n:
        remaining = corpus[~corpus["doc_id"].isin(sample["doc_id"])]
        shortfall = min(n - len(sample), len(remaining))
        if shortfall:
            index = rng.choice(len(remaining), size=shortfall, replace=False)
            sample = pd.concat([sample, remaining.iloc[np.sort(index)]])

    # Design weights at severity-stratum granularity, from realised counts.
    sample = sample.sort_values("doc_id").reset_index(drop=True)
    realised = sample["severity_level"].fillna(-1.0).value_counts().to_dict()
    sample["sampling_weight"] = sample["severity_level"].fillna(-1.0).map(
        lambda level: pop_counts.get(level, 0) / realised.get(level, 1)
    ).astype(float)
    return sample


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def inspect(path: Path, cfg: Config, limit: int = 3) -> None:
    """Print what the configured parser extracts, for calibration."""
    fmt = cfg.get("corpus.narrative_format", "csv")
    records = NARRATIVE_PARSERS[fmt](path, cfg)
    print(f"parser={fmt}  records={len(records)}")
    for record in records[:limit]:
        print("-" * 70)
        for key, value in record.items():
            shown = str(value)
            if key == "narrative" and len(shown) > 400:
                shown = shown[:400] + " ... [truncated]"
            print(f"  {key:16s} {shown!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the analysis corpus.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inspect", default=None,
                        help="print what the parser extracts from one archive")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path
    cfg = Config.load(config_path)
    set_seed(cfg.seed)
    ensure_dirs()

    if args.inspect:
        inspect(Path(args.inspect), cfg)
        return 0

    try:
        with timed("stage 02: corpus", LOG):
            corpus = build_corpus(cfg)
    except (CorpusError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 1
    out = DATA_PROCESSED / f"corpus_{cfg.fingerprint()}.parquet"
    corpus.to_parquet(out)
    LOG.info("written: %s (%d records)", out.name, len(corpus))
    return 0


__all__ = [
    "CORPUS_SCHEMA", "CorpusError", "NARRATIVE_PARSERS", "split_tag_blocks",
    "strip_boilerplate", "parse_narratives", "load_structured",
    "coverage_audit", "build_corpus", "stratified_sample", "inspect",
]


if __name__ == "__main__":
    sys.exit(main())
