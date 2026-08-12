"""
09_deposit.py — Building the citable, disclosure-reviewed data deposit.

Stage 9 of the pipeline. This journal applies the strictest of the publisher's
research-data tiers: depositing the research data in a repository and citing and
linking it in the article is *required*, not encouraged. This module exists so
that requirement is met deliberately, with a documented disclosure review,
rather than retrofitted at proof stage by uploading whatever files happen to be
on disk.

WHAT IS DEPOSITED, AND THE REASONING BEHIND IT
----------------------------------------------
Deposited — derived and non-reconstructive:
  * document-level annotations (the model's judgements, keyed to public ids)
  * the F-tag taxonomy mapping built for the gap analysis, a reusable artefact
  * human gold-standard labels and inter-annotator agreement data
  * facility-quarter aggregates used in the models
  * the acquisition manifest (source URLs, retrieval dates, SHA-256)

Not deposited:
  * the narrative texts. They are already public at source; consolidating
    resident-level care episodes into one machine-readable corpus would raise
    the ease of assembly for no scientific gain, since stages 01 and 02
    reconstruct the corpus exactly from the manifest.
  * verbatim evidence spans. Retained locally for audit, released only as
    character offsets.

THE HONEST ACCOUNT OF FACILITY IDENTITY
---------------------------------------
It is tempting to hash the facility identifier and call the release
de-identified. That would be theatre. There are on the order of fifteen thousand
certified facilities and the roster is public, so any salted hash whose salt is
published is invertible by exhaustive search in seconds, and any hash whose salt
is withheld simply makes the data unusable for the replication the deposit
exists to enable. Facility identity is *already public* — the regulator
publishes facility-level ratings and citations by name — so hashing adds no
protection and subtracts reuse value.

The real disclosure risk in this study is at the resident level, and it is
addressed where it actually lives: by not releasing narrative text or verbatim
spans at all, by scanning every released field for free-text residue, and by
suppressing small cells in derived aggregates where a facility-quarter cell
could isolate a single individual's incident. Hashing is therefore available
(:func:`salted_hash`) and off by default, with the trade-off recorded in the
deposit's own documentation rather than hidden in a parameter.

FAIL-CLOSED
-----------
:func:`write_deposit` runs the review first and writes nothing at all unless it
clears. A partial deposit is worse than none: it looks complete and is not.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils import (DEPOSIT, FAILURE_ROLE_DEFINITIONS, FAILURE_ROLES,
                   TECHNOLOGY_TYPES, Config, ensure_dirs, get_logger,
                   provenance, relpath_to_root, save_json, sha256_file)

LOG = get_logger("deposit")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FieldSpec:
    """One released column, documented well enough to be reused without us."""

    name: str
    dtype: str                       # "string" | "integer" | "number" | "boolean"
    description: str
    permitted: tuple[Any, ...] | None = None
    provenance: str = ""
    caveat: str = ""


RELEASE_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("doc_id", "string",
              "Stable identifier for one cited deficiency.",
              provenance="Derived deterministically from CCN, survey date, "
                         "F-tag, and citation ordinal (see utils.stable_doc_id).",
              caveat="Stable across rebuilds; join key for all released tables."),
    FieldSpec("ccn", "string",
              "CMS Certification Number identifying the facility.",
              provenance="Published by the regulator.",
              caveat="Public information; released unhashed by default so the "
                     "deposit can be linked to other public facility data."),
    FieldSpec("year", "integer", "Calendar year of the survey.",
              provenance="Survey date in the public deficiency file."),
    FieldSpec("quarter", "integer", "Calendar quarter of the survey.",
              permitted=(1, 2, 3, 4)),
    FieldSpec("state", "string", "Two-letter state or territory code.",
              provenance="Public provider file."),
    FieldSpec("f_tag", "string", "Regulatory citation tag assigned by the surveyor.",
              provenance="Public deficiency file."),
    FieldSpec("scope_severity", "string",
              "Scope and severity code assigned to the citation.",
              provenance="Public deficiency file."),
    FieldSpec("technology_present", "boolean",
              "Whether a care technology was implicated in the deficiency.",
              provenance="Model judgement; see the validation table for error rates.",
              caveat="A measured variable with quantified error, not ground truth."),
    FieldSpec("technology_type", "string",
              "Kind of care technology implicated.",
              permitted=TECHNOLOGY_TYPES,
              provenance="Model judgement. Vocabulary defined once in "
                         "utils.py and shared with the extraction validator, "
                         "so the released schema cannot drift from the "
                         "instrument that produced it."),
    FieldSpec("failure_role", "string",
              "Role the technology played in the failure. Operational "
              "definitions for every category are in the codebook's "
              "failure_role_definitions section.",
              permitted=FAILURE_ROLES,
              provenance="Model judgement. The category set extends the "
                         "technology-induced-error typology of Ash, Berg & "
                         "Coiera (2004); over_reliance follows the "
                         "automation-bias definition of Alon-Barkat & "
                         "Busuioc (2023) and requires deference DESPITE "
                         "contradicting signals.",
              caveat="The rare roles carry the theoretical weight; see the "
                     "per-class agreement figures before reusing them."),
    FieldSpec("harm_linked", "boolean",
              "Whether the narrative ties the technology to resident harm.",
              provenance="Model judgement."),
    FieldSpec("evidence_span_start", "integer",
              "Character offset of the supporting span in the source narrative.",
              caveat="Offsets only. The span text is deliberately not released; "
                     "resolve it against the public source if required."),
    FieldSpec("evidence_span_end", "integer",
              "End character offset of the supporting span."),
    FieldSpec("confidence", "number", "Model self-reported confidence in [0, 1]."),
    FieldSpec("model_id", "string", "Identifier of the extraction model."),
    FieldSpec("model_revision", "string", "Pinned model revision."),
    FieldSpec("prompt_hash", "string",
              "Hash of the extraction prompt that produced this row.",
              caveat="Rows produced by different prompts are not interchangeable."),
)

RELEASE_FIELDS: tuple[str, ...] = tuple(spec.name for spec in RELEASE_SCHEMA)

# Fields that must never appear in a release, whatever the caller passes.
FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "narrative", "evidence_span", "text", "resident_name", "surveyor_name",
    "raw_response", "narrative_text",
})


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReleasePolicy:
    """Disclosure-control settings, recorded verbatim in the deposit."""

    min_cell_size: int = 10
    quasi_identifiers: tuple[str, ...] = ("state", "year", "quarter", "technology_type")
    max_text_length: int = 64
    hash_facility_id: bool = False
    salt: str = ""
    license: str = "CC-BY-4.0"

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "ReleasePolicy":
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        if "quasi_identifiers" in known:
            known["quasi_identifiers"] = tuple(known["quasi_identifiers"])
        return cls(**known)

    def __post_init__(self) -> None:
        if self.min_cell_size < 1:
            raise ValueError(f"min_cell_size must be >= 1, got {self.min_cell_size}")
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be positive")
        if self.hash_facility_id and not self.salt:
            raise ValueError(
                "hash_facility_id is set but no salt was provided. Note that a "
                "published salt is invertible against the public facility "
                "roster; see this module's docstring before enabling hashing."
            )


# --------------------------------------------------------------------------- #
# Identifier hashing
# --------------------------------------------------------------------------- #

def salted_hash(value: Any, salt: str, length: int = 16) -> str:
    """Salted, deterministic hash of an identifier.

    Available but off by default. Against a public roster of roughly fifteen
    thousand facilities a published salt is invertible by exhaustive search, and
    a withheld salt breaks the replication the deposit exists to support; see
    the module docstring for why facility identity is released openly instead.
    """
    if not salt:
        raise ValueError("a non-empty salt is required")
    if not 8 <= length <= 64 or length % 2:
        raise ValueError(f"length must be even and in [8, 64], got {length}")
    digest = hashlib.blake2b(f"{salt}|{value}".encode("utf-8"),
                             digest_size=length // 2)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def validate_records(
    records: Sequence[Mapping[str, Any]],
    schema: Sequence[FieldSpec] = RELEASE_SCHEMA,
) -> dict[str, Any]:
    """Check the release table against the published schema.

    Returns a findings record rather than raising, so every problem in a large
    table is reported at once instead of one per run.
    """
    expected = {spec.name for spec in schema}
    by_name = {spec.name: spec for spec in schema}
    findings: list[str] = []

    if not records:
        return {"ok": False, "n": 0, "findings": ["release table is empty"]}

    extra_seen: set[str] = set()
    missing_seen: set[str] = set()
    bad_values: Counter[str] = Counter()
    bad_types: Counter[str] = Counter()

    for row in records:
        keys = set(row)
        extra_seen |= keys - expected
        missing_seen |= expected - keys
        for name in keys & expected:
            spec = by_name[name]
            value = row[name]
            if value is None:
                continue
            if not _type_ok(value, spec.dtype):
                bad_types[name] += 1
            if spec.permitted is not None and value not in spec.permitted:
                bad_values[name] += 1

    forbidden = extra_seen & FORBIDDEN_FIELDS
    if forbidden:
        findings.append(
            f"forbidden field(s) present in the release: {sorted(forbidden)}"
        )
    if extra_seen - FORBIDDEN_FIELDS:
        findings.append(
            f"undocumented field(s) not in the schema: "
            f"{sorted(extra_seen - FORBIDDEN_FIELDS)}"
        )
    if missing_seen:
        findings.append(f"missing schema field(s): {sorted(missing_seen)}")
    for name, count in sorted(bad_types.items()):
        findings.append(f"{count} row(s) have the wrong type for {name!r}")
    for name, count in sorted(bad_values.items()):
        findings.append(f"{count} row(s) have a value outside the permitted set "
                        f"for {name!r}")

    return {"ok": not findings, "n": len(records), "findings": findings}


def _type_ok(value: Any, dtype: str) -> bool:
    if dtype == "boolean":
        return isinstance(value, bool)
    if dtype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if dtype == "string":
        return isinstance(value, str)
    raise ValueError(f"unknown dtype {dtype!r}")


_PROSE = re.compile(r"\S+\s+\S+\s+\S+")   # three or more whitespace-separated tokens


def scan_free_text(
    records: Sequence[Mapping[str, Any]], policy: ReleasePolicy,
) -> dict[str, Any]:
    """Detect narrative residue in any released field.

    The single most consequential leak this deposit could carry is a fragment of
    surveyor narrative describing an identifiable resident's care. Rather than
    trusting the assembly step to have dropped those columns, every string value
    is scanned: anything longer than the policy's ceiling, or that reads as prose
    (three or more whitespace-separated tokens), is flagged with its field and
    row so it can be traced.
    """
    offenders: list[dict[str, Any]] = []
    by_field: Counter[str] = Counter()
    for index, row in enumerate(records):
        for name, value in row.items():
            if not isinstance(value, str):
                continue
            too_long = len(value) > policy.max_text_length
            prose_like = bool(_PROSE.search(value))
            if too_long or prose_like:
                by_field[name] += 1
                if len(offenders) < 20:      # bounded sample for the report
                    offenders.append({
                        "row": index, "field": name, "length": len(value),
                        "reason": "too_long" if too_long else "prose_like",
                    })
    return {
        "ok": not by_field,
        "fields_flagged": dict(by_field),
        "examples": offenders,
        "max_text_length": policy.max_text_length,
    }


def k_anonymity_report(
    records: Sequence[Mapping[str, Any]], quasi_identifiers: Sequence[str], k: int,
) -> dict[str, Any]:
    """Count combinations of quasi-identifiers that appear fewer than ``k`` times.

    Applied to derived aggregates rather than to the document-level annotation
    table. A single citation is a specific, already-public incident, so the
    annotation rows disclose nothing new; the concern is a cross-tabulation in
    which one facility-quarter cell isolates a single resident's incident.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    missing = [name for name in quasi_identifiers
               if records and name not in records[0]]
    if missing:
        raise KeyError(f"quasi-identifier(s) absent from the records: {missing}")

    counts = Counter(
        tuple(row.get(name) for name in quasi_identifiers) for row in records
    )

    violations = {combo: count for combo, count in counts.items() if count < k}
    return {
        "ok": not violations,
        "k": int(k),
        "quasi_identifiers": list(quasi_identifiers),
        "n_combinations": len(counts),
        "n_violating": len(violations),
        "n_rows_at_risk": int(sum(violations.values())),
        "violating_examples": [
            {"combination": list(combo), "count": count}
            for combo, count in
            sorted(violations.items(), key=lambda kv: str(kv[0]))[:20]
        ],
    }


def suppress_small_cells(
    rows: Sequence[Mapping[str, Any]], count_field: str, k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Suppress aggregate cells below the threshold.

    The count is replaced with ``None`` and the row marked ``suppressed`` rather
    than dropped, because silently removing rows changes the shape of a
    published table and invites a reader to mistake absence for zero.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    out: list[dict[str, Any]] = []
    suppressed = 0
    for row in rows:
        value = row.get(count_field)
        record = dict(row)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if numeric and value < k:
            record[count_field] = None
            record["suppressed"] = True
            suppressed += 1
        else:
            record["suppressed"] = False
        out.append(record)
    return out, {
        "count_field": count_field, "k": int(k),
        "n_rows": len(rows), "n_suppressed": suppressed,
        "policy": "counts below k are masked, not dropped, so table shape is "
                  "preserved and absence is never mistaken for zero",
    }


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #

def aggregate_cell_check(
    rows: Sequence[Mapping[str, Any]], count_field: str, k: int,
) -> dict[str, Any]:
    """Verify no released aggregate cell falls below the threshold.

    This, not k-anonymity, is the right control for this deposit, and the
    distinction matters enough to state plainly. k-anonymity protects
    individuals who could be re-identified by a combination of quasi-identifiers
    in a record-level release. Here the record-level identifiers — facility,
    survey date, citation tag — are *already published by the regulator*, so
    requiring them to repeat k times would be incoherent with this module's own
    reasoning about facility identity, and would suppress public information to
    no benefit.

    The genuine residual risk is a derived aggregate so finely cut that one cell
    corresponds to a single resident's incident. That is a cell-size problem, so
    it gets a cell-size control: every released cell is either suppressed or at
    or above ``k``.

    :func:`k_anonymity_report` remains available for reviewing any future
    release that does introduce non-public quasi-identifiers.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    offenders = [
        {"row": index, "count": row.get(count_field)}
        for index, row in enumerate(rows)
        if not row.get("suppressed", False)
        and isinstance(row.get(count_field), (int, float))
        and not isinstance(row.get(count_field), bool)
        and row[count_field] < k
    ]
    return {
        "ok": not offenders,
        "k": int(k),
        "count_field": count_field,
        "n_rows": len(rows),
        "n_below_threshold": len(offenders),
        "examples": offenders[:20],
    }


def disclosure_review(
    annotations: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    policy: ReleasePolicy,
    count_field: str = "n",
) -> dict[str, Any]:
    """Run every check and decide whether the deposit may be written.

    Clearance requires all of: the annotation table matches the published
    schema; no field in either table carries free-text residue; and every
    released aggregate cell is either suppressed or at or above the threshold.
    The verdict and its evidence are written into the deposit, so a reader can
    see what was checked rather than taking "de-identified" on trust.

    A k-anonymity report over the annotation table is computed and recorded as a
    *diagnostic*, not a clearance condition — see :func:`aggregate_cell_check`
    for why applying it to already-public identifiers would be incoherent.
    """
    schema_check = validate_records(annotations)
    text_annotations = scan_free_text(annotations, policy)
    text_aggregates = scan_free_text(aggregates, policy)
    cells = (
        aggregate_cell_check(aggregates, count_field, policy.min_cell_size)
        if aggregates else
        {"ok": True, "k": policy.min_cell_size, "n_rows": 0,
         "n_below_threshold": 0, "note": "no aggregates supplied"}
    )

    checks = {
        "schema": schema_check,
        "free_text_annotations": text_annotations,
        "free_text_aggregates": text_aggregates,
        "aggregate_cell_size": cells,
    }
    failed = [name for name, result in checks.items() if not result.get("ok")]

    diagnostics: dict[str, Any] = {}
    present = [name for name in policy.quasi_identifiers
               if annotations and name in annotations[0]]
    if present:
        diagnostics["k_anonymity_annotations"] = k_anonymity_report(
            annotations, present, policy.min_cell_size
        )
        diagnostics["k_anonymity_note"] = (
            "Reported for transparency only. Not a clearance condition: the "
            "record-level identifiers in this release are already published by "
            "the regulator."
        )

    return {
        "cleared": not failed,
        "failed_checks": failed,
        "checks": checks,
        "diagnostics": diagnostics,
        "policy": asdict(policy),
        "reviewed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def prepare_annotations(
    records: Sequence[Mapping[str, Any]], policy: ReleasePolicy,
) -> list[dict[str, Any]]:
    """Project raw annotation records onto the released schema.

    Drops every field not in the schema — a whitelist, not a blacklist, so a new
    internal column added upstream cannot leak into a future release simply
    because nobody remembered to exclude it.
    """
    out: list[dict[str, Any]] = []
    for row in records:
        projected = {name: row.get(name) for name in RELEASE_FIELDS}
        if policy.hash_facility_id and projected.get("ccn") is not None:
            projected["ccn"] = salted_hash(projected["ccn"], policy.salt)
        out.append(projected)
    return out


def build_codebook(
    policy: ReleasePolicy, schema: Sequence[FieldSpec] = RELEASE_SCHEMA,
) -> dict[str, Any]:
    """Field-by-field documentation shipped alongside the data."""
    return {
        "title": "Codebook: technology-mediated failures in nursing-home "
                 "inspection records",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": policy.license,
        "disclosure_policy": asdict(policy),
        "fields": [
            {
                "name": spec.name, "type": spec.dtype,
                "description": spec.description,
                "permitted_values": list(spec.permitted) if spec.permitted else None,
                "provenance": spec.provenance or None,
                "caveat": spec.caveat or None,
            }
            for spec in schema
        ],
        "failure_role_definitions": dict(FAILURE_ROLE_DEFINITIONS),
        "not_released": {
            "narrative_text": "Public at source; not redistributed. Stages 01 "
                              "and 02 reconstruct it exactly from the manifest.",
            "evidence_span_text": "Released as character offsets only.",
        },
        "interpretation_warning": (
            "The technology fields are measured variables produced by a language "
            "model, with error rates quantified in the accompanying validation "
            "table. They are not ground truth and should not be reused without "
            "reading those error rates."
        ),
    }


def citation_cff(meta: Mapping[str, Any]) -> str:
    """Render a CITATION.cff so the deposit is citable in the article."""
    authors = "\n".join(
        f"  - family-names: {author.get('family', 'ANONYMISED')}\n"
        f"    given-names: {author.get('given', 'ANONYMISED')}"
        for author in meta.get("authors", [{}])
    )
    return (
        "cff-version: 1.2.0\n"
        "message: If you use this dataset, please cite it as below.\n"
        f"title: {meta.get('title', 'Technology-mediated failure annotations')}\n"
        f"version: {meta.get('version', '1.0.0')}\n"
        f"date-released: {meta.get('date', time.strftime('%Y-%m-%d'))}\n"
        f"license: {meta.get('license', 'CC-BY-4.0')}\n"
        f"doi: {meta.get('doi', 'PENDING-ON-ACCEPTANCE')}\n"
        "type: dataset\n"
        "authors:\n" + authors + "\n"
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def write_deposit(
    annotations: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    policy: ReleasePolicy,
    cfg: Config | None = None,
    extras: Mapping[str, Any] | None = None,
    destination: Path | None = None,
    count_field: str = "n",
) -> dict[str, Any]:
    """Assemble, review, and write the deposit. Fails closed.

    Nothing is written unless the review clears. A partial deposit is worse than
    none, because it looks complete and is not.

    Returns the deposit manifest, which lists every file with its SHA-256 so the
    archived copy can be verified against the one described in the article.
    """
    destination = Path(destination) if destination else DEPOSIT
    prepared = prepare_annotations(annotations, policy)
    aggregates_suppressed, suppression = suppress_small_cells(
        aggregates, count_field, policy.min_cell_size
    ) if aggregates else ([], {"n_rows": 0, "n_suppressed": 0})

    review = disclosure_review(prepared, aggregates_suppressed, policy, count_field)
    if not review["cleared"]:
        LOG.error("Disclosure review FAILED: %s", review["failed_checks"])
        for name in review["failed_checks"]:
            LOG.error("  %s: %s", name, review["checks"][name])
        raise RuntimeError(
            f"disclosure review not cleared ({review['failed_checks']}); "
            f"nothing written"
        )

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(save_json(prepared, destination / "annotations.json"))
    if aggregates_suppressed:
        written.append(save_json(aggregates_suppressed,
                                 destination / "facility_quarter_aggregates.json"))
    written.append(save_json(build_codebook(policy), destination / "codebook.json"))
    written.append(save_json(review, destination / "disclosure_review.json"))
    for name, payload in (extras or {}).items():
        written.append(save_json(payload, destination / f"{name}.json"))

    cff = destination / "CITATION.cff"
    cff.write_text(citation_cff(extras.get("citation", {}) if extras else {}),
                   encoding="utf-8")
    written.append(cff)

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": policy.license,
        "doi": "PENDING-ON-ACCEPTANCE",
        "n_annotation_rows": len(prepared),
        "n_aggregate_rows": len(aggregates_suppressed),
        "suppression": suppression,
        "disclosure_cleared": True,
        "config_fingerprint": cfg.fingerprint() if cfg else None,
        "provenance": provenance(cfg),
        "files": [
            {"path": relpath_to_root(path), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in sorted(written)
        ],
    }
    written.append(save_json(manifest, destination / "MANIFEST.json"))
    LOG.info("Deposit written to %s (%d files, %d annotation rows).",
             destination, len(manifest["files"]) + 1, len(prepared))
    return manifest


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the citable, disclosure-reviewed data deposit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotations", default=None,
                        help="path to the extracted annotations JSON")
    parser.add_argument("--aggregates", default=None,
                        help="path to the facility-quarter aggregates JSON")
    parser.add_argument("--review-only", action="store_true",
                        help="run the disclosure review and report, writing nothing")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path
    cfg = Config.load(config_path)
    ensure_dirs()
    policy = ReleasePolicy.from_config(cfg.deposit)

    from utils import load_json
    if not args.annotations:
        LOG.error("No --annotations supplied. This stage consumes the output of "
                  "the extraction stage; run `make extract` first.")
        return 1
    annotations = load_json(args.annotations)
    aggregates = load_json(args.aggregates) if args.aggregates else []

    if args.review_only:
        prepared = prepare_annotations(annotations, policy)
        review = disclosure_review(prepared, aggregates, policy)
        LOG.info("Disclosure review: %s",
                 "CLEARED" if review["cleared"] else review["failed_checks"])
        save_json(review, DEPOSIT / "disclosure_review.json")
        return 0 if review["cleared"] else 1

    try:
        write_deposit(annotations, aggregates, policy, cfg)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    return 0


__all__ = [
    "FieldSpec", "RELEASE_SCHEMA", "RELEASE_FIELDS", "FORBIDDEN_FIELDS",
    "ReleasePolicy", "salted_hash", "validate_records", "scan_free_text",
    "k_anonymity_report", "suppress_small_cells", "aggregate_cell_check",
    "disclosure_review",
    "prepare_annotations", "build_codebook", "citation_cff", "write_deposit",
]


if __name__ == "__main__":
    sys.exit(main())
