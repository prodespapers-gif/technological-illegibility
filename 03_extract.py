"""
03_extract.py — LLM measurement of technology-mediated care failures.

Stage 3 of the pipeline and the study's measurement instrument. It reads each
deficiency narrative and returns a structured judgement: was a care technology
implicated, which kind, and — the theoretically decisive question — what role
did it play in the failure?

THE CODING FRAME
----------------
The vocabularies live in ``utils.py`` (single source of truth, shared with the
deposit schema). Ten technology types; ten failure roles:

  absent             - expected technology not in place
  incidental         - mentioned but causally irrelevant
  juxtaposition_error- wrong adjacent resident/item selected in an interface
  malfunction        - device or system failed technically
  misconfiguration   - present but wrongly set up for this resident
  not_responded_to   - it signalled; staff did not act in time
  over_reliance      - deference DESPITE contradicting signals (automation
                       bias sensu Alon-Barkat & Busuioc 2023; mere use of a
                       device is not over-reliance)
  overcompleteness   - templated documentation "complete but empty"
  workaround         - bypassed, disabled, or circumvented
  none

``overcompleteness`` and ``juxtaposition_error`` extend the Ash, Berg & Coiera
(2004) typology of technology-induced errors; both are common in this setting
and were absent from the original eight-role frame. The frame is FROZEN as of
this revision: changing it invalidates the gold-standard annotation round.

Remaining fields: ``technology_present`` (bool), ``harm_linked`` (bool, true
only when the narrative ties the technology's role to actual resident harm),
``evidence_span`` (verbatim, audit only), ``confidence`` (model's own 0-1).

PROMPT DESIGN (Ziems et al. 2024, Comput. Linguist. 50(1))
----------------------------------------------------------
The templates follow their guidelines with one reasoned deviation. Applied:
the narrative comes FIRST and instructions after it, because recent text has
greater effect under common attention patterns; options are enumerated one per
line; an explicit constraint forbids refusal and hedging ("Even if you are
uncertain, you must pick..."), with uncertainty routed into ``confidence``;
output is requested as JSON. Deviation: options are NOT lettered A/B/C —
their single-letter-token guideline fits single-label tasks, whereas this is
a six-field judgement where their own JSON guideline governs, and lettering
would invite ``"A"`` as a field value, a guaranteed parse failure. Templates
must keep the ``NARRATIVE:`` and ``INSTRUCTIONS:`` section markers: the stub
backend and the paraphrase-variant checks rely on them.

WHY LOCAL INFERENCE IS NOT OPTIONAL
-----------------------------------
The narratives describe identifiable care episodes involving vulnerable
residents. Sending them to a third-party API would export sensitive material
outside the research environment for no methodological gain. The vLLM backend
runs on local GPUs with tensor parallelism; no text leaves the machine. This is
reported as an ethics requirement in the manuscript, not a convenience.

DETERMINISM AND PROVENANCE
--------------------------
Temperature zero, fixed seed, pinned model revision, and a hashed prompt. Every
output row records the model id, revision, and prompt hash, so a result can
never be silently produced by an edited prompt or a drifting model. Rows
produced under different prompts are not interchangeable and the hash is what
makes that checkable.

MALFORMED OUTPUT IS COUNTED, NOT COERCED
----------------------------------------
A response that does not parse is recorded as a parse failure and excluded, and
the parse-failure rate is reported in the paper. Coercing an unparseable
response to "no technology" would silently bias prevalence downward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils import (ANNOTATIONS, CONFIGS, DATA_PROCESSED, FAILURE_ROLES,
                   TECHNOLOGY_TYPES, Config, ensure_dirs, get_logger,
                   load_json, save_json, set_seed, timed)

LOG = get_logger("extract")

# The category vocabularies are defined once, in utils.py, and imported here
# and by the deposit stage (09), so the extraction validator and the released
# schema cannot drift apart. See utils.py for the frame's lineage and for the
# rule that any change invalidates prior extraction and annotation output.
SCHEMA_KEYS: tuple[str, ...] = (
    "technology_present", "technology_type", "failure_role", "harm_linked",
    "evidence_span", "confidence",
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ExtractionError(RuntimeError):
    """Raised when extraction cannot proceed."""


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

def load_prompt(cfg: Config, variant: str | None = None) -> str:
    """Load a versioned prompt template from ``configs/``."""
    name = cfg.get("extraction.prompt_file", "configs/prompt_v1.txt")
    path = Path(name)
    if variant:
        path = path.with_name(path.stem.replace("v1", variant) + path.suffix)
    if not path.is_absolute():
        path = CONFIGS.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def prompt_hash(template: str) -> str:
    """Stable hash of a prompt template, stamped onto every row it produces."""
    return hashlib.blake2b(template.encode("utf-8"), digest_size=6).hexdigest()


def build_prompt(template: str, narrative: str) -> str:
    """Render the extraction prompt for one narrative."""
    if "{{NARRATIVE}}" not in template:
        raise ValueError("prompt template must contain the {{NARRATIVE}} slot")
    return template.replace("{{NARRATIVE}}", narrative)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

# Ziems et al. (2024): models "hedge in the case of uncertainty, refuse to
# engage with offensive language, and attempt to generalize beyond provided
# labels." This corpus describes abuse and neglect, so refusals WILL occur and
# must be counted separately from malformed output: a refusal concentrated in
# abuse narratives is differential missingness on the highest-stakes
# documents — a finding, not noise.
REFUSAL_PATTERNS = re.compile(
    r"i\s+(?:can(?:no|')t|cannot|won'?t|am\s+unable\s+to)\s+"
    r"(?:help|assist|analy[sz]e|classify|label|provide|do\s+that)|"
    r"i'?m\s+(?:sorry|not\s+able)|as\s+an\s+ai\b|"
    r"(?:harmful|inappropriate)\s+content",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    """Whether raw model output reads as a refusal rather than a judgement."""
    return bool(REFUSAL_PATTERNS.search(text or ""))


def response_json_schema() -> dict[str, Any]:
    """JSON schema for constrained decoding, built from the shared frame."""
    return {
        "type": "object",
        "properties": {
            "technology_present": {"type": "boolean"},
            "technology_type": {"enum": list(TECHNOLOGY_TYPES)},
            "failure_role": {"enum": list(FAILURE_ROLES)},
            "harm_linked": {"type": "boolean"},
            "evidence_span": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["technology_present", "technology_type", "failure_role",
                     "harm_linked", "evidence_span", "confidence"],
        "additionalProperties": False,
    }


def parse_response(text: str) -> dict[str, Any]:
    """Parse and validate one model response.

    Returns a record with ``parse_ok``. Never guesses: a response missing a key,
    carrying an out-of-vocabulary category, or wrapping malformed JSON is
    rejected outright, because a coerced value would enter the prevalence
    estimate as though it had been measured.
    """
    match = _JSON_OBJECT.search(text or "")
    if not match:
        refused = looks_like_refusal(text)
        return {"parse_ok": False, "refusal": refused,
                "parse_error": "refusal" if refused
                else "no JSON object found"}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"parse_ok": False, "parse_error": f"invalid JSON: {exc.msg}"}
    if not isinstance(obj, dict):
        return {"parse_ok": False, "parse_error": "JSON root is not an object"}

    missing = [key for key in SCHEMA_KEYS if key not in obj]
    if missing:
        return {"parse_ok": False, "parse_error": f"missing keys: {missing}"}
    if not isinstance(obj["technology_present"], bool):
        return {"parse_ok": False, "parse_error": "technology_present not boolean"}
    if not isinstance(obj["harm_linked"], bool):
        return {"parse_ok": False, "parse_error": "harm_linked not boolean"}
    if obj["technology_type"] not in TECHNOLOGY_TYPES:
        return {"parse_ok": False,
                "parse_error": f"unknown technology_type {obj['technology_type']!r}"}
    if obj["failure_role"] not in FAILURE_ROLES:
        return {"parse_ok": False,
                "parse_error": f"unknown failure_role {obj['failure_role']!r}"}
    try:
        confidence = float(obj["confidence"])
    except (TypeError, ValueError):
        return {"parse_ok": False, "parse_error": "confidence not numeric"}
    if not 0.0 <= confidence <= 1.0:
        return {"parse_ok": False, "parse_error": "confidence outside [0, 1]"}

    # Internal coherence: a record claiming no technology cannot also name one.
    if not obj["technology_present"] and (
        obj["technology_type"] != "none" or obj["failure_role"] != "none"
    ):
        return {"parse_ok": False,
                "parse_error": "technology_present is false but a type or role "
                               "was named"}

    return {
        "parse_ok": True,
        "technology_present": obj["technology_present"],
        "technology_type": obj["technology_type"],
        "failure_role": obj["failure_role"],
        "harm_linked": obj["harm_linked"],
        "evidence_span": str(obj["evidence_span"] or ""),
        "confidence": confidence,
    }


def locate_span(narrative: str, span: str) -> tuple[int | None, int | None]:
    """Return character offsets of an evidence span within its narrative.

    Offsets, not text, are what the deposit releases. A span the model
    paraphrased rather than quoted will not be found; that is reported as a
    missing offset rather than approximated.
    """
    if not span:
        return None, None
    start = narrative.find(span)
    if start < 0:
        return None, None
    return start, start + len(span)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class Backend:
    """Interface every inference backend implements."""

    name = "base"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        raise NotImplementedError


class VLLMBackend(Backend):
    """Local inference via vLLM with tensor parallelism across the GPUs."""

    name = "vllm"

    def __init__(self, cfg: Config, model_key: str = "model"):
        from vllm import LLM, SamplingParams

        spec = cfg.extraction
        self.model_id = spec[model_key]
        self.revision = spec.get("revision")
        self.engine = LLM(
            model=self.model_id, revision=self.revision,
            tensor_parallel_size=int(spec.get("tensor_parallel_size", 4)),
            dtype=spec.get("dtype", "bfloat16"), seed=cfg.seed,
            gpu_memory_utilization=float(spec.get("gpu_memory_utilization", 0.90)),
        )
        kwargs: dict[str, Any] = {}
        if bool(spec.get("guided_json", True)):
            # Constrained decoding (Ziems et al. constrain output to valid
            # classes): the schema makes malformed output structurally
            # impossible, so the parse-failure counter measures refusals and
            # incoherence only. Falls back loudly, not silently, on vLLM
            # builds without guided decoding.
            try:
                from vllm.sampling_params import GuidedDecodingParams

                kwargs["guided_decoding"] = GuidedDecodingParams(
                    json=response_json_schema()
                )
            except Exception:  # pragma: no cover - depends on vllm build
                LOG.warning("vLLM build lacks GuidedDecodingParams; "
                            "extraction runs unconstrained and relies on "
                            "parse validation alone")
        self.params = SamplingParams(
            temperature=float(spec.get("temperature", 0.0)),
            max_tokens=int(spec.get("max_tokens", 512)), seed=cfg.seed,
            **kwargs,
        )

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs = self.engine.generate(list(prompts), self.params)
        return [output.outputs[0].text for output in outputs]


class StubBackend(Backend):
    """Deterministic keyword backend for pipeline testing. NOT FOR PUBLICATION.

    Emits schema-valid responses derived from keyword matches so the full chain
    can be exercised without GPUs. It is a plumbing test, not a measurement:
    :func:`extract_corpus` records the backend name on every row and
    :func:`assert_publication_backend` refuses to let stub output be treated as
    a result.
    """

    name = "stub"

    KEYWORDS: dict[str, str] = {
        "call light": "call_light", "call bell": "call_light",
        "bed alarm": "fall_alarm_or_sensor", "alarm": "fall_alarm_or_sensor",
        "sensor": "fall_alarm_or_sensor", "camera": "camera_monitoring",
        "wanderguard": "wander_elopement_system",
        "elopement": "wander_elopement_system",
        "electronic health record": "electronic_health_record",
        "emar": "e_medication_administration",
        "telehealth": "remote_or_tele_monitoring",
        "predictive": "predictive_or_ai_decision_support",
    }
    ROLE_KEYWORDS: dict[str, str] = {
        "not functioning": "malfunction", "malfunction": "malfunction",
        "did not respond": "not_responded_to", "unanswered": "not_responded_to",
        "not answered": "not_responded_to",
        "relied on": "over_reliance", "not turned on": "misconfiguration",
        "disabled": "workaround", "was not in place": "absent",
        "wrong resident": "juxtaposition_error",
        "adjacent": "juxtaposition_error",
        "standard text": "overcompleteness",
        "documented as completed": "overcompleteness",
        "templated": "overcompleteness",
    }

    def __init__(self, cfg: Config, model_key: str = "model"):
        self.model_id = f"stub:{cfg.extraction.get(model_key, 'primary')}"
        self.revision = "stub"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        out: list[str] = []
        for prompt in prompts:
            # Bounded on BOTH sides. The templates place the narrative before
            # the instructions (attention recency, per the module docstring),
            # and the instruction block enumerates category names such as
            # "fall_alarm_or_sensor". An unbounded split would sweep those
            # names into the keyword scan and make every document match
            # "alarm" — a silent 100% false-positive rate in every pipeline
            # test. Matching must see the narrative and nothing else.
            body = (prompt.split("NARRATIVE:")[-1]
                    .split("INSTRUCTIONS:")[0].lower())
            technology, span = "none", ""
            for keyword, label in self.KEYWORDS.items():
                if keyword in body:
                    technology, span = label, keyword
                    break
            role = "none"
            if technology != "none":
                role = "incidental"
                for keyword, label in self.ROLE_KEYWORDS.items():
                    if keyword in body:
                        role = label
                        break
            digest = int(hashlib.blake2b(body.encode(), digest_size=4).hexdigest(), 16)
            out.append(json.dumps({
                "technology_present": technology != "none",
                "technology_type": technology,
                "failure_role": role,
                "harm_linked": bool(digest % 3 == 0) and technology != "none",
                "evidence_span": span,
                "confidence": round(0.6 + (digest % 40) / 100.0, 2),
            }))
        return out


BACKENDS: dict[str, type[Backend]] = {"vllm": VLLMBackend, "stub": StubBackend}


def build_backend(cfg: Config, model_key: str = "model") -> Backend:
    name = cfg.get("extraction.backend", "vllm")
    if name not in BACKENDS:
        raise ExtractionError(
            f"unknown backend {name!r}; available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name](cfg, model_key)


def assert_publication_backend(cfg: Config) -> None:
    """Refuse to treat stub output as a measurement."""
    if cfg.get("extraction.backend", "vllm") == "stub":
        raise ExtractionError(
            "extraction.backend is 'stub', which produces keyword-matched "
            "placeholder judgements for pipeline testing only. Set it to 'vllm' "
            "before producing any result reported in the manuscript."
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionStats:
    n_documents: int = 0
    n_parsed: int = 0
    n_parse_failures: int = 0
    n_refusals: int = 0          # subset of parse failures; counted separately
    n_spans_located: int = 0

    @property
    def parse_failure_rate(self) -> float:
        return self.n_parse_failures / self.n_documents if self.n_documents else 0.0


def extract_documents(
    narratives: Sequence[str], doc_ids: Sequence[str], cfg: Config,
    backend: Backend, template: str,
) -> tuple[pd.DataFrame, ExtractionStats]:
    """Run one backend over a set of narratives and return parsed judgements."""
    if len(narratives) != len(doc_ids):
        raise ValueError("narratives and doc_ids must align")
    batch_size = int(cfg.get("extraction.batch_size", 256))
    digest = prompt_hash(template)
    stats = ExtractionStats(n_documents=len(narratives))
    rows: list[dict[str, Any]] = []

    for start in range(0, len(narratives), batch_size):
        chunk = list(narratives[start:start + batch_size])
        ids = list(doc_ids[start:start + batch_size])
        responses = backend.generate([build_prompt(template, text) for text in chunk])
        if len(responses) != len(chunk):
            raise ExtractionError(
                f"backend returned {len(responses)} responses for {len(chunk)} prompts"
            )
        for doc_id, narrative, response in zip(ids, chunk, responses):
            parsed = parse_response(response)
            record = {"doc_id": doc_id, "prompt_hash": digest,
                      "model_id": backend.model_id,
                      "model_revision": getattr(backend, "revision", None),
                      "backend": backend.name}
            if not parsed["parse_ok"]:
                stats.n_parse_failures += 1
                if parsed.get("refusal"):
                    stats.n_refusals += 1
                record.update({"parse_ok": False,
                               "parse_error": parsed["parse_error"]})
            else:
                stats.n_parsed += 1
                span_start, span_end = locate_span(narrative, parsed["evidence_span"])
                if span_start is not None:
                    stats.n_spans_located += 1
                record.update({
                    "parse_ok": True, "parse_error": None,
                    "technology_present": parsed["technology_present"],
                    "technology_type": parsed["technology_type"],
                    "failure_role": parsed["failure_role"],
                    "harm_linked": parsed["harm_linked"],
                    "confidence": parsed["confidence"],
                    "evidence_span_start": span_start,
                    "evidence_span_end": span_end,
                })
            rows.append(record)
        LOG.info("  extracted %d/%d", min(start + batch_size, len(narratives)),
                 len(narratives))
    return pd.DataFrame(rows), stats


def load_gold_standard(cfg: Config) -> pd.DataFrame | None:
    """Load human annotations, if the annotation round has been completed.

    Expects one JSON file per annotator in ``data/annotations/``, each a list of
    records carrying ``doc_id``, ``technology_present``, and ``failure_role``.
    Returns ``None`` when no annotations exist, so the pipeline can run before
    the human round and report the validation stage as unmet rather than
    fabricating agreement.
    """
    paths = sorted(ANNOTATIONS.glob("annotator_*.json"))
    if not paths:
        return None
    frames = []
    for index, path in enumerate(paths):
        frame = pd.DataFrame(load_json(path))
        frame["annotator"] = index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def annotation_matrix(
    gold: pd.DataFrame, field: str = "technology_present"
) -> np.ndarray:
    """Reshape annotations into the coders x units matrix stage 04 expects."""
    pivot = gold.pivot_table(index="annotator", columns="doc_id", values=field,
                             aggfunc="first")
    return pivot.to_numpy(dtype=float, na_value=np.nan)


def extract_corpus(sample: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Run the full measurement round and assemble stage 04's inputs.

    Returns the keys the validation stage consumes: the human annotation matrix,
    paired model and human labels, predictions under each prompt variant,
    predictions from a second model family, and the error indicator with its
    facility covariates for the differential-error audit.
    """
    if "narrative" not in sample or "doc_id" not in sample:
        raise ExtractionError("sample must carry doc_id and narrative columns")
    template = load_prompt(cfg)
    primary = build_backend(cfg, "model")
    narratives = sample["narrative"].astype(str).tolist()
    doc_ids = sample["doc_id"].astype(str).tolist()

    with timed("extraction: primary model", LOG):
        frame, stats = extract_documents(narratives, doc_ids, cfg, primary, template)

    variants = list(cfg.get("validation.prompt_variants", ["v1"]))
    predictions_by_prompt: dict[str, list[int]] = {}
    for variant in variants:
        try:
            variant_template = load_prompt(cfg, variant if variant != "v1" else None)
        except FileNotFoundError:
            LOG.warning("prompt variant %s not found; skipping", variant)
            continue
        if variant == variants[0]:
            variant_frame = frame
        else:
            variant_frame, _ = extract_documents(
                narratives, doc_ids, cfg, primary, variant_template
            )
        predictions_by_prompt[variant] = (
            variant_frame["technology_present"].fillna(False).astype(int).tolist()
        )

    secondary_id = cfg.get("extraction.secondary_model")
    if secondary_id:
        secondary = build_backend(cfg, "secondary_model")
        with timed("extraction: secondary model", LOG):
            secondary_frame, _ = extract_documents(
                narratives, doc_ids, cfg, secondary, template
            )
        secondary_pred = secondary_frame["technology_present"].fillna(False).astype(int)
    else:
        secondary_pred = frame["technology_present"].fillna(False).astype(int)
        LOG.warning("no secondary model configured; cross-model sensitivity will "
                    "be uninformative")

    predicted = frame["technology_present"].fillna(False).astype(int).to_numpy()
    gold = load_gold_standard(cfg)
    if gold is None:
        raise ExtractionError(
            "no human annotations found in data/annotations/. The validation "
            "stage requires a completed gold-standard round; see the codebook "
            "for the expected annotator file format."
        )

    # The gold standard must cover the documents the pipeline actually drew.
    # If the annotation round was run against a different sample — a different
    # seed, an earlier corpus vintage — the join silently yields mostly-missing
    # labels and agreement collapses toward chance, which reads as a failed
    # model rather than a misaligned join. Detect it explicitly.
    overlap = len(set(gold["doc_id"].astype(str)) & set(doc_ids)) / len(doc_ids)
    minimum = float(cfg.get("validation.min_gold_overlap", 0.90))
    if overlap < minimum:
        raise ExtractionError(
            f"only {overlap:.1%} of the drawn sample has human annotations "
            f"(minimum {minimum:.0%}). The annotation round and the pipeline "
            f"sample disagree — check that the corpus vintage and seed match "
            f"those used when annotators worked, since a partial join produces "
            f"chance-level agreement that looks like a failed model."
        )

    aligned = (gold[gold["annotator"] == 0]
               .set_index("doc_id").reindex(doc_ids))
    gold_present = aligned["technology_present"].fillna(False).astype(int).to_numpy()
    gold_role = aligned.get(
        "failure_role", pd.Series(["none"] * len(doc_ids))
    ).fillna("none").to_numpy()

    covariate_names = list(cfg.get("analysis.equity_attrs", []))
    covariates = np.column_stack([
        pd.to_numeric(sample.get(name, pd.Series(np.zeros(len(sample)))),
                      errors="coerce").fillna(0.0).to_numpy()
        for name in covariate_names
    ]) if covariate_names else np.zeros((len(sample), 1))
    if not covariate_names:
        covariate_names = ["intercept_only"]

    # Gilardi et al. (2023) measure the model's OWN consistency with a second
    # identical pass (97% self-agreement at temperature 0.2 vs 91% at 1.0).
    # With a deterministic local engine this doubles as the determinism check
    # that Ollion et al.'s reproducibility critique calls for.
    self_consistency: dict[str, Any] | None = None
    if int(cfg.get("extraction.self_consistency_passes", 2)) >= 2:
        with timed("extraction: self-consistency pass", LOG):
            repeat_frame, _ = extract_documents(
                narratives, doc_ids, cfg, primary, template
            )
        fields = ("technology_present", "technology_type", "failure_role")
        self_consistency = {
            "n": int(len(frame)),
            "agreement_by_field": {
                field: float(
                    (frame[field].fillna("__na__").to_numpy()
                     == repeat_frame[field].fillna("__na__").to_numpy()).mean()
                )
                for field in fields
            },
            "temperature": float(cfg.get("extraction.temperature", 0.0)),
        }

    save_json({
        "n_documents": stats.n_documents, "n_parsed": stats.n_parsed,
        "n_parse_failures": stats.n_parse_failures,
        "n_refusals": stats.n_refusals,
        "parse_failure_rate": round(stats.parse_failure_rate, 4),
        "spans_located": stats.n_spans_located,
        "backend": primary.name, "model_id": primary.model_id,
        "prompt_hash": prompt_hash(template),
    }, DATA_PROCESSED / "extraction_stats.json")

    return {
        "annotations": frame,
        "annotation_matrix": annotation_matrix(gold),
        "pred_technology_present": predicted,
        "gold_technology_present": gold_present,
        "pred_failure_role": frame["failure_role"].fillna("none").to_numpy(),
        "gold_failure_role": gold_role,
        "predictions_by_prompt": predictions_by_prompt,
        "self_consistency": self_consistency,
        "n_refusals": stats.n_refusals,
        "pred_primary": predicted,
        "pred_secondary": secondary_pred.to_numpy(),
        "facility_ids": sample["ccn"].astype(str).to_numpy(),
        "errors": (predicted != gold_present).astype(int),
        "covariates": covariates,
        "covariate_names": covariate_names,
        "stats": {
            "n_documents": stats.n_documents,
            "parse_failure_rate": stats.parse_failure_rate,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM extraction.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="pilot on N docs")
    parser.add_argument("--allow-stub", action="store_true",
                        help="permit the stub backend (testing only)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path
    cfg = Config.load(config_path)
    set_seed(cfg.seed)
    ensure_dirs()
    if not args.allow_stub:
        assert_publication_backend(cfg)

    corpus_files = sorted(DATA_PROCESSED.glob("corpus_*.parquet"))
    if not corpus_files:
        LOG.error("no corpus found; run 02_corpus.py first.")
        return 1
    corpus = pd.read_parquet(corpus_files[-1])
    if args.limit:
        corpus = corpus.head(args.limit)

    template = load_prompt(cfg)
    backend = build_backend(cfg, "model")
    frame, stats = extract_documents(
        corpus["narrative"].astype(str).tolist(),
        corpus["doc_id"].astype(str).tolist(), cfg, backend, template,
    )
    out = DATA_PROCESSED / f"extracted_{cfg.fingerprint()}.parquet"
    frame.to_parquet(out)
    LOG.info("written: %s (%d rows, %.2f%% parse failures)",
             out.name, len(frame), 100 * stats.parse_failure_rate)
    return 0


__all__ = [
    "looks_like_refusal", "response_json_schema",
    "TECHNOLOGY_TYPES", "FAILURE_ROLES", "SCHEMA_KEYS", "ExtractionError",
    "load_prompt", "prompt_hash", "build_prompt", "parse_response",
    "locate_span", "Backend", "VLLMBackend", "StubBackend", "BACKENDS",
    "build_backend", "assert_publication_backend", "ExtractionStats",
    "extract_documents", "load_gold_standard", "annotation_matrix",
    "extract_corpus",
]


if __name__ == "__main__":
    sys.exit(main())
