"""
utils.py — Shared infrastructure for the technological-illegibility pipeline.

Every other module in ``src/`` imports from this file. Cross-cutting concerns —
repository layout, configuration, reproducibility, logging, atomic I/O, content
hashing, and run provenance — are deliberately concentrated here so that the
project's determinism guarantees can be audited in one place rather than
reconstructed from nine scattered files.

Nothing in this module is specific to the research question: it is engineering
scaffolding, and it is the only module with no domain assumptions.

Design commitments
------------------
1. **Fail loudly, never silently.** A mistyped configuration key raises rather
   than falling back to a default, because a silent fallback produces a result
   that looks valid and is not.
2. **Every artefact is traceable.** The configuration fingerprint and the run
   provenance record are stamped onto outputs, so any number in the manuscript
   can be traced to the exact settings, code revision, and environment that
   produced it.
3. **Writes are atomic.** A pipeline stage that dies mid-write leaves the
   previous file intact rather than a truncated one that the next stage would
   happily read.
4. **Optional dependencies stay optional.** Data-only stages must run on a
   machine with no GPU stack, so ``torch`` is imported lazily and its absence is
   never fatal.

Python 3.11+.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

__all__ = [
    "ROOT", "SRC", "CONFIGS", "DATA", "DATA_RAW", "DATA_PROCESSED",
    "ANNOTATIONS", "RESULTS", "FIGURES", "TABLES", "LOGS", "DEPOSIT",
    "TECHNOLOGY_TYPES", "FAILURE_ROLES", "FAILURE_ROLE_DEFINITIONS",
    "Config", "set_seed", "get_device", "get_logger", "save_json", "load_json",
    "sha256_file", "sha256_bytes", "stable_doc_id", "ensure_dirs",
    "relpath_to_root", "provenance", "timed",
]

# --------------------------------------------------------------------------- #
# 1. Repository layout
# --------------------------------------------------------------------------- #
# Resolved from this file's location rather than the working directory, so the
# pipeline behaves identically whether invoked as `python src/01_acquire.py`,
# via `make`, or from a scheduler in another directory.

ROOT: Path = Path(__file__).resolve().parents[1]
SRC: Path = ROOT / "src"
CONFIGS: Path = ROOT / "configs"

DATA: Path = ROOT / "data"
DATA_RAW: Path = DATA / "raw"                 # immutable downloads + manifest
DATA_PROCESSED: Path = DATA / "processed"     # corpus, extractions
ANNOTATIONS: Path = DATA / "annotations"      # human gold-standard labels

RESULTS: Path = ROOT / "results"
FIGURES: Path = RESULTS / "figures"
TABLES: Path = RESULTS / "tables"
LOGS: Path = RESULTS / "logs"

DEPOSIT: Path = ROOT / "deposit"              # DOI-ready derived dataset

_ALL_DIRS: tuple[Path, ...] = (
    DATA_RAW, DATA_PROCESSED, ANNOTATIONS, FIGURES, TABLES, LOGS, DEPOSIT,
)


def ensure_dirs() -> None:
    """Create the full output layout if it does not yet exist.

    Idempotent. Called at the top of each executable stage so that a fresh clone
    (in which the git-ignored data and results trees are absent) runs without
    manual setup.
    """
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def relpath_to_root(path: str | Path) -> str:
    """Return ``path`` relative to the repository root, as a POSIX-style string.

    Manifests and deposit records store repository-relative paths so they remain
    valid when the project is moved, archived, or unpacked by a replicator on a
    different machine. Paths outside the repository are returned absolute, since
    silently rewriting them would misrepresent where the file actually lives.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


# --------------------------------------------------------------------------- #
# 2. The coding frame (single source of truth)
# --------------------------------------------------------------------------- #
# These vocabularies define the study's measured variables. They are declared
# here, once, because they are load-bearing in two modules that must never
# drift: the extraction stage (03) validates every model response against
# them, and the deposit stage (09) publishes them as the released schema's
# permitted values. When they were independent copies, a category added in one
# place but not the other would have produced judgements the deposit rejects —
# a failure discovered only after the extraction run had been paid for.
#
# ANY change here invalidates prior extraction output AND the human annotation
# round, so the frame must be frozen before either begins.
#
# Lineage of the failure roles: the set extends the technology-induced-error
# typology of Ash, Berg & Coiera (2004, J. Am. Med. Inform. Assoc. 11(2)),
# from which `overcompleteness` (their "complete but empty" templated
# documentation) and `juxtaposition_error` (their wrong-adjacent-selection
# error) are drawn. `over_reliance` follows the automation-bias definition of
# Alon-Barkat & Busuioc (2023, J. Public Adm. Res. Theory 33(1)): deference to
# an automated system DESPITE warning signals or contradictory information —
# mere use of a device is not over-reliance.

TECHNOLOGY_TYPES: tuple[str, ...] = (
    "call_light",
    "camera_monitoring",
    "e_medication_administration",
    "electronic_health_record",
    "fall_alarm_or_sensor",
    "predictive_or_ai_decision_support",
    "remote_or_tele_monitoring",
    "wander_elopement_system",
    "other",
    "none",
)

FAILURE_ROLES: tuple[str, ...] = (
    "absent",
    "incidental",
    "juxtaposition_error",
    "malfunction",
    "misconfiguration",
    "not_responded_to",
    "over_reliance",
    "overcompleteness",
    "workaround",
    "none",
)

# One-line operational definitions. Shipped verbatim in the deposit codebook so
# reusers of the released annotations receive the frame, not just the labels;
# the prompt templates carry the same definitions and a verification check
# asserts every role name appears in every template.
FAILURE_ROLE_DEFINITIONS: dict[str, str] = {
    "absent": (
        "An expected care technology was not in place or not provided where "
        "the standard of care called for one."
    ),
    "incidental": (
        "A technology is mentioned in the narrative but played no causal part "
        "in the failure."
    ),
    "juxtaposition_error": (
        "The wrong resident, item, or entry was selected from adjacent options "
        "in an interface (e.g., wrong-patient order entry)."
    ),
    "malfunction": (
        "The device or system failed technically: broken, unpowered, out of "
        "order, or producing wrong output."
    ),
    "misconfiguration": (
        "The technology was present and functional but wrongly set up for this "
        "resident (e.g., alarm not turned on, wrong thresholds)."
    ),
    "not_responded_to": (
        "The technology signalled as designed, and staff did not act on the "
        "signal in time."
    ),
    "over_reliance": (
        "Staff deferred to the technology despite warning signals or "
        "contradictory information from other sources, substituting its output "
        "for direct assessment of the resident."
    ),
    "overcompleteness": (
        "Templated or auto-filled documentation presented care as complete "
        "while the narrative shows it was not delivered or not verifiable."
    ),
    "workaround": (
        "Staff bypassed, disabled, or circumvented the technology so it could "
        "not perform its protective function."
    ),
    "none": "No care technology was implicated in the deficiency.",
}


# --------------------------------------------------------------------------- #
# 3. Configuration
# --------------------------------------------------------------------------- #

# Scalar fields are validated against these types; all other fields are sections.
_SCALAR_TYPES: dict[str, type] = {"seed": int, "device": str}
_SECTION_NAMES: frozenset[str] = frozenset(
    {"data", "corpus", "extraction", "validation", "analysis", "forecast", "deposit"}
)


@dataclass(frozen=True)
class Config:
    """Typed, validated view over ``configs/*.yaml``.

    A frozen dataclass rather than a bare dict, for three reasons: attribute
    access documents the expected shape; unknown keys are rejected at load time;
    and immutability of the top-level fields means the object cannot drift apart
    from the fingerprint computed from it.

    Note the deliberate limitation: freezing prevents rebinding ``cfg.seed``, but
    the nested section dictionaries remain mutable. Downstream modules must
    therefore treat sections as read-only. This is documented rather than
    enforced — deep-freezing would complicate every call site for little gain —
    and is the one place where this module relies on convention.

    Attributes correspond one-to-one with the top-level blocks of the YAML file.
    """

    seed: int = 20260101
    device: str = "cuda"
    data: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)
    deposit: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load and validate a YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist — surfaced explicitly rather than as an
            opaque error deep inside the YAML parser.
        ValueError
            If the file is empty or its top level is not a mapping.
        KeyError
            If the file contains keys this dataclass does not define. A typo
            such as ``extration:`` would otherwise be silently ignored and the
            stage would run with default settings, producing a plausible but
            wrong result.
        TypeError
            If a field has the wrong type (e.g. ``seed: "abc"``).
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if raw is None:
            raise ValueError(f"Configuration file is empty: {path}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Configuration root must be a mapping, got "
                f"{type(raw).__name__}: {path}"
            )

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(
                f"Unknown configuration key(s) {sorted(unknown)} in {path}. "
                f"Permitted keys: {sorted(known)}."
            )

        cls._validate_types(raw, path)
        return cls(**raw)

    @staticmethod
    def _validate_types(raw: dict[str, Any], path: Path) -> None:
        """Check field types before construction, with actionable messages."""
        # bool is a subclass of int, so a boolean seed would pass a naive
        # isinstance check; reject it explicitly.
        if isinstance(raw.get("seed"), bool):
            raise TypeError(
                f"Configuration key 'seed' must be int, got bool in {path}."
            )

        for key, expected in _SCALAR_TYPES.items():
            if key in raw and not isinstance(raw[key], expected):
                raise TypeError(
                    f"Configuration key '{key}' must be {expected.__name__}, got "
                    f"{type(raw[key]).__name__} in {path}."
                )

        for key in _SECTION_NAMES & set(raw):
            if not isinstance(raw[key], dict):
                raise TypeError(
                    f"Configuration section '{key}' must be a mapping, got "
                    f"{type(raw[key]).__name__} in {path}."
                )

    # -- access ------------------------------------------------------------- #
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested value by dotted path, e.g. ``cfg.get('extraction.model')``.

        Returns ``default`` if any component of the path is missing. Intended for
        genuinely optional settings; required settings should be read directly
        (``cfg.extraction["model"]``) so that their absence raises.
        """
        node: Any = self.as_dict()
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the configuration, keyed by field name."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # -- provenance --------------------------------------------------------- #
    def fingerprint(self, length: int = 12) -> str:
        """Deterministic hash of the full configuration.

        Stamped onto every results file, figure, and log so an output can never
        be silently attributed to settings other than those that produced it.
        The hash is computed over a canonical JSON serialisation with sorted
        keys, so it is invariant to the order in which blocks appear in the YAML
        file but sensitive to any change of value.

        ``device`` is included deliberately: on this pipeline the device
        interacts with determinism settings, so a CPU run and a GPU run are not
        provenance-interchangeable.
        """
        if not 4 <= length <= 64:
            raise ValueError(f"length must be in [4, 64], got {length}")
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, default=str, ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# 4. Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True) -> None:
    """Seed every random number generator the project can reach.

    Seeds :mod:`random`, :mod:`numpy`, and — if installed — PyTorch (CPU and all
    CUDA devices). ``PYTHONHASHSEED`` is set for completeness, though note it
    only affects *child* processes: the hash seed of the running interpreter is
    fixed at start-up, which is why :func:`stable_doc_id` uses an explicit hash
    function rather than Python's built-in ``hash``.

    Parameters
    ----------
    seed
        Seed value. Must be a non-negative int below 2**32 to satisfy NumPy's
        legacy seeding API.
    deterministic
        Request deterministic algorithms from PyTorch and cuDNN. Costs some
        throughput; non-negotiable for published numbers.
    warn_only
        Passed through to ``torch.use_deterministic_algorithms``. When ``True``,
        an operation lacking a deterministic implementation warns instead of
        raising. Set ``False`` for a strict pre-publication verification run.

    Notes
    -----
    ``CUBLAS_WORKSPACE_CONFIG`` must be set *before* the CUDA context is created
    for deterministic cuBLAS matrix multiplications. If CUDA is already
    initialised when this function runs, setting it has no effect, so an
    explicit warning is emitted rather than leaving a false impression of
    determinism.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {seed}")

    logger = logging.getLogger("utils.seed")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return  # data-only stage on a machine without the GPU stack

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            logger.warning(
                "CUDA context already initialised; CUBLAS_WORKSPACE_CONFIG set "
                "too late to guarantee deterministic cuBLAS. Export it in the "
                "environment before launching Python for a strict run."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "cuda") -> str:
    """Resolve a requested device string to one that is actually available.

    Falls back to ``"cpu"`` when CUDA (or PyTorch itself) is unavailable, and
    logs the downgrade: a silent fallback would let a run intended for four GPUs
    quietly take days on CPU.
    """
    logger = logging.getLogger("utils.device")
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed; falling back to CPU.")
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        return "cpu"
    return requested


# --------------------------------------------------------------------------- #
# 5. Logging
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def get_logger(
    name: str,
    logfile: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger, safe to call repeatedly.

    Handlers are added at most once per destination: a second call with the same
    name attaches no duplicate stream handler, and a call adding a new
    ``logfile`` attaches only that file handler. Without this guard, importing a
    module twice — which the numbered-module import shim in ``07_experiments.py``
    can cause — would double every log line.

    Logs are written to ``stderr`` so that a stage's diagnostic output never
    contaminates piped stdout. ``propagate`` is disabled so records are not
    re-emitted by an ancestor logger if a third-party library configures the
    root logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(stream)

    if logfile is not None:
        target = Path(logfile).resolve()
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == target
            for h in logger.handlers
        )
        if not already:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
            )
            logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager logging the wall-clock duration of a pipeline stage.

    Stage timings support the manuscript's computational-cost note, and an
    unexpected duration is often the first sign that a stage silently processed
    the wrong number of records. The end message is emitted even if the body
    raises, so a failure's timing is still recorded.
    """
    log = logger or get_logger("utils.timing")
    start = time.perf_counter()
    log.info("START %s", label)
    try:
        yield
    finally:
        log.info("END   %s (%.2f s)", label, time.perf_counter() - start)


# --------------------------------------------------------------------------- #
# 6. Atomic I/O
# --------------------------------------------------------------------------- #

def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise ``obj`` to JSON atomically.

    The payload is written to a temporary file in the destination directory and
    then moved into place with :func:`os.replace`, which is atomic for
    same-filesystem renames. A run interrupted mid-write therefore leaves the
    previous file intact instead of a truncated file that a later stage would
    read as valid.

    Keys are sorted, so two runs producing equal content produce byte-identical
    files and ``diff`` becomes a meaningful check between runs.

    Returns
    -------
    Path
        The path written, for convenient logging at the call site.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=indent, sort_keys=True, default=str,
                      ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())   # durability before the rename
        os.replace(tmp, path)
    finally:
        # No debris if serialisation raised; a no-op after a successful replace.
        tmp.unlink(missing_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file, naming the offending path on a parse failure."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg} (while reading {path})", exc.doc, exc.pos
        ) from None


# --------------------------------------------------------------------------- #
# 7. Content hashing and identifiers
# --------------------------------------------------------------------------- #

def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of an in-memory payload, as lowercase hexadecimal."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file on disk.

    Read in chunks so multi-gigabyte narrative archives are hashed without being
    held in memory. Recorded in the acquisition manifest, this pins the exact
    vintage of each public source: upstream files are refreshed on a rolling
    schedule, so a URL alone does not identify what was analysed.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _normalise_key_part(value: Any) -> str:
    """Canonicalise one component of a composite identifier.

    Trims surrounding whitespace, collapses internal runs of whitespace, and
    upper-cases, so that ``" 015009 "`` and ``"015009"`` — both of which occur
    across the public files — yield the same identifier. Without this, the same
    deficiency parsed from two sources would receive two ids and be
    double-counted.
    """
    return " ".join(str(value).split()).upper()


def stable_doc_id(
    ccn: str, survey_date: str, tag: str, offset: int, length: int = 16
) -> str:
    """Deterministic identifier for one cited deficiency.

    Built from public keys only — facility CCN, survey date, F-tag, and the
    citation's ordinal position within its report — so the same deficiency
    receives the same id across rebuilds, machines, and Python versions. That
    stability is a precondition for the public deposit to be citable and for
    human annotations to remain joinable to model output.

    BLAKE2b is used rather than Python's built-in ``hash`` (randomised per
    process) or SHA-1 (needlessly flagged by security scanners). The choice is
    for stable, collision-resistant identifiers, not for any security property.

    Parameters
    ----------
    offset
        Ordinal position of the citation within its report. Required because a
        single survey can cite the same F-tag more than once, so the other three
        components are not jointly unique.
    length
        Number of hexadecimal characters returned. Must be even and in [8, 64];
        the default of 16 gives a 64-bit identifier, ample for a corpus of this
        size.
    """
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"offset must be int, got {type(offset).__name__}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if not 8 <= length <= 64:
        raise ValueError(f"length must be in [8, 64], got {length}")
    if length % 2:
        raise ValueError(f"length must be even, got {length}")

    parts = [_normalise_key_part(part) for part in (ccn, survey_date, tag)]
    parts.append(str(offset))
    # "|" does not occur in CCNs, dates, or F-tags after normalisation, so the
    # joined key is unambiguous.
    key = "|".join(parts).encode("utf-8")
    return hashlib.blake2b(key, digest_size=length // 2).hexdigest()


# --------------------------------------------------------------------------- #
# 8. Run provenance
# --------------------------------------------------------------------------- #

def _git_revision() -> dict[str, Any]:
    """Return the current git commit and working-tree cleanliness, if available.

    A dirty working tree is recorded explicitly: results produced from
    uncommitted code are not reproducible from the published revision, and the
    manuscript's reproducibility statement should not imply otherwise. All
    fields are ``None`` when the project is not a git checkout or git is absent.
    """
    def _run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True,
                timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    """Installed version of each named distribution, or ``None`` if absent."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_info() -> list[dict[str, Any]] | None:
    """Describe visible CUDA devices, or ``None`` if there are none."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "capability": ".".join(
                str(component) for component in torch.cuda.get_device_capability(index)
            ),
        }
        for index in range(torch.cuda.device_count())
    ]


_PROVENANCE_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "PyYAML", "torch", "vllm", "transformers",
    "scikit-learn", "statsmodels", "pyarrow", "krippendorff",
)

_PROVENANCE_ENV_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
    "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
)


def provenance(cfg: Config | None = None) -> dict[str, Any]:
    """Capture everything needed to attribute a result to its origin.

    Written alongside each results file. Together with the acquisition manifest
    (source URLs and SHA-256 checksums) and the configuration fingerprint, this
    record lets a third party establish whether a rerun that disagrees with the
    published numbers differs in data, in code, or in environment — the three
    explanations that must be distinguished before a discrepancy can be
    interpreted.
    """
    record: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "hostname": platform.node(),
        "git": _git_revision(),
        "packages": _package_versions(_PROVENANCE_PACKAGES),
        "gpus": _gpu_info(),
        "env": {key: os.environ.get(key) for key in _PROVENANCE_ENV_KEYS},
    }
    if cfg is not None:
        record["config_fingerprint"] = cfg.fingerprint()
        record["seed"] = cfg.seed
    return record
