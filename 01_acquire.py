"""
01_acquire.py — Reproducible acquisition of every public data source.

Stage 1 of the pipeline. Downloads the structured CMS provider files and the
inspection-narrative archives, verifies each transfer, and writes a manifest
recording the source URL, retrieval time, byte count, and SHA-256 of every file.

WHY THIS STAGE IS NOT TRIVIAL
-----------------------------
The public sources are refreshed on a rolling schedule and the CMS file URLs
contain a content hash that changes at every refresh, so a hard-coded link stops
working within weeks. Worse, the failure is not always loud: a partial transfer,
a silently truncated API response, or a gzip-encoded body can all yield a file
that parses cleanly and is wrong. This module therefore treats acquisition as a
measurement problem — resolve from a stable identifier, verify what arrived
against what the server declared, and record enough provenance that a replicator
can prove they analysed the same bytes.

The manifest is what makes the manuscript's Data Statement verifiable. Citing a
URL alone does not identify what was analysed; the checksum does.

DELIBERATELY OUT OF SCOPE
-------------------------
This stage never parses archive internals. It downloads and verifies bytes.
Interpreting the narrative format is the responsibility of ``02_corpus.py``,
which means acquisition can be completed and audited before the corpus schema is
finalised.

Usage
-----
    python src/01_acquire.py                        # fetch everything
    python src/01_acquire.py --dry-run              # resolve URLs, download nothing
    python src/01_acquire.py --only ownership       # fetch a subset
    python src/01_acquire.py --skip-narratives      # structured sources only
    python src/01_acquire.py --force                # re-fetch even if unchanged

Outputs
-------
    data/raw/<name>.csv
    data/raw/narratives/<filename>
    data/raw/_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils import (DATA_RAW, Config, ensure_dirs, get_logger, load_json,
                   provenance, relpath_to_root, save_json, set_seed, timed)

LOG = get_logger("acquire")

MANIFEST_PATH = DATA_RAW / "_manifest.json"

# The Provider Data Catalog exposes a stable identifier per dataset; the file
# URL behind it rotates at every monthly refresh. Always resolve, never pin.
PDC_METASTORE = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{id}"
)

# Identify the client honestly to the public API, as good-citizen practice for
# an automated collector against a government endpoint.
USER_AGENT = "technological-illegibility-research/1.0 (academic; contact in paper)"

# Transient conditions worth retrying. 404 and 403 are not: they mean the
# resource is gone or forbidden, and retrying only delays a necessary error.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = (
    "data.cms.gov", "www.cms.gov", "data.census.gov", "api.census.gov",
    "projects.propublica.org", "assets.propublica.org",
)


class AcquisitionError(RuntimeError):
    """Raised when a source cannot be obtained or fails verification."""


@dataclass(frozen=True)
class Target:
    """One file to acquire.

    Attributes
    ----------
    name
        Local basename, also the manifest key.
    url
        Direct download URL, or ``None`` if it must be resolved from ``dataset_id``.
    dest
        Destination path on disk.
    dataset_id
        Provider Data Catalog identifier, when the URL is resolved rather than given.
    expected_sha256
        Optional pin. When supplied, a mismatch aborts the run: this is how a
        replicator reproduces the exact vintage analysed in the paper rather than
        whatever the source currently serves.
    """

    name: str
    url: str | None
    dest: Path
    dataset_id: str | None = None
    expected_sha256: str | None = None


# --------------------------------------------------------------------------- #
# URL safety
# --------------------------------------------------------------------------- #

def validate_url(url: str, allowed_hosts: Iterable[str]) -> str:
    """Reject any URL that is not HTTPS on an expected host.

    This is a research-integrity control, not merely a security one. The
    manifest asserts that the corpus came from named public sources; a typo or
    an edited configuration that silently pulled bytes from somewhere else would
    make that assertion false while leaving every checksum internally
    consistent. Failing here keeps the manifest honest.

    Subdomains of an allowed host are accepted; look-alike suffixes are not
    (``evil-data.cms.gov.example.com`` does not match ``data.cms.gov``).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise AcquisitionError(
            f"Refusing non-HTTPS URL (scheme={parsed.scheme!r}): {url}"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise AcquisitionError(f"URL has no host: {url}")
    for allowed in allowed_hosts:
        allowed = allowed.lower()
        if host == allowed or host.endswith("." + allowed):
            return url
    raise AcquisitionError(
        f"Host {host!r} is not in the allow-list {sorted(allowed_hosts)}. "
        f"Add it to data.allowed_hosts in the configuration if it is legitimate."
    )


# --------------------------------------------------------------------------- #
# HTTP with bounded retries
# --------------------------------------------------------------------------- #

def _sleep_for_retry(attempt: int, base: float, retry_after: str | None) -> float:
    """Return the backoff delay, honouring a server-supplied ``Retry-After``.

    Exponential backoff with jitter. The jitter draws from the seeded global
    RNG, so even the retry schedule is reproducible in a replay.
    """
    if retry_after:
        try:
            return max(0.0, float(int(retry_after)))
        except (TypeError, ValueError):
            pass  # HTTP-date form; fall through to exponential backoff
    return base * (2 ** attempt) * (1.0 + random.random() * 0.25)


def _open_with_retries(
    url: str, timeout: float, retries: int, backoff: float
) -> Any:
    """Open a URL, retrying only on genuinely transient failures.

    Returns the open response object; the caller is responsible for closing it.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            # Ask for an unencoded body. urllib does not transparently inflate
            # gzip, so without this a compressed response would be written to
            # disk verbatim while Content-Length described the compressed size —
            # a mismatch that looks like corruption, or worse, does not.
            "Accept-Encoding": "identity",
        },
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRYABLE_STATUS or attempt == retries:
                raise AcquisitionError(
                    f"HTTP {exc.code} for {url}: {exc.reason}"
                ) from exc
            delay = _sleep_for_retry(attempt, backoff, exc.headers.get("Retry-After"))
            LOG.warning("HTTP %d for %s; retry %d/%d in %.1fs",
                        exc.code, url, attempt + 1, retries, delay)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            last = exc
            if attempt == retries:
                raise AcquisitionError(f"Network failure for {url}: {exc}") from exc
            delay = _sleep_for_retry(attempt, backoff, None)
            LOG.warning("Network error for %s (%s); retry %d/%d in %.1fs",
                        url, exc, attempt + 1, retries, delay)
        time.sleep(delay)
    raise AcquisitionError(f"Exhausted retries for {url}: {last}")


# --------------------------------------------------------------------------- #
# Distribution resolution
# --------------------------------------------------------------------------- #

def _extract_download_url(distribution: dict[str, Any]) -> str | None:
    """Pull the download URL out of one distribution entry.

    The catalogue nests the URL under ``data`` when references are dereferenced
    and exposes it at the top level in some responses, so both shapes are
    handled. Guessing one and silently returning ``None`` for the other would
    make the dataset appear to have no CSV.
    """
    if not isinstance(distribution, dict):
        return None
    nested = distribution.get("data")
    if isinstance(nested, dict) and nested.get("downloadURL"):
        return str(nested["downloadURL"])
    if distribution.get("downloadURL"):
        return str(distribution["downloadURL"])
    return None


def resolve_distribution(
    dataset_id: str, cfg_net: dict[str, Any], allowed_hosts: Iterable[str],
    metastore_template: str | None = None,
) -> str:
    """Resolve a Provider Data Catalog identifier to its current CSV URL.

    Parameters
    ----------
    metastore_template
        Overrides the default catalogue endpoint. Resolved at call time rather
        than bound as a default argument, so the pipeline can be pointed at an
        archived mirror of the catalogue — which is what a replicator needs when
        the live catalogue has moved on from the vintage analysed in the paper.

    Raises
    ------
    AcquisitionError
        If the catalogue entry cannot be read, or exposes no CSV distribution.

    Notes
    -----
    There is deliberately **no fallback to the datastore query endpoint**. That
    endpoint caps results per request, so using it as a silent fallback would
    return a truncated file that parses perfectly and understates every count in
    the paper. An unresolvable dataset must fail loudly instead.
    """
    template = metastore_template or PDC_METASTORE
    url = validate_url(template.format(id=dataset_id), allowed_hosts)
    with _open_with_retries(
        url, cfg_net["timeout"], cfg_net["retries"], cfg_net["backoff"]
    ) as response:
        try:
            meta = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcquisitionError(
                f"Catalogue entry for {dataset_id} was not valid JSON: {exc}"
            ) from None

    distributions = meta.get("distribution") or []
    if not isinstance(distributions, list):
        raise AcquisitionError(
            f"Catalogue entry for {dataset_id} has a malformed distribution block."
        )

    candidates = [u for u in map(_extract_download_url, distributions) if u]
    csv_urls = [u for u in candidates if u.lower().split("?")[0].endswith(".csv")]

    if csv_urls:
        chosen = csv_urls[0]
    elif candidates:
        chosen = candidates[0]
        LOG.warning("No .csv distribution for %s; using first available: %s",
                    dataset_id, chosen)
    else:
        raise AcquisitionError(
            f"No downloadable distribution found for dataset {dataset_id}. "
            f"Verify the identifier at data.cms.gov/provider-data."
        )

    LOG.info("Resolved %s -> %s", dataset_id, chosen)
    return validate_url(chosen, allowed_hosts)


# --------------------------------------------------------------------------- #
# Download and verification
# --------------------------------------------------------------------------- #

def download_file(
    url: str, dest: Path, cfg_net: dict[str, Any],
    expected_sha256: str | None = None, name: str | None = None,
) -> dict[str, Any]:
    """Stream a URL to disk, verifying the transfer, and return a manifest record.

    Parameters
    ----------
    name
        Canonical manifest key. Defaults to the destination stem, but callers
        pass the target's name explicitly so that the manifest key and the
        lookup key used for change detection are guaranteed to agree. When they
        disagree, every run re-downloads every file — which for multi-gigabyte
        narrative archives is an expensive silent failure rather than a visible
        one.

    Three verification steps, each catching a failure mode that would otherwise
    pass silently downstream:

    1. **Declared length.** If the server states ``Content-Length`` and fewer
       bytes arrive, the transfer was truncated. A CSV parser would happily read
       the partial file and every count in the paper would be too low.
    2. **Expected checksum.** When the configuration pins a SHA-256, a mismatch
       means the upstream file changed; the run aborts rather than mixing
       vintages within one analysis.
    3. **Atomic placement.** Bytes land in a ``.part`` file and are moved into
       place only after both checks pass, so an interrupted run can never leave
       a half-written file that a later stage treats as complete.

    The checksum is computed during the same pass that writes the file, so
    multi-gigabyte archives are never re-read or held in memory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + f".{os.getpid()}.part")

    digest = hashlib.sha256()
    written = 0
    chunk_size = int(cfg_net["chunk_size"])
    started = time.perf_counter()

    try:
        with _open_with_retries(
            url, cfg_net["timeout"], cfg_net["retries"], cfg_net["backoff"]
        ) as response:
            headers = response.headers
            declared_raw = headers.get("Content-Length")
            with part.open("wb") as handle:
                while chunk := response.read(chunk_size):
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())

        if declared_raw is not None:
            try:
                declared = int(declared_raw)
            except ValueError:
                declared = None
            if declared is not None and declared != written:
                raise AcquisitionError(
                    f"Truncated transfer for {url}: server declared {declared} "
                    f"bytes, received {written}."
                )

        if written == 0:
            raise AcquisitionError(f"Empty response body for {url}.")

        sha256 = digest.hexdigest()
        # `is not None`, not truthiness: a malformed pin that evaluates falsy
        # would otherwise be silently unenforced, and a pin that is silently
        # unenforced is worse than no pin at all — the manifest would assert a
        # vintage guarantee that was never checked.
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise AcquisitionError(
                f"Checksum mismatch for {url}.\n"
                f"  expected {expected_sha256}\n  received {sha256}\n"
                f"The upstream file has changed since it was pinned. Update the "
                f"pin deliberately, or analyse the pinned vintage."
            )

        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    LOG.info("  %-28s %8.2f MB  %6.1fs  sha256=%s...",
             dest.name, written / 1e6, elapsed, sha256[:12])

    return {
        "name": name if name is not None else dest.stem,
        "path": relpath_to_root(dest),
        "source_url": url,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": written,
        "sha256": sha256,
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "content_type": headers.get("Content-Type"),
        "elapsed_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# Target assembly
# --------------------------------------------------------------------------- #

def _normalise_pin(value: Any, key: str) -> str:
    """Validate and canonicalise one SHA-256 pin from the configuration.

    Pins are hexadecimal, so a value such as sixty-four zeros is parsed by YAML
    as an integer rather than a string. Coercing and validating here — instead of
    trusting the parsed type — means a malformed pin fails at start-up with a
    clear message, rather than being quietly ignored at verification time.
    """
    text = str(value).strip().lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise AcquisitionError(
            f"Invalid SHA-256 pin for {key!r}: expected 64 hexadecimal characters, "
            f"got {value!r}. Quote the value in YAML if it is all digits."
        )
    return text


def build_targets(cfg: Config, skip_narratives: bool) -> list[Target]:
    """Assemble the list of files to acquire from the configuration.

    ``Target.name`` is the canonical identity of a file: it is the manifest key,
    the ``--only`` selector, and the lookup key for change detection. Deriving
    all three from one field is what keeps idempotency correct.
    """
    raw_pins: dict[str, Any] = cfg.get("data.sha256_pins", {}) or {}
    pins = {key: _normalise_pin(value, key) for key, value in raw_pins.items()}
    targets: list[Target] = []

    for dataset_id, name in (cfg.data.get("pdc_datasets") or {}).items():
        targets.append(Target(
            name=name, url=None, dest=DATA_RAW / f"{name}.csv",
            dataset_id=dataset_id, expected_sha256=pins.get(name),
        ))

    if not skip_narratives:
        for url in cfg.data.get("narrative_urls") or []:
            filename = Path(urllib.parse.urlparse(url).path).name
            if not filename:
                raise AcquisitionError(f"Cannot derive a filename from URL: {url}")
            targets.append(Target(
                name=filename, url=url,
                dest=DATA_RAW / "narratives" / filename,
                expected_sha256=pins.get(filename),
            ))
        if not cfg.data.get("narrative_urls"):
            LOG.warning(
                "No narrative_urls configured. The narrative corpus is this "
                "study's primary measurement substrate; set data.narrative_urls "
                "in the configuration before building the corpus."
            )

    return targets


def _load_previous_manifest() -> dict[str, dict[str, Any]]:
    """Index the previous manifest by name, for change detection."""
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        previous = load_json(MANIFEST_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Could not read previous manifest (%s); re-fetching all.", exc)
        return {}
    return {rec["name"]: rec for rec in previous.get("files", []) if "name" in rec}


def _is_current(target: Target, previous: dict[str, dict[str, Any]]) -> dict | None:
    """Return the previous record if the local file is present and unchanged.

    Re-verifies the checksum on disk rather than trusting the manifest: the point
    of the manifest is to detect drift, so it cannot be its own evidence.
    """
    record = previous.get(target.name)
    if not record or not target.dest.is_file():
        return None
    from utils import sha256_file  # local import keeps the module's top clean

    if sha256_file(target.dest) != record.get("sha256"):
        LOG.warning("Local %s differs from manifest; re-fetching.", target.dest.name)
        return None
    return record


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def acquire(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    """Run the acquisition stage and return the manifest."""
    ensure_dirs()
    net = {
        "timeout": cfg.get("data.timeout_seconds", 120),
        "retries": cfg.get("data.retries", 4),
        "backoff": cfg.get("data.backoff_seconds", 2.0),
        "chunk_size": cfg.get("data.chunk_size", 1 << 20),
    }
    allowed_hosts = cfg.get("data.allowed_hosts", list(DEFAULT_ALLOWED_HOSTS))
    metastore = cfg.get("data.metastore_template") or PDC_METASTORE

    targets = build_targets(cfg, args.skip_narratives)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        targets = [t for t in targets if t.name in wanted]
        if not targets:
            raise AcquisitionError(f"No targets match --only {args.only!r}.")

    previous = {} if args.force else _load_previous_manifest()
    records: list[dict[str, Any]] = []
    skipped = 0

    for target in targets:
        url = target.url or resolve_distribution(
            target.dataset_id, net, allowed_hosts, metastore
        )
        url = validate_url(url, allowed_hosts)

        if args.dry_run:
            LOG.info("DRY RUN  %-28s <- %s", target.dest.name, url)
            continue

        unchanged = _is_current(target, previous)
        if unchanged is not None and unchanged.get("source_url") == url:
            LOG.info("  %-28s unchanged; skipping (use --force to re-fetch).",
                     target.dest.name)
            records.append(unchanged)
            skipped += 1
            continue

        records.append(download_file(
            url, target.dest, net, target.expected_sha256, name=target.name
        ))

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_fingerprint": cfg.fingerprint(),
        "provenance": provenance(cfg),
        "files": sorted(records, key=lambda r: r["name"]),
    }

    if not args.dry_run:
        save_json(manifest, MANIFEST_PATH)
        total_mb = sum(r["bytes"] for r in records) / 1e6
        LOG.info("Manifest written: %d files, %.1f MB total (%d unchanged).",
                 len(records), total_mb, skipped)
    return manifest


def fetch_archived(cfg: Config, args) -> dict[str, Any]:
    """Download archived CMS monthly snapshots for the long deficiency series.

    ProPublica's narrative corpus is a rolling ~3-year window — roughly 12
    quarters — which cannot identify a three-parameter diffusion model
    (Mahajan, Muller & Bass: pre-inflection windows leave the saturation
    parameter unidentifiable). RQ4's long COUNT series therefore comes from
    the archived structured deficiency files at
    data.cms.gov/provider-data/archived-data/nursing-homes; narratives are
    used only to classify within the current window. Entries are configured
    as ``data.archived_snapshots: [{name, url}, ...]`` and pass through the
    same host allow-list, retry, checksum-manifest, and truncation machinery
    as every other download.
    """
    snapshots = cfg.get("data.archived_snapshots", []) or []
    if not snapshots:
        LOG.warning("no data.archived_snapshots configured; the long series "
                    "for RQ4 cannot be built from narratives alone")
        return {"downloaded": [], "skipped": [], "failed": []}
    allowed_hosts = cfg.get("data.allowed_hosts", list(DEFAULT_ALLOWED_HOSTS))
    destination_root = DATA_RAW / "archived"
    destination_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, list] = {"downloaded": [], "skipped": [], "failed": []}
    previous = _load_previous_manifest()
    for snap in snapshots:
        name, url = snap.get("name"), snap.get("url")
        if not name or not url:
            report["failed"].append({"entry": snap,
                                     "error": "name and url required"})
            continue
        destination = destination_root / str(name)
        try:
            url = validate_url(url, allowed_hosts)
            entry = previous.get(f"archived/{name}")
            if destination.is_file() and entry and not getattr(
                    args, "force", False):
                report["skipped"].append(name)
                continue
            record = download_file(
                url, destination, cfg.get("data.network", {}) or {},
                expected_sha256=snap.get("sha256"),
                name=f"archived/{name}",
            )
            report["downloaded"].append(record)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            report["failed"].append({"entry": name, "error": str(exc)})
    if report["failed"]:
        LOG.warning("archived snapshots with errors: %d",
                    len(report["failed"]))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and verify all public data sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="path to the YAML configuration")
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of target names")
    parser.add_argument("--skip-narratives", action="store_true",
                        help="fetch structured CMS files only")
    parser.add_argument("--archived", action="store_true",
                        help="also fetch archived monthly snapshots for the "
                             "long deficiency series (RQ4)")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even if the local copy is unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print URLs without downloading")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        from utils import ROOT
        config_path = ROOT / config_path

    cfg = Config.load(config_path)
    set_seed(cfg.seed)   # makes the retry jitter reproducible

    try:
        with timed("stage 01: acquire", LOG):
            acquire(cfg, args)
            if getattr(args, "archived", False):
                fetch_archived(cfg, args)
    except AcquisitionError as exc:
        LOG.error("Acquisition failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.error("Interrupted; no partial file was left in place.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
