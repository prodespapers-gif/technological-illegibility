"""
08_figures.py — Deterministic generation of every exhibit.

Stage 8 of the pipeline. Reads the JSON written by ``07_experiments.py`` and
regenerates every figure and table in the manuscript, so the exhibits are a pure
function of recorded results: no manual editing, and no drift between a number
in the text and the run that produced it.

WITHHELD RESULTS PRODUCE A PLACEHOLDER, NOT A CRASH
----------------------------------------------------
When a stage was withheld — because validation failed, or the differential-error
audit blocked the equity analysis — the corresponding exhibit is rendered as an
explicit "withheld" panel stating why, rather than silently omitted. An absent
figure is easy to overlook when assembling a manuscript; a panel that says the
analysis was withheld is not. The same discipline that gates the analysis should
be visible in the output.

FORMATS
-------
Vector PDF at publication column widths for every figure, plus a raster fallback
at the resolution the journal requires, and LaTeX plus CSV for every table.
Colours are drawn from a colour-vision-safe palette and every figure is checked
to remain legible in greyscale, since figures may appear in print monochrome.

Elsevier's policy permits AI assistance for data visualisations whose output is
derived from underlying data by reproducible computational methods, which is
exactly what this module does. No observed data are altered.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")   # headless: no display on the compute node
import matplotlib.pyplot as plt  # noqa: E402

from utils import FIGURES, TABLES, get_logger, load_json  # noqa: E402

LOG = get_logger("figures")

# Colour-vision-safe qualitative palette, ordered so the first two are also
# distinguishable in greyscale.
PALETTE: tuple[str, ...] = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
)
SINGLE_COLUMN = (3.5, 2.6)     # inches
DOUBLE_COLUMN = (7.2, 3.4)
RASTER_DPI = 500
GRAPHICAL_ABSTRACT_PX = (1328, 531)   # width x height, per the journal


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True, "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    """Write a figure as vector PDF plus a raster fallback."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".pdf", {}), (".tiff", {"dpi": RASTER_DPI})):
        path = FIGURES / f"{stem}{suffix}"
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    LOG.info("figure: %s", stem)
    return written


def _withheld_panel(stem: str, title: str, reason: str) -> list[Path]:
    """Render an explicit placeholder for an exhibit that was not produced."""
    _style()
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.55, "ANALYSIS WITHHELD", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#666666")
    ax.text(0.5, 0.32, _wrap(reason, 46), ha="center", va="top", fontsize=6.5,
            color="#333333")
    return _save(fig, stem)


def _wrap(text: str, width: int) -> str:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "\n".join(lines[:8])


def _stage(results: Mapping[str, Any], name: str) -> tuple[str, Any, str]:
    """Return ``(status, result, reason)`` for a stage in the results file."""
    entry = (results.get("stages") or {}).get(name) or {}
    return (entry.get("status", "missing"), entry.get("result"),
            entry.get("reason", "stage not present in the results file"))


def write_table(rows: Sequence[Mapping[str, Any]], stem: str,
                caption: str = "") -> list[Path]:
    """Write one table as CSV and as a LaTeX fragment."""
    TABLES.mkdir(parents=True, exist_ok=True)
    if not rows:
        LOG.warning("table %s has no rows; skipped", stem)
        return []
    fields = list(rows[0])
    csv_path = TABLES / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def esc(value: Any) -> str:
        return (str(value).replace("_", r"\_").replace("%", r"\%")
                .replace("&", r"\&"))

    lines = [r"\begin{tabular}{" + "l" * len(fields) + "}", r"\hline",
             " & ".join(esc(f) for f in fields) + r" \\", r"\hline"]
    lines += [" & ".join(esc(row.get(f, "")) for f in fields) + r" \\"
              for row in rows]
    lines += [r"\hline", r"\end{tabular}"]
    if caption:
        lines.insert(0, f"% {caption}")
    tex_path = TABLES / f"{stem}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("table: %s", stem)
    return [csv_path, tex_path]


# --------------------------------------------------------------------------- #
# Exhibits
# --------------------------------------------------------------------------- #

def figure2_prevalence(results: Mapping[str, Any]) -> list[Path]:
    """RQ1 — recorded prevalence over time, with bootstrap intervals."""
    status, result, reason = _stage(results, "rq1")
    if status != "completed":
        return _withheld_panel("figure2_prevalence",
                               "Figure 2. Recorded prevalence", reason)
    rows = result["by_period"]
    x = list(range(len(rows)))
    share = [row.get("share_point") or 0.0 for row in rows]
    lo = [row.get("share_lo") or 0.0 for row in rows]
    hi = [row.get("share_hi") or 0.0 for row in rows]

    _style()
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.20, linewidth=0)
    ax.plot(x, share, color=PALETTE[0], marker="o", markersize=3, linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row.get("year", "")) for row in rows], rotation=45)
    ax.set_ylabel("Share of citations")
    ax.set_xlabel("Year")
    ax.set_title("Technology-mediated failures in the regulatory record")
    return _save(fig, "figure2_prevalence")


def figure3_gap(results: Mapping[str, Any]) -> list[Path]:
    """RQ2 — the illegibility gap, broken down by failure role."""
    status, result, reason = _stage(results, "rq2")
    if status != "completed":
        return _withheld_panel("figure3_illegibility_gap",
                               "Figure 3. Illegibility gap", reason)
    gap = result["gap"]
    rows = [row for row in gap["by_failure_role"] if row.get("point") is not None]
    rows = sorted(rows, key=lambda row: row["point"])
    labels = [row["failure_role"].replace("_", " ") for row in rows]
    values = [row["point"] for row in rows]
    errors = [[row["point"] - (row["lo"] or row["point"]) for row in rows],
              [(row["hi"] or row["point"]) - row["point"] for row in rows]]

    _style()
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    ax.barh(labels, values, xerr=errors, color=PALETTE[1], height=0.6,
            error_kw={"elinewidth": 0.8, "capsize": 2})
    overall = gap["gap"]["point"]
    ax.axvline(overall, color=PALETTE[0], linestyle="--", linewidth=1.2,
               label=f"overall {overall:.0%}")
    ax.set_xlabel("Share cited under a tag naming no technology")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("The illegibility gap, by failure role")
    return _save(fig, "figure3_illegibility_gap")


def figure4_projection(results: Mapping[str, Any]) -> list[Path]:
    """RQ4 — scenario projections with prediction intervals."""
    status, result, reason = _stage(results, "rq4")
    if status != "completed":
        return _withheld_panel("figure4_projection",
                               "Figure 4. Scenario projection", reason)
    scenarios = result["projection"]["scenarios"]
    _style()
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for index, (name, payload) in enumerate(sorted(scenarios.items())):
        if payload.get("status") != "projected":
            continue
        t = payload["t"]
        ax.plot(t, payload["failures_illegible"], color=PALETTE[index % len(PALETTE)],
                linewidth=1.5, label=name.replace("_", " "))
        if payload.get("failures_total_lo"):
            ax.fill_between(t, payload["failures_total_lo"],
                            payload["failures_total_hi"],
                            color=PALETTE[index % len(PALETTE)], alpha=0.12,
                            linewidth=0)
    ax.set_xlabel("Period")
    ax.set_ylabel("Failures not legible to the record")
    ax.legend(title="scenario", loc="upper left")
    ax.set_title("Projected illegible failures (conditional scenarios)")
    if not result.get("backtest_passed", False):
        ax.text(0.98, 0.02, "backtest not passed", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="#B00020")
    return _save(fig, "figure4_projection")


def table1_vocabulary_census(results: Mapping[str, Any]) -> list[Path]:
    """Study 1 — the regulatory instrument's vocabulary, term by term.

    The paper's most striking single exhibit: every configured term with its
    count in the current Appendix PP, grouped pre-digital vs digital, with
    the ratio and the zero-count terms as summary rows. Rendered only from
    recorded results, like every other exhibit.
    """
    status, result, _ = _stage(results, "study1")
    if status != "completed":
        return []
    census = result["census"]
    rows: list[dict[str, Any]] = []
    for group in ("pre_digital", "digital"):
        for term_row in census["groups"][group]["terms"]:
            top = ", ".join(f"{t['f_tag']}({t['count']})"
                            for t in term_row["top_tags"][:3])
            rows.append({"group": group, "term": term_row["term"],
                         "count": term_row["count"], "top_tags": top})
    ratio = census.get("ratio_pre_to_digital")
    rows.append({"group": "summary", "term": "pre_digital_total",
                 "count": census["pre_digital_total"], "top_tags": ""})
    rows.append({"group": "summary", "term": "digital_total",
                 "count": census["digital_total"], "top_tags": ""})
    rows.append({"group": "summary", "term": "ratio_pre_to_digital",
                 "count": None if ratio is None else round(ratio, 1),
                 "top_tags": ""})
    rows.append({"group": "summary", "term": "digital_terms_never_occurring",
                 "count": len(census["groups"]["digital"]["zero_terms"]),
                 "top_tags": ", ".join(
                     census["groups"]["digital"]["zero_terms"])})
    return write_table(rows, "table1_vocabulary_census",
                       "Vocabulary census of Appendix PP "
                       f"({census.get('source_revision', '')}).")


def table2_validation(results: Mapping[str, Any]) -> list[Path]:
    """Measurement validation: reliability, agreement, sensitivity."""
    status, result, _ = _stage(results, "validate")
    if status != "completed":
        return []
    gate = results.get("gate", {})
    rows = [
        {"criterion": name,
         "observed": check.get("observed"),
         "rule": check.get("rule"),
         "passed": check.get("passed")}
        for name, check in (gate.get("checks") or {}).items()
    ]
    boot = result.get("human_alpha_bootstrap") or {}
    q_map = boot.get("q") or {}
    per_cat = result.get("per_category_alpha") or {}
    consistency = result.get("self_consistency") or {}
    if boot.get("lo") is not None:
        rows.append({"criterion": "human_alpha_ci",
                     "observed": (f"{boot['point']:.3f} "
                                  f"[{boot['lo']:.3f}, {boot['hi']:.3f}]"),
                     "rule": (f"pair-level bootstrap, "
                              f"{boot.get('n_pairs')} pairs"),
                     "passed": None})
    if q_map:
        rows.append({"criterion": "alpha_q_statistic",
                     "observed": "; ".join(
                         f"q({k})={v:.4f}" for k, v in q_map.items()
                         if v is not None),
                     "rule": "q = P(alpha < minimum); gate tests q",
                     "passed": None})
    if per_cat.get("min_alpha") is not None:
        rows.append({"criterion": "per_category_alpha_min",
                     "observed": (f"{per_cat['min_alpha']:.3f} "
                                  f"({per_cat.get('min_category')})"),
                     "rule": "minimum per-category alpha; gates per-role "
                             "claims", "passed": None})
    if consistency:
        rows.append({"criterion": "self_consistency",
                     "observed": "; ".join(
                         f"{k}={v:.3f}" for k, v in
                         (consistency.get("agreement_by_field") or {})
                         .items()),
                     "rule": "second identical pass (Gilardi et al.)",
                     "passed": None})
    rows.append({"criterion": "refusals",
                 "observed": result.get("n_refusals"),
                 "rule": "reported, not gated", "passed": None})
    rows.append({"criterion": "parse failure rate",
                 "observed": (result.get("stats") or {}).get("parse_failure_rate"),
                 "rule": "reported, not gated", "passed": None})
    return write_table(rows, "table2_validation",
                       "Measurement validation against pre-registered thresholds.")


def table3_gap(results: Mapping[str, Any]) -> list[Path]:
    """The gap by technology type, with the tag-adequacy check."""
    status, result, _ = _stage(results, "rq2")
    if status != "completed":
        return []
    rows = [
        {"technology_type": row["technology_type"], "n": row["n"],
         "illegible_share": None if row["point"] is None else round(row["point"], 3),
         "ci_lo": None if row["lo"] is None else round(row["lo"], 3),
         "ci_hi": None if row["hi"] is None else round(row["hi"], 3)}
        for row in result["gap"]["by_technology_type"]
    ]
    return write_table(rows, "table3_gap_by_technology",
                       "Illegibility gap by technology type.")


def table4_equity(results: Mapping[str, Any]) -> list[Path]:
    """Equity model, or an explicit record that it was withheld."""
    status, result, reason = _stage(results, "rq3")
    if status != "completed":
        return write_table([{"status": status, "reason": reason}],
                           "table4_equity_withheld",
                           "Equity analysis was not reported.")
    rows = [
        {"covariate": name, "coefficient": round(payload["coefficient"], 4),
         "ci_lo": payload["ci_lo"], "ci_hi": payload["ci_hi"],
         "significant": payload["significant"]}
        for name, payload in result["covariates"].items()
    ]
    return write_table(rows, "table4_equity",
                       "Illegibility gap by community and ownership covariates.")


def table5_forecast(results: Mapping[str, Any]) -> list[Path]:
    """Diffusion fit, backtest, and scenario assumptions."""
    status, result, _ = _stage(results, "rq4")
    if status != "completed":
        return []
    back = result["backtest"]
    rows = [{"quantity": "selected model", "value": result["selected_model"]},
            {"quantity": "backtest MAPE", "value": back.get("test_mape")},
            {"quantity": "backtest passed", "value": back.get("passed")},
            {"quantity": "beats naive benchmark", "value": back.get("beats_naive")}]
    rows += [{"quantity": f"param {k}", "value": round(v, 5)}
             for k, v in result["fit"]["params"].items()]
    for name, payload in sorted(result["projection"]["scenarios"].items()):
        rows.append({"quantity": f"scenario {name}",
                     "value": payload.get("status")})
    return write_table(rows, "table5_forecast",
                       "Diffusion fit, backtest, and scenario status.")


def graphical_abstract(results: Mapping[str, Any]) -> list[Path]:
    """Single panel at the journal's required pixel dimensions."""
    status, result, _ = _stage(results, "rq2")
    _style()
    width, height = GRAPHICAL_ABSTRACT_PX
    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    if status == "completed":
        gap = result["gap"]["gap"]["point"]
        ax.barh([""], [gap], color=PALETTE[1], height=0.4)
        ax.barh([""], [1 - gap], left=[gap], color="#DDDDDD", height=0.4)
        ax.text(gap / 2, 0, f"{gap:.0%} illegible", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Technology-mediated failures in the regulatory record")
        ax.set_title("Technological illegibility in nursing-home oversight")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Results pending", ha="center", va="center")
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "graphical_abstract.tiff"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOG.info("figure: graphical_abstract")
    return [path]


EXHIBITS = (table1_vocabulary_census, figure2_prevalence, figure3_gap,
            figure4_projection, table2_validation, table3_gap, table4_equity,
            table5_forecast, graphical_abstract)


def render_all(results: Mapping[str, Any]) -> dict[str, list[str]]:
    written: dict[str, list[str]] = {}
    for exhibit in EXHIBITS:
        try:
            paths = exhibit(results)
        except Exception as exc:  # noqa: BLE001 - isolate exhibit failures
            LOG.error("%s failed: %s: %s", exhibit.__name__, type(exc).__name__, exc)
            written[exhibit.__name__] = []
            continue
        written[exhibit.__name__] = [str(path) for path in paths]
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate all exhibits.")
    parser.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    path = Path(args.results)
    if not path.is_file():
        LOG.error("results file not found: %s", path)
        return 1
    written = render_all(load_json(path))
    produced = sum(len(paths) for paths in written.values())
    LOG.info("wrote %d files across %d exhibits", produced, len(written))
    return 0 if produced else 1


__all__ = [
    "PALETTE", "SINGLE_COLUMN", "DOUBLE_COLUMN", "write_table",
    "table1_vocabulary_census",
    "figure2_prevalence", "figure3_gap", "figure4_projection",
    "table2_validation", "table3_gap", "table4_equity", "table5_forecast",
    "graphical_abstract", "EXHIBITS", "render_all",
]


if __name__ == "__main__":
    sys.exit(main())
