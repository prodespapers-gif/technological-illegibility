# Technological Illegibility — Replication Package

**Paper:** *Technological illegibility: how pre-digital regulatory categories render technology-mediated care failures invisible.*

This repository contains the complete data pipeline, measurement instrument, statistical
analyses, and figure-generation code for the paper. Everything reported in the manuscript
is produced by the scripts below from public data.

> **Anonymized for double-blind review.** This repository contains no author names,
> institutional identifiers, acknowledgements, or funding information. Please do not
> attempt to deanonymize the authors. A permanent, de-anonymized archive with a DOI will
> replace this repository on acceptance.

---

## 1. What this repository does

The study measures *technological illegibility* — the systematic invisibility, within a
regulator's statistical output, of failures that frontline surveyors nonetheless recorded
as technology-mediated — across the complete U.S. certified nursing-home record. The
pipeline:

1. acquires the federal surveyor manual, the full census of deficiency citations, the
   underlying surveyor narratives, and county demographics;
2. builds a linked analysis corpus;
3. runs a transparent **lexical instrument** (primary) that detects care technologies and
   scores the naming capacity of the category each finding was filed under, plus a
   specified **LLM confirmatory layer** governed by a pre-registered validation gate;
4. estimates prevalence (RQ1), the illegibility gap (RQ2), its distribution across
   ownership and community disadvantage (RQ3), and its diffusion (RQ4); and
5. renders the seven publication figures and deposits the derived tables.

---

## 2. Repository structure

```
.
├── 01_acquire.py      # Download all public sources into data/raw/ (idempotent, checksummed)
├── 02_corpus.py       # Build the linked analysis corpus; attach facility + ACS covariates
├── 03_extract.py      # Study 1 vocabulary census + 206-tag taxonomy; Study 2 lexical detector
├── 04_validate.py     # LLM confirmatory layer, human gold standard, and the validation gate
├── 05_analysis.py     # RQ1 prevalence, RQ2 gap, RQ3 fixed-effects equity model, descriptives
├── 06_forecast.py     # RQ4 diffusion: logistic + Bass fits, rolling-origin backtest
├── 07_experiments.py  # Robustness and sensitivity checks; subgroup breakdowns
├── 08_figures.py      # The seven publication figures (vector PDF, embedded fonts)
├── 09_deposit.py      # Package derived tables + manifest for permanent archival
├── utils.py           # Shared I/O, seeding, lexicons, taxonomy loader, plotting style, stats
├── config.yaml        # Paths, random seed, model id, and gate thresholds
└── requirements.txt   # Pinned dependencies
```

Generated directories (not tracked): `data/raw/`, `data/interim/`, `data/derived/`,
`outputs/tables/`, `outputs/figures/`.

---

## 3. Data sources

All primary data are public. `01_acquire.py` fetches them; raw files are **not**
redistributed here, in keeping with each provider's terms. Acquisition is fully scripted
and checksum-verified, so the corpus is reconstructible byte-for-byte.

| Source (provider) | Scale | Vintage | Role |
|---|---|---|---|
| State Operations Manual, Appendix PP (CMS) | 347,189 words; 206 categories | Rev. 225 (2024) | Regulatory instrument; vocabulary census and taxonomy |
| Health Deficiencies (CMS) | 418,344 citations | 2017–2026 | Structured outcome corpus |
| Full Statement of Deficiencies narratives (CMS) | 418,090 findings | 2017–2026 | Free-text corpus for the instrument |
| Provider Information (CMS) | 14,693 facilities | 2026 | Ownership, beds, county, staffing |
| Ownership (CMS) | 14,629 facilities | 2026 | Ownership structure |
| ACS 5-year, tables B03002 & B19013 (U.S. Census Bureau) | 3,144 counties | 2019–2023 | Community % minority; median household income |
| SSA → FIPS county crosswalk (NBER) | county codes | 2026 | Link CMS county codes to Census FIPS |
| Nursing Home Inspect (ProPublica) | facility level | 2026 | External coverage cross-check |

Facilities link to community covariates for 99.0% of citations; the analytic facility
panel is **N = 14,515** with complete covariates.

---

## 4. Installation

Requires **Python ≥ 3.11**.

```bash
git clone <this-anonymized-repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `squarify`,
`pyarrow`, `openpyxl`, `requests`, `pyyaml`, `tqdm`. The confirmatory layer
(`04_validate.py`) additionally requires a local open-weight model runtime
(`vllm` or `transformers`) and is optional for reproducing the reported population results
(see §6).

---

## 5. Reproducing the paper

**One command** (runs the full pipeline in order):

```bash
python 01_acquire.py && python 02_corpus.py && python 03_extract.py \
  && python 05_analysis.py && python 06_forecast.py \
  && python 07_experiments.py && python 08_figures.py
```

**Stage by stage:**

```bash
python 01_acquire.py      # → data/raw/           fetch + verify checksums
python 02_corpus.py       # → data/interim/       linked corpus (parquet)
python 03_extract.py      # → data/derived/       taxonomy, vocab census, flagged findings
python 04_validate.py     # → outputs/tables/     confirmatory layer + gate report (optional)
python 05_analysis.py     # → outputs/tables/     T1–T8 and RQ1–RQ3 results
python 06_forecast.py     # → outputs/tables/     diffusion fits + backtest
python 07_experiments.py  # → outputs/tables/     robustness / sensitivity
python 08_figures.py      # → outputs/figures/    fig1–fig7 (PDF)
python 09_deposit.py      # → dist/               archival bundle + manifest
```

Paths, the global seed, the confirmatory model id, and all gate thresholds live in
`config.yaml`; no code edits are needed to point at a different data directory.

---

## 6. The measurement instrument

**Primary — lexical detector (`03_extract.py`).** Deterministic, dictionary-based,
whole-word and sense-disambiguated. For every narrative finding it records: (i) whether any
technology vocabulary is present; (ii) whether a *named care technology* is present,
resolved to one of eight families — call light / nurse-call, electronic health or medical
record, eMAR, bed/fall alarm or sensor, wander/elopement system, camera/video monitoring,
tele/remote monitoring, predictive/AI decision support; (iii) the technology's apparent
causal role (malfunction, present-but-not-responded-to, over-reliance/workaround,
incidental); and (iv) the naming level of the F-tag it was cited under, from the Study 1
taxonomy (`full` / `partial` / `none`). The population results in the paper are the lexical
estimates and reproduce end-to-end with no model weights.

**Confirmatory — LLM layer + validation gate (`04_validate.py`).** Re-reads a stratified,
severity-floored sample under a structured schema and is benchmarked against a two-coder
human gold standard. Confirmatory estimates are released **only if every gate passes**:

| Gate | Threshold |
|---|---|
| Human inter-coder reliability (Krippendorff α) | ≥ 0.80 |
| Human–model agreement (macro-F1; Cohen's κ reported) | ≥ 0.80 |
| Prompt-variant robustness (max label shift) | ≤ 0.05 |
| Cross-model stability (Spearman ρ, two open-weight models) | reported |
| Differential-error audit across ownership and community strata | required |

Running this layer requires local model weights and the annotated gold set; inference is
performed locally with no data egress and the model is pinned by version.

---

## 7. Outputs

**Tables** (`outputs/tables/`, CSV): data sources (T1), taxonomy (T2), vocabulary gap (T3),
corpus composition (T4), covariates (T5), gap by technology type and failure role (T6),
fixed-effects equity estimates (T7), diffusion comparison (T8).

**Figures** (`outputs/figures/`, vector PDF with embedded non–Type-3 fonts, Paul Tol
high-contrast palette, grayscale-safe):

| File | Figure |
|---|---|
| `fig1_framework.pdf` | The illegibility mechanism |
| `fig2_vocab_gap.pdf` | Vocabulary gap (regulation vs. practice) |
| `fig3_taxonomy.pdf` | Citation-weighted taxonomy treemap |
| `fig4_architecture.pdf` | Measurement architecture and validation gate |
| `fig5_flow.pdf` | Technology → naming-level flows |
| `fig6_severity_equity.pdf` | Severity gradient + equity coefficient forest |
| `fig7_diffusion.pdf` | Diffusion fit and rolling-origin backtest |

---

## 8. Headline results (for quick verification)

Running the pipeline reproduces, among others:

- **Vocabulary.** *Electronic medical record* appears **0** times in Appendix PP and
  **46,674** times in the narratives; the manual's pre-digital-to-digital term ratio is
  **≈ 34:1** versus **≈ 3.9:1** in practice.
- **Taxonomy.** Only **8 of 206** categories (3.9%) can name a technology in their binding
  text; they carry just **2.5%** of citations.
- **Illegibility gap (RQ2).** Among the **64,521** technology-referencing findings,
  **95.9%** are filed under categories that cannot name the technology (strict); 27.0%
  under categories with no technology vocabulary at all (lenient).
- **Distribution (RQ3).** Fixed-effects LPM (N = 413,878; 14,509 facility clusters;
  state + year FE): **no** ownership, % minority, or income gradient (all *p* > 0.1);
  actual-harm/immediate-jeopardy findings are **+26.1 pp** more likely to be illegible
  (*p* < 0.001). Illegibility is structural, and concentrated in severe harms
  (13.3% at no-harm → 51.6% at immediate-jeopardy level J).
- **Diffusion (RQ4).** Best fit is logistic; Bass collapses to it (innovation coefficient
  ≈ 0). Rolling-origin backtest MAPE = **188.5%**, which does not beat naive persistence,
  so a certified point projection is withheld.

Exact figures may shift trivially if a provider re-posts a source file; `01_acquire.py`
records the retrieval date and checksum of every input for provenance.

---

## 9. Reproducibility and ethics

- **Deterministic.** All sampling and inference are seeded from `config.yaml`; the corpus
  is rebuilt from checksummed inputs.
- **Local and pinned.** Any model inference runs locally with no data egress; the model is
  version-pinned to guard against silent drift.
- **Public data only.** No private or patient-identifiable data are used. Surveyor
  narratives are public CMS records; facilities are identified by public CCN.
- **Figures.** All figures are generated from data by reproducible code (no
  generative-AI imagery), in line with Elsevier's policy.

---

## 10. License

Code is released under the **MIT License** (`LICENSE`). Data remain under their original
providers' terms: CMS and U.S. Census Bureau products are U.S. Government public-domain
works; the NBER crosswalk and ProPublica dataset are used under their respective terms.
Redistribution of raw provider files is avoided; use `01_acquire.py` to obtain them.
