# Paper-to-project verification — 21 September 2026

This checks **transcription and implementation scope**, not the validity of
every underlying experiment. It does not claim a new full reproduction of the
paper's benchmark results or a clinical validation.

## Sources and version boundaries

- The project title, preprint date, DOI, and citation follow
  [arXiv:2603.20314v1](https://arxiv.org/abs/2603.20314v1), submitted 19 March
  2026. That version has five authors and studies two datasets.
- The website's three-dataset Table 1, expanded appendix, and cases come from
  the author-supplied `acl_latex_aug.tex`, whose title is *Token-Level Visual
  Dependency Probing and Decoding for Hallucination Mitigation in Medical
  Vision-Language Models*. The supplied TeX has placeholder author metadata;
  it is not the source for the website's affiliation list.
- The verified TeX snapshot has SHA-256
  `ad440b570aabdb4bddb80db714b10922afd682c8fbb11c3560062193b6bc602e`.
- The seven-author website list, equal-first-author marker, additional MBZUAI
  affiliations, and removal of Govinda's second affiliation are author-requested
  project updates, not facts independently established by the five-author
  arXiv record. The BibTeX retains the actual five-author record.
- The website abstract is a project summary in the authors' voice, not a verbatim
  paper abstract; this distinction is documented in `CONTENT_SOURCES.md`.
  The new interactive demo is a software addition, not a
  paper experiment or evidence for new performance claims.

## Verified against the supplied manuscript

| Item | Check |
| --- | --- |
| Main tables | All 135 open/closed/overall score cells match Table 1; all 45 method rows are retained. |
| Main-table differences | Calculated from the displayed overall values, in percentage points. |
| Appendix | All 17 appendix tables match after the explicit corrections below. |
| Qualitative cases | All six questions, reference answers and five-method response sets match, ignoring typesetting whitespace. |
| Original figures | All 12 supplied assets are byte-identical to the manuscript image directory. PNG renders of the PDFs and lab branding are additional assets. |
| VGS rule | Normalized clean/perturbed probability difference, epsilon=1e-8, multiplicative scaling, floor and renormalization agree with the supplied method/algorithm. |
| Released defaults | alpha=1, Gaussian sigma=0.07, Poisson scale=70, floor=0.01, one perturbed view, deterministic token argmax. |
| Release scope | LLaVA-Med and MedGemma generation on VQA-RAD only; no CheXagent runner or benchmark evaluation is implied. |

The supplied Table 1 has VGS above greedy in 9/9 model–dataset cells and best
overall among the listed methods in 8/9; DoLA is 0.26 points higher than VGS on
MedGemma–SLAKE. This distinction resolves the source abstract's ambiguous
“8 of 9” wording; it is not a new evaluation.

### Explicit corrections, not silent transcription changes

Computing a score without applying it cannot change greedy token selection.
Both component tables therefore label **VGS probe only (= Greedy)** and use:

| Backbone | Open recall | Closed accuracy | Overall | Delta vs greedy |
| --- | ---: | ---: | ---: | ---: |
| LLaVA-Med | 34.45 | 68.92 | 53.64 | 0.00 |
| MedGemma | 49.50 | 61.75 | 56.32 | 0.00 |

The correction is consistent with the saved-answer identity checks documented
in `CONTENT_SOURCES.md`. The original nonzero probe-only numbers and their
benefit interpretation are not retained as evidence. Separately, the region
comparator is explicitly labeled **ARCD-style port**, not native ARCD.

## Source-table issues requiring author reconciliation

Five LLaVA-Med rows do not reconcile with the manuscript's stated
question-count-weighted overall-score definition beyond ordinary two-decimal
rounding. The website preserves the submitted values; it does **not** silently
substitute the arithmetic estimates below.

| Dataset | Method | Manuscript overall | Weighted from displayed open/closed |
| --- | --- | ---: | ---: |
| VQA-RAD | VCD | 47.71 | 47.7410 |
| VQA-RAD | OPERA | 49.05 | 49.0647 |
| VQA-RAD | VGS | 57.75 | 57.8280 |
| MIMIC-Diff-VQA | VCD | 33.33 | 33.3656 |
| MIMIC-Diff-VQA | OPERA | 30.14 | 30.1554 |

The check uses 200/251 open/closed questions for VQA-RAD, 8,380/4,741 for
MIMIC-Diff-VQA, and 645/416 for SLAKE, as stated in the manuscript. Forty of
45 rows agree within 0.011 percentage points; the five above do not. Raw
records, denominators and scoring settings must determine which source
entries need correction. Accurate transcription is not proof of internal
numerical consistency or an independent reproduction.

## Interpretation and software boundaries

Open-answer recall is not a direct hallucination-rate metric; overall is a
mixed score rather than uniform exact-match accuracy. Selected examples and
token sensitivity are not evidence of clinical safety. These qualifications
remain visible in the project page and code documentation.

The three original inference modules are unchanged from the first public
release (`c51f473`). The demo calls that core in an isolated inference
environment. Its user-selected seed is an example-level perturbation seed;
the dataset CLI uses `base_seed + dataset_index`. The demo does not claim to
reproduce every historical table, and its alpha=0 control is not a timing
benchmark. Full hosted-demo inference still requires the selected GPU,
model access, and a deployment smoke test.
