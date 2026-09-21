# Website content and version notes

## Title, authors, and affiliations

The project-page title is **VGS-Decoding: Visual Grounding Score Guided Decoding
for Hallucination Mitigation in Medical VLMs**, matching the linked
[arXiv preprint](https://arxiv.org/abs/2603.20314).

At the project author's request, the website retains seven authors in its
existing order and marks Govinda Kolli and Adinath Madhavrao Dukre as equal first
authors. This requested credit is not presented as a transcription of the
arXiv v1 contribution statement.

Institution names for the five arXiv authors were checked against the first
page of [arXiv:2603.20314v1](https://arxiv.org/pdf/2603.20314v1). The website
uses the following updated affiliation list: the project author requested
MBZUAI only for Govinda Kolli and confirmed MBZUAI for Yifan Lu and Ziyun Zou.
The remaining affiliation numbers were renumbered consistently.

| Author                  | Affiliation markers |
| ----------------------- | ------------------- |
| Govinda Kolli           | 1                   |
| Adinath Madhavrao Dukre | 1                   |
| Yifan Lu                | 1                   |
| Ziyun Zou               | 1                   |
| Behzad Bozorgtabar      | 2                   |
| Dwarikanath Mahapatra   | 3                   |
| Imran Razzak            | 1                   |

1. MBZUAI, Abu Dhabi, UAE.
2. Aarhus University, Aarhus, Denmark.
3. Khalifa University, Abu Dhabi, UAE.

The BibTeX block follows the five-author arXiv record rather than changing that bibliographic
record to match the seven-author project-page update.

## Abstract and paper versions

The website abstract is an adapted project summary, not a verbatim quotation of
either abstract. The paper buttons link to arXiv; the results and appendix
material come from the author-supplied extended `acl_latex_aug.tex` manuscript.
The arXiv v1 study covers VQA-RAD and MIMIC-Diff-VQA. The supplied submission adds
SLAKE and the expanded appendix. The authors are preparing an updated arXiv
version. The page presents their expanded work in the authors' voice; repeated
source-attribution captions are omitted. Until the update is published, its
citation retains the available arXiv record. The version distinctions,
detailed provenance and verification remain in these documentation files
rather than repeated beneath each figure or the citation.

## Tables

All 18 numbered table environments in the supplied manuscript are represented:
the main comparison is split into three dataset views, and 17 appendix tables
are expandable. Numbers are transcribed, not newly evaluated using the release.
Open recall, closed accuracy, and the weighted mixed overall score remain
distinct. Overall-score deltas in the main tables are calculated from the
displayed rounded values. Bold main-table values are determined numerically
within each model and metric rather than copied from inconsistent source bolding.
Main tables group methods under backbone headers, pin the method column during
horizontal scrolling, and label overall-score changes in percentage points.
Appendix comparisons use grouped model headers and expanded metric names.
These presentation changes do not alter any reported values or remove baselines.

Two explicit presentation corrections are made:

- Probe-only rows in both component ablations are replaced by greedy equality.
  Saved-output checks established identical answers for 451/451 examples per
  backbone. The original nonzero probe-only gains are not valid evidence of an
  effect from computing an unused score. Notes beside both tables identify the
  correction.
- The same-backbone region comparator is named **ARCD-style port**, not native
  ARCD. Its original mask column is retained as a manuscript-reported setting.

The recall-based results should not be interpreted as direct hallucination-rate
or clinical-safety measurements. The page does not turn the selected appendix
examples into an aggregate success-rate claim. Ablation and significance values
are identified as manuscript-reported, not independently revalidated here.

See [the paper-to-project verification](PAPER_VERIFICATION.md) for the checked
source snapshot, explicit corrections, and five unresolved source-table
overall-score inconsistencies. Transcription verification does not validate
the underlying experiments or resolve inconsistent source aggregates.

## Figures and appendix examples

All 12 files in the supplied manuscript image directory are retained unchanged:

- `VGSDecoding.pdf`: architecture figure; PNG rendering for the web.
- `Frame 2 (1).png`: introductory CT example.
- `vgs_bar.png`: token-category mean/standard-deviation figure.
- `results.pdf`: original results plot; PNG rendering for the web.
- `vgs_violin.png`: additional token-category distribution asset.
- `image.png`: additional mean-score rendering.
- `llava_case_013.jpg`, `llava_case_381.jpg`, `llava_case_545.jpg`,
  `llava_case_620.jpg`, `llava_case_849.jpg`, `llava_case_881.jpg`: six appendix
  case images, paired with their source questions, reference answers, and all
  five decoding responses.

The extra distribution/plot assets are identified as supplied figures rather
than assigned figure numbers that the extended TeX source does not give them.
Images and responses are research illustrations supplied by the authors, not
new clinical interpretations or a redistribution of the complete dataset.

The separate `genmilab-logo.png` branding asset is the public avatar of
[GenMI-Lab's GitHub organization](https://github.com/genmilab), reproduced
unchanged at the project author's request. The twelve manuscript assets remain
unchanged; the lab logo is not counted as a paper figure.

## Code and publishing

The three Python release files remain unchanged: LLaVA-Med/VQA-RAD and
MedGemma/VQA-RAD entry points plus shared VGS decoding. No evaluation scripts
are added. The optional `hf_demo/` interface invokes the same core in a separate
environment; it is a software addition rather than a manuscript experiment.
Repository and Pages links target `genmilab/VGS-Decoding`. The first public
release and Pages deployment were verified on 21 September 2026; the live
Hugging Face Space remains pending hosting/GPU confirmation.
