# VGS-Decoding: Visual Grounding Score Guided Decoding for Hallucination Mitigation in Medical VLMs

A small, inference-only implementation of **Visual Grounding Score (VGS)
decoding** for two medical vision-language models. Given a VQA-RAD image and
question, it generates an answer by reweighting next-token probabilities using
the original image and a Gaussian-plus-Poisson perturbed copy.

Code accompanying [**VGS-Decoding: Visual Grounding Score Guided Decoding for
Hallucination Mitigation in Medical VLMs**](https://arxiv.org/abs/2603.20314).

Project-page authors: **Govinda Kolli\***, **Adinath Madhavrao Dukre\***,
Yifan Lu, Ziyun Zou, Dwarikanath Mahapatra, Behzad Bozorgtabar, and Imran Razzak. \* Equal first authors. See the website for the author-requested affiliation list and
[version notes](docs/CONTENT_SOURCES.md) for the linked preprint's author list.

The supplied extended manuscript studies LLaVA-Med, CheXagent, and MedGemma on
VQA-RAD, SLAKE, and MIMIC-Diff-VQA. This release provides **LLaVA-Med and MedGemma generation on
VQA-RAD**, with a shared VGS implementation.

## Project layout

```text
vgs_llavamed_vqarad.py  LLaVA-Med VQA-RAD entry point
vgs_medgemma_vqarad.py  MedGemma VQA-RAD entry point
vgs_decoding.py         Shared decoder, image perturbation, model/input loading
pyproject.toml         Pinned dependencies and installable commands
docs/decoding.md        Formula, prompts, cache handling, and configuration
docs/index.html         Paper project website with original manuscript figures
hf_demo/                Optional Hugging Face-ready interactive demo
```

The two entry points select their model and call the same core. In
`vgs_decoding.py`, clearly labeled sections cover configuration, seeded image
perturbation, dual-cache decoding, model adapters, VQA-RAD loading, and output
provenance. No benchmark scoring is performed.

## Supported checkpoints

| CLI name    | Hugging Face checkpoint                                                                                   | CUDA precision |
| ----------- | --------------------------------------------------------------------------------------------------------- | -------------- |
| `llava-med` | [chaoyinshe/llava-med-v1.5-mistral-7b-hf](https://huggingface.co/chaoyinshe/llava-med-v1.5-mistral-7b-hf) | float16        |
| `medgemma`  | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)                                     | bfloat16       |

The LLaVA-Med checkpoint is a community Hugging Face-format conversion; this
loader is not intended for an unconverted original LLaVA repository checkpoint.
Both default model revisions are pinned in `vgs_decoding.py`. MedGemma requires
accepting the model's access terms using your own Hugging Face account.

The dataset is [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad).
The default is its test split. Only image, question, and optional question ID
columns are retained; dataset answers are not passed to the model or saved.
Weights and data remain subject to their own licenses and access conditions.

## Installation: separate environments

Use Python 3.10. The dependency pins follow the existing CUDA implementation:
PyTorch 2.7.1, torchvision 0.22.1, and Transformers 4.53.0. CUDA inference is the
default. A GPU with sufficient memory is required; MedGemma additionally needs
bfloat16 support. The source implementation was run on A100 40 GB GPUs. That is
not a claim about minimum memory on other devices.

Run the following from the project root. The CUDA 12.8 wheels require a
compatible NVIDIA driver; consult [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/)
if your system needs a different wheel build.

```bash
# LLaVA-Med environment
python3.10 -m venv .venv/llava-med
.venv/llava-med/bin/python -m pip install --upgrade pip
.venv/llava-med/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/llava-med/bin/python -m pip install -e .

# MedGemma environment
python3.10 -m venv .venv/medgemma
.venv/medgemma/bin/python -m pip install --upgrade pip
.venv/medgemma/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/medgemma/bin/python -m pip install -e .
```

Authenticate interactively, after accepting MedGemma's terms:

```bash
.venv/medgemma/bin/huggingface-cli login
```

Alternatively, use this environment-variable placeholder for MedGemma:

```bash
# Replace HF_Token with your own token locally; never commit the real value.
export HF_TOKEN="HF_Token"
```

`HF_Token` is a placeholder, not a working credential. If using cached login,
leave `HF_TOKEN` unset instead of exporting the placeholder.

Do not put tokens in source files, command arguments, commits, or screenshots.
Normal Hugging Face cached login or the `HF_TOKEN` environment variable is used;
there is deliberately no token command-line option. No custom model code is
downloaded for execution (`trust_remote_code=False`).

## Generate answers

Start with a small run. Every output directory must be **new**; existing runs
are never silently overwritten.

```bash
.venv/llava-med/bin/python vgs_llavamed_vqarad.py \
  --device cuda:0 --limit 3 \
  --output-dir outputs/llava-med-smoke

.venv/medgemma/bin/python vgs_medgemma_vqarad.py \
  --device cuda:0 --limit 3 \
  --output-dir outputs/medgemma-smoke
```

Remove `--limit` to generate answers for the complete test split:

```bash
.venv/llava-med/bin/python vgs_llavamed_vqarad.py \
  --output-dir outputs/llava-med-test

.venv/medgemma/bin/python vgs_medgemma_vqarad.py \
  --output-dir outputs/medgemma-test
```

Each command loads **one model on one device**, processing one question at a
time. On a machine with two allocated GPUs, run the commands in separate shells
with `--device cuda:0` and `--device cuda:1`. On a cluster, run only within a GPU
allocation, not on the login node. Scheduler-specific scripts are not bundled.

### Existing local models/data

```bash
HF_HUB_OFFLINE=1 .venv/medgemma/bin/python vgs_medgemma_vqarad.py \
  --model-path /path/to/medgemma-hf-snapshot \
  --dataset-arrow /path/to/vqa-rad-test.arrow \
  --local-files-only --limit 3 \
  --output-dir outputs/medgemma-offline
```

The same options work for LLaVA-Med with a compatible local snapshot. An Arrow
file is treated as the requested split; select the correct file yourself.
For remote datasets, `--dataset-revision COMMIT` can pin an exact revision.

## Configuration

| Option               |       Default | Meaning                                                   |
| -------------------- | ------------: | --------------------------------------------------------- |
| `--alpha`            |           1.0 | Strength of token reweighting                             |
| `--min-factor`       |          0.01 | Positive lower bound on the multiplicative factor         |
| `--gaussian-std`     |          0.07 | Gaussian standard deviation on RGB values in [0, 1]       |
| `--poisson-scale`    |          70.0 | Poisson rate scale                                        |
| `--max-new-tokens`   |            64 | Maximum generated tokens, including a possible stop token |
| `--seed`             |      20260920 | Base seed; example noise uses seed + dataset index        |
| `--start`            |             0 | First dataset index                                       |
| `--limit`            | all remaining | Number of questions to process                            |
| `--save-token-trace` |           off | Save chosen-token probabilities and VGS values            |

Run `python vgs_llavamed_vqarad.py --help` or
`python vgs_medgemma_vqarad.py --help` inside its environment for all options.
`--device cpu` is explicit opt-in and uses float32; full-model CPU generation
can be very slow and memory-intensive. It is not the default fallback.

## Interactive demo

An optional [Hugging Face-ready demo](hf_demo/README.md) wraps the same shared
decoder for MedGemma and LLaVA-Med. It supports public VQA-RAD test examples,
de-identified research-image uploads, adjustable guidance, an alpha=0 greedy
control, and selected-token traces. It does not add benchmark evaluation code.

The demo's Gradio interface and pinned inference dependencies run in separate
environments. **Hosting/GPU selection is pending; no live Space is claimed.**
The three command-line inference files and their default dependencies are
unchanged. The website's Demo section links to setup instructions until a
live Space is confirmed.

## Output and failure handling

Each run creates:

- `predictions.jsonl`: question ID/index, question, generated answer, model,
  seed, generated-token count, and stop reason. Optional traces include stop
  tokens, although the returned answer excludes them.
- `run.json`: settings, checkpoint revision or local source, dataset fingerprint,
  software versions, device, completed count, and completion/failure status.

No accuracy, recall, F1, hallucination metrics, or significance tests are
computed. `outputs/` is ignored by Git.

The run stops with a nonzero exit code on an error. Already generated lines
are flushed and preserved. A per-example error is recorded as `status: error`,
not as a fabricated answer. After resolving a failure, use a **new** output
directory and `--start` at the failed/next unfinished index. Keep the same seed
and settings. Hard termination (for example, scheduler SIGKILL) can prevent
`run.json` from being written; do not assume such a partial run completed.

## Implementation verification

The decoder has passed synthetic implementation checks for cache separation,
shared answer prefixes, stopping, deterministic perturbations, and output
handling. It was also smoke-tested on an NVIDIA A100 with three VQA-RAD
questions per model; all six answers matched the existing corrected decoder.
These are implementation checks, not full benchmark or clinical validation.
See [the decoding notes](docs/decoding.md) for details and [NOTICE](NOTICE.md)
for provenance. This is research software, not a clinical decision tool.

## Code contributors

- [Govinda Kolli (@govindakolli)](https://github.com/govindakolli)
- [Adinath Madhavrao Dukre (@adinathdukre)](https://github.com/adinathdukre)

Repository: [genmilab/VGS-Decoding](https://github.com/genmilab/VGS-Decoding).
The manuscript author list appears on the project page and in its citation.

See [publishing and contributor setup](docs/GITHUB_SETUP.md) for creating the
lab-owned repository and letting both contributors push with their own accounts.

## Project website

The static research-project page is in `docs/`. It presents the abstract,
method, main results, all supplied figure assets, 17 appendix tables, and six
qualitative case studies, followed by the released VQA-RAD code. All figures and
case responses come from the supplied manuscript materials. See
[website instructions](docs/WEBSITE.md) for preview and GitHub Pages deployment,
and [content sources](docs/CONTENT_SOURCES.md) for version and metric definitions.
The [paper-to-project verification](docs/PAPER_VERIFICATION.md) distinguishes
checked transcriptions from author-requested updates, explicit corrections,
and unresolved source-table consistency issues.
