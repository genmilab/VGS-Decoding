# Interactive research demo

This is an optional UI for the existing `vgs_decoding.py`, not a different VGS
implementation and not benchmark evaluation code. The three original inference
files remain unchanged. **The public HF namespace and GPU have not yet been
selected: this directory is not evidence of a deployed or GPU-tested Space.**

## What it does

- Choose MedGemma 4B or LLaVA-Med 7B, using the release's pinned checkpoints.
- Load a public VQA-RAD test example (index 0–450); only its image and question
  are used. Or upload a de-identified research image and enter a question.
- Adjust alpha, generation length and perturbation seed. Gaussian sigma=0.07,
  Poisson scale=70, floor=0.01 and one perturbed view match the released defaults.
- Inspect the answer, selected-token VGS/probability trace, and run settings.
- Optionally generate a greedy control by setting alpha=0 with the same
  decoder, precision and prompt. This retains both branches; it is **not a
  greedy speed benchmark**. No accuracy, recall, F1, or clinical score is computed.

The maximum output length is 128 tokens, questions are limited to 1,000
characters, uploads to 10 MB, and decoded images to 16 million pixels.
Requests share a one-job queue (at most eight waiting requests) and a process
lock. Each request has a 600-second timeout. One model is loaded per worker;
the process exits after the request. Weights are cached on disk but reloaded
into GPU memory each time, which trades response latency for simple isolation.

## Separate environments

The current UI uses Gradio 6.28.0, which requires a newer `huggingface-hub`
than the inference release. **Do not install it into the existing model
environments or loosen the decoder pins to make it fit.**

`Dockerfile` uses `/opt/ui` for Gradio and `/opt/inference` for the original
model stack. `app.py` passes bounded JSON via stdin to `worker.py`; the worker
imports the original core. There is no network-facing inference-worker port.

Build from the project root, not from the full research workspace:

```bash
docker build -f hf_demo/Dockerfile -t vgs-decoding-demo .
docker run --rm --gpus all -p 127.0.0.1:7860:7860 \
  -e HF_TOKEN vgs-decoding-demo
```

Set `HF_TOKEN` securely in your local environment first when MedGemma is needed.
Do not put the token literally in a command, Docker build argument, file in
this repository, or screenshot. Mount a private cache volume if downloads
should survive container removal. No model downloads or GPU inference happen
at Docker build time.

For non-Docker development, create a separate UI environment and install
`hf_demo/ui-requirements.txt`. Point `VGS_INFERENCE_PYTHON` at an interpreter
where the original project is installed, then run `hf_demo/app.py` using the
UI interpreter. Local binding defaults to `127.0.0.1`; HF Docker hosting sets
`GRADIO_SERVER_NAME=0.0.0.0`. No public Gradio sharing tunnel is enabled.

## Deploy to Hugging Face later

1. Confirm the exact owner/organization and obtain permission to create a Space.
2. Create a **Docker / blank** Space. Review the current account/plan and
   hardware requirements; do not assume a free CPU Space provides inference.
3. Upload the prepared Space bundle, or copy these exact files:
   - `hf_demo/SPACE.md` → root `README.md`
   - `hf_demo/Dockerfile` → root `Dockerfile`
   - root `.dockerignore`, `pyproject.toml`, `LICENSE`, `NOTICE.md`, and the
     three `vgs_*.py` files, preserving filenames
   - `hf_demo/app.py`, `hf_demo/worker.py`, `hf_demo/ui-requirements.txt`,
     and this guide, preserving their `hf_demo/` paths
4. Choose GPU hardware only after explicit budget/access approval. MedGemma
   requires bfloat16; the original release was run on A100 40 GB. Smaller
   tiers need a real memory/latency smoke test. **Docker Spaces do not support
   ZeroGPU**; this implementation does not claim ZeroGPU compatibility.
5. Accept `google/medgemma-4b-it` access terms with the hosting token's account.
   In **Space Settings → Variables and secrets**, add **secret** `HF_TOKEN`.
   A read token permitted to access that model is sufficient for inference;
   a separate write credential is needed to publish the Space repository.
6. Check build logs, launch the UI, load an example, and generate with each
   backbone on the actual allocated GPU. Check the alpha=0 equality control.
7. Only then replace the website Demo link with the verified Space URL and
   remove its hosting-pending note. The website lives under `docs/` on GitHub
   Pages; it does not run Python or host GPU inference itself.

Deployment references: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[GPU hardware](https://huggingface.co/docs/hub/spaces-gpus),
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).

## Privacy and scientific scope

Do not use identifiable patient data. A public demo is not a clinical/privacy
compliance environment. Gradio temporarily caches images, with an hourly sweep
of files older than one hour; deletion is not immediate. This app adds no
prediction log or analytics and does not forward credentials to the browser.
Model-library/network caching and the hosting provider's infrastructure still
apply. Exceptions are mapped to fixed messages instead of exposing server
paths, raw exception strings or potentially credential-bearing URLs.

The existing manuscript tables are reported results, not results measured by
this demo. Outputs can be unchanged or worse. Token sensitivity is not clinical
correctness or calibrated confidence. No claim is made that the demo's chosen
seed or arbitrary uploaded images reproduce a manuscript table.
