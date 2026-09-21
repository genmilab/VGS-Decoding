---
title: VGS-Decoding
emoji: 🩻
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: Explore visually grounded medical VLM decoding
models:
  - google/medgemma-4b-it
  - chaoyinshe/llava-med-v1.5-mistral-7b-hf
datasets:
  - flaviagiammarino/vqa-rad
---

# VGS-Decoding research demo

Code for [VGS-Decoding: Visual Grounding Score Guided Decoding for Hallucination
Mitigation in Medical VLMs](https://arxiv.org/abs/2603.20314).

[GitHub](https://github.com/genmilab/VGS-Decoding) ·
[Project page](https://genmilab.github.io/VGS-Decoding/)

Load a public VQA-RAD example or upload a de-identified research image, select
MedGemma or LLaVA-Med, and inspect generated answers and selected-token scores.
An optional alpha=0 control uses the same decoder and prompt. No evaluation
metrics are calculated and no reference answers are passed to the model.

**Research only. Not medical advice, a diagnostic tool, or a clinically
validated system.** Outputs may be inaccurate. Do not upload identifiable
patient information. The host receives uploaded images/questions, and Gradio
temporarily caches image files; the cache is swept hourly for files older than
one hour. Do not interpret this as a clinical privacy/compliance guarantee.

## Hosting requirements

- A CUDA GPU; MedGemma also requires bfloat16 support. The underlying release
  was exercised on A100 40 GB hardware, not benchmarked on every Space tier.
- Configure `HF_TOKEN` as a **Space secret** belonging to an account that has
  accepted the MedGemma model terms. Never put a real token in repository files.
- The Docker image isolates the current Gradio UI from the pinned model stack.
  It does not include model weights or data; those download at runtime.
- Requests are serialized. One subprocess loads one model per request and
  exits afterward; downloads are cached, but each request reloads model weights.
- CPU can serve the interface/load dataset examples, but generation reports
  a GPU-required message. This Docker implementation does not support ZeroGPU.
- Selecting paid hardware is an explicit hosting decision, not automated here.

Model/data licenses and access terms remain applicable; the MIT code license
does not relicense them. See the original model and dataset cards.

Code contributors: [Adinath Madhavrao Dukre](https://github.com/adinathdukre)
and [Govinda Kolli](https://github.com/govindakolli), GenMI-Lab.
