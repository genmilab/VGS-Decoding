"""Gradio UI; inference runs in a separate, pinned Python environment."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
if not os.environ.get("GRADIO_TEMP_DIR"):
    # Avoid another user's /tmp/gradio on shared research servers.
    os.environ["GRADIO_TEMP_DIR"] = tempfile.mkdtemp(prefix="vgs-gradio-")
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "hf_demo" / "worker.py"
INFERENCE_PYTHON = os.environ.get("VGS_INFERENCE_PYTHON", "/opt/inference/bin/python")
REQUEST_LOCK = threading.Lock()
MAX_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 24 * 1024 * 1024
ERRORS = {
    "gpu_required": "Inference needs a configured CUDA GPU. The interface alone can run on CPU.",
    "bf16_required": "MedGemma needs a GPU with bfloat16 support, such as L4, A10G or A100.",
    "model_or_data_access": "Model/dataset access failed. The host must check connectivity and, for MedGemma, accept its terms and configure the HF_TOKEN Space secret.",
    "gpu_memory": "The GPU ran out of memory. Try fewer output tokens or ask the host for more GPU memory.",
    "invalid_input": "Check the image, question, and bounded generation settings.",
    "inference_failed": "Inference failed. No answer was produced; the host should check the deployment.",
}


def call_worker(payload: dict) -> dict:
    encoded = json.dumps(payload, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise gr.Error("Image is too large; use a smaller research image.")
    try:
        # A second guard complements Gradio's shared single-job queue.
        with REQUEST_LOCK:
            result = subprocess.run([INFERENCE_PYTHON, str(WORKER)], input=encoded,
                                    text=True, capture_output=True, cwd=ROOT, timeout=600,
                                    env={**os.environ, "PYTHONNOUSERSITE": "1"})
        if result.returncode != 0:
            raise gr.Error("The inference worker stopped without completing the request.")
        output = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise gr.Error("The request timed out. First-time model downloads can take several minutes; ask the host to check the model cache.") from None
    except (OSError, json.JSONDecodeError):
        raise gr.Error("The inference environment is unavailable. Ask the host to check VGS_INFERENCE_PYTHON and the Docker build.") from None
    if not output.get("ok"):
        raise gr.Error(ERRORS.get(output.get("code"), ERRORS["inference_failed"]))
    return output


def load_example(index):
    if isinstance(index, bool) or not isinstance(index, (int, float)) or not 0 <= index <= 450 or int(index) != index:
        raise gr.Error("Choose a VQA-RAD test index from 0 to 450.")
    result = call_worker({"action": "example", "index": int(index)})
    image = Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")
    note = f"Loaded VQA-RAD test example {result['index']}. Reference answers are not sent to the model."
    return image, result["question"], note


def generate(image, question, model, alpha, max_new_tokens, seed, compare, consent):
    if not consent:
        raise gr.Error("Confirm research-only use and that the image contains no identifiable patient information.")
    if image is None or not isinstance(image, Image.Image):
        raise gr.Error("Load a VQA-RAD example or upload a research image first.")
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise gr.Error("Enter a question of 1–1000 characters.")
    if image.width * image.height > MAX_PIXELS:
        raise gr.Error("Use an image with at most 16 million pixels.")
    # Strip metadata rather than forwarding the uploaded file or path.
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    result = call_worker({"action": "generate", "consent": True, "model": model,
                          "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
                          "question": question.strip(), "alpha": alpha, "seed": seed,
                          "max_new_tokens": max_new_tokens, "compare": compare})
    guided = result["guided"]
    trace = [[step["step"], step["token"], round(step["clean_probability"], 6),
              round(step["vgs"], 6), step["is_stop"]] for step in guided["trace"]]
    baseline = result["baseline"]["answer"] if result["baseline"] is not None else "Comparison not requested."
    metadata = {**result["metadata"], "vgs_stop_reason": guided["stop_reason"]}
    return guided["answer"], baseline, trace, metadata


def build_demo():
    with gr.Blocks(title="VGS-Decoding · Medical VLM Research Demo", delete_cache=(3600, 3600), analytics_enabled=False) as demo:
        gr.Markdown("# VGS-Decoding\n### Visual Grounding Score Guided Decoding for Hallucination Mitigation in Medical VLMs\n"
                    "**GenMI-Lab** · [Paper](https://arxiv.org/abs/2603.20314) · "
                    "[Project](https://genmilab.github.io/VGS-Decoding/) · "
                    "[Code](https://github.com/genmilab/VGS-Decoding)")
        gr.Markdown("Explore the released decoder with **MedGemma** or **LLaVA-Med**. "
                    "Load a public VQA-RAD test example or provide a de-identified research image. "
                    "The first request downloads the selected checkpoint; later requests reuse cached weights.")
        gr.Markdown("**Research demonstration—not medical advice or a clinically validated system.** "
                    "Answers may be wrong. Do not upload identifiable patient information or use outputs for diagnosis or treatment. "
                    "Uploads are processed on the hosting server and may remain in Gradio’s temporary cache for up to roughly two hours.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", sources=["upload"], label="Research image", format="png", buttons=["fullscreen"], height=310)
                with gr.Row():
                    index = gr.Number(value=0, minimum=0, maximum=450, precision=0, label="VQA-RAD test index")
                    example = gr.Button("Load VQA-RAD example")
                source = gr.Textbox(label="Example source", interactive=False)
                question = gr.Textbox(label="Question", placeholder="Ask a concise question about the image.", lines=2, max_length=1000)
            with gr.Column():
                model = gr.Dropdown(choices=[("MedGemma · 4B", "medgemma"), ("LLaVA-Med · 7B", "llava-med")], value="medgemma", label="Model")
                alpha = gr.Slider(0, 2, value=1, step=0.1, label="VGS strength α", info="α = 0 leaves greedy token selection unchanged.")
                tokens = gr.Slider(1, 128, value=64, step=1, label="Maximum new tokens")
                seed = gr.Number(value=20260920, precision=0, minimum=0, maximum=2**32 - 1, label="Perturbation seed")
                compare = gr.Checkbox(value=True, label="Also generate the α = 0 greedy control")
                consent = gr.Checkbox(value=False, label="Research only; my image contains no identifiable patient information.")
                run = gr.Button("Generate with VGS", variant="primary")
        with gr.Row():
            answer = gr.Textbox(label="VGS answer", lines=5, interactive=False)
            greedy = gr.Textbox(label="Greedy control (α = 0)", lines=5, interactive=False)
        gr.Markdown("Identical or worse answers are possible: this demo does not score accuracy or prove hallucination reduction. "
                    "The greedy control uses the same two-branch implementation with α = 0, so it is not a speed benchmark.")
        with gr.Accordion("Inspect selected-token VGS and run settings", open=False):
            trace = gr.Dataframe(headers=["Step", "Selected token", "Clean probability", "VGS", "Stop token"],
                                 datatype=["number", "str", "number", "number", "bool"], interactive=False)
            metadata = gr.JSON(label="Run provenance (no credentials)")
        example.click(load_example, inputs=[index], outputs=[image, question, source], concurrency_id="inference", concurrency_limit=1, api_visibility="private")
        run.click(generate, inputs=[image, question, model, alpha, tokens, seed, compare, consent],
                  outputs=[answer, greedy, trace, metadata], concurrency_id="inference", concurrency_limit=1, api_visibility="private")
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_demo().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                        server_port=int(os.environ.get("PORT", "7860")), share=False,
                        max_file_size="10mb", show_error=False, enable_monitoring=False,
                        footer_links=[], run_history=False,
                        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"))
