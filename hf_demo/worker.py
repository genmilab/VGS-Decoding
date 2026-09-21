"""One isolated, bounded inference request using the unchanged release decoder.

Run with the pinned inference environment, never the Gradio environment.
JSON goes through stdin/stdout; model-library messages go to stderr. No input,
answer, token trace, or credential is intentionally written to a results file.
"""

from __future__ import annotations

import base64
import contextlib
from dataclasses import asdict, replace
import io
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image

MAX_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def integer(value, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected an integer")
    if not math.isfinite(value) or value != int(value) or not minimum <= value <= maximum:
        raise ValueError("Integer outside the supported range")
    return int(value)


def encode_image(image: Image.Image) -> str:
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image too large")
    # Create a new image to drop metadata, including EXIF.
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(encoded: str) -> Image.Image:
    if not isinstance(encoded, str) or len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Invalid image payload")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image too large")
        return image.convert("RGB")


def handle(request: dict) -> dict:
    from vgs_decoding import DecodingConfig, MODEL_SPECS, decode, load_model, load_vqa_rad, perturb_image
    import torch

    if not isinstance(request, dict):
        raise ValueError("Expected a request object")
    if request.get("action") == "example":
        index = integer(request.get("index"), 0, 450)
        dataset = load_vqa_rad(split="test")
        if index >= len(dataset):
            raise ValueError("Example index not present")
        sample = dataset[index]
        return {"ok": True, "image": encode_image(sample["image"]),
                "question": str(sample["question"]), "index": index,
                "dataset_fingerprint": dataset._fingerprint}
    if request.get("action") != "generate" or request.get("consent") is not True:
        raise ValueError("Research-use confirmation is required")
    name = request.get("model")
    if name not in MODEL_SPECS:
        raise ValueError("Unsupported model")
    question = request.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise ValueError("Question must contain 1–1000 characters")
    alpha = request.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 2:
        raise ValueError("Alpha must be in [0, 2]")
    if not isinstance(request.get("compare"), bool):
        raise ValueError("Invalid comparison option")
    config = DecodingConfig(alpha=float(alpha),
                            seed=integer(request.get("seed"), 0, 2**32 - 1),
                            max_new_tokens=integer(request.get("max_new_tokens"), 1, 128))
    image = decode_image(request.get("image"))
    if not torch.cuda.is_available():
        return {"ok": False, "code": "gpu_required"}
    device = torch.device("cuda:0")
    with torch.cuda.device(device):
        if name == "medgemma" and not torch.cuda.is_bf16_supported():
            return {"ok": False, "code": "bf16_required"}
    torch.manual_seed(config.seed)
    # Optional local snapshots are administrator-controlled, never UI inputs.
    local = os.environ.get("VGS_MEDGEMMA_PATH" if name == "medgemma" else "VGS_LLAVA_PATH")
    bundle = load_model(name, device, model_path=Path(local) if local else None)
    distorted = perturb_image(image, config, config.seed)
    stop_ids = bundle.stop_token_ids()

    def generate(current):
        result = decode(bundle.model, bundle.prepare(image, question.strip()),
                        bundle.prepare(distorted, question.strip()), current, stop_ids)
        answer = bundle.processor.tokenizer.decode(result.token_ids[0], skip_special_tokens=True).strip()
        trace = [{**asdict(step), "token": bundle.processor.tokenizer.decode([step.token_id])}
                 for step in result.steps]
        return {"answer": answer, "trace": trace, "stop_reason": result.stop_reason}

    guided = generate(config)
    # Same decoder/prompt/precision; alpha=0 leaves greedy token selection intact.
    # This runs two branches and is NOT a greedy speed benchmark.
    baseline = (guided if config.alpha == 0 else generate(replace(config, alpha=0))) if request["compare"] else None
    return {"ok": True, "guided": guided, "baseline": baseline,
            "metadata": {"model": name, "checkpoint": MODEL_SPECS[name].model_id,
                         "revision": "administrator-provided local snapshot" if local else MODEL_SPECS[name].revision,
                         "decoding": asdict(config), "device": torch.cuda.get_device_name(device),
                         "dtype": str(bundle.dtype), "comparison": "alpha=0 identity; not a timing benchmark",
                         "scope": "Single-example generation; no benchmark scoring or clinical validation."}}


def main() -> None:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Request too large")
        request = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            response = handle(request)
    except Exception as error:
        # Never expose exception strings: URLs/paths can contain private data.
        kind = type(error).__name__
        if kind in {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError", "OSError"}:
            code = "model_or_data_access"
        elif kind == "OutOfMemoryError":
            code = "gpu_memory"
        elif kind in {"ValueError", "TypeError", "UnidentifiedImageError", "DecompressionBombError", "Error"}:
            code = "invalid_input"
        else:
            code = "inference_failed"
        response = {"ok": False, "code": code}
    print(json.dumps(response, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
