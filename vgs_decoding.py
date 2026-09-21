"""VGS generation for LLaVA-Med and MedGemma on VQA-RAD.

Shared implementation used by vgs_llavamed_vqarad.py and
vgs_medgemma_vqarad.py. Sections below cover configuration, perturbation,
dual-cache decoding, model adapters, input loading, and output provenance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from datasets import Dataset, DownloadConfig, load_dataset
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, LlavaForConditionalGeneration


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass(frozen=True)
class DecodingConfig:
    alpha: float = 1.0
    min_factor: float = 0.01
    gaussian_std: float = 0.07
    poisson_scale: float = 70.0
    max_new_tokens: int = 64
    seed: int = 20260920

    def __post_init__(self) -> None:
        for name in ("alpha", "min_factor", "gaussian_std", "poisson_scale"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if not 0 < self.min_factor <= 1:
            raise ValueError("min_factor must be in (0, 1]")
        if self.gaussian_std < 0 or self.poisson_scale <= 0:
            raise ValueError("gaussian_std must be >= 0 and poisson_scale > 0")
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")


# ==========================================================================
# Seeded image perturbation
# ==========================================================================

def perturb_image(image: Image.Image, config: DecodingConfig, seed: int) -> Image.Image:
    """Return a new RGB image; leave both the input and global RNG unchanged.

    The same distorted image is used for every token of an answer. The seed is
    local to this example, so splitting a dataset run does not change its noise.
    """
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pixels = np.clip(pixels + rng.normal(0, config.gaussian_std, pixels.shape), 0, 1)
    pixels = np.clip(rng.poisson(pixels * config.poisson_scale) / config.poisson_scale, 0, 1)
    return Image.fromarray(np.rint(pixels * 255).astype(np.uint8))


# ==========================================================================
# VGS rule and dual-cache decoding
# ==========================================================================

@dataclass(frozen=True)
class TokenStep:
    step: int
    token_id: int
    clean_probability: float
    vgs: float
    is_stop: bool


@dataclass(frozen=True)
class DecodeResult:
    token_ids: torch.Tensor
    steps: tuple[TokenStep, ...]
    stop_reason: str


def reweight_probabilities(
    clean_logits: torch.Tensor,
    distorted_logits: torch.Tensor,
    alpha: float,
    min_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute bounded VGS and normalized next-token probabilities in float32."""
    clean = torch.softmax(clean_logits.float(), dim=-1)
    distorted = torch.softmax(distorted_logits.float(), dim=-1)
    eps = 1e-8
    vgs = (clean - distorted) / (clean + distorted + eps)
    factors = torch.clamp(1 + alpha * vgs, min=min_factor)
    probabilities = clean * factors
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite VGS probabilities; check model precision and inputs")
    return probabilities, clean, vgs


def _initial_state(inputs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    kwargs = dict(inputs)
    ids = kwargs.pop("input_ids")
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("VGS decoding supports one nonempty prompt at a time")
    if kwargs.get("past_key_values") is not None:
        raise ValueError("Start each VGS branch with a fresh KV cache")
    kwargs.update(
        use_cache=True,
        cache_position=torch.arange(ids.shape[1], device=ids.device),
        logits_to_keep=1,
    )
    return ids, kwargs


def _forward_step(model: Any, ids: torch.Tensor, kwargs: dict) -> tuple[torch.Tensor, dict]:
    prepared = model.prepare_inputs_for_generation(ids, **kwargs)
    outputs = model(**prepared, return_dict=True)
    # Pinned Transformers version: this internal helper updates image-aware
    # generation state, including attention masks and cache positions.
    updated = model._update_model_kwargs_for_generation(outputs, kwargs, is_encoder_decoder=False)
    return outputs.logits[:, -1, :].float(), updated


@torch.inference_mode()
def decode(
    model: Any,
    clean_inputs: dict[str, Any],
    distorted_inputs: dict[str, Any],
    config: DecodingConfig,
    eos_ids: set[int],
) -> DecodeResult:
    """Generate a single answer using deterministic VGS-reweighted argmax.

    Stop tokens are recorded in the trace, but excluded from returned answer
    tokens. Setting alpha=0 is an identity check, not a separate evaluation mode.
    """
    clean_ids, clean_state = _initial_state(clean_inputs)
    distorted_ids, distorted_state = _initial_state(distorted_inputs)
    if not torch.equal(clean_ids, distorted_ids):
        raise ValueError("Clean and distorted images produced different prompt token IDs")
    response_start = clean_ids.shape[1]
    steps: list[TokenStep] = []
    stop_reason = "max_new_tokens"

    for step in range(config.max_new_tokens):
        clean_logits, clean_state = _forward_step(model, clean_ids, clean_state)
        distorted_logits, distorted_state = _forward_step(model, distorted_ids, distorted_state)
        final, clean, vgs = reweight_probabilities(
            clean_logits, distorted_logits, config.alpha, config.min_factor
        )
        next_token = torch.argmax(final, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        is_stop = token_id in eos_ids
        steps.append(TokenStep(
            step=step,
            token_id=token_id,
            clean_probability=float(clean.gather(1, next_token).item()),
            vgs=float(vgs.gather(1, next_token).item()),
            is_stop=is_stop,
        ))
        if is_stop:
            stop_reason = "eos"
            break
        # Never generate an independent text history for the distorted branch.
        clean_ids = torch.cat((clean_ids, next_token), dim=1)
        distorted_ids = torch.cat((distorted_ids, next_token), dim=1)

    return DecodeResult(clean_ids[:, response_start:], tuple(steps), stop_reason)


# ==========================================================================
# Model loading and prompt adapters
# ==========================================================================

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    dtype: torch.dtype


MODEL_SPECS = {
    "llava-med": ModelSpec(
        "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "627be53734c667cbb1669608dac747a4485a22d7",
        torch.float16,
    ),
    "medgemma": ModelSpec(
        "google/medgemma-4b-it",
        "290cda5eeccbee130f987c4ad74a59ae6f196408",
        torch.bfloat16,
    ),
}


@dataclass
class ModelBundle:
    name: str
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype

    def prepare(self, image: Image.Image, question: str) -> dict[str, Any]:
        """Use identical wording and processor settings for both image branches."""
        prompt = f"Answer this question as concisely as possible based on the provided image: {question}"
        if self.name == "llava-med":
            batch = self.processor(
                text=f"USER: <image>\n{prompt}\nASSISTANT:",
                images=image,
                return_tensors="pt",
            )
        else:
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": "You are a medical image analysis expert."}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ]},
            ]
            batch = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        moved = {}
        for key, value in dict(batch).items():
            if torch.is_tensor(value):
                value = value.to(
                    device=self.device,
                    dtype=self.dtype if value.is_floating_point() else value.dtype,
                )
            moved[key] = value
        return moved

    def stop_token_ids(self) -> set[int]:
        """Include model EOS and genuine end-of-turn tokens, never an UNK lookup."""
        result: set[int] = set()
        for source in (self.processor.tokenizer, self.model.generation_config):
            eos = getattr(source, "eos_token_id", None)
            if eos is not None:
                result.update(int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]))
        end_turn = self.processor.tokenizer.get_vocab().get("<end_of_turn>")
        if end_turn is not None:
            result.add(int(end_turn))
        return result


def load_model(
    name: str,
    device: torch.device,
    model_path: Path | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ModelBundle:
    """Load only the selected backbone. Authentication uses normal HF login."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {name}")
    spec = MODEL_SPECS[name]
    if model_path is not None and not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    if model_path is not None and revision is not None:
        raise ValueError("Use either a local model path or a remote revision, not both")
    if model_path is None and os.environ.get("HF_TOKEN") == "HF_Token":
        raise ValueError(
            "HF_Token is a placeholder. Set HF_TOKEN to your own token locally, "
            "or unset HF_TOKEN and use your cached Hugging Face login."
        )
    source = str(model_path) if model_path is not None else spec.model_id
    shared = {"local_files_only": local_files_only, "trust_remote_code": False}
    if model_path is None:
        shared["revision"] = revision or spec.revision
    dtype = spec.dtype if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(source, use_fast=False, **shared)
    model_cls = LlavaForConditionalGeneration if name == "llava-med" else AutoModelForImageTextToText
    model = model_cls.from_pretrained(
        source, torch_dtype=dtype, device_map={"": str(device)},
        low_cpu_mem_usage=True, **shared,
    ).eval()
    return ModelBundle(name, model, processor, device, dtype)


# ==========================================================================
# VQA-RAD input loading
# ==========================================================================

DATASET_ID = "flaviagiammarino/vqa-rad"


def load_vqa_rad(
    split: str = "test",
    arrow_path: Path | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Dataset:
    if split not in {"train", "test"}:
        raise ValueError("VQA-RAD split must be train or test")
    if arrow_path is not None:
        if not arrow_path.is_file():
            raise FileNotFoundError(f"Dataset Arrow file does not exist: {arrow_path}")
        dataset = Dataset.from_file(str(arrow_path))
    else:
        dataset = load_dataset(
            DATASET_ID, split=split, revision=revision,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
    required = {"image", "question"}
    if not required.issubset(dataset.column_names):
        raise ValueError("Dataset must have image and question columns")
    columns = ["image", "question"]
    if "question_id" in dataset.column_names:
        columns.append("question_id")
    return dataset.select_columns(columns)


def select_range(dataset_size: int, start: int, limit: int | None) -> range:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not 0 <= start < dataset_size:
        raise ValueError(f"start must be in [0, {dataset_size})")
    stop = dataset_size if limit is None else min(dataset_size, start + limit)
    return range(start, stop)


# ==========================================================================
# Generation runner and provenance
# ==========================================================================

def make_parser(model_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if model_name is None:
        parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    elif model_name in MODEL_SPECS:
        parser.set_defaults(model=model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    parser.add_argument("--model-path", type=Path, help="Optional local HF-compatible snapshot directory")
    parser.add_argument("--model-revision", help="Override the pinned remote model revision")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--dataset-arrow", type=Path, help="Optional existing VQA-RAD Arrow file")
    parser.add_argument("--dataset-revision", help="Optional HF dataset commit/revision")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model or dataset files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True, help="A NEW directory; existing runs are never overwritten")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Generate only this many answers")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-factor", type=float, default=0.01)
    parser.add_argument("--gaussian-std", type=float, default=0.07)
    parser.add_argument("--poisson-scale", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--save-token-trace", action="store_true", help="Include per-token VGS values in output")
    return parser


def run(args: argparse.Namespace) -> Path:
    config = DecodingConfig(**{key: getattr(args, key) for key in DecodingConfig.__dataclass_fields__})
    device = torch.device(args.device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices are cuda:N and cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a GPU node or explicitly pass --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA index does not exist")
        with torch.cuda.device(device):
            if args.model == "medgemma" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("MedGemma CUDA inference requires a GPU supporting bfloat16")
    if args.model_path is not None and args.model_revision is not None:
        raise ValueError("--model-path and --model-revision cannot be combined")
    if args.dataset_arrow is not None and args.dataset_revision is not None:
        raise ValueError("--dataset-arrow and --dataset-revision cannot be combined")
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--start must be >= 0 and --limit must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "predictions.jsonl"
    spec = MODEL_SPECS[args.model]
    metadata: dict[str, Any] = {
        "status": "initializing", "model": args.model,
        "model_source": str(args.model_path) if args.model_path is not None else spec.model_id,
        "model_revision": None if args.model_path is not None else args.model_revision or spec.revision,
        "dataset": DATASET_ID, "split": args.split,
        "dataset_arrow": str(args.dataset_arrow) if args.dataset_arrow is not None else None,
        "dataset_revision": args.dataset_revision,
        "start": args.start, "limit": args.limit, "completed": 0,
        "device": str(device), "decoding": asdict(config),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "datasets", "numpy", "Pillow"
        )},
    }
    try:
        torch.manual_seed(config.seed)
        dataset = load_vqa_rad(args.split, args.dataset_arrow, args.dataset_revision, args.local_files_only)
        indices = select_range(len(dataset), args.start, args.limit)
        metadata.update(dataset_fingerprint=dataset._fingerprint, stop=indices.stop, requested=len(indices))
        bundle = load_model(args.model, device, args.model_path, args.model_revision, args.local_files_only)
        metadata["dtype"] = str(bundle.dtype)
        if device.type == "cuda":
            metadata["cuda_device"] = torch.cuda.get_device_name(device)
        stop_ids = bundle.stop_token_ids()
        metadata["eos_token_ids"] = sorted(stop_ids)
        metadata["status"] = "running"

        with output.open("x", encoding="utf-8") as handle:
            for idx in indices:
                sample = dataset[idx]
                question = str(sample["question"])
                # Stable across --start/--limit partitions; no reference answer is read.
                sample_seed = config.seed + idx
                row: dict[str, Any] = {
                    "sample_index": idx, "question_id": str(sample.get("question_id", idx)),
                    "question": question, "model": args.model, "seed": sample_seed,
                }
                try:
                    image = sample["image"].convert("RGB")
                    distorted = perturb_image(image, config, sample_seed)
                    result = decode(
                        bundle.model, bundle.prepare(image, question),
                        bundle.prepare(distorted, question), config, stop_ids,
                    )
                    row.update(
                        status="ok",
                        answer=bundle.processor.tokenizer.decode(result.token_ids[0], skip_special_tokens=True).strip(),
                        generated_tokens=int(result.token_ids.shape[1]),
                        stop_reason=result.stop_reason,
                    )
                    if args.save_token_trace:
                        row["token_trace"] = [asdict(step) for step in result.steps]
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    raise
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                metadata["completed"] += 1
                if metadata["completed"] % 25 == 0 or metadata["completed"] == len(indices):
                    print(f"{args.model}: {metadata['completed']}/{len(indices)} answers generated", flush=True)
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Persist run provenance even on a normal Python failure/interruption.
        # A scheduler SIGKILL cannot execute this block; JSONL is flushed per row.
        with (args.output_dir / "run.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return output


def main(model_name: str | None = None) -> None:
    args = make_parser(model_name).parse_args()
    try:
        output = run(args)
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
