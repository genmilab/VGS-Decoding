# Decoding details

## Token rule

For the same question and previously selected answer tokens, let `p` be the
next-token distribution under the original image and `q` the distribution
under the perturbed image. For each candidate token:

```text
vgs = (p - q) / (p + q + 1e-8)
factor = max(1 + alpha * vgs, min_factor)
weighted = p * factor
next_distribution = weighted / sum(weighted)
next_token = argmax(next_distribution)
```

Probabilities and reweighting use float32 even when model weights use lower
precision. VGS lies between -1 and 1, up to floating-point precision. A positive
score means a token is more probable under the clean image than the particular
perturbed image. It does not certify that a token is correct or grounded.
Setting `alpha=0` leaves the clean distribution unchanged apart from numerical
roundoff. No sampling or repetition penalty is added.

## Two independent branches

1. Convert the original image to RGB. Add Gaussian noise, clip to [0, 1], apply
   Poisson noise, clip again, and round to 8-bit RGB.
2. Prepare both images using the same question, prompt, processor, and dtype.
   Reject the inputs if their text token IDs differ.
3. Initialize two independent generation states and KV caches.
4. Forward each branch once, apply the rule above, and append the **same**
   selected token to both histories.
5. Stop at the model's EOS/end-of-turn token or the configured token budget.

The distorted image is fixed for one answer. Noise uses a local NumPy random
generator seeded by `base_seed + dataset_index`, which makes partitioned runs
use the same perturbation. Identical seeds do not promise bitwise identity
across different GPU architectures, package versions, or model revisions.

The decoder relies on Transformers 4.53.0 generation helpers to maintain
attention masks and image-aware cache positions. Its private helper interface
is why the Transformers dependency is pinned. Only batch size 1 and the two
listed Hugging Face-compatible decoder-only multimodal architectures are
supported. Upgrading Transformers requires rechecking the adapters and cache
tests; arbitrary model compatibility is not claimed.

## Prompt adapters

LLaVA-Med uses the existing research implementation's prompt:

```text
USER: <image>
Answer this question as concisely as possible based on the provided image: {question}
ASSISTANT:
```

MedGemma uses its processor chat template, with a system message identifying
the medical image-analysis task and a user turn containing the same instruction,
question, and image. Reference answers are never included. The normal model
image processor is used; there is no manual resize or asymmetric text penalty.

`<end_of_turn>` is added to the stop set only if it actually exists in the
tokenizer vocabulary. This avoids treating an unknown-token fallback as EOS.
Stop tokens appear in optional traces but are not returned as answer text.

## Reproducibility and scope

The core is adapted from the corrected dual-cache research implementation.
Checkpoint revisions, prompts, defaults, dataset indices, and software versions
are recorded or documented. This packaging does not claim to reproduce every
historical paper table; the earlier project contained other generation settings
and scoring protocols that are intentionally not shipped here.

The Gaussian/Poisson configuration is an empirical perturbation, not a validated
physical acquisition model for every imaging modality. Clinical correctness,
hallucination rates, and effectiveness require separate validation. This
repository deliberately implements generation only.
