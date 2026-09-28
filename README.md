# Official implementation of “Object Hallucination Mitigation in Large Vision-Language Models via Self-Vision Dual Masking and Uncertainty-Triggered Assembly”

**Accepted to NeurIPS 2026 as a Poster.**

[![Conference](https://img.shields.io/badge/NeurIPS-2026%20Poster-8A2BE2)](https://openreview.net/forum?id=k3bNFdhhNj)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> This repository currently releases only an **initial subset of the SIGMA core implementation**. The fully organized final implementation, model-specific adapters, and evaluation/reproduction pipeline are still being prepared and will be released soon. **Coming soon.**

## Overview

SIGMA is a training-free framework for mitigating object hallucination in large vision-language models (LVLMs). It operates on the target LVLM's own visual evidence and consists of three main stages:

1. **Self-vision dual masking** separates query-supporting evidence from query-inconsistent or shortcut-prone context.
2. **Tri-stream visual encoding** forms global, support-masked, and complement-masked visual streams in the same latent space.
3. **Uncertainty-triggered assembly** follows the native global stream by default, activates support correction when the prediction is uncertain, and invokes complement suppression only when the support-corrected state remains risky.

The current release contains a model-agnostic implementation of the central SIGMA equations and routing policy. It does not yet provide an end-to-end benchmark runner.

## Currently available

The code under [`src/sigma_core`](src/sigma_core) includes:

- polarity-corrected image and query relevance;
- agreement-focused support fusion;
- context-aware support and complement mask construction;
- multi-entity mask aggregation;
- soft patch-token reweighting for masked self-vision injection;
- full-vocabulary and candidate-answer uncertainty estimation;
- support-first and complement-on-demand routing;
- positive calibration, structured residual assembly, and global plausibility constraints;
- cache synchronization for selectively activated decoding branches;
- the paper-aligned default configuration.

## Release status

| Component | Status |
|---|---|
| Model-agnostic SIGMA core | Available |
| Paper-aligned default configuration | Available |
| LLaVA adapter | Coming soon |
| InstructBLIP adapter | Coming soon |
| Qwen3-VL and InternVL adapters | Coming soon |
| Evaluation and reproduction scripts | Coming soon |
| Dataset preparation instructions | Coming soon |
| Full paper-table reproduction | Coming soon |

## Repository structure

```text
src/sigma_core/
├── assembly.py      # Structured residual assembly and plausibility constraint
├── config.py        # Paper-aligned SIGMA configuration
├── decoder.py       # Selective-branch cache synchronization interface
├── masks.py         # Self-vision support/complement mask construction
├── routing.py       # Support-first, complement-on-demand routing
├── uncertainty.py   # Entropy, calibration, and scheduling utilities
└── vision.py        # Backbone-neutral patch-token reweighting
```

## Using the preview core

This initial release is provided as source code. Add `src` to `PYTHONPATH` before importing the package:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Example:

```python
import torch

from sigma_core import SUBMITTED_CONFIG, build_dual_masks

image_relevance = torch.rand(1, 576)
query_relevance = torch.rand(1, 576)
cls_patch_similarity = torch.randn(1, 576)

masks = build_dual_masks(
    image_relevance=image_relevance,
    query_relevance=query_relevance,
    cls_patch_similarity=cls_patch_similarity,
    config=SUBMITTED_CONFIG.mask,
)

support_mask = masks.support_mask
complement_mask = masks.complement_mask
```

The relevance tensors in this example are placeholders. In the complete release, model-specific adapters will construct them directly from each LVLM's vision encoder and text projection.

## License

This project is released under the [MIT License](LICENSE).
