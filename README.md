<div align="center">

# DyTox Modification

Dynamic Token Expansion for Continual Learning — refactored and extended for industrial experimentation.

[arXiv:2111.11326](https://arxiv.org/abs/2111.11326) · CVPR 2022

</div>

## Overview
- Original DyTox: ViT/ConViT backbone with task-specific tokens (TAB) that expand per task, rehearsal + distillation to control forgetting.
- This fork: production-grade `adl` package, adds a ResNet18 feature-extractor path, optional joint token routing, divergence head controls, Sharpness-Aware Minimization (SAM/Look-SAM), and broader backbone registry (ResNet/ResNeXt/SENet/VGG/Inception/SCS variants) alongside ViT/ConViT.
- Goal: improve stability/speed on continual CIFAR/ImageNet by combining CNN inductive bias, token routing variants, diversity pressure, and sharpness-aware optimization.

## Architecture (mermaid)
```mermaid
flowchart LR
    subgraph Backbone
        A[Input Image]
        B[Patch/Conv Stem]
        C[Feature Tokens]
    end
    subgraph TokenRouting
        T1[Task Tokens (per task)]
        JT[Joint Tokens (optional)]
        CA[Class/Joint Attention Blocks]
    end
    subgraph Heads
        H1[Main Classifier]
        H2[Divergence Head (optional)]
    end
    subgraph Losses
        L1[CE / Label Smoothing / Soft BCE]
        L2[KD to Teacher]
        L3[POD Feature Distillation]
        L4[SAM / Look-SAM Optimization]
    end
    subgraph Memory
        M1[Rehearsal Buffer]
        M2[Herding & Replay Sampling]
    end

    A --> B --> C --> CA
    T1 --> CA
    JT -. joint_tokens .-> CA
    CA --> H1
    CA --> H2
    H1 --> L1
    H2 --> L1
    C --> L3
    L2 --> L1
    M1 <---> M2
    M2 --> L1
    L4 -. updates .-> CA
    L4 -. updates .-> H1
    L4 -. updates .-> H2
```

## What Changed (MODIFIED FROM ORIGINAL)
- DyTox backbone swap: `resnet=True` path uses ResNet18 + 1×1 projection to embed dim (see [src/adl/models/dytox.py](src/adl/models/dytox.py)).
- Joint token routing: `joint_tokens` processes all task tokens together for faster passes (masked attention).
- Divergence head: `head_div` with `head_div_mode` (train/finetune) to diversify logits for new classes.
- Backbone registry expansion: SCS ResNet variants (`resnet18_scs`, `resnet18_scs_avg`, `resnet18_scs_max`), SENet/VGG/Inception/ResNeXt exposed in [src/adl/models/factory.py](src/adl/models/factory.py) and [src/adl/models/backbones/cnn](src/adl/models/backbones/cnn).
- SAM / Look-SAM: memory-aware first/second steps and look-ahead cadence in [src/adl/training/engine.py](src/adl/training/engine.py) and [src/adl/training/sam.py](src/adl/training/sam.py).
- Packaging: refactored into `adl` package with config-driven runs and preserved upstream docs.

## Original Components (ORIGINAL DYTOX CODE)
- Task-token TAB blocks (ClassAttention/JointCA) in [src/adl/models/backbones/convit.py](src/adl/models/backbones/convit.py).
- POD feature distillation in [src/adl/training/pod.py](src/adl/training/pod.py).
- KD, rehearsal memory, finetuning hooks in [src/adl/training/rehearsal.py](src/adl/training/rehearsal.py) and loss wrappers in [src/adl/training/losses.py](src/adl/training/losses.py).
- Continuum-based datasets and loaders in [src/adl/datasets/datasets.py](src/adl/datasets/datasets.py).

## Tech Stack
- PyTorch (torch>=2.9.0), torchvision>=0.24.0
- timm==0.4.12, continuum==1.0.27, pyyaml>=6.0.3
- Distributed training, AMP, SAM/Look-SAM supported.

## Project Structure
- [src/adl/cli.py](src/adl/cli.py) — entrypoint, args, task loop, logging.
- [src/adl/models/dytox.py](src/adl/models/dytox.py) — DyTox module, resnet path, divergence head.
- [src/adl/models/classifier.py](src/adl/models/classifier.py) — classifiers and head expansion.
- [src/adl/models/factory.py](src/adl/models/factory.py) — backbone registry, loaders, DyTox updater.
- [src/adl/models/backbones/](src/adl/models/backbones/) — ViT/ConViT, CNNs (ResNet/SCS/ResNeXt/SENet/VGG/Inception/Rebuffi), weight init.
- [src/adl/training/](src/adl/training/) — engine, losses, mixup, POD, rehearsal, SAM/Look-SAM, samplers, scaler.
- [src/adl/datasets/datasets.py](src/adl/datasets/datasets.py) — CIFAR100, ImageNet100/1000 loaders, transforms, class orders.
- [src/adl/utils/](src/adl/utils/) — logging, distributed helpers, freezing, metrics.
- Configs: [src/adl/configs/model](src/adl/configs/model) (DyTox/DyTox+/DyTox++), [src/adl/configs/data](src/adl/configs/data) (class orders and increments), [src/adl/configs/arthur.yaml](src/adl/configs/arthur.yaml) (machine paths).
- Scripts: [src/adl/scripts/train.sh](src/adl/scripts/train.sh) (distributed launch), [src/adl/scripts/convert_memory.py](src/adl/scripts/convert_memory.py) (memory path rewrite).
- Checkpoints: [src/adl/data/checkpoints/25-10-21_CIFAR-2-2_dytox_singleGPU_1](src/adl/data/checkpoints/25-10-21_CIFAR-2-2_dytox_singleGPU_1) (copied from logs_singleGPU).
- Upstream docs preserved: [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md), [docs/UPSTREAM_LICENSE](docs/UPSTREAM_LICENSE).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Training Examples
- ConViT DyTox, CIFAR100 2-2:
```bash
python -m adl.cli \
  --options src/adl/configs/model/cifar_dytox.yaml src/adl/configs/data/cifar100_2-2.yaml \
  --data-path /path/to/cifar \
  --dytox --model convit --input-size 32 --epochs 500 --batch-size 128 \
  --head-div 0.1 --joint-tokens \
  --sam-rho 0.05 --sam-first main --sam-second main
```

- ResNet18 DyTox path:
```bash
python -m adl.cli \
  --options src/adl/configs/model/cifar_dytox.yaml src/adl/configs/data/cifar100_2-2.yaml \
  --data-path /path/to/cifar \
  --dytox --model resnet18 --resnet --input-size 32 \
  --head-div 0.1 --joint-tokens
```

- Distributed wrapper:
```bash
bash src/adl/scripts/train.sh 0,1,2,3 --options ...
```

## Evaluation
```bash
python -m adl.cli --eval --resume /path/to/checkpoint.pth --options <model_yaml> <data_yaml>
```

## Results & Logs
- Available run: CIFAR100 2-2, metrics/logs in logs_singleGPU/25-10-21_CIFAR-2-2_dytox_singleGPU_1 and mirrored checkpoints in [src/adl/data/checkpoints/25-10-21_CIFAR-2-2_dytox_singleGPU_1](src/adl/data/checkpoints/25-10-21_CIFAR-2-2_dytox_singleGPU_1). Example from logs: avg_acc ≈62% after 50 increments, forgetting ≈19 (top-5 ≈88–90%).
- No side-by-side baseline vs modified table yet; add future experiment sheets for comparison.

## Future Work
- Ablate joint_tokens and divergence head modes; benchmark ResNet/SCS vs ConViT on CIFAR 2-2/5-5/10-10 and ImageNet100 incremental.
- Tune SAM/Look-SAM schedules per task; profile speed/accuracy tradeoffs of joint token routing.
- Add experiment registry and automated result tables; unit tests for backbone registry and DyTox updates.

## Citation
If you use this code, cite the original paper:

```
@inproceedings{douillard2021dytox,
  title     = {DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion},
  author    = {Douillard, Arthur and Ram\'e, Alexandre and Couairon, Guillaume and Cord, Matthieu},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
