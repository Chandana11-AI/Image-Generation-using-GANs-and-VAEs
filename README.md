# Image Generation with a DCGAN Prototype

A PyTorch learning project exploring adversarial image generation: a generator maps random noise to images, while a discriminator learns to distinguish generated examples from real images.

The repository name mentions GANs and VAEs, but the committed implementation contains **DCGAN-style generator/discriminator code only**. No VAE implementation is included.

## What is implemented

- A generator built from transposed convolutions, batch normalization, ReLU and a final Tanh layer.
- A discriminator built from convolutions, LeakyReLU, batch normalization and Sigmoid.
- A training loop with separate Adam optimizers and binary cross-entropy losses.
- CelebA dataset loading and image preprocessing.
- Periodic loss printing and generated-image export.

## Adversarial training flow

For each batch, the training script first updates the discriminator using real images and detached generated images. It then updates the generator using the discriminator's predictions on generated images, with the objective of making them look real to the discriminator.

| Setting | Source configuration |
|---|---|
| Latent vector | 100 values, sampled from a normal distribution |
| Generator/discriminator feature width | 64 by default |
| Channels | 3 |
| Input image preprocessing | Resize/crop to 64 × 64; tensor conversion and normalization |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Epochs | 10 |
| Image export interval | Every 100 batches |
| Device | CUDA when available; otherwise CPU |

## Files

| File | Purpose |
|---|---|
| [dcgan.py](dcgan.py) | Generator and discriminator definitions |
| [train_dcgan.py](train_dcgan.py) | Dataset loading, training and sample-export loop |
| [README.md](README.md) | Architecture explanation and prototype status |

No trained weights, evaluation report, sample-output directory or VAE model is committed.

## Prototype status and known blockers

This snapshot needs architecture validation before it can be presented as a working training run:

- With a latent input shaped `N × 100 × 1 × 1`, the generator's current layer sequence produces 32 × 32 images, while the training preprocessing produces 64 × 64 real images.
- The discriminator produces spatial predictions that are flattened, while the loss targets contain one value per image. Its output dimensions must be aligned with those targets.
- The training script writes to `results/` without creating that directory.
- Dataset download/access requirements and package compatibility must be checked in the chosen environment.

These observations come from the committed layer definitions and training code. They are not the result of a completed training run.

## Setup reference

The source imports `torch` and `torchvision`. Use an isolated environment and an appropriate PyTorch installation for your hardware. CelebA must be obtained through an authorized dataset route.

```bash
python -m pip install torch torchvision
```

After resolving the dimension and output-directory issues, the intended training entry point is `python train_dcgan.py`. It should not be treated as a verified end-to-end command for the current revision.

## How quality should be assessed

Track generator/discriminator losses alongside fixed-noise samples across epochs. Inspect mode collapse and image diversity, and use a documented held-out evaluation protocol before reporting quantitative image-quality results.

The earlier description of “high-quality face generation” is not supported by committed outputs, so no image-quality or successful-training claim is made here.

## Learning outcomes and extensions

This project exposes the mechanics of adversarial optimization, convolutional image synthesis, tensor-shape reasoning and experiment logging. Next steps are shape checks, a small smoke test, reproducible sampling, checkpoint saving and documented outputs. A VAE could be a separate, explicitly implemented comparison.
