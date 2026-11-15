
# SDXL Turbo — Generated Image Dataset

**Occupational Demographic Bias Assessment in Text-to-Image Models**

![Model](https://img.shields.io/badge/Model-SDXL%20Turbo-7950F2)
![Images](https://img.shields.io/badge/Images-21%2C000-087F8C)
![Resolution](https://img.shields.io/badge/Resolution-512%C3%97512-2563EB)
![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853)

---

## Overview

This directory documents the **SDXL Turbo** baseline image dataset generated for a systematic investigation of demographic bias in text-to-image diffusion models.

SDXL Turbo is an accelerated latent diffusion model based on adversarial diffusion distillation. Its few-step image generation capability provides an important architectural comparison against conventional diffusion models and alternative generation pipelines.

## Dataset Specifications

| Property | Specification |
|:---|:---|
| Generative model | SDXL Turbo |
| Architecture | Distilled latent diffusion / U-Net |
| Text encoders | CLIP + OpenCLIP-G |
| Model checkpoint | `stabilityai/sdxl-turbo` |
| Professions | 20 |
| Prompt variants per profession | 21 |
| Total unique prompts | 420 |
| Images per prompt | 50 |
| Total baseline images | **21,000** |
| Image resolution | 512 × 512 |
| Inference steps | 8 |
| Guidance scale | 0.0 |
| Generation hardware | NVIDIA Tesla T4 (16 GB) |

## Prompt Design

Each profession is evaluated using 21 controlled prompt variants distributed across five categories.

| Category | Number of variants |
|:---|---:|
| Neutral | 1 |
| Religion | 5 |
| Nationality | 5 |
| Positive traits | 5 |
| Negative traits | 5 |
| **Total** | **21** |

**Prompt template:**

`A [modifier] [profession] at work`

Each unique prompt generates 50 images using distinct random seeds, producing 21,000 baseline images.

## Dataset Access

The full image collection is maintained externally due to its storage requirements.

### [📂 Access SDXL Turbo Dataset](https://drive.google.com/drive/folders/1jFdMFxutqaVoa0QUALDKks2oAjI49fZp?usp=drive_link)

**Dataset:** SDXL Turbo  
**Baseline images:** 21,000  
**Access platform:** Google Drive

## Research Applications

The dataset supports:

- Model-level and occupation-level gender-bias assessment
- Age and racial representation analysis
- Context-conditioned demographic evaluation
- Cross-classifier comparison
- Before-and-after comparison with gender-debiased generations

The baseline images are assessed using DeepFace, FairFace, and zero-shot CLIP, with FairFace serving as the primary source of gender statistics.

## Reproducibility

The corresponding generation script is maintained in:

[`image_generation/sdxl_turbo_batch_generator.py`](../../image_generation/sdxl_turbo_batch_generator.py)

**Main repository:** [T2I Demographic Bias Assessment and Mitigation](https://github.com/SHITICHY21/T2I-Demographic-Bias-Assessment-and-Mitigation)

## Responsible Use

The dataset contains synthetic imagery generated for demographic representation research. Automated demographic classifications are estimates of visual presentation and must not be treated as definitive indicators of personal identity.

---

**Dataset type:** AI-generated research imagery  
**Purpose:** Reproducible evaluation of demographic bias in T2I models

