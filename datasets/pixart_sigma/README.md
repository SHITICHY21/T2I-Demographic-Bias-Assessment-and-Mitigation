
# PixArt-Σ — Generated Image Dataset

**Occupational Demographic Bias Assessment in Text-to-Image Models**

![Model](https://img.shields.io/badge/Model-PixArt--Sigma-7950F2)
![Images](https://img.shields.io/badge/Images-21%2C000-087F8C)
![Resolution](https://img.shields.io/badge/Resolution-512%C3%97512-2563EB)
![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853)

---

## Overview

This directory provides access to the **PixArt-Σ** baseline image dataset, developed as part of a cross-architecture study of demographic bias in text-to-image generative models.

Unlike U-Net-based diffusion systems, PixArt-Σ employs a Diffusion Transformer (DiT) architecture conditioned on a T5-XXL text encoder. Its inclusion enables investigation of whether occupational demographic stereotypes persist across fundamentally different generative architectures.

## Dataset Specifications

| Property | Specification |
|:---|:---|
| Generative model | PixArt-Σ |
| Architecture | Diffusion Transformer (DiT) |
| Text encoder | T5-XXL |
| Model checkpoint | `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` |
| Professions | 20 |
| Prompt variants per profession | 21 |
| Total unique prompts | 420 |
| Images per prompt | 50 |
| Total baseline images | **21,000** |
| Image resolution | 512 × 512 |
| Inference steps | 6 |
| Guidance scale | 4.0 |
| Generation hardware | NVIDIA Tesla T4 (16 GB) |
| Memory optimization | CPU offloading |

## Prompt Design

The dataset uses a structured set of occupational prompts.

| Prompt category | Variants |
|:---|---:|
| Neutral | 1 |
| Religion-conditioned | 5 |
| Nationality-conditioned | 5 |
| Positive-trait | 5 |
| Negative-trait | 5 |
| **Total per occupation** | **21** |

**Prompt template:**

`A [modifier] [profession] at work`

For every occupation–variant combination, 50 independently seeded images are generated.

**Dataset composition:**

20 occupations × 21 variants × 50 images = **21,000 images**

## Dataset Access

Because the full four-model research collection exceeds 37 GB, the generated images are hosted on Google Drive.

### [📂 Access PixArt-Σ Dataset](https://drive.google.com/drive/folders/1B6CRG0M_X_WF3DOAr1HQJOsovD3kRJNY?usp=drive_link)

**Dataset:** PixArt-Σ  
**Baseline images:** 21,000  
**Access platform:** Google Drive

## Research Applications

- Demographic bias assessment across transformer-based image generators
- Occupational gender representation analysis
- Age and race representation analysis
- Contextual prompt sensitivity evaluation
- Comparison against U-Net and prior–decoder architectures
- Evaluation of prompt-based gender-bias mitigation

## Reproducibility

The corresponding generation script is maintained in:

[`image_generation/pixart_sigma_batch_generator.py`](../../image_generation/pixart_sigma_batch_generator.py)

**Main repository:** [T2I Demographic Bias Assessment and Mitigation](https://github.com/SHITICHY21/T2I-Demographic-Bias-Assessment-and-Mitigation)

## Responsible Use

The generated images are intended for research on representational bias in generative AI. Demographic inference reflects the outputs of automated classification systems and should not be interpreted as a definitive assessment of identity or group membership.

---

**Dataset type:** AI-generated research imagery  
**Purpose:** Cross-architecture demographic bias evaluation

