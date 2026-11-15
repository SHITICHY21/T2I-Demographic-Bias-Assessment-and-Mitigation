
# Stable Diffusion 1.5 — Generated Image Dataset

**Occupational Demographic Bias Assessment in Text-to-Image Models**

![Model](https://img.shields.io/badge/Model-Stable%20Diffusion%201.5-7950F2)
![Images](https://img.shields.io/badge/Images-21%2C000-087F8C)
![Resolution](https://img.shields.io/badge/Resolution-512%C3%97512-2563EB)
![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853)

---

## Overview

This directory documents the **Stable Diffusion 1.5** baseline image dataset used for occupational demographic bias assessment in text-to-image generative models.

Stable Diffusion 1.5 employs a U-Net-based latent diffusion architecture conditioned on CLIP text embeddings. It serves as a conventional latent diffusion baseline for comparing demographic representation against SDXL Turbo, PixArt-Σ, and Kandinsky 2.2.

## Dataset Specifications

| Property | Specification |
|:---|:---|
| Generative model | Stable Diffusion 1.5 |
| Architecture | Latent diffusion / U-Net |
| Text encoder | CLIP ViT-L |
| Model checkpoint | `runwayml/stable-diffusion-v1-5` |
| Professions | 20 |
| Prompt variants per profession | 21 |
| Total unique prompts | 420 |
| Images per prompt | 50 |
| Total baseline images | **21,000** |
| Image resolution | 512 × 512 |
| Inference steps | 20 |
| Guidance scale | 7.5 |
| Generation hardware | NVIDIA Tesla T4 (16 GB) |

## Prompt Design

A standardized prompt set is used to evaluate occupational stereotypes and contextual demographic variation.

| Category | Variants |
|:---|---:|
| Neutral occupational prompts | 1 |
| Religion-based prompts | 5 |
| Nationality-based prompts | 5 |
| Positive-trait prompts | 5 |
| Negative-trait prompts | 5 |
| **Total per profession** | **21** |

**Prompt template:**

`A [modifier] [profession] at work`

For each prompt, 50 images are generated with distinct random seeds.

**Total:** 20 × 21 × 50 = **21,000 images**

## Dataset Access

The image dataset is hosted externally to avoid storing large binary files directly in GitHub.

### [📂 Access Stable Diffusion 1.5 Dataset](https://drive.google.com/drive/folders/17zDPIt0AmLPR-7laB5nOXMPKfhnwsXD_?usp=drive_link)

**Dataset:** Stable Diffusion 1.5  
**Baseline images:** 21,000  
**Access platform:** Google Drive

## Research Applications

- Occupational gender-bias quantification
- Age and racial representation analysis
- Contextual bias analysis across five prompt categories
- Multi-classifier evaluation with DeepFace, FairFace, and CLIP
- Comparison of prompt-based and embedding-based gender debiasing
- Cross-architecture evaluation of demographic disparities

## Reproducibility

The corresponding generation script is maintained in:

[`image_generation/sd15_batch_generator.py`](../../image_generation/sd15_batch_generator.py)

**Main repository:** [T2I Demographic Bias Assessment and Mitigation](https://github.com/SHITICHY21/T2I-Demographic-Bias-Assessment-and-Mitigation)

## Responsible Use

The images are synthetic and were generated for academic research. The demographic attributes assessed in this study correspond to perceived visual characteristics inferred by automated classifiers, rather than self-reported or verified personal identities.

---

**Dataset type:** AI-generated research imagery  
**Purpose:** Reproducible occupational demographic bias assessment

