
# Kandinsky 2.2 — Generated Image Dataset

**Occupational Demographic Bias Assessment in Text-to-Image Models**

![Model](https://img.shields.io/badge/Model-Kandinsky%202.2-7950F2)
![Images](https://img.shields.io/badge/Images-21%2C000-087F8C)
![Resolution](https://img.shields.io/badge/Resolution-512%C3%97512-2563EB)
![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853)

---

## Overview

This directory provides access to the **Kandinsky 2.2** baseline image dataset used in our research on demographic bias assessment and mitigation in text-to-image (T2I) diffusion models.

Kandinsky 2.2 employs a two-stage generation architecture consisting of an image prior and a latent diffusion decoder. The dataset supports systematic investigation of demographic representation across occupational and context-conditioned prompts.

## Dataset Specifications

| Property | Specification |
|:---|:---|
| Generative model | Kandinsky 2.2 |
| Architecture | Image prior + U-Net decoder |
| Text encoder | CLIP ViT-bigG |
| Professions | 20 |
| Prompt variants per profession | 21 |
| Total unique prompts | 420 |
| Images per prompt | 50 |
| Total baseline images | **21,000** |
| Image resolution | 512 × 512 |
| Decoder inference steps | 30 |
| Generation hardware | NVIDIA Tesla T4 (16 GB) |
| Memory optimization | CPU offloading |

## Prompt Design

The dataset covers five contextual categories:

| Category | Variants |
|:---|---:|
| Neutral occupational prompts | 1 |
| Religion-conditioned prompts | 5 |
| Nationality-conditioned prompts | 5 |
| Positive-trait prompts | 5 |
| Negative-trait prompts | 5 |
| **Total per profession** | **21** |

The standard prompt structure is:

`A [modifier] [profession] at work`

Each prompt is sampled 50 times using distinct random seeds.

## Dataset Access

The images are hosted on Google Drive because the complete four-model collection exceeds 37 GB.

### [📂 Access Kandinsky 2.2 Dataset](https://drive.google.com/drive/folders/1schJt2NQSiVQLp26c2RFyykOuKdnMXKC?usp=drive_link)

**Dataset:** Kandinsky 2.2  
**Baseline images:** 21,000  
**Access platform:** Google Drive

## Research Applications

- Occupational gender-bias quantification
- Perceived age and racial representation analysis
- Cross-model demographic bias comparison
- Context-sensitive bias evaluation
- Multi-classifier demographic assessment using FairFace, DeepFace, and zero-shot CLIP

## Reproducibility

The generation implementation and demographic evaluation code are maintained in the main repository.

**Repository:** [T2I Demographic Bias Assessment and Mitigation](https://github.com/SHITICHY21/T2I-Demographic-Bias-Assessment-and-Mitigation)

## Responsible Use

Demographic labels in this study represent automated estimates of visual presentation, not self-identified personal attributes. Age, race, and gender estimates should be interpreted in light of classifier limitations and the simplified demographic categories used for analysis.

---

**Dataset type:** AI-generated research imagery  
**Purpose:** Academic evaluation of demographic bias in generative AI

