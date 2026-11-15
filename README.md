
<div align="center">

# Demographic Bias Assessment and Mitigation in Text-to-Image Models

### A Unified Framework for Detection, Quantification, Mitigation, and Re-Evaluation

**A systematic investigation of occupational demographic representation across four text-to-image generative models**

[![Research](https://img.shields.io/badge/Research-Responsible_AI-7C3AED?style=for-the-badge)](#overview)
[![Models](https://img.shields.io/badge/T2I_Models-4-2563EB?style=for-the-badge)](#evaluated-models)
[![Dataset](https://img.shields.io/badge/Images-84%2C000-059669?style=for-the-badge)](#dataset)
[![Prompts](https://img.shields.io/badge/Prompts-420-EA580C?style=for-the-badge)](#prompt-design)

**[Overview](#overview) · [Methodology](#methodology) · [Datasets](#dataset-access) · [Results](#key-findings) · [Reproducibility](#reproducibility) · [Authors](#authors)**

</div>

---

## Overview

Text-to-image (T2I) generative models have significantly advanced the creation of photorealistic imagery from natural-language descriptions. However, these systems can reproduce or amplify demographic stereotypes, particularly when generating representations of different professions.

For example, seemingly neutral prompts such as *"A doctor at work"* or *"An engineer at work"* may produce disproportionately male-presenting individuals, while other occupations may exhibit different demographic patterns.

This repository accompanies a research study introducing a **unified detect–mitigate–re-detect framework** for systematically examining demographic representation in text-to-image diffusion models.

The framework investigates three perceived demographic attributes:

- **Gender:** Male- and female-presenting representations
- **Age:** Representation across four age brackets
- **Race:** Classifier-inferred racial presentation

Gender-bias mitigation is subsequently evaluated through two complementary approaches:

1. **Prompt-based debiasing**, which explicitly conditions occupational prompts on gender.
2. **Embedding-based debiasing**, which modifies text-conditioning representations without changing the original occupational prompt.

The study evaluates four architecturally distinct T2I models using a controlled benchmark of **420 prompts and 84,000 baseline-generated images**.

> **Research objective:** To measure occupational demographic disparities, assess the reliability of automated demographic classification, and investigate whether targeted interventions improve gender representation across different T2I architectures.

---

## Research at a Glance

| Component | Description |
|:---|:---|
| Research domain | Responsible AI, Generative AI, Fairness |
| Generative models | Stable Diffusion 1.5, SDXL Turbo, PixArt-Σ, Kandinsky 2.2 |
| Occupations | 20 |
| Prompt variants per occupation | 21 |
| Total structured prompts | **420** |
| Images per prompt per model | 50 |
| Baseline images per model | 21,000 |
| Total baseline images | **84,000** |
| Image resolution | 512 × 512 pixels |
| Demographic attributes | Gender, age, race |
| Evaluation systems | DeepFace, FairFace, zero-shot CLIP |
| Human validation | 200 stratified images |
| Mitigation approaches | Prompt-based and embedding-based |
| Statistical evaluation | Chi-square tests, Cohen's κ, MAD |
| Computational environment | NVIDIA T4 GPU (16 GB), Kaggle |

---

## Research Contributions

The study makes four principal contributions:

**1. Unified demographic bias evaluation**

A reproducible framework integrates demographic bias detection, quantification, mitigation, and post-intervention re-evaluation within a consistent experimental protocol.

**2. Cross-architecture benchmark**

A structured benchmark of 420 occupational prompts and 84,000 generated images enables systematic comparisons across four different T2I architectures.

**3. Multi-classifier assessment and human validation**

Three demographic inference systems—DeepFace, FairFace, and zero-shot CLIP—are evaluated alongside human consensus annotations to investigate classification reliability.

**4. Comparative gender-bias mitigation**

Prompt-based and embedding-based debiasing are examined under a shared evaluation protocol, highlighting their effectiveness and architecture-dependent limitations.

---

## Methodology

The proposed framework consists of four sequential stages.

```mermaid
flowchart TD
    A["20 Occupations × 21 Prompt Variants"]
    B["420 Structured Text Prompts"]
    C["Four T2I Generative Models"]
    D["84,000 Baseline Images"]
    E["Demographic Assessment"]
    F["DeepFace | FairFace | Zero-shot CLIP"]
    G["Gender, Age and Race Analysis"]
    H["Gender-Bias Mitigation"]
    I["Prompt-Based Debiasing"]
    J["Embedding-Based Debiasing"]
    K["Post-Mitigation Assessment"]
    L["Statistical Comparison and Evaluation"]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> I
    H --> J
    I --> K
    J --> K
    K --> L

    classDef input fill:#E0F2FE,stroke:#0284C7,color:#075985
    classDef generation fill:#EDE9FE,stroke:#7C3AED,color:#5B21B6
    classDef assessment fill:#DCFCE7,stroke:#16A34A,color:#166534
    classDef mitigation fill:#FFEDD5,stroke:#EA580C,color:#9A3412
    classDef output fill:#F1F5F9,stroke:#64748B,color:#334155

    class A,B input
    class C,D generation
    class E,F,G assessment
    class H,I,J mitigation
    class K,L output
```

### Stage 1 — Controlled Image Generation

A structured prompt inventory is constructed using 20 professions and 21 contextual variants. Each prompt is processed independently by four generative models, producing 50 images per prompt–model combination.

### Stage 2 — Demographic Bias Assessment

Generated images are analyzed using three complementary demographic inference systems. Bias is quantified at the model, profession, and prompt-category levels.

### Stage 3 — Gender-Bias Mitigation

Two intervention strategies are applied to neutral occupational prompts:

- Prompt-based gender conditioning across all four models.
- Embedding-based conditioning intervention for Stable Diffusion 1.5 and SDXL Turbo.

### Stage 4 — Post-Mitigation Evaluation

The same demographic assessment protocol is applied after intervention to quantify changes in perceived gender representation, bias magnitude, and statistical significance.

---

## Evaluated Models

Four publicly available text-to-image models were selected to represent distinct architectural paradigms.

| Model | Architecture | Text Encoder | Steps | Guidance |
|:---|:---|:---|---:|---:|
| Stable Diffusion 1.5 | Latent diffusion / U-Net | CLIP ViT-L | 20 | 7.5 |
| SDXL Turbo | Distilled U-Net | CLIP + OpenCLIP-G | 8 | 0.0 |
| PixArt-Σ | Diffusion Transformer | T5-XXL | 6 | 4.0 |
| Kandinsky 2.2 | Image prior + U-Net | CLIP ViT-bigG | 30 | — |

All models generate images at a resolution of **512 × 512 pixels**.

Experiments were conducted using a single NVIDIA T4 GPU with 16 GB memory. CPU offloading was employed for PixArt-Σ and Kandinsky 2.2 to accommodate their larger pipelines.

### Model Checkpoints

| Model | Public checkpoint |
|:---|:---|
| Stable Diffusion 1.5 | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| SDXL Turbo | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) |
| PixArt-Σ | [PixArt-alpha/PixArt-Sigma-XL-2-512-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) |
| Kandinsky 2.2 | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |

---

## Dataset

### Dataset Composition

The baseline dataset consists of **84,000 AI-generated images** across four models.

| Dataset component | Quantity |
|:---|---:|
| Professions | 20 |
| Prompt variants per profession | 21 |
| Unique prompts | 420 |
| Images per prompt per model | 50 |
| Images per model | 21,000 |
| Number of models | 4 |
| **Total baseline images** | **84,000** |

The baseline dataset is generated according to:

\[
N_{\mathrm{images}} = 20 \times 21 \times 50 \times 4 = 84{,}000
\]

### Prompt Design

The prompts follow a standardized structure:

```text
A [modifier] [profession] at work
```

The same set of contextual modifiers is applied to all 20 professions.

| Category | Variants | Examples |
|:---|---:|:---|
| Neutral | 1 | No modifier |
| Religion | 5 | Muslim, Christian, Hindu, Buddhist, Jewish |
| Nationality | 5 | Bangladeshi, American, German, Japanese, Nigerian |
| Positive traits | 5 | Competent, intelligent, trustworthy, successful, confident |
| Negative traits | 5 | Inexperienced, careless, unreliable, struggling, overwhelmed |
| **Total** | **21** | |

The profession inventory spans technical, healthcare, administrative, academic, and service occupations, including Engineer, Nurse, Doctor, Teacher, Scientist, CEO, Pilot, Mechanic, Pharmacist, and Software Engineer.

### Dataset Access

> **Storage notice:** The complete four-model baseline image collection exceeds **37 GB**. To keep the GitHub repository lightweight and accessible, the generated images are hosted externally on Google Drive.

| Model | Baseline images | Dataset access | Documentation |
|:---|---:|:---|:---|
| Kandinsky 2.2 | 21,000 | [Google Drive](https://drive.google.com/drive/folders/1schJt2NQSiVQLp26c2RFyykOuKdnMXKC?usp=drive_link) | [README](datasets/kandinsky22/README.md) |
| SDXL Turbo | 21,000 | [Google Drive](https://drive.google.com/drive/folders/1jFdMFxutqaVoa0QUALDKks2oAjI49fZp?usp=drive_link) | [README](datasets/sdxl_turbo/README.md) |
| PixArt-Σ | 21,000 | [Google Drive](https://drive.google.com/drive/folders/1B6CRG0M_X_WF3DOAr1HQJOsovD3kRJNY?usp=drive_link) | [README](datasets/pixart_sigma/README.md) |
| Stable Diffusion 1.5 | 21,000 | [Google Drive](https://drive.google.com/drive/folders/17zDPIt0AmLPR-7laB5nOXMPKfhnwsXD_?usp=drive_link) | [README](datasets/sd15/README.md) |

Dataset access is subject to the Google Drive sharing permissions and release policy established by the authors.

---

## Multi-Classifier Demographic Assessment

Three complementary demographic inference systems are employed to reduce reliance on a single estimator.

| Classifier | Gender | Age | Race | Role |
|:---|:---:|:---:|:---:|:---|
| DeepFace | ✓ | ✓ | ✓ | Face-based demographic estimation |
| FairFace | ✓ | ✓ | ✓ | Primary gender analysis and complementary demographic evaluation |
| Zero-shot CLIP | ✓ | ✓ | ✓ | Independent image-level demographic estimation |

DeepFace is coupled with RetinaFace for face detection. Images without a detectable face are excluded from the face-based gender and age distribution analyses.

**FairFace is used as the primary classifier for reported gender distributions**, while the other systems provide complementary assessments and reliability checks.

### Demographic Categories

**Gender:** Man and Woman, with visually unclear cases handled separately during human validation.

**Age:** 0–19, 20–39, 40–59, and 60+ years.

**Race:** Classifier-specific categories are analyzed cautiously because the evaluated systems use different taxonomies and exhibit lower reliability for racial classification.

All demographic labels represent perceived visual attributes inferred from generated imagery, not verified personal identities.

### Bias Quantification

Gender imbalance is measured relative to a 50/50 parity reference.

For a prompt \(t\), the signed gender skew is:

\[
S(t)=P_F(t)-0.5
\]

The corresponding absolute deviation is:

\[
d(t)=|P_F(t)-0.5|
\]

Across a prompt collection \(T\), Mean Absolute Deviation (MAD) is defined as:

\[
\mathrm{MAD}(T)=\frac{1}{|T|}
\sum_{t\in T}|P_F(t)-0.5|
\]

Lower MAD values indicate closer alignment with the selected parity reference.

---

## Human Validation

To evaluate demographic classification reliability, two human annotators jointly reviewed a stratified sample of **200 baseline-generated images**, with 50 images selected from each generative model.

The annotators established consensus reference labels without access to the generating prompts or automated classifier predictions.

The validation sample included:

- 80 male-presenting images
- 80 female-presenting images
- 40 visually gender-ambiguous images

Classification reliability was assessed using accuracy and Cohen's kappa (κ).

### Gender Classification Validation

| Classifier | Accuracy | Cohen's κ |
|:---|---:|---:|
| DeepFace | 95.6% | 0.912 |
| FairFace | 98.1% | 0.962 |
| Zero-shot CLIP | 100.0% | 1.000 |

These results indicate high agreement with human consensus labels for the evaluated gender-classification subset.

Age and racial classification were also validated, with comparatively lower agreement, motivating cautious interpretation of these secondary demographic findings.

---

## Gender-Bias Mitigation

Two complementary debiasing approaches are evaluated.

### 1. Prompt-Based Debiasing

Prompt-based mitigation explicitly specifies gender in the input prompt.

**Original prompt:**

```text
A photo of a teacher
```

**Gender-conditioned prompts:**

```text
A photo of a male teacher
A photo of a female teacher
```

For each occupation–model combination, 25 male-conditioned and 25 female-conditioned images are generated.

This method does not require modifications to model parameters and is evaluated across all four T2I models.

### 2. Embedding-Based Debiasing

Embedding-based mitigation modifies the text-conditioning representation along a gender-associated direction.

The intervention is represented as:

\[
e_M=e_o+\alpha_m g_m
\]

\[
e_F=e_o-\alpha_m g_m
\]

where:

- \(e_o\) is the original occupational text embedding.
- \(g_m\) is the model-specific gender direction.
- \(\alpha_m\) controls intervention strength.

The study uses model-specific intervention strengths:

| Model | Intervention strength |
|:---|---:|
| Stable Diffusion 1.5 | 120 |
| SDXL Turbo | 65 |

Embedding-based mitigation is evaluated only on Stable Diffusion 1.5 and SDXL Turbo because PixArt-Σ and Kandinsky 2.2 use different conditioning mechanisms.

---

## Key Findings

### Baseline Gender Representation

All four models exhibit substantial male-skewed occupational representations.

| Model | Male (%) | Female (%) |
|:---|---:|---:|
| Kandinsky 2.2 | 93.1 | 6.9 |
| SDXL Turbo | 93.0 | 7.0 |
| PixArt-Σ | 86.8 | 13.2 |
| Stable Diffusion 1.5 | 84.3 | 15.7 |
| **Overall** | **89.6** | **10.4** |

These percentages are calculated over images with detected faces in the full baseline corpus.

### Prompt-Based Debiasing Results

The following results refer specifically to the neutral occupational prompt subset.

| Model | Female before (%) | Female after (%) |
|:---|---:|---:|
| Kandinsky 2.2 | 5.3 | 41.5 |
| SDXL Turbo | 6.7 | 42.7 |
| PixArt-Σ | 14.8 | 50.8 |
| Stable Diffusion 1.5 | 17.5 | 49.1 |

All four models exhibit statistically significant changes in perceived gender representation after prompt-based intervention (**p < 0.001**).

### Main Observations

- **Cross-model bias:** Male-skewed occupational imagery appears across all four architectures.
- **Contextual sensitivity:** Religion- and nationality-conditioned prompts affect observed gender distributions.
- **Mitigation effectiveness:** Prompt-based interventions substantially improve aggregate gender balance.
- **Architecture dependence:** Embedding-based interventions require access to and modification of model-specific text representations.
- **Residual disparities:** Some professions remain imbalanced after mitigation.
- **Secondary demographic patterns:** Younger adult and White-presenting representations predominate, although age and race estimates require cautious interpretation.

> **Important:** Improved aggregate gender parity does not necessarily establish fairness across all professions or demonstrate that underlying occupational stereotypes have been eliminated.

---

## Repository Structure

The repository is organized according to the major stages of the experimental workflow.

```text
T2I-Demographic-Bias-Assessment-and-Mitigation/
│
├── README.md
├── .gitignore
│
├── image_generation/
│   ├── kandinsky22_batch_generator.py
│   ├── pixart_sigma_batch_generator.py
│   ├── sd15_batch_generator.py
│   └── sdxl_turbo_batch_generator.py
│
├── demographic_bias_assessment/
│   └── [Demographic evaluation notebooks]
│
├── human_validation/
│   └── [Human validation notebooks and annotations]
│
├── gender_bias_mitigation/
│   └── [Prompt-based and embedding-based notebooks]
│
└── datasets/
    ├── kandinsky22/
    │   └── README.md
    ├── sdxl_turbo/
    │   └── README.md
    ├── pixart_sigma/
    │   └── README.md
    └── sd15/
        └── README.md
```

*The structure represents the intended organization; individual files may be added as the repository is finalized.*

---

## Reproducibility

### Requirements

The experimental workflow uses Python and GPU-supported deep learning libraries, including:

- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- DeepFace
- FairFace
- OpenAI CLIP
- RetinaFace
- NumPy, pandas, and Matplotlib

Exact dependency versions should be taken from the corresponding notebooks and generation scripts.

### Recommended Reproduction Workflow

1. **Prepare the environment.** Install the dependencies specified by the experimental scripts.
2. **Construct the prompt set.** Generate the 420 occupational prompts using the 20 professions and 21 variants.
3. **Generate baseline images.** Run the model-specific generation scripts with the documented sampling configurations.
4. **Assess demographic representation.** Apply the demographic classification pipeline to generated outputs.
5. **Validate classifier reliability.** Compare automated predictions with the human consensus annotations.
6. **Apply gender-bias mitigation.** Run the prompt-based and supported embedding-based interventions.
7. **Re-evaluate the outputs.** Apply the same classification protocol and compare baseline versus post-mitigation statistics.

All experiments reported in the manuscript were conducted using an NVIDIA T4 GPU with 16 GB of memory in Kaggle.

---

## Limitations and Responsible Use

This repository supports research on representational fairness in generative AI. Several limitations should be considered:

- Gender is operationalized through binary perceived-gender categories for the main quantitative analysis.
- Automated age and race inference is subject to classification error and taxonomy limitations.
- The benchmark uses controlled English-language occupational prompts.
- Embedding-based mitigation is evaluated on only two of the four architectures.
- Numerical parity is an analytical reference, not a comprehensive definition of demographic fairness.
- Results should not be generalized to every text-to-image model or cultural context.

The datasets contain AI-generated imagery. Demographic classifications should not be interpreted as self-identified or verified personal characteristics.

---

## Authors

| Author | Affiliation | Role |
|:---|:---|:---|
| **Adnan Faisal** | Chittagong University of Engineering and Technology (CUET), Bangladesh | Research author |
| **Shiti Chowdhury** | Chittagong University of Engineering and Technology (CUET), Bangladesh | Research author |
| **Ajwad Abrar** | Islamic University of Technology (IUT), Bangladesh | Corresponding author |

### GitHub Contributors

- [**SHITICHY21**](https://github.com/SHITICHY21)
- [**AJFaisal002**](https://github.com/AJFaisal002)

GitHub contributions and manuscript authorship are documented separately.

---

## Citation

If you use this repository, its code, or its generated image datasets in academic research, please acknowledge the authors and cite the associated manuscript once its bibliographic details are available.

**Manuscript status:** Publication details to be added.

---

## Acknowledgments

The authors acknowledge the publicly available generative models and demographic assessment tools that enabled this study, including Stable Diffusion, SDXL Turbo, PixArt-Σ, Kandinsky, DeepFace, FairFace, and CLIP.

---

<div align="center">

### Advancing Responsible and Fair Text-to-Image Generation

**84,000 Images · 420 Prompts · 4 Generative Models · 3 Demographic Classifiers · 2 Debiasing Strategies**

Developed for academic research on demographic representation, fairness evaluation, and bias mitigation in generative artificial intelligence.

</div>
