# pcbrom/trace-llm: TraCE-LLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17929531.svg)](https://doi.org/10.5281/zenodo.17929531)

Pedro Carvalho Brom, Di Oliveira, V., & Weigang, L. (2025). TraCE-LLM: Evaluation datasets and pipeline (v2.2) (2.2). Zenodo. https://doi.org/10.5281/zenodo.17929531

```bibtex
@misc{brom_oliveira_weigang_2025_tracellm,
  author       = {Brom, Pedro Carvalho and Di Oliveira, V. and Weigang, L.},
  title        = {{TraCE-LLM: Evaluation datasets and pipeline (v2.2)}},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {2.2},
  doi          = {10.5281/zenodo.17929531},
  url          = {https://doi.org/10.5281/zenodo.17929531}
}
```

# Overview

This repository contains the core assets used in the TraCE-LLM study described in the article  
*“The Adversarial Compensation Effect: Identifying Hidden Instabilities in Large Language Model Evaluation”*.

The project is centered on the **TraCE-LLM protocol**, which measures latent behavioral traits of Large Language Models (LLMs) using a multidimensional rubric with two primary dimensions:

- **Depth of Reasoning (DoR)** – logical structure and coherence of the model's reasoning.
- **Originality (ORI)** – novelty and creativity of the model's output.

The full experimental pipeline (data generation, judgment, and analysis) is documented in the notebooks in this repository and in the datasets released on Zenodo. This repository itself only tracks a **subset of the data files** (benchmark test splits) and all **analysis / pipeline notebooks**.

---

## Repository Structure (Tracked Files)

At the root of the repository:

- `first_step_and_sample.ipynb` – first-step pipeline notebook (prompt engineering and sampling).
- `second_step.ipynb` – second-step rubric evaluation notebook (model-internal evaluation).
- `aed_and_model.ipynb` – analysis of ensemble vs individual models and trait structure.
- `classical_metrics.ipynb` – computation of classical metrics (accuracy, F1, etc.) on tidy TraCE-LLM outputs.
- `LICENSE` – project license.
- `README.md` – this file.

Under `data/` (benchmark test sets used as sources for TraCE-LLM items):

- `data/ARC_test/test-00000-of-00001.parquet`  
  Multiple-choice ARC-Challenge style items with:
  - `id`, `question`, `choices`, `answerKey`.
- `data/MMLU_test/test-00000-of-00001.parquet`  
  MMLU test split with:
  - `question`, `subject`, `choices`, `answer`.
- `data/hellaswag_test/hellaswag_val.jsonl`  
  HellaSwag validation split in JSONL format, containing:
  - `ind`, `activity_label`, `ctx_a`, `ctx_b`, `ctx`, `split`, `split_type`, `label`, `endings`, `source_id`.

> The *derived* TraCE-LLM datasets (e.g., CoT classifications, outlier views, tidy prediction tables) are **not stored in this Git repository**. They are available via the Zenodo record and can also be regenerated locally by running the notebooks, as described below.

No `figures/` directory is tracked in this repo; figures referenced in the paper are generated on demand by the analysis notebooks.

---

## Notebooks and Their Roles

This section documents what each notebook in the repository does and how it relates to the datasets.

### `first_step_and_sample.ipynb`

**Goal:** Build unified multiple-choice items from the benchmark datasets and define the first-step prompting scheme for LLMs.

- Loads benchmark test splits from `data/`:
  - ARC-Challenge items from `data/ARC_test/test-00000-of-00001.parquet`.
  - MMLU items from `data/MMLU_test/test-00000-of-00001.parquet`.
  - HellaSwag items from `data/hellaswag_test/hellaswag_val.jsonl`.
- Normalizes each dataset to a common schema, conceptually producing tables with:
  - `source` (ARC, MMLU, HellaSwag),
  - `item` (question/context plus options),
  - `answer` (A/B/C/D).
- Demonstrates how to construct balanced samples across sources for evaluation (e.g., equal number of items per dataset).
- Defines prompt templates for three prompting regimes:
  - **`cot`** – chain-of-thought prompts that request step-by-step reasoning and JSON output with fields such as `CoT`, `answer`, `justification`.
  - **`naive`** – straightforward question-answer prompts with JSON output containing `answer` and `justification`.
  - **`adversarial`** – prompts that explicitly ask the model to question its own assumptions and, if necessary, choose an extra option `E` with an alternative answer.
- Configures API clients for external LLMs via environment variables (see *Reproducibility* below) and sketches the loop that:
  - iterates over models and prompts,
  - applies the prompts to sampled items,
  - collects JSON outputs for further processing.

The notebook is written to be re-runnable with the available benchmark files and user-provided API keys; it **does not commit any generated CSV/Parquet files** to this repository.

### `second_step.ipynb`

**Goal:** Apply the zero-shot semantic interval rubric using model-internal evaluation (LLMs as judges of DoR and ORI).

- Describes the **Second Step** of TraCE-LLM: judge-models score each Chain-of-Thought on:
  - Depth of Reasoning (DoR),
  - Originality (ORI),
  according to a semantic interval rubric.
- Loads API keys from a local `.env` file and configures judge-clients for:
  - GPT-4 family,
  - Claude 3.5 Haiku,
  - xAI Grok,
  - DeepSeek Chat.
- Maintains a `models` registry and helper functions like:
  - `step_two(model_name, prompt)` – calls the appropriate client and returns the judge’s textual output.
  - `step_two_recovered(model_name, prompt)` – a self-contained variant that reconstructs the client from `model_name`.
- Expects as input a tidy table of CoT outputs from the first step, with columns conceptually including:
  - `source`, `item`, `answer`, `r`,
  - `model`, `prompt_type`,
  - `model_answer`, `hit`,
  - `model_alternative_answer`, `hit_alternative`,
  - `CoT`, `cot_steps`.
- For each row, constructs a rubric prompt, queries judge-models and parses their outputs into:
  - DoR scores (`gr_dor_*`) and explanations,
  - ORI scores (`gr_ori_*`) and explanations,
  - optional rubric criteria labels.

The resulting enriched tables (with `gr_dor_*` and `gr_ori_*` columns) are **generated at runtime** and are not version-controlled here; they are part of the Zenodo distribution and can be reconstructed by re-running this notebook with the appropriate inputs.

### `aed_and_model.ipynb`

**Goal:** Analyze trait scores across models and ensembles, and study the latent structure of DoR/ORI.

- Assumes access to a trait-augmented dataset produced by the second step, containing:
  - base columns from the CoT evaluation (source, item, model, prompt type, etc.),
  - judge-model DoR and ORI scores (`gr_dor_*`, `gr_ori_*`),
  - optionally, rubric criteria (`criterion_x`, `criterion_y`).
- Computes ensemble statistics such as:
  - per-observation medians (`gr_dor_median`, `gr_ori_median`) across judge-models,
  - per-model descriptive statistics (mean, std, median, min, max, coefficient of variation) for DoR and ORI.
- Performs multivariate analyses of trait profiles, e.g.:
  - PCA projections of models in trait space,
  - hierarchical clustering of models,
  - correlation analyses (e.g., Kendall correlation between models or between traits).
- Generates figures used in the article (PCA, dendrograms, correlograms, comparisons of DoR vs ORI).  
  These figures are **not stored** in this Git repo by default but can be saved locally when running the notebook.

### `classical_metrics.ipynb`

**Goal:** Compute and compare classical evaluation metrics over TraCE-LLM prediction tables.

- Expects as input a tidy table of multiple-choice predictions with, at minimum:
  - true answer labels (A/B/C/D),
  - model predictions,
  - metadata such as `model`, `prompt_type`, and `source`.
- Builds derived columns like `model_prompt_type` (e.g., `"model_gpt_4o_mini_cot"`).
- Implements helper routines to:
  - compute overall accuracy, macro and weighted precision, recall and F1,
  - produce confusion matrices (per class and aggregated),
  - aggregate metrics by:
    - prompt type (`cot`, `naive`, `adversarial`),
    - dataset source (`ARC`, `MMLU`, `HellaSwag`),
    - model or model–prompt combinations.
- Performs statistical comparisons of prompt regimes, including:
  - Friedman tests on class-level F1 scores,
  - Nemenyi post-hoc tests for pairwise comparisons.
- Optionally creates visualization-ready tables and plots (e.g., weighted F1 by condition, facet plots by hit status); these artifacts are generated at runtime and are not checked into this repository.

---

## How This Repo Relates to the Zenodo Datasets

Because this Git repository only contains benchmark test splits and notebooks, the complete TraCE-LLM datasets described in the paper are hosted on Zenodo:

- CoT classification tables (with DoR/ORI traits and categories),
- outlier views (including hit = 1 subsets),
- tidy prediction tables used for classical metrics.

When working locally, you can:

1. Use the benchmark test files in `data/` to reconstruct the item pool.  
2. Run `first_step_and_sample.ipynb` to generate model responses under different prompt regimes.  
3. Run `second_step.ipynb` to obtain DoR and ORI scores from judge-models.  
4. Use the resulting tables as input to `aed_and_model.ipynb` and `classical_metrics.ipynb` to reproduce analyses and figures.

The naming of intermediate CSV/Parquet files in the notebooks is part of the *example pipeline* and may differ from the file layout in the Zenodo deposit. The README therefore only documents **files that are actually present in this repository**.

---

## Reproducibility Notes

- **Environment variables:**  
  Notebooks that query external APIs (`first_step_and_sample.ipynb`, `second_step.ipynb`) expect a `.env` file (outside this repo or at a user-specified path) defining keys such as:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `XAI_API_KEY`
  - `DEEPSEEK_API_KEY`

- **Python environment:**  
  The notebooks assume a recent Python 3 version with standard data/ML libraries (e.g., `pandas`, `numpy`, `scikit-learn`, plotting libraries) and the respective API client libraries for the LLM providers.

- **Generated artifacts:**  
  All CSV/Parquet outputs, tidy datasets, and figures produced by the notebooks are generated locally when you run them and are not committed to this repository. To obtain the exact datasets used in the article, use the Zenodo record referenced above.

---

## Suggested Uses

- **LLM behaviour auditing:**  
  Study stability and variability of model reasoning beyond simple accuracy metrics.

- **Prompt robustness testing:**  
  Compare model performance and stability across naive, chain-of-thought, and adversarial prompting strategies.

- **Trait-based benchmarking:**  
  Use Depth of Reasoning and Originality as explicit evaluation dimensions alongside classical metrics.

- **Outlier and instability analysis:**  
  Investigate cases where models give correct answers with shallow reasoning, or where adversarial prompts cause abrupt changes in behaviour.
