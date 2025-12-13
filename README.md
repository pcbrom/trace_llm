# pcbrom/trace-llm: TraCE-LLM

[![DOI](https://zenodo.org/badge/1115968829.svg)](https://doi.org/10.5281/zenodo.17925148)

Pedro Carvalho Brom. (2025). pcbrom/trace-llm: datasets. Zenodo. https://doi.org/10.5281/zenodo.17925148

```bibtex
@misc{brom_2025_tracellm,
  author       = {Brom, Pedro Carvalho},
  title        = {{pcbrom/trace-llm: datasets}},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17925148},
  url          = {https://doi.org/10.5281/zenodo.17925148}
}
```

# Datasets Used in the TraCE-LLM Study

The datasets provided correspond to the evaluation outputs and analytical views generated during the empirical study described in the article *"The Adversarial Compensation Effect: Identifying Hidden Instabilities in Large Language Model Evaluation"*.

These datasets are the structured results of the **TraCE-LLM** protocol, which was designed to assess latent behavioral traits of Large Language Models (LLMs) through a multidimensional rubric. The evaluation framework measured two primary dimensions:

* **Depth of Reasoning (DoR)** – logical structure and coherence of the model's reasoning.
* **Originality (ORI)** – novelty and creativity of the model's output.

The study involved five LLMs, three benchmark datasets (MMLU, ARC-Challenge, HellaSwag), three prompt styles (Naive, Chain-of-Thought, Adversarial), and five independent replications per condition, totaling **10,125 observations**【25†source】.

---

## Relation of Datasets to the Study

1. **`cot_classification_df.csv`**

   * Contains CoT (Chain-of-Thought) classification results for a balanced subset of items.
   * Includes raw scores assigned by model-judges for DoR and ORI, the prompt type, and the final category assigned to the CoT.
   * Directly used to validate research hypotheses such as *Asymmetric Trait Stability* (H1) and *Partial Trait Separability* (H4).
2. **`cot_classification_df_hit1.csv`**

   * Same structure as above, but restricted to instances where the model produced the correct answer (hit = 1).
   * Used to study phenomena such as *Compressed Reasoning* (H5), where correct answers had low DoR scores.
3. **`df_outliers_view.csv`**

   * Focused on identifying and categorizing outlier responses in the CoT classification dataset.
   * Supports the detection of instability patterns, particularly under adversarial prompts (*Adversarial Collapse*, H2).
4. **`df_outliers_view_hit1.csv`**

   * Outlier analysis limited to the correct-answer subset.
   * Useful to highlight cases where a correct answer was accompanied by extreme ORI or DoR values, revealing instability masked by accuracy.
5. **`tidydata2cmr.csv`**

   * Canonical “tidy” table used to compute **classical machine learning metrics** (accuracy, precision, recall, F1).
   * Each row corresponds to a combination of:
     * dataset source (`ARC`, `MMLU`, `HellaSwag`),
     * item text,
     * ground-truth answer,
     * replication index (`r`),
     * model identity,
     * prompt type (`naive`, `cot`, `adversarial`),
     * model answer and correctness flags.
   * Serves as the input for the statistical analyses in `classical_metrics.ipynb`, including confusion matrices, prompt-wise comparisons and non-parametric tests (Friedman + Nemenyi).

---

## Structure of the Datasets

Each dataset shares a common set of key columns:

| Column              | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `source`          | Origin or source of the evaluated data                                      |
| `item`            | Unique identifier for the evaluated item                                    |
| `model`           | Name of the AI model used for the evaluation                                |
| `prompt_type`     | Prompt style used (Naive, CoT, Adversarial)                                 |
| `CoT`             | Chain-of-Thought reasoning text or ID                                       |
| `gr_dor_*`        | Depth of Reasoning score by each model-judge                                |
| `gr_ori_*`        | Originality score by each model-judge                                       |
| `outlier_columns` | Columns flagged as containing outlier values                                |
| `CoT Category`    | (Only in classification files) Category assigned based on rubric evaluation |

---

## Usage in the Article's Analyses

* **Variability and Stability Analysis:** By comparing DoR and ORI distributions, the study confirmed that reasoning depth is inherently more stable than originality.
* **Outlier Detection:** Outlier datasets enabled the quantification of volatility in latent traits, especially under adversarial stress.
* **Compressed Reasoning Identification:** The `*_hit1.csv` datasets isolated cases where a correct answer had superficial reasoning, supporting the claim that accuracy alone is insufficient.
* **Model Agreement and Baselines:** All datasets contributed to computing the *median ensemble baseline*, which showed higher stability and alignment with high-performing models than any single judge.

---

## Repository Structure

- `data/`
  - `cot_classification_df.csv`, `cot_classification_df_hit1.csv` – rubric-based CoT classifications and trait scores.
  - `df_outliers_view.csv`, `df_outliers_view_hit1.csv` – filtered views emphasizing extreme trait configurations and instability patterns.
  - `tidydata2cmr.csv` – tidy table of model predictions for computing classical metrics (see above).
- `figures/`
  - Publication-ready figures derived from the notebooks (see **Notebook Guide**), such as:
    - `pca_plot.png`, `pca_and_dendrogram_plot.png`,
    - `kendall_correlogram.png`, `kendall_correlogram_combined.png`,
    - `kendall_correlogram_combined_source_prompt.png`,
    - `hierarchical_clustering_dendrogram.png`,
    - `comparison_deep_of_reasoning_and_originality_scores.png`,
    - `faceted_scores_by_hit_status.png`,
    - `cmr_weighted_f1_scores.png`.
- Root notebooks (analysis & pipeline):
  - `first_step_and_sample.ipynb`
  - `second_step.ipynb`
  - `aed_and_model.ipynb`
  - `classical_metrics.ipynb`

---

## Notebook Guide

This section summarizes the purpose of each notebook and how it fits in the TraCE-LLM pipeline.The full experimental pipeline has three conceptual stages:

1. **Item preparation and prompt generation** (multiple-choice items from ARC, MMLU, HellaSwag; prompts: naive, CoT, adversarial).
2. **Model runs and internal evaluation** (LLMs answer questions and generate CoT; judge-models score Depth of Reasoning and Originality).
3. **Analytical views and statistical tests** (trait stability, outliers, classical metrics).

### `first_step_and_sample.ipynb`

**Goal:** Build the unified item dataset and generate prompts and responses for the first experimental step.

- Loads the original benchmark datasets:
  - ARC-Challenge (`db_arc`),
  - MMLU (`db_mmlu`),
  - HellaSwag (`db_hellaswag`).
- Normalizes them into a common multiple-choice format:
  - Creates per-dataset tables with `item` (question + options) and `answer` (A/B/C/D).
  - Concatenates them into `data/all_sources.csv` with a `source` column (`ARC`, `MMLU`, `HellaSwag`).
- Draws a balanced sample across sources and writes `data/sample.csv`, which is used as the evaluation pool.
- Defines prompt templates for three regimes:
  - **`cot`** – asks the model to solve the item with step-by-step reasoning and to return a JSON object with `CoT`, `answer`, and `justification`.
  - **`naive`** – asks only for the answer + justification in JSON, without explicit reasoning steps.
  - **`adversarial`** – asks for step-by-step reasoning while warning that the question may be adversarial or ambiguous; allows the model to pick an `E` option when none of A–D is adequate.
- Configures client objects for the different LLM APIs (OpenAI, Anthropic, xAI Grok, DeepSeek) using environment variables loaded from a `.env` file.
- Implements `step_one(...)` and a parallelized loop (`process_model`) to:
  - Call each model with each prompt type over all sampled items.
  - Repeat each condition 5 times (`r = 0..4`), to capture intra-condition variability.
  - Store raw JSON outputs (`output` column) along with metadata (`source`, `item`, `answer`, `r`, `model`, `prompt_type`).
- Produces the raw and cleaned first-step result tables:
  - `data/res_step_one.csv` – full raw outputs (including models not used in the final analyses).
  - `data/res_step_one_clean.csv` – filtered version without deprecated configurations (`model_deepseek_reasoner`).

### `second_step.ipynb`

**Goal:** Apply the zero-shot semantic interval rubric using model-internal evaluation (judge-models scoring DoR and ORI).

- Starts with a markdown description:
  - “Second Step: Zero-Shot Semantic Interval Rubric with model-internal evaluation”.
- Loads API keys from the `.env` file and configures multiple judge-model clients:
  - GPT-4.1 family (`gpt-4o-mini`, `gpt-4.1-nano`),
  - Claude 3.5 Haiku,
  - Grok-3-mini (xAI),
  - DeepSeek Chat.
- Defines a `models` registry and the helper functions:
  - `step_two(model_name, prompt)` – generic interface that calls the appropriate API and returns the judge’s textual output.
  - `step_two_recovered(model_name, prompt)` – self-contained variant that reconstructs the client from `model_name` (useful for resumptions).
- Uses a tidy table of CoT responses (derived from the first step) with columns such as:
  - `source`, `item`, `answer`, `r`, `model`, `prompt_type`,
  - `model_answer`, `hit`, `model_alternative_answer`, `hit_alternative`,
  - `CoT`, `cot_steps`.
- For each CoT, prompts the judge-models with the rubric and collects:
  - DoR scores (`gr_dor_*`) and justifications,
  - ORI scores (`gr_ori_*`) and justifications,
  - criteria labels (`criterion_x`, `criterion_y`), depending on the specific rubric variant.
- Produces an enriched evaluation dataset (stored as a parquet file in the original workflow) with one row per (item, model, prompt_type, replication) and trait scores from each judge-model.

### `aed_and_model.ipynb`

**Goal:** Analyze trait scores using the model ensemble as a reference and derive latent-structure visualizations.

- Loads the trait-augmented dataset (e.g., `data/tidy_data_2_aed_model.parquet`) which contains:
  - Original columns from the CoT evaluation,
  - DoR and ORI grades from each judge-model (`gr_dor_*`, `gr_ori_*`),
  - rubric-based criteria (`criterion_x`, `criterion_y`).
- Computes ensemble statistics:
  - `gr_dor_median`, `gr_ori_median` – median scores across judge-models for each observation.
  - Per-model summary table (mean, std, median, min, max, coefficient of variation) for both DoR and ORI, including the median ensemble.
- Generates plots stored under `figures/`, including:
  - PCA projections and hierarchical clustering of models based on trait profiles (`pca_plot.png`, `pca_and_dendrogram_plot.png`, `hierarchical_clustering_dendrogram.png`).
  - Kendall correlation heatmaps (`kendall_correlogram.png`, `kendall_correlogram_combined.png`, `kendall_correlogram_combined_source_prompt.png`).
  - Comparative visualizations of Depth of Reasoning vs Originality (`comparison_deep_of_reasoning_and_originality_scores.png`).
- Provides the basis for claims about:
  - ensemble stability vs individual judge variability,
  - trait separability and inter-model similarity.

### `classical_metrics.ipynb`

**Goal:** Compute and compare classical evaluation metrics over the TraCE-LLM prediction table.

- Loads `data/tidydata2cmr.csv` and constructs a `model_prompt_type` column (`model` + `_` + `prompt_type`).
- Defines helper functions to:
  - compute overall accuracy, macro/weighted precision, recall and F1,
  - build confusion matrices,
  - aggregate metrics by:
    - prompt type (`cot`, `naive`, `adversarial`),
    - dataset source (`ARC`, `MMLU`, `HellaSwag`),
    - model, and model–prompt combinations.
- Produces:
  - overall metrics table and confusion matrix,
  - prompt-wise aggregated metrics and class-level F1 matrices,
  - `cmr_weighted_f1_scores.png` (comparison of weighted F1 by condition),
  - faceted views of hit rates (`faceted_scores_by_hit_status.png`).
- Runs a **Friedman test** followed by **Nemenyi post-hoc** comparisons on class-level F1-scores to quantify differences between prompt regimes.

---

## Recommended Applications

* **LLM Behavior Auditing** – Detect behavioral instability not visible through accuracy scores.
* **Prompt Robustness Testing** – Compare performance and stability across Naive, CoT, and Adversarial prompts.
* **Outlier Analysis** – Identify unusual patterns for further investigation.
* **Multidimensional Benchmarking** – Incorporate rubric-based traits (DoR, ORI) in model evaluations instead of relying on single scalar scores.

---

## Reproducibility Notes

- To re-run the notebooks that query external APIs (`first_step_and_sample.ipynb`, `second_step.ipynb`), create a `.env` file with the appropriate keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`) and adjust the path if needed.
- Some intermediate parquet/CSV files used in the notebooks may be distributed via the Zenodo release referenced above rather than directly in this repository; the notebook descriptions here document how those files are produced and consumed within the TraCE-LLM pipeline.
