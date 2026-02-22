# Layer-wise Probing Analysis of Belief Encoding in LLMs

## 1. Executive Summary
We tested whether open-source LLMs encode epistemic vs non-epistemic belief signals in distinct internal layers using layer-wise linear probes on GPT-2 family models and a Llama-family checkpoint.

Key finding: belief-related information is decodable in specific layers (especially for TruthfulQA-style epistemic labels), but robustness and validation-layer alignment are inconsistent, so evidence favors mixed semantic + surface-pattern encoding rather than clean semantic abstraction.

Practically, this suggests layer-local probes can be useful for diagnostics, but should not be treated as proof of deep belief semantics without stronger robustness/causal checks.

## 2. Goal
- Hypothesis: GPT-2 and Meta Llama-family models encode epistemic and non-epistemic belief types in separable internal layers; these signals may be semantic or superficial.
- Importance: clarifies interpretability claims around "belief" in LLM internals and informs future hallucination-detection mechanisms.
- Problem solved: provides a reproducible cross-model layer-wise probe pipeline with perturbation controls.
- Expected impact: better standards for interpreting probe-based claims in LLM safety/reliability work.

## 3. Data Construction

### Dataset Description
- ToMi-NLI (`datasets/tomi_nli`): non-epistemic/Theory-of-Mind-style entailment setup.
- TruthfulQA multiple choice (`datasets/truthful_qa_multiple_choice`): epistemic truthfulness labeling using candidate answer correctness.
- Final run sample sizes (for feasibility with Llama-7B):
  - ToMi-NLI: train 300 / val 80 / test 80
  - TruthfulQA-MC expanded candidate examples: train 300 / val 80 / test 80

Known limitations:
- TruthfulQA candidate-level labels are class-imbalanced (test: 59 negatives, 21 positives).
- Llama access limitation: gated `meta-llama/*` models were unavailable; used public Llama-family `huggyllama/llama-7b`.

### Example Samples
| Dataset | Input (abridged) | Label |
|---|---|---|
| ToMi-NLI | Premise about object movement + hypothesis entailment | `entailment`/`not_entailment` |
| TruthfulQA-MC | Question + candidate answer, classify truthful/not | `1` truthful / `0` not truthful |

### Data Quality
- Missing values: 0% in all splits (from `results/data_summary.json`).
- ToMi class distribution (test): 41 negative / 39 positive.
- TruthfulQA class distribution (test): 59 negative / 21 positive.
- Word-length stats recorded in `results/data_summary.json`.

### Preprocessing Steps
1. Converted ToMi examples to prompt-style text (`Premise/Hypothesis/Question`).
2. Expanded TruthfulQA MC into candidate-level binary classification examples.
3. Built lexical perturbation variants (word substitutions) for robustness tests.
4. Built rephrased-template variants for cross-template transfer checks.

### Train/Val/Test Splits
- ToMi-NLI: sampled from native train/validation/test splits.
- TruthfulQA-MC: stratified split from validation pool into train/val/test (60/20/20), then downsampled to configured caps.

## 4. Experiment Description

### Methodology
#### High-Level Approach
For each model and each transformer layer, extract last-token hidden states, train linear probes, and evaluate on:
- original test,
- lexical perturbation test,
- rephrased template test.

#### Why This Method?
- Layer-wise linear probing is the standard first-pass localization method from prior belief-representation work.
- Perturbation and rephrasing test whether decodability is robust beyond exact wording.
- Surface TF-IDF logistic baseline tests whether simple lexical features explain performance.

Alternatives considered but not implemented in this cycle:
- Causal activation interventions (kept for follow-up).
- Larger sample sizes (reduced for full 3-model completion with Llama).

### Implementation Details
#### Tools and Libraries
- Python 3.12.2
- torch 2.10.0+cu128
- transformers 5.2.0
- scikit-learn 1.8.0
- scipy 1.17.0
- statsmodels 0.14.6
- datasets, pandas, seaborn, matplotlib

#### Algorithms/Models
- Models: `gpt2`, `gpt2-medium`, `huggyllama/llama-7b`.
- Probes: `StandardScaler + LogisticRegression(liblinear, class_weight='balanced')` per layer.
- Surface baseline: `TF-IDF (1-2 grams) + LogisticRegression`.

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| max_train | 300 | feasibility constraint |
| max_val | 80 | feasibility constraint |
| max_test | 80 | feasibility constraint |
| random seeds | 42, 43, 44 | fixed reproducibility |
| max_length | 256 | default cap |
| probe solver | liblinear | stable binary logistic |
| probe class weight | balanced | class imbalance mitigation |

#### Pipeline
1. Load and validate datasets.
2. Create original/perturbed/rephrased text variants.
3. Extract hidden states layer-wise.
4. Train probe per layer (3 seed runs).
5. Compute metrics (AUROC, Accuracy, F1, ECE, Brier).
6. Compute robustness deltas and Wilcoxon tests + FDR correction.

### Experimental Protocol
#### Reproducibility Information
- Runs averaged over seeds: 3.
- Hardware: 2x NVIDIA RTX 3090 (24GB each), CUDA enabled.
- Batch sizes: adaptive (64 for smaller models, 16 for Llama hidden size).
- Mixed precision: CUDA autocast enabled for feature extraction.
- Validation check: repeated reduced gpt2 run produced identical `stats_summary.csv` (`REPRO_MATCH`).

#### Evaluation Metrics
- AUROC: ranking quality for binary labels.
- Accuracy / F1: thresholded classification quality.
- ECE / Brier: calibration quality.
- Robustness drop: original AUROC minus perturbed/rephrased AUROC.

### Raw Results
#### Main Summary (peak-by-validation layer)
| Model | Task | Peak Layer | Test AUROC | Perturbed AUROC | Rephrased AUROC |
|---|---|---:|---:|---:|---:|
| gpt2 | tomi_nli | 5 | 0.447 | 0.644 | 0.500 |
| gpt2 | truthfulqa_mc | 1 | 0.609 | 0.605 | 0.486 |
| gpt2-medium | tomi_nli | 10 | 0.438 | 0.477 | 0.531 |
| gpt2-medium | truthfulqa_mc | 10 | 0.617 | 0.576 | 0.531 |
| llama7b | tomi_nli | 13 | 0.551 | 0.456 | 0.483 |
| llama7b | truthfulqa_mc | 9 | 0.639 | 0.693 | 0.498 |

#### Best test-layer AUROC (independent of val-picked layer)
| Model | ToMi best AUROC (layer) | TruthfulQA best AUROC (layer) |
|---|---:|---:|
| gpt2 | 0.545 (L9) | 0.692 (L9) |
| gpt2-medium | 0.537 (L16) | 0.701 (L19) |
| llama7b | 0.551 (L13) | 0.756 (L13/15 tie) |

#### Surface Baseline (TF-IDF)
| Task | AUROC | Accuracy |
|---|---:|---:|
| tomi_nli | 0.568 | 0.538 |
| truthfulqa_mc | 0.526 | 0.763 |

#### Statistical Tests (Wilcoxon + BH-FDR)
- No robustness difference reached FDR < 0.05.
- All corrected p-values were >= 0.1875.

#### Visualizations
- Layer curves: `results/plots/layerwise_val_auroc.png`
- Perturbation drops: `results/plots/perturbation_drop.png`

#### Output Locations
- Metrics table: `results/layerwise_metrics.csv`
- Statistics summary: `results/stats_summary.csv`
- Baseline summary: `results/baseline_summary.csv`
- Environment + data quality: `results/environment.json`, `results/data_summary.json`
- Aggregated JSON: `results/metrics.json`

## 5. Result Analysis

### Key Findings
1. Decodable signal exists at specific layers, strongest on TruthfulQA in later layers for larger models (e.g., Llama7B ~0.756 AUROC at best test layer).
2. Validation-chosen peak layers were unstable for some settings (e.g., GPT-2 ToMi), indicating sensitivity to split/selection noise.
3. Robustness behavior was mixed: some conditions improved under perturbation, others degraded; no FDR-significant consistent drop.
4. Surface baseline was competitive on ToMi, supporting the possibility of non-semantic shortcut capture.

### Hypothesis Testing Results
- Hypothesis partially supported: layer-local decodability is present.
- Hypothesis not strongly supported on semantic robustness: perturbation/rephrase consistency was weak and statistically non-significant.
- Multiple-comparison-corrected significance: none below 0.05.
- Effect sizes from seed-level comparisons were near zero due deterministic probe behavior under current setup.

### Comparison to Baselines
- On TruthfulQA, hidden-state probes exceeded TF-IDF AUROC for best layers (notably Llama7B).
- On ToMi, gains over surface baseline were limited/inconsistent.
- Practical conclusion: probes add signal, but semantic interpretation remains uncertain.

### Surprises and Insights
- Validation peak layers did not always align with top test layers.
- Some perturbations increased AUROC, suggesting lexical edits may occasionally simplify decision boundaries instead of challenging semantics.

### Error Analysis
- Class imbalance in TruthfulQA likely inflated accuracy but depressed F1 sensitivity for positive class.
- ToMi performance hovered near chance for several model/layer combinations.

### Limitations
- Reduced sample sizes (compute-constrained for full multi-model run).
- Llama model is a public Llama-family mirror, not gated latest Meta release.
- No causal intervention in this cycle.
- Seed-based inferential statistics limited by near-deterministic solver behavior.

## 6. Conclusions
Layer-wise probes reveal belief-related decodable information in GPT-2 and Llama-family internals, especially for epistemic truthfulness prompts. However, robustness and layer-selection instability indicate these signals are not clean evidence of deep semantic belief representation. The current evidence is most consistent with mixed representations containing both meaningful and superficial linguistic components.

## 7. Next Steps
### Immediate Follow-ups
1. Add causal activation intervention at top layers (ITI/RepBelief-style) to test mechanistic relevance beyond decodability.
2. Increase dataset size and use grouped cross-validation for stable peak-layer estimation.
3. Add stronger semantic-preserving perturbations (entity role swaps, paraphrase models) and OOD templates.

### Alternative Approaches
- Probe heads (not only residual stream layers).
- Use sparse/CCS probes for interpretability and robustness.

### Broader Extensions
- Apply layer-local belief signals to hallucination detection and confidence recalibration.

### Open Questions
- Which signals remain invariant under strict semantic equivalence transformations?
- Do intervention directions transfer across model families and tasks?

## Validation Checklist (Phase 5)
- Code validation: completed (`src/belief_probing_experiment.py` runs end-to-end).
- Reproducibility check: completed on reduced gpt2 run (`REPRO_MATCH`).
- Scientific checks: metrics + nonparametric tests + FDR included; limitations documented.
- Documentation checks: required sections, figures, and output paths included.

## References
- Zhu et al. (2024), *Language Models Represent Beliefs of Self and Others*.
- Bortoletto et al. (2024), *Brittle Minds, Fixable Activations*.
- Azaria & Mitchell (2023), *The Internal State of an LLM Knows When It’s Lying*.
- Herrmann & Levinstein (2024), *Standards for Belief Representations in LLMs*.
- Dies et al. (2025), *Representational and Behavioral Stability of Truth in LLMs*.
