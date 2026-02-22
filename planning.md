# Research Plan: Layer-wise Probing Analysis of Belief Encoding in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding where and how belief-related signals are encoded in LLM internals is important for interpretability and reliability: if epistemic beliefs are represented distinctly, we can target those features for diagnosis and control. This directly benefits safety work on hallucination mitigation and confidence calibration. It also clarifies whether apparent “belief reasoning” is mechanistic or merely a surface pattern effect.

### Gap in Existing Work
Prior work (e.g., RepBelief, brittle-activation analyses, truth-probe papers) shows decodability of belief/truth signals, but cross-family comparisons under one standardized pipeline are limited, especially between GPT-2 and Meta Llama checkpoints on both ToM and epistemic truth tasks. Robustness checks against lexical/surface perturbations are still underused, creating uncertainty about semantic vs superficial encoding.

### Our Novel Contribution
We run a unified layer-wise probe pipeline across GPT-2 and Meta Llama models on both non-epistemic (ToMi-NLI belief-state) and epistemic (TruthfulQA MC framing) tasks, then stress-test probes with controlled perturbations. We combine decodability, cross-template generalization, and perturbation-stability metrics to separate semantic signal from lexical shortcut signal.

### Experiment Justification
- Experiment 1: Layer-wise linear probing across models/tasks to localize where belief-type information is decodable.
- Experiment 2: Lexical/surface perturbation robustness to test whether probe performance survives wording changes.
- Experiment 3: Cross-template transfer and control-feature baselines to estimate semantic generalization vs spurious dependence.

## Research Question
Do GPT-2 and Meta Llama models encode epistemic and non-epistemic belief types in distinct internal layers, and do those encodings reflect semantic understanding rather than superficial linguistic cues?

## Background and Motivation
Literature indicates linear probes can decode belief/truth signals from mid-to-late layers, but conceptual critiques warn that decodability alone does not imply genuine representation. A stronger design requires controls and perturbation tests. This project targets that gap with a shared, reproducible protocol across two model families.

## Hypothesis Decomposition
- H1 (Layer localization): Belief labels are significantly decodable above controls, peaking in specific middle/late layers.
- H2 (Task-dependent localization): Peak layers differ between non-epistemic belief-state reasoning (ToMi-NLI) and epistemic truthfulness (TruthfulQA-derived labels).
- H3 (Semantic robustness): If encoding is semantic, probe performance should remain relatively stable under lexical/syntactic perturbations and cross-template transfer.
- H4 (Surface-pattern alternative): If performance collapses under perturbation while in-template performance stays high, representations are likely superficial.

Independent variables:
- Model family/checkpoint (GPT-2 variants, Meta Llama variants)
- Layer index
- Task type (ToMi-NLI vs TruthfulQA)
- Input condition (original vs perturbation)

Dependent variables:
- Probe AUROC, Accuracy, F1, ECE, Brier
- Robustness deltas (original minus perturbed)

Success criteria:
- Statistically significant layer-wise separation from controls (p < 0.05 with multiple-comparison correction)
- Reproducible localization patterns across seeds
- Quantified robustness profile that supports or refutes semantic encoding claim

## Proposed Methodology

### Approach
Extract hidden states per layer for each example, train lightweight linear probes per layer, and evaluate across tasks and perturbation conditions. Use identical data splits and probe settings across models to ensure fair comparison.

### Experimental Steps
1. Data loading and validation for ToMi-NLI and TruthfulQA MC, with class-balance checks and schema validation.
2. Build two classification targets:
   - ToMi-NLI label mapping for belief-state/NLI signal.
   - TruthfulQA MC label from best true vs best false option scoring setup.
3. Generate controlled perturbations (entity renaming, syntactic alternation, lexical substitution while preserving meaning).
4. Extract final-token (and pooled-token sensitivity check) hidden states for every layer.
5. Train layer-wise logistic probes with train/val splits; test on held-out and perturbed sets.
6. Add baselines: random labels, bag-of-words/logistic control, and random projection control.
7. Run across seeds {42, 43, 44}; aggregate mean/std and confidence intervals.
8. Statistical testing: paired tests across layers/conditions with BH-FDR correction.

### Baselines
- Random-label probe baseline (sanity floor)
- Surface baseline: bag-of-words TF-IDF logistic model
- Random projection baseline from hidden states
- Majority-class baseline

### Evaluation Metrics
- Primary: AUROC, Accuracy, Macro-F1
- Calibration: ECE, Brier
- Robustness: performance drop under perturbation; cross-template transfer gap
- Localization: peak-layer index and area-under-layer-curve

### Statistical Analysis Plan
- Null hypothesis H0: no difference between original and perturbed probe performance; no layer effect beyond controls.
- Tests:
  - Repeated-measures comparisons across layers (nonparametric Friedman if normality fails; otherwise RM-ANOVA).
  - Paired permutation or Wilcoxon tests for original vs perturbed metrics.
- Multiple comparisons: Benjamini-Hochberg FDR at q=0.05.
- Effect sizes: Cohen’s d (paired), Kendall’s W/Friedman effect as appropriate.

## Expected Outcomes
Support for hypothesis if:
- Distinct peak layers emerge and differ by task/model.
- Probe performance remains meaningfully above controls under perturbation.
Refutation/qualification if:
- Gains disappear under perturbation or are matched by surface baselines.

## Timeline and Milestones
- M1 (Planning complete): design + metrics finalized.
- M2 (Setup): environment, dependencies, GPU config, reproducibility config.
- M3 (Implementation): data processing, activation extraction, probing pipeline.
- M4 (Experiments): baseline + full runs across models and seeds.
- M5 (Analysis): stats, plots, robustness and error analysis.
- M6 (Documentation): REPORT.md and README.md with reproducible commands.

## Potential Challenges
- Meta Llama gated access on Hugging Face.
  - Mitigation: detect and log access status; use available Llama checkpoints in workspace/cache and keep GPT-2 runs complete.
- GPU memory pressure for larger Llama checkpoints.
  - Mitigation: bf16/fp16 loading, gradient-free inference, capped batch sizes.
- Label construction ambiguity for TruthfulQA.
  - Mitigation: document deterministic label mapping and sensitivity checks.

## Success Criteria
- End-to-end reproducible scripts in `src/` run without manual edits.
- Results artifacts produced in `results/` (JSON + plots).
- REPORT.md contains actual quantitative findings and statistical tests.
