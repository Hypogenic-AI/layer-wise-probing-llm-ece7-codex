# Literature Review: Layer-wise Probing Analysis of Belief Encoding in LLMs

## Review Scope

### Research Question
Do open-source LLMs (especially GPT-2-family and Llama-family models) encode epistemic and non-epistemic belief states in distinct internal layers, and are these representations genuinely semantic vs superficial linguistic artifacts?

### Inclusion Criteria
- Paper studies internal LLM representations, probing, or activation interventions.
- Paper studies truthfulness, belief states, uncertainty, or Theory-of-Mind style reasoning.
- Paper provides implementation detail, dataset details, or actionable baselines.

### Exclusion Criteria
- Purely prompt-level studies with no internal representation analysis.
- Non-LLM work without transfer value for probing/intervention design.

### Time Frame
- Primary: 2023-2026

### Sources
- Paper-finder search service (`.claude/skills/paper-finder/scripts/find_papers.py`)
- Semantic Scholar metadata via paper-finder output
- arXiv PDF retrieval (when available)
- ACL anthology (for selected papers)

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-02-22 | "Layer-wise probing belief encoding LLM epistemic" | paper-finder | 70 papers | Used fast mode after diligent mode stalled |
| 2026-02-22 | Relevance filtering (>=2) | local filtering | 18 papers | 17 PDFs downloaded; 1 unresolved |
| 2026-02-22 | Dataset search (truthful, ToM, belief) | HuggingFace datasets | 3 selected datasets | Downloaded and validated |
| 2026-02-22 | Code search (belief probing/intervention) | GitHub API + paper links | 5 repos cloned | Includes official RepBelief + probing baselines |

## Screening Results

- Total discovered: 70
- Included for detailed review: 18 (relevance >= 2)
- Downloaded full PDFs: 17
- Not fully accessible: 1 (`Truth-value judgment in language models: 'truth directions' are context sensitive`)

## Key Papers

### 1) Language Models Represent Beliefs of Self and Others (ICML 2024)
- **Authors**: Wentao Zhu, Zhining Zhang, Yizhou Wang
- **Source**: arXiv / ICML
- **Key Contribution**: Shows belief states (self/other) can be linearly decoded from internal activations.
- **Methodology**: Layer/head probing with logistic and multinomial probes; activation intervention along belief directions.
- **Datasets Used**: BigToM; ToMi (generalization section).
- **Baselines**: Random directions, uninfluenced model behavior.
- **Results**: Strong middle-layer/head separability and causal behavior shifts under intervention.
- **Code Available**: Yes (`code/RepBelief/`).
- **Relevance**: Directly aligned with layer-wise belief probing hypothesis.

### 2) Brittle Minds, Fixable Activations (2024)
- **Authors**: Matteo Bortoletto et al.
- **Source**: arXiv
- **Key Contribution**: Probing-detected belief representations are sensitive to prompt perturbation but can be improved with interventions.
- **Methodology**: Probe analysis across Llama/Pythia variants; causal interventions (CAA/ITI-style comparisons).
- **Datasets Used**: False-belief style prompts and controlled prompt variants.
- **Baselines**: No intervention, random intervention, control tasks.
- **Results**: Mid/late-layer signal exists but brittleness indicates nontrivial spurious sensitivity.
- **Code Available**: paper code + compatible intervention repos.
- **Relevance**: Important caution for semantic-vs-surface interpretation.

### 3) The Internal State of an LLM Knows When It’s Lying (Findings EMNLP 2023)
- **Authors**: A. Azaria, Tom Mitchell
- **Key Contribution**: Internal states reveal truthful vs false outputs better than output text alone.
- **Methodology**: Supervised probing on hidden states.
- **Datasets Used**: Author-constructed truth/lie prompts.
- **Results**: Detectable latent truthfulness signal.
- **Code Available**: Related probing code in `code/Probes/`.
- **Relevance**: Foundational for epistemic probe design.

### 4) Still No Lie Detector for Language Models (2023)
- **Authors**: B. Levinstein, D. Herrmann
- **Key Contribution**: Critiques empirical/conceptual limits of current lie-detection probes.
- **Methodology**: Probe evaluation under distribution shifts and conceptual constraints.
- **Results**: Highlights fragility and interpretation risk.
- **Relevance**: Essential negative/control framing for this project.

### 5) Standards for Belief Representations in LLMs (2024)
- **Authors**: D. Herrmann, B. Levinstein
- **Key Contribution**: Proposes standards for claiming belief representations in LLM internals.
- **Methodology**: Conceptual + methodological framework.
- **Relevance**: Useful criteria for avoiding overclaiming semantic understanding.

### 6) Representational and Behavioral Stability of Truth in LLMs (2025)
- **Authors**: Samantha Dies et al.
- **Key Contribution**: P-StaT framework measures belief stability under semantic perturbation.
- **Methodology**: Probe-level and zero-shot perturbation stability across multiple LLMs.
- **Datasets Used**: City/medical/word-definition truth-value statements.
- **Results**: Synthetic perturbations induce largest epistemic retractions.
- **Relevance**: Adds robustness lens beyond raw probe accuracy.

### 7) How Post-Training Reshapes LLMs (2025)
- **Authors**: Hongzhe Du et al.
- **Key Contribution**: Mechanistic view of post-training effects on truthfulness/confidence.
- **Relevance**: Suggests representation shifts to track across base vs instruct models.

### 8) Enhancing Uncertainty Estimation with Aggregated Internal Belief (2025)
- **Authors**: Zeguan Xiao et al.
- **Key Contribution**: Uses internal belief signals for better uncertainty estimation.
- **Relevance**: Connects probing features to calibration metrics.

### 9) Survey of Theory of Mind in LLMs (2025)
- **Authors**: H. Nguyen
- **Key Contribution**: Consolidates ToM evaluations and representation-level findings.
- **Relevance**: Broader benchmark/task map for selecting datasets.

### 10) Hallucination Survey (2023)
- **Authors**: Lei Huang et al.
- **Key Contribution**: Taxonomy of hallucinations and evaluation challenges.
- **Relevance**: Useful for failure mode categorization in epistemic probing experiments.

## Deep Reading Notes (Full-Chunk Reads)

Deep-read papers were chunked with `pdf_chunker.py` and all chunks were processed:
- `papers/2024_language_models_represent_beliefs_of_self_and_others.pdf` (15 chunks)
- `papers/2024_brittle_minds_fixable_activations_understanding_belief_representations_in_langua.pdf` (8 chunks)
- `papers/2025_representational_and_behavioral_stability_of_truth_in_large_language_models.pdf` (9 chunks)
- Extracted chunk notes: `papers/deep_reading_extracts.md`

Key detailed takeaways from full reads:
- Belief signal is typically strongest in middle layers/selected heads, not uniformly distributed.
- Linear probing often works surprisingly well, but perturbation studies show this can be brittle.
- Causal interventions along probe-derived directions can improve task behavior, supporting at least partial mechanistic relevance.
- Robustness checks (prompt variants, synthetic perturbations) are necessary to separate semantics from lexical shortcuts.

## Common Methodologies

- **Linear probing (logistic / multinomial)**: Used in RepBelief and lie-detection work.
- **Head/layer localization**: Identify sparse informative heads/layers.
- **Activation intervention / steering**: ITI/CAA-style edits to test causal effect.
- **Perturbation stability testing**: Evaluate whether representations stay stable under semantic reframing.

## Standard Baselines

- Random-direction intervention baseline.
- Random-label or control-task probe baseline.
- No-intervention model output baseline.
- Base-vs-chat (or pre-vs-post-training) comparison.

## Evaluation Metrics

- Probe accuracy/F1/AUROC for belief label decoding.
- Calibration metrics (ECE, Brier score) for epistemic reliability.
- Belief retraction/expansion rates under perturbations (P-StaT style).
- Downstream task accuracy to measure collateral effects after intervention.

## Datasets in the Literature

- **BigToM / ToMi**: Frequently used for false-belief and ToM reasoning.
- **TruthfulQA**: Standard truthfulness benchmark.
- **Custom true/false/synthetic statement sets**: Used in perturbation stability and uncertainty papers.

## Gaps and Opportunities

- Probe success does not guarantee semantic belief representation; robustness evidence is still limited.
- Many studies use single-model or narrow prompt templates; cross-model generalization remains under-tested.
- Few papers jointly compare GPT-2 and modern Llama checkpoints under identical probing pipelines.
- Need stronger causal tests linking localized internal features to behavior across multiple tasks.

## Recommendations for Our Experiment

- **Recommended datasets**:
  - `tasksource/tomi-nli` for belief-state reasoning.
  - `truthfulqa/truthful_qa` (generation + MC) for epistemic truthfulness.
  - `onyrotssih/false-true-belief-tasks-general-331` for fast sanity checks.
- **Recommended baselines**:
  - Linear probe at each layer + random-control probes.
  - No-intervention vs probe-direction intervention.
  - Base vs instruct checkpoints where available.
- **Recommended metrics**:
  - Layer-wise probe AUROC/accuracy.
  - Stability score under prompt perturbations.
  - Calibration (ECE/Brier) before and after intervention.
- **Methodological considerations**:
  - Use strict train/test separation across prompt templates.
  - Include lexical-control perturbations to rule out superficial heuristics.
  - Treat high probe accuracy as evidence of decodability, not necessarily “belief possession.”

## Local Artifacts Created During Review

- Paper index: `papers/README.md`
- Download log: `papers/paper_download_log.json`
- Full-chunk deep notes: `papers/deep_reading_extracts.md`
- Dataset catalog: `datasets/README.md`
- Code catalog: `code/README.md`
