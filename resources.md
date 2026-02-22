## Resources Catalog

### Summary
This document catalogs all resources gathered for:
**Layer-wise Probing Analysis of Belief Encoding in LLMs**.

### Papers
Total papers discovered (high/medium relevance): 18  
Total papers downloaded: 17  
Unresolved access/download: 1

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Language Models Represent Beliefs of Self and Others | Zhu et al. | 2024 | `papers/2024_language_models_represent_beliefs_of_self_and_others.pdf` | Direct belief probing + intervention |
| Brittle Minds, Fixable Activations | Bortoletto et al. | 2024 | `papers/2024_brittle_minds_fixable_activations_understanding_belief_representations_in_langua.pdf` | Probe brittleness + fixable activations |
| The Internal State of an LLM Knows When its Lying | Azaria, Mitchell | 2023 | `papers/2023_the_internal_state_of_an_llm_knows_when_its_lying.pdf` | Foundational truthful-vs-lying probe result |
| Still no lie detector for language models | Levinstein, Herrmann | 2023 | `papers/2023_still_no_lie_detector_for_language_models_probing_empirical_and_conceptual_roadb.pdf` | Critical limits of lie probes |
| Standards for Belief Representations in LLMs | Herrmann, Levinstein | 2024 | `papers/2024_standards_for_belief_representations_in_llms.pdf` | Methodological standards |
| Representational and Behavioral Stability of Truth in LLMs | Dies et al. | 2025 | `papers/2025_representational_and_behavioral_stability_of_truth_in_large_language_models.pdf` | P-StaT perturbation stability |
| How Post-Training Reshapes LLMs | Du et al. | 2025 | `papers/2025_how_post_training_reshapes_llms_a_mechanistic_view_on_knowledge_truthfulness_ref.pdf` | Post-training mechanistic shifts |
| Survey on Hallucination in LLMs | Huang et al. | 2023 | `papers/2023_a_survey_on_hallucination_in_large_language_models_principles_taxonomy_challenge.pdf` | Error/factuality taxonomy |
| Additional 9 papers (belief/uncertainty/truth) | Various | 2025-2026 | `papers/*.pdf` | See `papers/README.md` and `papers/paper_download_log.json` |

See `papers/README.md` for full per-paper annotations.

### Datasets
Total datasets downloaded: 4 local dataset artifacts

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA (generation) | `truthfulqa/truthful_qa` | 817 examples | Epistemic truthfulness | `datasets/truthful_qa_generation/` | Open-ended truthful response analysis |
| TruthfulQA (multiple_choice) | `truthfulqa/truthful_qa` | 817 examples | Truthfulness MC evaluation | `datasets/truthful_qa_multiple_choice/` | Probe/intervention scoring |
| ToMi-NLI | `tasksource/tomi-nli` | 17,982 examples | Theory-of-mind / belief NLI | `datasets/tomi_nli/` | Belief-state reasoning benchmark |
| False/True Belief Tasks (331) | `onyrotssih/false-true-belief-tasks-general-331` | 331 examples | Fast false-belief checks | `datasets/false_true_belief_tasks_general_331/` | Lightweight debugging benchmark |

See `datasets/README.md` for download and loading instructions.

### Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RepBelief | https://github.com/Walter0807/RepBelief | Official belief probing/intervention implementation | `code/RepBelief/` | Most directly aligned with topic |
| Probes | https://github.com/balevinstein/Probes | Probe training (supervised + CCS) | `code/Probes/` | Strong baseline probe workflow |
| honest_llama | https://github.com/likenneth/honest_llama | ITI truthfulness intervention | `code/honest_llama/` | Causal intervention baseline |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability toolkit | `code/TransformerLens/` | Activation extraction/patching backbone |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | Standardized benchmark evaluation | `code/lm-evaluation-harness/` | Reproducible reporting |

See `code/README.md` for key scripts and entry points.

### Resource Gathering Notes

#### Search Strategy
- Ran paper-finder with belief/probing query and filtered to relevance >= 2.
- Downloaded available PDFs via Semantic Scholar metadata + arXiv fallback.
- Performed deep reading on top 3 core papers using PDF chunker across all chunks.
- Selected datasets to cover epistemic truthfulness + non-epistemic/theory-of-mind belief tasks.
- Collected official and high-utility code repos for probing, intervention, and evaluation.

#### Selection Criteria
- Direct relevance to layer-wise probing and belief representation.
- Availability of code and reproducible methods.
- Benchmark utility for automated experiments.
- Coverage across probing, causal intervention, and robustness testing.

#### Challenges Encountered
- Paper-finder diligent mode stalled; fast mode succeeded.
- One relevant paper had no retrievable PDF from available metadata.
- Some legacy HF datasets (script-based) were incompatible with current `datasets` version; alternatives selected.

#### Gaps and Workarounds
- Missing paper download documented in `papers/README.md`.
- Used dual TruthfulQA configs + ToMi-NLI + false-belief compact set to maintain task coverage.

### Recommendations for Experiment Design

1. **Primary dataset(s)**:
   - `tasksource/tomi-nli` for belief-state probing (core).
   - `truthfulqa/truthful_qa` for epistemic truthfulness probing and intervention effects.
2. **Baseline methods**:
   - Layer-wise linear probes (logistic/multinomial) + random/control probes.
   - ITI/activation steering baseline using `honest_llama` methodology.
3. **Evaluation metrics**:
   - Probe AUROC/accuracy per layer.
   - Perturbation stability (retraction/expansion rates).
   - Calibration metrics (ECE/Brier) and downstream task retention.
4. **Code to adapt/reuse**:
   - Start with `code/RepBelief/` for belief probing scripts.
   - Use `code/TransformerLens/` for GPT-2/Llama instrumentation.
   - Use `code/Probes/` for alternate probe training controls.
