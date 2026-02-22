# Cloned Repositories

## Repo 1: RepBelief
- URL: https://github.com/Walter0807/RepBelief
- Purpose: Official code for ICML 2024 paper on belief representations of self/others in LLMs.
- Location: `code/RepBelief/`
- Key files:
  - `code/RepBelief/probe.py`
  - `code/RepBelief/probe_multinomial.py`
  - `code/RepBelief/scripts/save_reps.sh`
  - `code/RepBelief/scripts/0_forward_belief.sh`
  - `code/RepBelief/scripts/0_forward_belief_interv_o0p1.sh`
- Notes:
  - Supports representation extraction, binary/multinomial probing, and activation intervention.
  - Uses BigToM-style forward/backward belief and action tasks.

## Repo 2: Probes
- URL: https://github.com/balevinstein/Probes
- Purpose: Probe training/evaluation code referenced in lie-detection probing literature.
- Location: `code/Probes/`
- Key files:
  - `code/Probes/Generate_Embeddings.py`
  - `code/Probes/TrainProbes.py`
  - `code/Probes/Train_CCSProbe.py`
  - `code/Probes/Generate_CCS_predictions.py`
- Notes:
  - Includes both supervised probes and CCS-style unsupervised probing workflow.
  - Useful baseline for linear probe pipelines over layer activations.

## Repo 3: honest_llama
- URL: https://github.com/likenneth/honest_llama
- Purpose: Inference-Time Intervention (ITI) for truthful responses in LLaMA-family models.
- Location: `code/honest_llama/`
- Key files:
  - `code/honest_llama/get_activations/`
  - `code/honest_llama/validation/validate_2fold.py`
  - `code/honest_llama/interveners.py`
- Notes:
  - Strong baseline for intervention-based truthfulness changes.
  - Directly relevant for comparing probe-only vs intervention-guided outcomes.

## Repo 4: TransformerLens
- URL: https://github.com/TransformerLensOrg/TransformerLens
- Purpose: Mechanistic interpretability library for activation extraction, patching, and circuit analysis.
- Location: `code/TransformerLens/`
- Key files:
  - `code/TransformerLens/transformer_lens/HookedTransformer.py`
  - `code/TransformerLens/transformer_lens/patching.py`
  - `code/TransformerLens/transformer_lens/ActivationCache.py`
- Notes:
  - Best practical framework for layer/head-level activation probing on GPT-2 and compatible models.
  - Supports quick prototyping of causal interventions.

## Repo 5: lm-evaluation-harness
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Purpose: Standardized benchmark evaluation framework for LLM tasks.
- Location: `code/lm-evaluation-harness/`
- Key files:
  - `code/lm-evaluation-harness/lm_eval/tasks/`
  - `code/lm-evaluation-harness/docs/interface.md`
- Notes:
  - Provides reproducible evaluation and reporting infrastructure.
  - Useful for adding TruthfulQA and auxiliary sanity benchmarks.

## Quick Validation Performed
- All repositories cloned successfully with `--depth 1`.
- README files inspected for installation requirements and entry points.
- No full runtime test executed yet because most workflows require GPU/model checkpoints and (for some repos) external API keys.

## Potential Application to This Project
- `RepBelief` + `Probes`: direct templates for layer-wise belief decoding and probe training.
- `TransformerLens`: shared instrumentation backbone across GPT-2 and Llama checkpoints.
- `honest_llama`: intervention baseline to test whether decoded belief directions are causally meaningful.
- `lm-evaluation-harness`: standardized reporting for truthfulness/robustness transfer tasks.
