# Layer-wise Probing Analysis of Belief Encoding in LLMs

This project runs layer-wise linear probing on GPT-2 and Llama-family models to test whether epistemic vs non-epistemic belief signals are encoded in distinct internal layers, and whether these signals are robust to lexical/template changes.

## Key Findings
- Belief-related signals are decodable in specific layers across all tested models.
- Strongest epistemic decoding was on `llama7b` for TruthfulQA (best test-layer AUROC ~0.756).
- Robustness evidence was mixed; perturbation/rephrase effects were inconsistent and not FDR-significant.
- Results suggest mixed semantic + surface-pattern encoding rather than clean semantic abstraction.

## Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Run experiment:
```bash
python src/belief_probing_experiment.py --models gpt2,gpt2_medium,llama7b --max-train 300 --max-val 80 --max-test 80
```
3. See outputs in `results/` and full write-up in `REPORT.md`.

## File Structure
- `src/belief_probing_experiment.py`: end-to-end experiment pipeline.
- `planning.md`: motivation, novelty, and experimental design.
- `REPORT.md`: full research report with methods/results/analysis.
- `results/layerwise_metrics.csv`: per-layer, per-seed metrics.
- `results/stats_summary.csv`: summarized robustness/statistics.
- `results/plots/`: visualization artifacts.
- `logs/experiment_run.log`: run log.
