# Downloaded Datasets

This directory contains datasets for belief-probing and truthfulness experiments.
Data files are intentionally excluded from git; use the instructions below to re-download.

## Dataset 1: TruthfulQA (generation)

### Overview
- **Source**: `truthfulqa/truthful_qa` (config: `generation`)
- **Size**: validation split, 817 examples
- **Format**: HuggingFace Dataset (Arrow on disk)
- **Task**: truthfulness / epistemic correctness analysis
- **Splits**: validation (817)
- **License**: see dataset card on HuggingFace

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds.save_to_disk("datasets/truthful_qa_generation")
```

### Loading the Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/truthful_qa_generation")
print(ds["validation"][0])
```

### Sample Data
- Saved samples: `datasets/truthful_qa_generation/samples/samples.json`

### Notes
- Useful for probing truth-related latent directions.
- Pairs naturally with ITI/honesty interventions.

## Dataset 2: TruthfulQA (multiple_choice)

### Overview
- **Source**: `truthfulqa/truthful_qa` (config: `multiple_choice`)
- **Size**: validation split, 817 examples
- **Format**: HuggingFace Dataset
- **Task**: MC truthfulness scoring
- **Splits**: validation (817)
- **License**: see dataset card on HuggingFace

### Download Instructions

```python
from datasets import load_dataset

ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
ds.save_to_disk("datasets/truthful_qa_multiple_choice")
```

### Loading the Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/truthful_qa_multiple_choice")
print(ds["validation"][0])
```

### Sample Data
- Saved samples: `datasets/truthful_qa_multiple_choice/samples/samples.json`

### Notes
- Useful for calibrating and comparing probe-guided truth predictions.

## Dataset 3: ToMi-NLI

### Overview
- **Source**: `tasksource/tomi-nli`
- **Size**: 17,982 examples total
- **Format**: HuggingFace Dataset
- **Task**: theory-of-mind / belief-state NLI (epistemic reasoning)
- **Splits**: train (5994), validation (5994), test (5994)
- **License**: see dataset card on HuggingFace

### Download Instructions

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tomi-nli")
ds.save_to_disk("datasets/tomi_nli")
```

### Loading the Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/tomi_nli")
print(ds["train"][0])
```

### Sample Data
- Saved samples: `datasets/tomi_nli/samples/samples.json`

### Notes
- Matches the literature around ToM/belief decoding in internal states.

## Dataset 4: False/True Belief Tasks (General-331)

### Overview
- **Source**: `onyrotssih/false-true-belief-tasks-general-331`
- **Size**: train split, 331 examples
- **Format**: HuggingFace Dataset
- **Task**: compact false-belief evaluation set
- **Splits**: train (331)
- **License**: see dataset card on HuggingFace

### Download Instructions

```python
from datasets import load_dataset

ds = load_dataset("onyrotssih/false-true-belief-tasks-general-331")
ds.save_to_disk("datasets/false_true_belief_tasks_general_331")
```

### Loading the Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/false_true_belief_tasks_general_331")
print(ds["train"][0])
```

### Sample Data
- Saved samples: `datasets/false_true_belief_tasks_general_331/samples/samples.json`

### Notes
- Small, practical sanity-check benchmark for quick probe iteration.

## Validation Summary

- Validation metadata: `datasets/dataset_validation.json`
- Downloaded sizes (local):
  - `truthful_qa_generation`: ~276 KB
  - `truthful_qa_multiple_choice`: ~328 KB
  - `tomi_nli`: ~8 KB metadata + Arrow shards
  - `false_true_belief_tasks_general_331`: ~4 KB metadata + Arrow shards

Quick schema check used:
```python
from datasets import load_from_disk
for path in [
    "datasets/truthful_qa_generation",
    "datasets/truthful_qa_multiple_choice",
    "datasets/tomi_nli",
    "datasets/false_true_belief_tasks_general_331",
]:
    ds = load_from_disk(path)
    print(path, ds)
```
