import json
import os
import random
import subprocess
import time
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def get_gpu_info() -> Dict[str, object]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        )
        gpus = []
        for line in out.strip().splitlines():
            name, total, free = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "memory_total": total, "memory_free": free})
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def lexical_perturb(text: str) -> str:
    swaps = {
        "entered": "went into",
        "exited": "left",
        "moved": "relocated",
        "believes": "thinks",
        "question": "query",
        "candidate": "proposed",
        "truthful": "factually correct",
        "premise": "context",
        "hypothesis": "claim",
    }
    out = text
    for src, tgt in swaps.items():
        out = out.replace(src, tgt)
        out = out.replace(src.capitalize(), tgt.capitalize())
    return out


def tomi_text(premise: str, hypothesis: str) -> str:
    return f"Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: Does the premise entail the hypothesis?"


def tomi_text_rephrased(premise: str, hypothesis: str) -> str:
    return f"Context: {premise}\nClaim: {hypothesis}\nDecide if the claim logically follows from the context."


def truthfulqa_text(question: str, choice: str) -> str:
    return (
        f"Question: {question}\nCandidate answer: {choice}\n"
        "Task: Is this candidate answer truthful? Reply yes/no."
    )


def truthfulqa_text_rephrased(question: str, choice: str) -> str:
    return (
        f"Query: {question}\nProposed response: {choice}\n"
        "Judge whether this response is factually correct."
    )


def load_tomi(max_train: int = 1200, max_val: int = 300, max_test: int = 300) -> Dict[str, Dict[str, List]]:
    ds = load_from_disk("datasets/tomi_nli")

    def convert(split_name: str, max_n: int, text_fn):
        split = ds[split_name]
        idx = list(range(len(split)))
        random.shuffle(idx)
        idx = idx[: max_n]
        texts = [text_fn(split[i]["premise"], split[i]["hypothesis"]) for i in idx]
        labels = [1 if split[i]["label"] == "entailment" else 0 for i in idx]
        return {"texts": texts, "labels": labels}

    return {
        "train": convert("train", max_train, tomi_text),
        "val": convert("validation", max_val, tomi_text),
        "test": convert("test", max_test, tomi_text),
        "test_perturbed": convert("test", max_test, lambda p, h: lexical_perturb(tomi_text(p, h))),
        "test_rephrased": convert("test", max_test, tomi_text_rephrased),
    }


def load_truthfulqa(max_train: int = 1200, max_val: int = 300, max_test: int = 300) -> Dict[str, Dict[str, List]]:
    ds = load_from_disk("datasets/truthful_qa_multiple_choice")["validation"]

    texts, labels = [], []
    texts_rephrased = []
    for row in ds:
        q = row["question"]
        choices = row["mc1_targets"]["choices"]
        y = row["mc1_targets"]["labels"]
        for c, l in zip(choices, y):
            texts.append(truthfulqa_text(q, c))
            texts_rephrased.append(truthfulqa_text_rephrased(q, c))
            labels.append(int(l))

    idx = np.arange(len(texts))
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=np.array(labels)[temp_idx],
        random_state=42,
    )

    train_idx_sel = np.array(train_idx)
    val_idx_sel = np.array(val_idx)
    test_idx_sel = np.array(test_idx)
    if len(train_idx_sel) > max_train:
        train_idx_sel = np.random.RandomState(42).choice(train_idx_sel, size=max_train, replace=False)
    if len(val_idx_sel) > max_val:
        val_idx_sel = np.random.RandomState(43).choice(val_idx_sel, size=max_val, replace=False)
    if len(test_idx_sel) > max_test:
        test_idx_sel = np.random.RandomState(44).choice(test_idx_sel, size=max_test, replace=False)

    def build(indices, source_texts):
        return {"texts": [source_texts[i] for i in indices], "labels": [labels[i] for i in indices]}

    test = build(test_idx_sel, texts)

    return {
        "train": build(train_idx_sel, texts),
        "val": build(val_idx_sel, texts),
        "test": test,
        "test_perturbed": {"texts": [lexical_perturb(t) for t in test["texts"]], "labels": test["labels"]},
        "test_rephrased": {
            "texts": [texts_rephrased[i] for i in test_idx_sel],
            "labels": test["labels"],
        },
    }


def summarize_data(data_bundle: Dict[str, Dict[str, Dict[str, List]]]) -> Dict[str, object]:
    summary = {}
    for task, splits in data_bundle.items():
        task_info = {}
        for split, d in splits.items():
            arr = np.array(d["labels"])
            text_lengths = [len(x.split()) for x in d["texts"]]
            task_info[split] = {
                "n": len(arr),
                "class_counts": dict(Counter(arr.tolist())),
                "missing_text": int(sum(1 for x in d["texts"] if (x is None or x == ""))),
                "avg_word_length": float(np.mean(text_lengths)),
                "std_word_length": float(np.std(text_lengths)),
                "min_word_length": int(np.min(text_lengths)),
                "max_word_length": int(np.max(text_lengths)),
            }
        summary[task] = task_info
    return summary


def configure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"


def extract_features(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> Dict[int, np.ndarray]:
    model.eval()
    all_layer_features = None
    n = len(texts)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_text = texts[start : start + batch_size]
            enc = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(**enc, output_hidden_states=True, use_cache=False)

            hs = out.hidden_states[1:]
            attn = enc["attention_mask"]
            last_idx = attn.sum(dim=1) - 1

            if all_layer_features is None:
                all_layer_features = {i: [] for i in range(len(hs))}

            for li, layer_h in enumerate(hs):
                sel = layer_h[torch.arange(layer_h.size(0), device=layer_h.device), last_idx]
                all_layer_features[li].append(sel.float().cpu().numpy())

            del enc, out, hs
            if device == "cuda":
                torch.cuda.empty_cache()

    return {li: np.concatenate(chunks, axis=0) for li, chunks in all_layer_features.items()}


def train_probe_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            solver="liblinear",
        ),
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_eval)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1": float(f1_score(y_eval, pred)),
        "auroc": float(roc_auc_score(y_eval, prob)),
        "ece": expected_calibration_error(y_eval, prob),
        "brier": float(brier_score_loss(y_eval, prob)),
    }


def run_surface_baseline(train_texts, train_labels, test_texts, test_labels) -> Dict[str, float]:
    clf = make_pipeline(
        TfidfVectorizer(max_features=15000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42),
    )
    clf.fit(train_texts, train_labels)
    prob = clf.predict_proba(test_texts)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(test_labels, pred)),
        "f1": float(f1_score(test_labels, pred)),
        "auroc": float(roc_auc_score(test_labels, prob)),
        "ece": expected_calibration_error(np.array(test_labels), prob),
        "brier": float(brier_score_loss(test_labels, prob)),
    }


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    denom = np.std(diff, ddof=1)
    if denom == 0:
        return 0.0
    return float(np.mean(diff) / denom)


def plot_layer_curves(df: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="task", hue="model", height=4.0, aspect=1.5, sharey=False)
    g.map(sns.lineplot, "layer", "auroc", marker="o")
    g.add_legend()
    g.set_axis_labels("Layer index", "Validation AUROC")
    g.fig.suptitle("Layer-wise probe AUROC (validation)", y=1.03)
    g.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def plot_drop_bars(df: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 4))
    sns.barplot(data=df, x="task", y="auroc_drop", hue="model", errorbar="sd")
    plt.title("AUROC drop under lexical perturbation at peak layer")
    plt.ylabel("Original AUROC - Perturbed AUROC")
    plt.xlabel("Task")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Layer-wise probing analysis for belief encoding.")
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--max-test", type=int, default=300)
    parser.add_argument(
        "--models",
        type=str,
        default="gpt2,gpt2_medium,llama7b",
        help="Comma-separated subset of [gpt2,gpt2_medium,llama7b].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = get_gpu_info()

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "sklearn": __import__("sklearn").__version__,
        "device": device,
        "gpu_info": gpu_info,
        "seeds": [42, 43, 44],
        "max_train": args.max_train,
        "max_val": args.max_val,
        "max_test": args.max_test,
        "models_requested": args.models,
    }

    with open("results/environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    data_bundle = {
        "tomi_nli": load_tomi(args.max_train, args.max_val, args.max_test),
        "truthfulqa_mc": load_truthfulqa(args.max_train, args.max_val, args.max_test),
    }

    data_summary = summarize_data(data_bundle)
    with open("results/data_summary.json", "w") as f:
        json.dump(data_summary, f, indent=2)

    all_models = [
        {"name": "gpt2", "hf_id": "gpt2"},
        {"name": "gpt2_medium", "hf_id": "gpt2-medium"},
        {"name": "llama7b", "hf_id": "huggyllama/llama-7b"},
    ]
    requested = {m.strip() for m in args.models.split(",") if m.strip()}
    models = [m for m in all_models if m["name"] in requested]

    seeds = [42, 43, 44]
    run_log = []
    all_rows = []
    summary_rows = []
    stats_rows = []

    for model_cfg in models:
        mname = model_cfg["name"]
        hf_id = model_cfg["hf_id"]
        t0 = time.time()
        print(f"\n=== Loading model: {hf_id} ===", flush=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            configure_tokenizer(tokenizer)
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            n_layers = model.config.num_hidden_layers
            hidden_size = model.config.hidden_size
            batch_size = 16 if hidden_size >= 3000 else 64
            if device == "cpu":
                batch_size = 8

            run_log.append({
                "model": mname,
                "status": "loaded",
                "num_layers": n_layers,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
            })
        except Exception as e:
            run_log.append({"model": mname, "status": "failed", "error": str(e)})
            print(f"Failed {mname}: {e}", flush=True)
            continue

        for task_name, splits in data_bundle.items():
            print(f"--- Task {task_name} with model {mname} ---", flush=True)
            y_train = np.array(splits["train"]["labels"])
            y_val = np.array(splits["val"]["labels"])
            y_test = np.array(splits["test"]["labels"])
            y_test_pert = np.array(splits["test_perturbed"]["labels"])
            y_test_reph = np.array(splits["test_rephrased"]["labels"])

            feat_train = extract_features(model, tokenizer, splits["train"]["texts"], batch_size, 256, device)
            feat_val = extract_features(model, tokenizer, splits["val"]["texts"], batch_size, 256, device)
            feat_test = extract_features(model, tokenizer, splits["test"]["texts"], batch_size, 256, device)
            feat_test_pert = extract_features(model, tokenizer, splits["test_perturbed"]["texts"], batch_size, 256, device)
            feat_test_reph = extract_features(model, tokenizer, splits["test_rephrased"]["texts"], batch_size, 256, device)

            # Surface baseline for control.
            surface = run_surface_baseline(
                splits["train"]["texts"],
                y_train,
                splits["test"]["texts"],
                y_test,
            )

            layer_seed_results = defaultdict(lambda: defaultdict(list))

            for layer_idx in sorted(feat_train.keys()):
                for seed in seeds:
                    val_m = train_probe_metrics(feat_train[layer_idx], y_train, feat_val[layer_idx], y_val, seed)
                    test_m = train_probe_metrics(feat_train[layer_idx], y_train, feat_test[layer_idx], y_test, seed)
                    pert_m = train_probe_metrics(
                        feat_train[layer_idx], y_train, feat_test_pert[layer_idx], y_test_pert, seed
                    )
                    reph_m = train_probe_metrics(
                        feat_train[layer_idx], y_train, feat_test_reph[layer_idx], y_test_reph, seed
                    )

                    row = {
                        "model": mname,
                        "task": task_name,
                        "layer": int(layer_idx),
                        "seed": int(seed),
                        "val_auroc": val_m["auroc"],
                        "test_auroc": test_m["auroc"],
                        "pert_auroc": pert_m["auroc"],
                        "reph_auroc": reph_m["auroc"],
                        "test_acc": test_m["accuracy"],
                        "test_f1": test_m["f1"],
                        "test_ece": test_m["ece"],
                        "test_brier": test_m["brier"],
                    }
                    all_rows.append(row)
                    layer_seed_results[layer_idx]["val_auroc"].append(val_m["auroc"])
                    layer_seed_results[layer_idx]["test_auroc"].append(test_m["auroc"])
                    layer_seed_results[layer_idx]["pert_auroc"].append(pert_m["auroc"])
                    layer_seed_results[layer_idx]["reph_auroc"].append(reph_m["auroc"])

            # Peak layer by mean validation AUROC.
            peak_layer = max(
                layer_seed_results.keys(),
                key=lambda l: float(np.mean(layer_seed_results[l]["val_auroc"])),
            )

            test_scores = np.array(layer_seed_results[peak_layer]["test_auroc"])
            pert_scores = np.array(layer_seed_results[peak_layer]["pert_auroc"])
            reph_scores = np.array(layer_seed_results[peak_layer]["reph_auroc"])

            try:
                p_pert = float(wilcoxon(test_scores, pert_scores, alternative="greater").pvalue)
            except Exception:
                p_pert = float("nan")
            try:
                p_reph = float(wilcoxon(test_scores, reph_scores, alternative="greater").pvalue)
            except Exception:
                p_reph = float("nan")

            stats_rows.append(
                {
                    "model": mname,
                    "task": task_name,
                    "peak_layer": int(peak_layer),
                    "test_auroc_mean": float(np.mean(test_scores)),
                    "pert_auroc_mean": float(np.mean(pert_scores)),
                    "reph_auroc_mean": float(np.mean(reph_scores)),
                    "auroc_drop_pert": float(np.mean(test_scores - pert_scores)),
                    "auroc_drop_reph": float(np.mean(test_scores - reph_scores)),
                    "p_wilcoxon_pert": p_pert,
                    "p_wilcoxon_reph": p_reph,
                    "d_pert": cohens_d_paired(test_scores, pert_scores),
                    "d_reph": cohens_d_paired(test_scores, reph_scores),
                }
            )

            summary_rows.append(
                {
                    "model": mname,
                    "task": task_name,
                    "peak_layer": int(peak_layer),
                    "surface_baseline_auroc": surface["auroc"],
                    "surface_baseline_acc": surface["accuracy"],
                }
            )

            del feat_train, feat_val, feat_test, feat_test_pert, feat_test_reph
            if device == "cuda":
                torch.cuda.empty_cache()

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        run_log.append({"model": mname, "elapsed_sec": elapsed})

    all_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)
    stats_df = pd.DataFrame(stats_rows)

    if not stats_df.empty:
        pvals = stats_df["p_wilcoxon_pert"].fillna(1.0).values
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        stats_df["p_wilcoxon_pert_fdr"] = pvals_adj

        pvals_r = stats_df["p_wilcoxon_reph"].fillna(1.0).values
        _, pvals_r_adj, _, _ = multipletests(pvals_r, alpha=0.05, method="fdr_bh")
        stats_df["p_wilcoxon_reph_fdr"] = pvals_r_adj

    all_df.to_csv("results/layerwise_metrics.csv", index=False)
    summary_df.to_csv("results/baseline_summary.csv", index=False)
    stats_df.to_csv("results/stats_summary.csv", index=False)

    with open("results/run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    # Plot layer curves from mean over seeds.
    if not all_df.empty:
        plot_df = (
            all_df.groupby(["model", "task", "layer"], as_index=False)["val_auroc"]
            .mean()
            .rename(columns={"val_auroc": "auroc"})
        )
        plot_layer_curves(plot_df, "results/plots/layerwise_val_auroc.png")

    if not stats_df.empty:
        drop_df = stats_df[["model", "task", "auroc_drop_pert"]].rename(
            columns={"auroc_drop_pert": "auroc_drop"}
        )
        plot_drop_bars(drop_df, "results/plots/perturbation_drop.png")

    merged = {
        "environment": env_info,
        "data_summary": data_summary,
        "summary": summary_df.to_dict(orient="records"),
        "stats": stats_df.to_dict(orient="records"),
        "notes": [
            "meta-llama gated checkpoints may be unavailable; public Llama-family mirror used when needed.",
            "Probe decodability is evidence of encoded information, not direct proof of agent-level belief possession.",
        ],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(merged, f, indent=2)

    print("Done. Results written to results/", flush=True)


if __name__ == "__main__":
    main()
