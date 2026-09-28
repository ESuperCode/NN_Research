"""
compare_models.py

THE file to run for a full baseline-vs-simplified comparison.

    python compare_models.py

Trains each baseline network and its simplified surrogate once (training
happens, but nothing about training is measured or reported), then runs a
battery of TEST-SET-ONLY evaluations across every dataset/config this
project defines, and writes:

    comparison_results.csv        raw per-run numbers, for your own analysis
    comparison_summary.csv        one row per config, means/stds only
    plots/accuracy_fidelity.png
    plots/latency_vs_batch_size.png
    plots/speedup_summary.png
    plots/accuracy_vs_latency_pareto.png
    plots/model_complexity.png

Every number in every plot and CSV comes from evaluating already-trained
models on held-out test data that was not used to fit either the baseline
or the surrogate. No training-time metric (loss curves, training accuracy,
epochs, convergence) is measured or reported anywhere in this script --
that's deliberate, per your instruction to keep this to test-time stats only.

This is meant to be the evidence base for a claim like "surrogate X is/isn't
a genuine improvement over baseline network architecture Y" -- so every run
reports accuracy AND latency AND model size together, because a result that
only reports one of the three is not a complete comparison and shouldn't be
the basis of a publishable claim.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:0.4f}")

from data_utils import load_text_classification_data
from text_datasets_extra import load_scam_detection_data, load_sentiment_data
from synthetic_3d_datasets import load_3d_dataset
from baseline_model import BaselineTextClassifier
from boundary_simplifier import autotune_surrogate

MAX_ACCURACY_DROP = 0.001  # 0.1 percentage points, per your requirement

OUT_DIR = "/mnt/user-data/outputs"
PLOT_DIR = f"{OUT_DIR}/plots"

BATCH_SIZES = [1, 16, 64, 256]
N_LATENCY_REPEATS = 30


# ---------------------------------------------------------------------------
# Experiment definitions: each is (label, loader, baseline kwargs, surrogate kwargs)
# ---------------------------------------------------------------------------

def _experiments():
    # Per your request: most configs are real-world-style TEXT classification
    # tasks (topic, scam/spam detection, sentiment) at TF-IDF dimensionality;
    # only two are the synthetic numeric 3D stress-test patterns. Each config
    # no longer hardcodes a surrogate basis/degree -- `autotune_surrogate`
    # (see boundary_simplifier.py) searches a small grid of increasingly
    # expensive configs and picks the FASTEST one whose validation accuracy
    # is within MAX_ACCURACY_DROP of the baseline's own accuracy.
    return [
        # --- TEXT (5 configs) ---
        dict(label="topic (20 Newsgroups)", data=lambda: _load_text(),
             hidden_layers=(64, 32), skip_pca=False, clip_non_negative=True),
        dict(label="scam detection", data=lambda: _load_scam(),
             hidden_layers=(64, 32), skip_pca=False, clip_non_negative=True),
        dict(label="sentiment", data=lambda: _load_sentiment(),
             hidden_layers=(64, 32), skip_pca=False, clip_non_negative=True),
        # --- NUMERIC 3D (2 configs only) ---
        dict(label="[numeric] spheres", data=lambda: _load_3d("spheres"),
             hidden_layers=(200, 100, 50), skip_pca=True, clip_non_negative=False),
        dict(label="[numeric] helix", data=lambda: _load_3d("helix"),
             hidden_layers=(200, 100, 50), skip_pca=True, clip_non_negative=False),
    ]


def _load_text():
    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    return X_train, X_test, y_train, y_test


def _load_scam():
    X_train, X_test, y_train, y_test, source = load_scam_detection_data()
    return X_train, X_test, y_train, y_test


def _load_sentiment():
    X_train, X_test, y_train, y_test, source = load_sentiment_data()
    return X_train, X_test, y_train, y_test


def _load_3d(pattern):
    X_train, X_test, y_train, y_test, source = load_3d_dataset(pattern, n_samples=1500)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# McNemar's exact test: is the accuracy DIFFERENCE between the two models on
# THIS test set statistically significant, or within noise? This is the
# correct paired test for "two classifiers evaluated on the same samples"
# (as opposed to an unpaired t-test, which would be wrong here since the
# two models are not evaluated on independent samples).
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # n01: A correct & B wrong.  n10: A wrong & B correct.
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n_disagree = n01 + n10
    if n_disagree == 0:
        return {"n_a_only_correct": n01, "n_b_only_correct": n10,
                "mcnemar_statistic": 0.0, "mcnemar_pvalue": 1.0}
    k = min(n01, n10)
    result = sstats.binomtest(k, n_disagree, p=0.5, alternative="two-sided")
    return {"n_a_only_correct": n01, "n_b_only_correct": n10,
            "mcnemar_statistic": float(abs(n01 - n10)),
            "mcnemar_pvalue": float(result.pvalue)}


# ---------------------------------------------------------------------------
# Model complexity (a "size" stat independent of both accuracy and latency)
# ---------------------------------------------------------------------------

def _baseline_param_count(baseline):
    mlp = baseline.clf
    return int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))


def _simplified_param_count(simplified):
    """Counts every number the fused closed-form model needs to store:
    the PCA projection (if any) plus whichever fused representation
    (quadratic matrix, monomial coefficients, or RBF weights) it holds."""
    n = 0
    if simplified.P is not None:
        n += simplified.P.size + simplified.mean.size
    p = simplified.params
    if simplified.basis == "poly2":
        n += p["a"].size + p["W"].size + 1
    elif simplified.basis == "polyN":
        n += p["coef"].size + 1  # idx is fixed structure, not a learned parameter
    elif simplified.basis == "rbf":
        n += p["Omega"].size + p["b"].size + p["w"].size + 1
    return int(n)


# ---------------------------------------------------------------------------
# Test-set-only evaluation
# ---------------------------------------------------------------------------

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _latency_stats(model, X, n_repeats=N_LATENCY_REPEATS):
    times = model.forward_pass_latency(X, n_repeats=n_repeats)
    return {
        "latency_mean_s": float(times.mean()),
        "latency_std_s": float(times.std()),
        "latency_median_s": float(np.median(times)),
        "throughput_samples_per_s": float(X.shape[0] / times.mean()) if times.mean() > 0 else float("inf"),
    }


def evaluate_on_test_set(model, model_name, config_label, X_test, y_test, param_count):
    """Every number here comes from the held-out test set only."""
    X_test_dense = _to_dense(X_test)
    y_pred = model.predict(X_test_dense)

    rows = []
    n = X_test_dense.shape[0]
    # de-duplicate: small test sets can make a requested batch size collapse
    # onto the full-test-set size, which would otherwise produce two rows
    # with the same batch_size and break the "pick the full-batch row" logic
    # downstream.
    unique_batches = sorted(set(min(b, n) for b in BATCH_SIZES + [n]))
    for b in unique_batches:
        Xb = X_test_dense[:b]
        lat = _latency_stats(model, Xb)
        rows.append({
            "config": config_label,
            "model": model_name,
            "batch_size": b,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "n_test_samples": X_test_dense.shape[0],
            "n_parameters": param_count,
            **lat,
        })
    return rows


def run_all_experiments():
    all_rows = []
    fidelity_rows = []
    mcnemar_rows = []
    chosen_config_rows = []

    for exp in _experiments():
        print(f"Training and converting: {exp['label']} ...")
        X_train, X_test, y_train, y_test = exp["data"]()

        # --- training happens here, but nothing from it is measured ---
        baseline = BaselineTextClassifier(hidden_layer_sizes=exp["hidden_layers"], max_iter=500)
        baseline.fit(X_train, y_train)
        simplified, tuning_report = autotune_surrogate(
            baseline, X_train, y_train, max_accuracy_drop=MAX_ACCURACY_DROP,
            skip_pca=exp["skip_pca"], clip_non_negative=exp["clip_non_negative"],
        )
        chosen = tuning_report["chosen_config"]
        chosen_str = ", ".join(f"{k}={v}" for k, v in chosen.items())
        print(f"  chosen surrogate config: {chosen_str}  [{tuning_report['status']}]")
        chosen_config_rows.append({
            "config": exp["label"],
            "chosen_surrogate_config": chosen_str,
            "autotune_status": tuning_report["status"],
            "baseline_val_accuracy": tuning_report["baseline_val_accuracy"],
        })
        # --- end training; everything below is test-set-only evaluation ---

        base_params = _baseline_param_count(baseline)
        simp_params = _simplified_param_count(simplified)

        all_rows += evaluate_on_test_set(baseline, "baseline", exp["label"], X_test, y_test, base_params)
        all_rows += evaluate_on_test_set(simplified, "simplified", exp["label"], X_test, y_test, simp_params)

        X_test_dense = _to_dense(X_test)
        y_test_arr = np.asarray(y_test)
        y_pred_base = baseline.predict(X_test_dense)
        y_pred_simp = simplified.predict(X_test_dense)

        fidelity_rows.append({
            "config": exp["label"],
            "fidelity_agreement": float((y_pred_base == y_pred_simp).mean()),
        })

        mc = mcnemar_test(y_test_arr, y_pred_base, y_pred_simp)
        mc["config"] = exp["label"]
        mcnemar_rows.append(mc)

    return (pd.DataFrame(all_rows), pd.DataFrame(fidelity_rows), pd.DataFrame(mcnemar_rows),
            pd.DataFrame(chosen_config_rows))


# ---------------------------------------------------------------------------
# Master comparison table: one row per config, every metric for both models
# side by side, plus the derived "how much better/worse" comparison scores.
# ---------------------------------------------------------------------------

def build_comparison_table(df, fidelity_df, mcnemar_df, chosen_config_df):
    full_batch = df.groupby(["config", "model"])["batch_size"].transform("max")
    d = df[df["batch_size"] == full_batch].set_index(["config", "model"]).sort_index()

    rows = []
    for config in df["config"].unique():
        b = d.loc[(config, "baseline")]
        s = d.loc[(config, "simplified")]
        fid = fidelity_df.loc[fidelity_df.config == config, "fidelity_agreement"].iloc[0]
        mc = mcnemar_df.loc[mcnemar_df.config == config].iloc[0]
        cc = chosen_config_df.loc[chosen_config_df.config == config].iloc[0]

        rows.append({
            "config": config,
            "chosen_surrogate_config": cc.chosen_surrogate_config,
            "autotune_status": cc.autotune_status,
            "baseline_accuracy": b.test_accuracy,
            "simplified_accuracy": s.test_accuracy,
            "accuracy_delta_pp": (s.test_accuracy - b.test_accuracy) * 100,
            "accuracy_retention_pct": (s.test_accuracy / b.test_accuracy * 100) if b.test_accuracy > 0 else float("nan"),
            "baseline_f1": b.test_f1,
            "simplified_f1": s.test_f1,
            "f1_delta": s.test_f1 - b.test_f1,
            "fidelity_agreement": fid,
            "mcnemar_pvalue": mc.mcnemar_pvalue,
            "accuracy_diff_significant_at_0.05": bool(mc.mcnemar_pvalue < 0.05),
            "baseline_latency_s": b.latency_mean_s,
            "simplified_latency_s": s.latency_mean_s,
            "speedup_x": b.latency_mean_s / s.latency_mean_s if s.latency_mean_s > 0 else float("inf"),
            "baseline_throughput_sps": b.throughput_samples_per_s,
            "simplified_throughput_sps": s.throughput_samples_per_s,
            "baseline_n_params": int(b.n_parameters),
            "simplified_n_params": int(s.n_parameters),
            "param_compression_x": b.n_parameters / s.n_parameters if s.n_parameters > 0 else float("inf"),
            "n_test_samples": int(b.n_test_samples),
        })
    return pd.DataFrame(rows)


def print_full_report(df, comparison):
    """Prints every number to the terminal -- per-config detail tables plus
    the master comparison table plus an aggregate headline summary."""
    pd.set_option("display.max_rows", None)

    print("\n" + "=" * 100)
    print("PER-CONFIG, PER-BATCH-SIZE RAW RESULTS (every number behind the plots/CSVs)")
    print("=" * 100)
    cols = ["config", "model", "batch_size", "test_accuracy", "test_f1", "test_precision",
            "test_recall", "latency_mean_s", "latency_std_s", "latency_median_s",
            "throughput_samples_per_s", "n_parameters", "n_test_samples"]
    print(df[cols].sort_values(["config", "model", "batch_size"]).to_string(index=False))

    print("\n" + "=" * 100)
    print("MASTER COMPARISON TABLE (one row per dataset/config, full test-batch size)")
    print("=" * 100)
    display_cols = [
        "config", "chosen_surrogate_config", "baseline_accuracy", "simplified_accuracy",
        "accuracy_delta_pp", "accuracy_retention_pct", "fidelity_agreement", "mcnemar_pvalue",
        "accuracy_diff_significant_at_0.05", "speedup_x", "param_compression_x",
    ]
    print(comparison[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("HEADLINE SUMMARY")
    print("=" * 100)
    n = len(comparison)
    n_sig_worse = int(((comparison.accuracy_delta_pp < 0) & comparison["accuracy_diff_significant_at_0.05"]).sum())
    n_sig_better = int(((comparison.accuracy_delta_pp > 0) & comparison["accuracy_diff_significant_at_0.05"]).sum())
    n_not_sig = n - n_sig_worse - n_sig_better
    n_faster = int((comparison.speedup_x >= 1).sum())
    n_slower = n - n_faster
    both_better = int(((comparison.speedup_x >= 1) &
                        (~comparison["accuracy_diff_significant_at_0.05"] | (comparison.accuracy_delta_pp >= 0))).sum())

    print(f"Configs tested:                              {n}")
    print(f"  Accuracy significantly WORSE (p<0.05):      {n_sig_worse}")
    print(f"  Accuracy significantly BETTER (p<0.05):     {n_sig_better}")
    print(f"  Accuracy difference NOT significant:        {n_not_sig}")
    print(f"  Simplified surrogate FASTER at full batch:  {n_faster}")
    print(f"  Baseline network FASTER at full batch:      {n_slower}")
    print(f"  Simplified surrogate always smaller (params): "
          f"{int((comparison.param_compression_x > 1).sum())}/{n}")
    print(f"  Configs where surrogate wins on speed AND doesn't lose accuracy significantly: "
          f"{both_better}/{n}")
    print("\nMean +/- std across all configs (unweighted across very different problems --")
    print("read the per-config table above for the real picture, this row is a rough gauge only):")
    print(f"  accuracy_delta_pp:    {comparison.accuracy_delta_pp.mean():+.2f} +/- {comparison.accuracy_delta_pp.std():.2f}")
    print(f"  speedup_x:            {comparison.speedup_x.mean():.2f} +/- {comparison.speedup_x.std():.2f}")
    print(f"  param_compression_x:  {comparison.param_compression_x.mean():.2f} +/- {comparison.param_compression_x.std():.2f}")
    print(f"  fidelity_agreement:   {comparison.fidelity_agreement.mean():.4f} +/- {comparison.fidelity_agreement.std():.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_accuracy_fidelity(df, fidelity_df, path):
    configs = df["config"].unique()
    full_batch = df.groupby(["config", "model"])["batch_size"].transform("max")
    d = df[df["batch_size"] == full_batch]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(configs))
    width = 0.35
    base_acc = [d[(d.config == c) & (d.model == "baseline")]["test_accuracy"].iloc[0] for c in configs]
    simp_acc = [d[(d.config == c) & (d.model == "simplified")]["test_accuracy"].iloc[0] for c in configs]

    ax.bar(x - width/2, base_acc, width, label="Baseline network", color="#3aa0ff")
    ax.bar(x + width/2, simp_acc, width, label="Simplified surrogate", color="#ff7a3a")
    for i, c in enumerate(configs):
        fid = fidelity_df[fidelity_df.config == c]["fidelity_agreement"].iloc[0]
        ax.annotate(f"fidelity\n{fid:.2f}", (x[i], max(base_acc[i], simp_acc[i]) + 0.02),
                    ha="center", fontsize=8, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Test-set accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Held-out test accuracy: baseline vs. simplified surrogate")
    ax.legend()
    ax.axhline(1.0, color="#999", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency_vs_batch(df, path):
    configs = df["config"].unique()
    n = len(configs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, c in enumerate(configs):
        ax = axes[i // cols][i % cols]
        for model_name, color in [("baseline", "#3aa0ff"), ("simplified", "#ff7a3a")]:
            sub = df[(df.config == c) & (df.model == model_name)].sort_values("batch_size")
            ax.errorbar(sub["batch_size"], sub["latency_mean_s"], yerr=sub["latency_std_s"],
                        marker="o", label=model_name, color=color, capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency (s), mean ± std over "
                       f"{N_LATENCY_REPEATS} runs")
        ax.set_title(c, fontsize=10)
        ax.legend(fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Test-time inference latency vs. batch size (log-log)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speedup_summary(df, path):
    configs = df["config"].unique()
    full_batch = df.groupby(["config", "model"])["batch_size"].transform("max")
    d = df[df["batch_size"] == full_batch]

    speedups = []
    for c in configs:
        t_base = d[(d.config == c) & (d.model == "baseline")]["latency_mean_s"].iloc[0]
        t_simp = d[(d.config == c) & (d.model == "simplified")]["latency_mean_s"].iloc[0]
        speedups.append(t_base / t_simp)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3fae5c" if s >= 1 else "#d13b3b" for s in speedups]
    x = np.arange(len(configs))
    bars = ax.bar(x, speedups, color=colors)
    ax.axhline(1.0, color="#333", linewidth=1, linestyle="--")
    ax.set_ylabel("Speedup (baseline latency / simplified latency)\n>1 = simplified faster, <1 = baseline faster")
    ax.set_title("Inference speedup at full test-batch size, by config")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    for bar, s in zip(bars, speedups):
        ax.annotate(f"{s:.2f}x", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pareto(df, path):
    full_batch = df.groupby(["config", "model"])["batch_size"].transform("max")
    d = df[df["batch_size"] == full_batch]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    markers = {"baseline": "o", "simplified": "^"}
    colors = plt.cm.tab10(np.linspace(0, 1, d["config"].nunique()))
    color_map = dict(zip(d["config"].unique(), colors))

    for _, row in d.iterrows():
        ax.scatter(row["latency_mean_s"], row["test_accuracy"],
                   marker=markers[row["model"]], s=110,
                   color=color_map[row["config"]],
                   edgecolor="black", linewidth=0.6,
                   label=f"{row['config']} ({row['model']})")

    ax.set_xscale("log")
    ax.set_xlabel("Test-time latency, full test batch (s, log scale) — lower is better →")
    ax.set_ylabel("Test-set accuracy — higher is better ↑")
    ax.set_title("Accuracy vs. latency (○ = baseline, ▲ = simplified surrogate)\n"
                 "Up-and-to-the-left = genuinely better on both axes")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", label="baseline", markersize=9),
               plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", label="simplified", markersize=9)]
    for c, col in color_map.items():
        handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=col, label=c, markersize=9))
    ax.legend(handles=handles, fontsize=7, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_complexity(df, path):
    configs = df["config"].unique()
    d = df.drop_duplicates(subset=["config", "model"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    width = 0.35
    base_p = [d[(d.config == c) & (d.model == "baseline")]["n_parameters"].iloc[0] for c in configs]
    simp_p = [d[(d.config == c) & (d.model == "simplified")]["n_parameters"].iloc[0] for c in configs]

    ax.bar(x - width/2, base_p, width, label="Baseline network", color="#3aa0ff")
    ax.bar(x + width/2, simp_p, width, label="Simplified surrogate", color="#ff7a3a")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Number of parameters / coefficients (log scale)")
    ax.set_title("Model size: baseline weights vs. surrogate coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scorecard(comparison, path):
    """The single 'is this actually better' plot for a paper: accuracy delta
    and speedup, side by side per config, on a shared x-axis."""
    configs = comparison["config"]
    x = np.arange(len(configs))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    colors1 = ["#3fae5c" if v >= 0 else "#d13b3b" for v in comparison.accuracy_delta_pp]
    bars1 = ax1.bar(x, comparison.accuracy_delta_pp, color=colors1)
    ax1.axhline(0, color="#333", linewidth=1)
    ax1.set_ylabel("Accuracy delta\n(percentage points,\nsimplified - baseline)")
    ax1.set_title("Surrogate improvement scorecard: accuracy change and speedup together")
    for bar, v, sig in zip(bars1, comparison.accuracy_delta_pp, comparison["accuracy_diff_significant_at_0.05"]):
        marker = "*" if sig else ""
        ax1.annotate(f"{v:+.1f}{marker}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.text(0.99, 0.995, "* = statistically significant (McNemar, p<0.05)",
              fontsize=8, ha="right", va="top", color="#666")

    colors2 = ["#3fae5c" if v >= 1 else "#d13b3b" for v in comparison.speedup_x]
    bars2 = ax2.bar(x, comparison.speedup_x, color=colors2)
    ax2.axhline(1.0, color="#333", linewidth=1, linestyle="--")
    ax2.set_ylabel("Speedup\n(baseline / simplified latency)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=30, ha="right")
    for bar, v in zip(bars2, comparison.speedup_x):
        ax2.annotate(f"{v:.2f}x", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    df, fidelity_df, mcnemar_df, chosen_config_df = run_all_experiments()
    comparison = build_comparison_table(df, fidelity_df, mcnemar_df, chosen_config_df)

    df.to_csv(f"{OUT_DIR}/comparison_results.csv", index=False)
    comparison.to_csv(f"{OUT_DIR}/comparison_scores.csv", index=False)

    full_batch = df.groupby(["config", "model"])["batch_size"].transform("max")
    summary = df[df["batch_size"] == full_batch].merge(fidelity_df, on="config", how="left")
    summary_cols = ["config", "model", "n_test_samples", "test_accuracy", "test_f1",
                     "test_precision", "test_recall", "fidelity_agreement",
                     "latency_mean_s", "latency_std_s", "throughput_samples_per_s",
                     "n_parameters"]
    summary[summary_cols].to_csv(f"{OUT_DIR}/comparison_summary.csv", index=False)

    print_full_report(df, comparison)

    plot_accuracy_fidelity(df, fidelity_df, f"{PLOT_DIR}/accuracy_fidelity.png")
    plot_latency_vs_batch(df, f"{PLOT_DIR}/latency_vs_batch_size.png")
    plot_speedup_summary(df, f"{PLOT_DIR}/speedup_summary.png")
    plot_pareto(df, f"{PLOT_DIR}/accuracy_vs_latency_pareto.png")
    plot_model_complexity(df, f"{PLOT_DIR}/model_complexity.png")
    plot_scorecard(comparison, f"{PLOT_DIR}/scorecard.png")

    print("\n" + "=" * 100)
    print("FILES WRITTEN")
    print("=" * 100)
    print(f"  {OUT_DIR}/comparison_results.csv   ({len(df)} rows: every batch size, every config, both models)")
    print(f"  {OUT_DIR}/comparison_summary.csv   ({len(summary)} rows: one per config/model, full test batch)")
    print(f"  {OUT_DIR}/comparison_scores.csv    ({len(comparison)} rows: one per config, all comparison scores)")
    print(f"  {PLOT_DIR}/ (6 plots: accuracy_fidelity, latency_vs_batch_size, speedup_summary,")
    print(f"                        accuracy_vs_latency_pareto, model_complexity, scorecard)")


if __name__ == "__main__":
    main()
