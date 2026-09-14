import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results_summary.json") as f:
    summary = json.load(f)
with open("results_history.json") as f:
    history = json.load(f)

# ---- 1. Bar chart: accuracy per task per model ----
tasks = list(summary.keys())
fig, ax = plt.subplots(figsize=(11, 6))

all_models = []
for t in tasks:
    for m in summary[t]:
        if m not in all_models:
            all_models.append(m)

colors = {"MLP (scalar baseline)": "#4C72B0"}
palette = ["#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
ci = 0
for m in all_models:
    if m not in colors:
        colors[m] = palette[ci % len(palette)]
        ci += 1

n_models = len(all_models)
width = 0.8 / n_models
x = np.arange(len(tasks))

for i, m in enumerate(all_models):
    accs = [summary[t][m]["val_acc"] if m in summary[t] else np.nan for t in tasks]
    ax.bar(x + i * width - 0.4 + width / 2, accs, width, label=m, color=colors[m])

ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=20, ha="right")
ax.set_ylabel("Validation accuracy")
ax.set_title("Standard MLP vs Vector-NN (squash) across tasks")
ax.set_ylim(0.5, 1.02)
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.close()

# ---- 2. Training curves for digits + vector_structured task ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, task in zip(axes, ["digits", "vector_structured"]):
    for m, hist in history[task].items():
        losses = hist["train_loss"]
        ax.plot(losses, label=m, color=colors.get(m, None))
    ax.set_title(f"Training loss: {task}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# ---- 3. Params vs accuracy scatter (efficiency) ----
fig, ax = plt.subplots(figsize=(7, 6))
for t in tasks:
    for m in summary[t]:
        p = summary[t][m]["n_params"]
        a = summary[t][m]["val_acc"]
        ax.scatter(p, a, color=colors[m], s=60)
        ax.annotate(f"{t[:4]}", (p, a), fontsize=7, xytext=(3, 3), textcoords="offset points")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], label=m, markersize=8)
           for m in all_models]
ax.legend(handles=handles, fontsize=8)
ax.set_xlabel("parameter count")
ax.set_ylabel("validation accuracy")
ax.set_title("Accuracy vs parameter count (matched architectures)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("params_vs_accuracy.png", dpi=150)
plt.close()

print("Saved: accuracy_comparison.png, training_curves.png, params_vs_accuracy.png")
