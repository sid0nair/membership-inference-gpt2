"""Redraw figures/progression.png from the recorded results (numbers copied from the notebooks)."""
import matplotlib.pyplot as plt
import numpy as np

steps = ["Mean loss\n(no reference)", "Base GPT-2\nreference", "Sliding\nwindows", "1 shadow\nreference",
         "1 shadow +\nwindows", "Mean gap,\n3 refs", "Mean gap,\n5 refs", "Gaussian LR\n4 IN/OUT pairs"]
dev = [0.3, 1.7, 2.6, 19.2, 36.4, 61.7, 67.9, 71.2]
held = [None, None, None, None, 32.2, 66.8, 69.1, 76.6]

x = np.arange(len(steps)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.bar(x - w / 2, dev, w, label="development target (labels public)", color="#4C72B0")
hx = [i for i, v in enumerate(held) if v is not None]
ax.bar(np.array(hx) + w / 2, [held[i] for i in hx], w, label="held-out target (labels hidden)", color="#DD8452")
for i, v in enumerate(dev):
    ax.text(i - w / 2, v + 1, f"{v:g}", ha="center", fontsize=8)
for i in hx:
    ax.text(i + w / 2, held[i] + 1, f"{held[i]:g}", ha="center", fontsize=8)
ax.set_xticks(x, steps, fontsize=8)
ax.set_ylabel("TPR (%) at FPR < 1%")
ax.set_ylim(0, 85)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Membership inference on LoRA fine-tuned GPT-2: TPR at FPR < 1%", fontsize=10)
fig.tight_layout()
fig.savefig("figures/progression.png")
