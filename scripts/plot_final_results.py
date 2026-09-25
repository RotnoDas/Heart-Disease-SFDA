import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "results" / "final" / "master_multiseed_summary.csv"

if not csv_path.exists():
    print(f"Error: {csv_path} not found. Please run the full pipeline first.")
    exit()

df = pd.read_csv(csv_path)

# We want to compare 'source_only' vs the best adaptation method (e.g., 'reliability_gated_sfda' or 'conservative_candidate')
baseline = df[df['method'] == 'source_only'].copy()
sfda = df[df['method'] == 'conservative_candidate'].copy()

# Ensure same order of targets
targets = baseline['target'].unique()

baseline_aucs = []
sfda_aucs = []

for t in targets:
    b_val = baseline[baseline['target'] == t]['mean_roc_auc'].values[0] * 100
    s_val = sfda[sfda['target'] == t]['mean_roc_auc'].values[0] * 100
    baseline_aucs.append(b_val)
    sfda_aucs.append(s_val)

import numpy as np
x = np.arange(len(targets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_aucs, width, label='Baseline (Source Only)', color='lightcoral')
rects2 = ax.bar(x + width/2, sfda_aucs, width, label='Proposed SFDA (Conservative)', color='mediumseagreen')

ax.set_ylabel('Mean ROC-AUC (%)')
ax.set_title('Robust Cross-Domain Performance (Averaged over 600 runs)')
ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in targets])
ax.set_ylim(0, 100)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
out_dir = ROOT / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "original_pipeline_accuracy_comparison.png")
plt.close()

print(f"Successfully generated comparison graph from CSV data!")
print(f"Saved to: {out_dir / 'original_pipeline_accuracy_comparison.png'}")
