import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sfda.models.dual_head_mlp import DualHeadHeartMLP
from heart_sfda.utils.device import get_device
from heart_sfda.utils.seed import set_seed

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['target']).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    feature_names = df.drop(columns=['target']).columns.tolist()
    return X, y, feature_names

def train_source(X, y, device):
    model = DualHeadHeartMLP(input_dim=X.shape[1], hidden_dims=(128, 64), dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model.train()
    for _ in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            _, l1, l2 = model(bx)
            loss = F.cross_entropy(l1, by) + F.cross_entropy(l2, by)
            loss.backward()
            opt.step()
    return model

class ShapWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        _, l1, l2 = self.model(x)
        # DeepExplainer expects the exact logits
        return (l1 + l2) / 2.0

def main():
    set_seed(42)
    device = get_device()
    
    # Load CDC Race Data
    X_src_raw, y_src, f_names = load_data(ROOT / "data" / "processed" / "cdc_source_minority.csv")
    X_tgt_raw, y_tgt, _ = load_data(ROOT / "data" / "processed" / "cdc_target_white.csv")
    
    scaler = StandardScaler()
    X_src = scaler.fit_transform(X_src_raw).astype(np.float32)
    X_tgt = scaler.transform(X_tgt_raw).astype(np.float32)
    
    print("Training source model for SHAP evaluation...")
    source_model = train_source(X_src, y_src, device)
    source_model.eval()
    
    wrapper = ShapWrapper(source_model).to(device)
    wrapper.eval()
    
    # Background data for SHAP (100 samples)
    bg_tensor = torch.tensor(X_src[:100]).to(device)
    # Test data for SHAP (300 samples for dense, beautiful beeswarm)
    test_tensor = torch.tensor(X_tgt[:300]).to(device)
    
    print("Computing SHAP values (this may take 10-20 seconds)...")
    explainer = shap.DeepExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)
    
    # shap_values is typically a list: [shap_class_0, shap_class_1] for binary
    if not isinstance(shap_values, list):
        # Fallback if SHAP returns a single array for some reason (depends on version)
        print("Note: SHAP returned a single array.")
        shap_values = [shap_values, shap_values]
        
    out_dir = ROOT / "figures" / "race_sfda"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating SHAP summary plots...")
    
    # Combined side-by-side plot matching user's image exactly!
    fig = plt.figure(figsize=(16, 7))
    
    # Subplot 1: Negative (Healthy)
    ax1 = fig.add_subplot(121)
    shap.summary_plot(shap_values[0], X_tgt[:300], feature_names=f_names, show=False)
    ax1.set_title("SHAP value (Negative / Healthy)", fontsize=16, fontweight='bold', pad=20)
    
    # Subplot 2: Positive (Disease)
    ax2 = fig.add_subplot(122)
    shap.summary_plot(shap_values[1], X_tgt[:300], feature_names=f_names, show=False)
    ax2.set_title("SHAP value (Positive / Heart Disease)", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(out_dir / "paper_style_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("SHAP beeswarm plots generated successfully in 'figures/race_sfda/'")

if __name__ == "__main__":
    main()
