import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import copy
from pathlib import Path
import sys
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sfda.models.dual_head_mlp import DualHeadHeartMLP
from heart_sfda.adaptation.dlar_lcl import adapt_dlar_lcl
from heart_sfda.utils.device import get_device
from heart_sfda.utils.seed import set_seed

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['target']).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    return X, y

def train_source(X, y, device):
    model = DualHeadHeartMLP(input_dim=X.shape[1], hidden_dims=(128, 64), dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model.train()
    for _ in range(40):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            _, l1, l2 = model(bx)
            loss = F.cross_entropy(l1, by) + F.cross_entropy(l2, by) + 0.5 * torch.mean(torch.abs(F.softmax(l1, dim=1) - F.softmax(l2, dim=1)))
            loss.backward()
            opt.step()
    return model

def main():
    set_seed(42)
    device = get_device()
    
    # Load CDC Race Data
    X_src, y_src = load_data(ROOT / "data" / "processed" / "cdc_source_minority.csv")
    X_tgt, y_tgt = load_data(ROOT / "data" / "processed" / "cdc_target_white.csv")
    
    # Downsample target slightly to make silhouette score (O(N^2)) computation faster for this script
    idx = np.random.choice(len(X_tgt), 2000, replace=False)
    X_tgt_small = X_tgt[idx]
    y_tgt_small = y_tgt[idx]
    
    scaler = StandardScaler()
    X_src = scaler.fit_transform(X_src).astype(np.float32)
    X_tgt_small = scaler.transform(X_tgt_small).astype(np.float32)
    
    print("Training source model...")
    source_model = train_source(X_src, y_src, device)
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: Varying K in LCL
    # ---------------------------------------------------------
    print("Running Experiment 1: Varying k...")
    k_values = [0, 5, 10, 15, 20]
    k_accs, k_times = [], []
    
    for k in k_values:
        m_copy = copy.deepcopy(source_model)
        start = time.time()
        
        # If k=0, it's equivalent to standard DLAR without LCL neighbors
        d_eps = 5
        l_eps = 0 if k == 0 else 5
        
        ad_model, _ = adapt_dlar_lcl(
            source_model=m_copy, X_target_unlabeled=X_tgt_small, device=device,
            learning_rate=5e-4, weight_decay=1e-4, dlar_epochs=d_eps, lcl_epochs=l_eps,
            beta=1.0, gamma=1.0, k_neighbors=max(1, k)
        )
        t = time.time() - start
        
        # Eval
        ad_model.eval()
        with torch.no_grad():
            _, l1, l2 = ad_model(torch.tensor(X_tgt_small).to(device))
            p = ((F.softmax(l1, dim=1) + F.softmax(l2, dim=1)) / 2.0).argmax(dim=1).cpu().numpy()
            acc = np.mean(p == y_tgt_small) * 100
            
        k_accs.append(acc)
        k_times.append(t)
        print(f"  k={k}: Acc={acc:.2f}%, Time={t:.2f}s")
        
    # ---------------------------------------------------------
    # EXPERIMENT 2: Epoch tracking (Silhouette vs Accuracy)
    # ---------------------------------------------------------
    print("Running Experiment 2: Epoch tracking...")
    m_copy = copy.deepcopy(source_model)
    opt = torch.optim.Adam(m_copy.parameters(), lr=5e-4, weight_decay=1e-4)
    X_t = torch.tensor(X_tgt_small).to(device)
    
    epochs = [0, 1, 2, 3, 4, 5]
    e_accs, e_sils = [], []
    
    for ep in epochs:
        m_copy.eval()
        with torch.no_grad():
            feats, l1, l2 = m_copy(X_t)
            probs = (F.softmax(l1, dim=1) + F.softmax(l2, dim=1)) / 2.0
            preds = probs.argmax(dim=1).cpu().numpy()
            f_np = feats.cpu().numpy()
            
            # Metrics
            acc = np.mean(preds == y_tgt_small) * 100
            if len(np.unique(preds)) > 1:
                sil = silhouette_score(f_np, preds)
            else:
                sil = 0.0
            
            e_accs.append(acc)
            e_sils.append(sil)
            
        if ep < 5:
            # Simulate 1 epoch of adaptation (DLAR logic)
            m_copy.train()
            opt.zero_grad()
            feats, l1, l2 = m_copy(X_t)
            probs1 = F.softmax(l1, dim=1)
            probs2 = F.softmax(l2, dim=1)
            mean_p = (probs1 + probs2) / 2.0
            
            loss_mcd = torch.mean(torch.norm(probs1 - probs2, p=2, dim=1))
            entropy = -torch.mean(torch.sum(mean_p * torch.log(mean_p + 1e-9), dim=1))
            loss = loss_mcd + 0.5 * entropy
            loss.backward()
            opt.step()
            
    print(f"  Final Epoch Acc={e_accs[-1]:.2f}%, Sil={e_sils[-1]:.4f}")
    
    # ---------------------------------------------------------
    # PLOTTING (Matching Emotion Paper Figure 6 style)
    # ---------------------------------------------------------
    print("Generating plots...")
    out_dir = ROOT / "figures" / "race_sfda"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Varying K (Green/Blue Dual Axis)
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(k_values, k_accs, 'g.-', markersize=12, label='Accuracy (%)')
    ax2.plot(k_values, k_times, 'b.-', markersize=12, label='Training Time (s)')
    
    ax1.set_xlabel('k (Nearest Neighbors)')
    ax1.set_ylabel('Accuracy (%)', color='g')
    ax2.set_ylabel('Training Time (s)', color='b')
    ax1.set_xticks(k_values)
    plt.title('Model performance for varying values of k (Source: Minority -> Target: White)')
    
    fig.tight_layout()
    plt.savefig(out_dir / "paper_style_k_variation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Epoch Tracking (Blue/Magenta Dual Axis)
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(epochs, e_accs, 'b.-', markersize=12, label='Pseudo-Label Accuracy (%)')
    ax2.plot(epochs, e_sils, 'm.-', markersize=12, label='Silhouette Score')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pseudo-Label Accuracy (%)', color='b')
    ax2.set_ylabel('Silhouette Score', color='m')
    ax1.set_xticks(epochs)
    plt.title('Epoch vs Pseudo-Label Accuracy & Silhouette Score')
    
    fig.tight_layout()
    plt.savefig(out_dir / "paper_style_epoch_tracking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All done! Check 'figures/race_sfda/'")

if __name__ == "__main__":
    main()
