import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import copy

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
    feature_names = df.drop(columns=['target']).columns.tolist()
    return X, y, feature_names

def train_source_model(X_source, y_source, device, epochs=50, batch_size=256):
    print(f"--> Training Source Model on {len(X_source)} minority (Black/Asian) patients...")
    model = DualHeadHeartMLP(input_dim=X_source.shape[1], hidden_dims=(128, 64), dropout=0.30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.tensor(X_source), torch.tensor(y_source))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, logits_1, logits_2 = model(X_batch)
            loss_1 = F.cross_entropy(logits_1, y_batch)
            loss_2 = F.cross_entropy(logits_2, y_batch)
            supervised_loss = loss_1 + loss_2
            prob_1, prob_2 = F.softmax(logits_1, dim=1), F.softmax(logits_2, dim=1)
            discrepancy_loss = torch.mean(torch.abs(prob_1 - prob_2))
            total_loss = torch.exp(supervised_loss) + (0.5 * discrepancy_loss)
            total_loss.backward()
            optimizer.step()
            
    print("--> Source Model Training Complete.")
    return model

def predict_baseline(model, X, device):
    model.eval()
    X_tensor = torch.tensor(X).to(device)
    with torch.no_grad():
        _, l1, l2 = model(X_tensor)
        p1, p2 = F.softmax(l1, dim=1), F.softmax(l2, dim=1)
        return ((p1 + p2) / 2.0).argmax(dim=1).cpu().numpy()

def predict_with_pctta(model, X, device, entropy_threshold=0.9):
    model.eval()
    X_tensor = torch.tensor(X).to(device)
    with torch.no_grad():
        _, l1, l2 = model(X_tensor)
        p_cmb = (F.softmax(l1, dim=1) + F.softmax(l2, dim=1)) / 2.0
        entropy = -torch.sum(p_cmb * torch.log2(p_cmb + 1e-9), dim=1)
        predictions = p_cmb.argmax(dim=1).cpu().numpy()
        
        uncertain_idx = torch.where(entropy >= entropy_threshold)[0]
        if len(uncertain_idx) > 0:
            print(f"    * Applying PC-TTA on {len(uncertain_idx)} highly uncertain samples...")
            X_un = X_tensor[uncertain_idx]
            all_preds = [p_cmb[uncertain_idx].argmax(dim=1)]
            for std in [0.1, 0.3, 0.5]:
                X_noisy = X_un + torch.randn_like(X_un) * std
                _, nl1, nl2 = model(X_noisy)
                np_cmb = (F.softmax(nl1, dim=1) + F.softmax(nl2, dim=1)) / 2.0
                all_preds.append(np_cmb.argmax(dim=1))
            final_uncertain, _ = torch.mode(torch.stack(all_preds, dim=0), dim=0)
            predictions[uncertain_idx.cpu().numpy()] = final_uncertain.cpu().numpy()
    return predictions

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy (0)', 'Disease (1)'], yticklabels=['Healthy (0)', 'Disease (1)'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    out_dir = ROOT / "figures" / "race_sfda"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename)
    plt.close()

def main():
    set_seed(42)
    device = get_device()
    
    print("=" * 70)
    print(" DEMOGRAPHIC SFDA (FAIRNESS) | Source: Minority (Black/Asian) -> Target: White")
    print("=" * 70)
    
    # Load data
    src_path = ROOT / "data" / "processed" / "cdc_source_minority.csv"
    tgt_path = ROOT / "data" / "processed" / "cdc_target_white.csv"
    
    X_src_raw, y_src, f_names = load_data(src_path)
    X_tgt_raw, y_tgt, _ = load_data(tgt_path)
    
    # Scale features
    scaler = StandardScaler()
    X_src = scaler.fit_transform(X_src_raw).astype(np.float32)
    X_tgt = scaler.transform(X_tgt_raw).astype(np.float32)
    
    # Train Source
    source_model = train_source_model(X_src, y_src, device)
    
    # Baseline Eval
    print("\n--> Evaluating Baseline Model on White patients...")
    b_preds = predict_baseline(source_model, X_tgt, device)
    b_acc = np.mean(b_preds == y_tgt) * 100
    print(f"[ BASELINE ] White Patients Accuracy: {b_acc:.2f}%")
    plot_cm(y_tgt, b_preds, "Baseline (Source Only) - Target: White", "cm_white_baseline.png")
    
    # SFDA Eval
    print(f"\n--> Starting SFDA (DLAR-LCL) on White patients ({len(X_tgt)} samples)...")
    model_to_adapt = copy.deepcopy(source_model)
    adapted_model, _ = adapt_dlar_lcl(
        source_model=model_to_adapt, X_target_unlabeled=X_tgt, device=device,
        learning_rate=5e-4, weight_decay=1e-4, dlar_epochs=5, lcl_epochs=10,
        beta=1.0, gamma=1.0, k_neighbors=5
    )
    
    print("--> Evaluating Adapted Model on White patients with PC-TTA...")
    a_preds = predict_with_pctta(adapted_model, X_tgt, device)
    a_acc = np.mean(a_preds == y_tgt) * 100
    print(f"[ PROPOSED SFDA ] White Patients Accuracy: {a_acc:.2f}%")
    plot_cm(y_tgt, a_preds, "Proposed SFDA - Target: White", "cm_white_adapted.png")
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"Target: WHITE | Baseline Accuracy: {b_acc:.2f}% | Adapted Accuracy: {a_acc:.2f}%")
    
    # 1. Accuracy Bar Chart
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['Baseline', 'Proposed SFDA'], y=[b_acc, a_acc], hue=['Baseline', 'Proposed SFDA'], palette=['lightcoral', 'mediumseagreen'], legend=False)
    plt.title('Demographic Domain Shift: Minority -> White')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate([b_acc, a_acc]):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.savefig(ROOT / "figures" / "race_sfda" / "accuracy_comparison.png")
    plt.close()
    
    # 2. Dataset Demographics Plot
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['Source (Black/Asian)', 'Target (White)'], y=[len(X_src_raw), len(X_tgt_raw)], hue=['Source (Black/Asian)', 'Target (White)'], palette=['skyblue', 'gold'], legend=False)
    plt.title('Patient Demographics for Adaptation')
    plt.ylabel('Number of Patients')
    plt.savefig(ROOT / "figures" / "race_sfda" / "demographics_count.png")
    plt.close()
    
    # 3. Feature Importance Plot (First Layer Weights absolute sum)
    try:
        weights = source_model.feature_extractor[0].weight.detach().cpu().numpy()
        importance = np.sum(np.abs(weights), axis=0)
        # Sort and take top 10
        idx = np.argsort(importance)[::-1][:10]
        top_features = [f_names[i] for i in idx]
        top_importance = importance[idx]
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_importance, y=top_features, hue=top_features, palette='viridis', legend=False)
        plt.title("Top 10 Most Important Health Factors (Source Model)")
        plt.xlabel("Importance (Absolute Weight Sum)")
        plt.tight_layout()
        plt.savefig(ROOT / "figures" / "race_sfda" / "feature_importance.png")
        plt.close()
    except Exception as e:
        print("Could not generate feature importance plot:", e)
    
    # 4. Model Confidence (KDE) Plot
    try:
        model_to_adapt.eval()
        adapted_model.eval()
        with torch.no_grad():
            b_l1, b_l2 = model_to_adapt(torch.tensor(X_tgt).to(device))[1:]
            b_probs = ((F.softmax(b_l1, dim=1) + F.softmax(b_l2, dim=1)) / 2.0).max(dim=1)[0].cpu().numpy()
            
            a_l1, a_l2 = adapted_model(torch.tensor(X_tgt).to(device))[1:]
            a_probs = ((F.softmax(a_l1, dim=1) + F.softmax(a_l2, dim=1)) / 2.0).max(dim=1)[0].cpu().numpy()
            
        plt.figure(figsize=(7, 5))
        sns.kdeplot(b_probs, label="Baseline Confidence", fill=True, color='lightcoral')
        sns.kdeplot(a_probs, label="Adapted Confidence", fill=True, color='mediumseagreen')
        plt.title("Model Confidence Distribution on White Patients")
        plt.xlabel("Prediction Confidence (Probability)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(ROOT / "figures" / "race_sfda" / "confidence_distribution.png")
        plt.close()
    except Exception as e:
        print("Could not generate KDE plot:", e)
    
    print("\n* All presentation graphs have been saved to 'figures/race_sfda/'")

if __name__ == "__main__":
    main()
