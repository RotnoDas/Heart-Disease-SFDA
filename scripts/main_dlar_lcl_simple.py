import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sfda.data.preprocessing import CORE8, build_linear_preprocessor
from heart_sfda.models.dual_head_mlp import DualHeadHeartMLP
from heart_sfda.adaptation.dlar_lcl import adapt_dlar_lcl, predict_dlar_lcl_probability
from heart_sfda.evaluation.metrics import compute_binary_metrics
from heart_sfda.utils.device import get_device
from heart_sfda.utils.seed import set_seed

def load_hospital_data(domain_name):
    """Loads dataset for a specific hospital"""
    data_path = ROOT / "data" / "processed" / f"{domain_name}.csv"
    df = pd.read_csv(data_path)
    X_raw = df[CORE8].copy()
    y = df["target"].to_numpy(dtype=np.int64)
    return X_raw, y

def train_source_model(X_source, y_source, device, epochs=20, batch_size=64):
    """Pretrains the dual-head model on the source domain (similar to pretrain.py)"""
    print(f"--> Training Source Model on {len(X_source)} samples...")
    model = DualHeadHeartMLP(input_dim=X_source.shape[1], hidden_dims=(32, 16), dropout=0.20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    # Simple DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.tensor(X_source, dtype=torch.float32), torch.tensor(y_source, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, logits_1, logits_2 = model(X_batch)
            
            # Supervised Loss
            loss_1 = F.cross_entropy(logits_1, y_batch)
            loss_2 = F.cross_entropy(logits_2, y_batch)
            supervised_loss = loss_1 + loss_2
            
            # Discrepancy Loss (mean |softmax(C1)-softmax(C2)|)
            prob_1, prob_2 = F.softmax(logits_1, dim=1), F.softmax(logits_2, dim=1)
            discrepancy_loss = torch.mean(torch.abs(prob_1 - prob_2))
            
            total_loss = torch.exp(supervised_loss) + (0.5 * discrepancy_loss)
            total_loss.backward()
            optimizer.step()
            
    print("--> Source Model Training Complete.")
    return model

class DualHeadLogitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, logits_1, logits_2 = self.model(x)
        return (logits_1 + logits_2) / 2.0

def compute_and_print_xai(model, X, feature_names, device):
    print("\n" + "=" * 60)
    print(" EXPLAINABLE AI (xAI) - SHAP FEATURE IMPORTANCE")
    print("=" * 60)
    
    wrapper = DualHeadLogitWrapper(model).to(device)
    wrapper.eval()
    
    # Use a small background dataset to speed up SHAP
    bg_size = min(32, len(X))
    np.random.seed(42)
    bg_indices = np.random.choice(len(X), bg_size, replace=False)
    X_bg = torch.tensor(X[bg_indices], dtype=torch.float32).to(device)
    
    # Explain a subset to keep it fast
    exp_size = min(100, len(X))
    exp_indices = np.random.choice(len(X), exp_size, replace=False)
    X_exp = torch.tensor(X[exp_indices], dtype=torch.float32).to(device)
    
    explainer = shap.DeepExplainer(wrapper, X_bg)
    shap_values = explainer.shap_values(X_exp, check_additivity=False)
    
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1] # Importance for Heart Disease (class 1)
    else:
        shap_values_class1 = shap_values[..., 1]
        
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]
    
    print("Top 5 Most Important Features for Target Hospital:")
    for i in range(5):
        if i < len(top_indices):
            idx = top_indices[i]
            print(f"  {i+1}. {feature_names[idx]:<25} (Importance Score: {mean_abs_shap[idx]:.4f})")
    print("=" * 60)

def plot_confusion_matrix(y_true, y_pred, target_name, title_prefix="SFDA", filename_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy (0)', 'Disease (1)'],
                yticklabels=['Healthy (0)', 'Disease (1)'])
    plt.title(f'{title_prefix} Confusion Matrix - {target_name.upper()}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    out_dir = ROOT / "figures" / "simple_sfda"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"cm_{target_name}{filename_suffix}.png")
    plt.close()

def plot_grouped_accuracy_bar(targets, baseline_accs, adapted_accs, source_name):
    import numpy as np
    x = np.arange(len(targets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline (Source Only)', color='lightcoral')
    rects2 = ax.bar(x + width/2, adapted_accs, width, label='Proposed SFDA', color='mediumseagreen')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Baseline vs SFDA Performance (Source: {source_name.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets])
    ax.set_ylim(0, 100)
    ax.legend()
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    out_dir = ROOT / "figures" / "simple_sfda"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"accuracy_comparison_{source_name}.png")
    plt.close()

def predict_baseline(model, X, device):
    """Simple inference without PC-TTA for baseline evaluation."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, l1, l2 = model(X_tensor)
        p1, p2 = F.softmax(l1, dim=1), F.softmax(l2, dim=1)
        p_cmb = (p1 + p2) / 2.0
        return p_cmb.argmax(dim=1).cpu().numpy()

def predict_with_pctta(model, X, device, entropy_threshold=0.9):
    """Prediction Confidence-aware Test-Time Augmentation (PC-TTA) for inference."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, logits_1, logits_2 = model(X_tensor)
        prob_1 = F.softmax(logits_1, dim=1)
        prob_2 = F.softmax(logits_2, dim=1)
        prob_combined = (prob_1 + prob_2) / 2.0
        
        # Calculate Entropy (base 2)
        entropy = -torch.sum(prob_combined * torch.log2(prob_combined + 1e-9), dim=1)
        
        predictions = prob_combined.argmax(dim=1).cpu().numpy()
        
        # Identify high entropy (uncertain) samples
        uncertain_idx = torch.where(entropy >= entropy_threshold)[0]
        
        if len(uncertain_idx) > 0:
            print(f"    * Applying PC-TTA on {len(uncertain_idx)} highly uncertain samples...")
            X_uncertain = X_tensor[uncertain_idx]
            
            # Augmentations (Adding Gaussian noise to standardized tabular features)
            noise_stds = [0.1, 0.3, 0.5]
            all_preds = [prob_combined[uncertain_idx].argmax(dim=1)]
            
            for std in noise_stds:
                noise = torch.randn_like(X_uncertain) * std
                X_noisy = X_uncertain + noise
                _, l1, l2 = model(X_noisy)
                p1, p2 = F.softmax(l1, dim=1), F.softmax(l2, dim=1)
                p_cmb = (p1 + p2) / 2.0
                all_preds.append(p_cmb.argmax(dim=1))
                
            # Majority vote
            all_preds_stacked = torch.stack(all_preds, dim=0) # Shape: (4, num_uncertain)
            final_uncertain_preds, _ = torch.mode(all_preds_stacked, dim=0)
            
            # Update predictions
            predictions[uncertain_idx.cpu().numpy()] = final_uncertain_preds.cpu().numpy()
            
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Standalone SF-UDA for Heart Disease (DLAR-LCL)")
    parser.add_argument("--source", type=str, default="cleveland", help="Source hospital name")
    parser.add_argument("--target", type=str, default="all", help="Target hospital name, or 'all' for all other hospitals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    ALL_DOMAINS = ["cleveland", "hungary", "switzerland", "va_long_beach"]
    
    if args.target.lower() == "all":
        targets = [d for d in ALL_DOMAINS if d != args.source.lower()]
    else:
        targets = [args.target.lower()]

    print("=" * 60)
    print(f" Source-Free Domain Adaptation (SFDA) | Source: {args.source.upper()}")
    print("=" * 60)

    # 1. Load & Process Source Data
    X_source_raw, y_source = load_hospital_data(args.source)
    preprocessor = build_linear_preprocessor()
    X_source = preprocessor.fit_transform(X_source_raw).astype(np.float32)

    # 2. Train Pre-trained Source Model
    source_model = train_source_model(X_source, y_source, device)
    
    baseline_accuracies = []
    adapted_accuracies = []
    import copy

    for target in targets:
        print("\n" + "*" * 60)
        print(f" TESTING ON TARGET: {target.upper()}")
        print("*" * 60)
        
        # Load Target Data
        X_target_raw, y_target = load_hospital_data(target)
        X_target = preprocessor.transform(X_target_raw).astype(np.float32)
        
        # Baseline Evaluation (Source-only model on Target)
        print(f"--> Evaluating Baseline (Source-Only) Model on {target.upper()}...")
        baseline_preds = predict_baseline(source_model, X_target, device)
        baseline_acc = np.mean(baseline_preds == y_target) * 100
        baseline_accuracies.append(baseline_acc)
        print(f"[ BASELINE ] {target.upper()} Accuracy: {baseline_acc:.2f}%\n")
        
        # Plot Baseline Confusion Matrix
        plot_confusion_matrix(y_target, baseline_preds, target, title_prefix="Baseline", filename_suffix="_baseline")
        
        # Deepcopy the source model so adaptation on one target doesn't affect another
        model_to_adapt = copy.deepcopy(source_model)
        
        # 3. Source-Free Domain Adaptation (DLAR & LCL)
        print(f"--> Starting SFDA (DLAR-LCL) on {target.upper()} ({len(X_target)} samples)...")
        adapted_model, diagnostics = adapt_dlar_lcl(
            source_model=model_to_adapt,
            X_target_unlabeled=X_target,
            device=device,
            learning_rate=1e-4,
            weight_decay=5e-4,
            dlar_epochs=5,
            lcl_epochs=10,
            beta=1.0,
            gamma=1.0,
            k_neighbors=5,
        )
        
        # 4. Evaluate with PC-TTA
        print(f"--> Evaluating Adapted Model on {target.upper()} with PC-TTA...")
        adapted_predictions = predict_with_pctta(adapted_model, X_target, device, entropy_threshold=0.9)
        
        accuracy = np.mean(adapted_predictions == y_target) * 100
        adapted_accuracies.append(accuracy)
        
        print(f"\n[ PROPOSED SFDA ] {target.upper()} Accuracy: {accuracy:.2f}%")
        
        # Plot Adapted Confusion Matrix
        plot_confusion_matrix(y_target, adapted_predictions, target, title_prefix="Proposed SFDA", filename_suffix="_adapted")
        
        # 5. Explainable AI (SHAP)
        feature_names = preprocessor.get_feature_names_out()
        compute_and_print_xai(adapted_model, X_target, feature_names, device)

    # 6. Print Final Summary
    print("\n" + "=" * 60)
    print(" FINAL CROSS-DOMAIN EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Source Model: {args.source.upper()}")
    for t, b_acc, a_acc in zip(targets, baseline_accuracies, adapted_accuracies):
        print(f"  -> Target: {t.upper():<15} | Baseline: {b_acc:.2f}% | Proposed: {a_acc:.2f}%")
    print("-" * 60)
    avg_b = np.mean(baseline_accuracies)
    avg_a = np.mean(adapted_accuracies)
    print(f"  AVERAGE ACCURACY | Baseline: {avg_b:.2f}% | Proposed: {avg_a:.2f}%")
    print("=" * 60)

    # 7. Plot Accuracy Bar Chart
    plot_grouped_accuracy_bar(targets, baseline_accuracies, adapted_accuracies, args.source)
    print("\n* All graphs (Confusion Matrices & Grouped Bar Chart) have been saved to 'figures/simple_sfda/'")

if __name__ == "__main__":
    main()
