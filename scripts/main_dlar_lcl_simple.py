import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import shap

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
    parser.add_argument("--target", type=str, default="hungary", help="Target hospital name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print(f" Source-Free Domain Adaptation: {args.source.upper()} -> {args.target.upper()}")
    print("=" * 60)

    # 1. Load Data
    X_source_raw, y_source = load_hospital_data(args.source)
    X_target_raw, y_target = load_hospital_data(args.target)

    # 2. Source-only Preprocessing (Target data is transformed using Source fitted scaler)
    preprocessor = build_linear_preprocessor()
    X_source = preprocessor.fit_transform(X_source_raw).astype(np.float32)
    X_target = preprocessor.transform(X_target_raw).astype(np.float32)

    # 3. Train Pre-trained Source Model (Equivalent to loading weights in Emotion SF-UDA)
    source_model = train_source_model(X_source, y_source, device)
    
    # 4. Source-Free Domain Adaptation (DLAR & LCL) on Target Data
    print(f"\n--> Starting SFDA (DLAR-LCL) on Target Data ({len(X_target)} samples)...")
    print("    * No Source labels or Source data is used here!")
    
    adapted_model, diagnostics = adapt_dlar_lcl(
        source_model=source_model,
        X_target_unlabeled=X_target, # Unlabeled Target Data
        device=device,
        learning_rate=1e-4,
        weight_decay=5e-4,
        dlar_epochs=5,
        lcl_epochs=10,
        beta=1.0,
        gamma=1.0,
        k_neighbors=5,
    )
    print("--> Adaptation Complete.")

    # 5. Evaluate on Target Data with PC-TTA
    print("\n--> Evaluating Adapted Model with PC-TTA (Prediction Confidence-aware Test-Time Augmentation)...")
    adapted_predictions = predict_with_pctta(adapted_model, X_target, device, entropy_threshold=0.9)
    
    # Compute basic accuracy for quick view
    accuracy = np.mean(adapted_predictions == y_target) * 100
    
    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print(f"Target Hospital: {args.target.upper()}")
    print(f"Test Accuracy:   {accuracy:.2f}%")
    print("=" * 60)

    # 6. Explainable AI (SHAP)
    feature_names = preprocessor.get_feature_names_out()
    compute_and_print_xai(adapted_model, X_target, feature_names, device)

if __name__ == "__main__":
    main()
