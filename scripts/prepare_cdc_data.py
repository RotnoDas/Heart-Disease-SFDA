import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

def process_cdc_data():
    print("Loading 2020 data...")
    df_2020 = pd.read_csv(RAW_DIR / "heart_2020.csv")
    print("Loading 2022 data...")
    df_2022 = pd.read_csv(RAW_DIR / "heart_2022.csv")
    
    print("Mapping and extracting common features...")
    # Target 2020
    target_2020 = (df_2020['HeartDisease'] == 'Yes').astype(int)
    
    # Features 2020
    features_2020 = pd.DataFrame({
        'numeric__bmi': df_2020['BMI'],
        'numeric__physical_health': df_2020['PhysicalHealth'],
        'categorical__sex_male': (df_2020['Sex'] == 'Male').astype(int),
        'categorical__stroke': (df_2020['Stroke'] == 'Yes').astype(int),
        'categorical__diabetic': df_2020['Diabetic'].apply(lambda x: 1 if 'Yes' in str(x) else 0),
        'categorical__kidney_disease': (df_2020['KidneyDisease'] == 'Yes').astype(int),
        'categorical__skin_cancer': (df_2020['SkinCancer'] == 'Yes').astype(int),
        'categorical__asthma': (df_2020['Asthma'] == 'Yes').astype(int),
        'categorical__diff_walking': (df_2020['DiffWalking'] == 'Yes').astype(int)
    })
    
    df_2020_clean = pd.concat([features_2020, target_2020.rename('target')], axis=1)
    
    # Target 2022
    target_2022 = ((df_2022['HadHeartAttack'] == 'Yes') | (df_2022['HadAngina'] == 'Yes')).astype(int)
    
    # Features 2022
    features_2022 = pd.DataFrame({
        'numeric__bmi': df_2022['BMI'],
        'numeric__physical_health': df_2022['PhysicalHealthDays'],
        'categorical__sex_male': (df_2022['Sex'] == 'Male').astype(int),
        'categorical__stroke': (df_2022['HadStroke'] == 'Yes').astype(int),
        'categorical__diabetic': df_2022['HadDiabetes'].apply(lambda x: 1 if 'Yes' in str(x) else 0),
        'categorical__kidney_disease': (df_2022['HadKidneyDisease'] == 'Yes').astype(int),
        'categorical__skin_cancer': (df_2022['HadSkinCancer'] == 'Yes').astype(int),
        'categorical__asthma': (df_2022['HadAsthma'] == 'Yes').astype(int),
        'categorical__diff_walking': (df_2022['DifficultyWalking'] == 'Yes').astype(int)
    })
    
    df_2022_clean = pd.concat([features_2022, target_2022.rename('target')], axis=1)
    
    # Downsample to prevent GPU Out-Of-Memory (OOM) during SFDA KNN
    # We will balance the dataset so it has enough positive (disease) cases to learn from
    def sample_data(df, n_samples=10000):
        pos = df[df['target'] == 1]
        neg = df[df['target'] == 0]
        n_pos = min(len(pos), n_samples // 2)
        n_neg = n_samples - n_pos
        pos_sampled = pos.sample(n=n_pos, random_state=42)
        neg_sampled = neg.sample(n=n_neg, random_state=42)
        return pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    print("Sampling 10,000 patients from each year for robust testing...")
    final_2020 = sample_data(df_2020_clean, 10000)
    final_2022 = sample_data(df_2022_clean, 10000)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_2020.to_csv(PROCESSED_DIR / "cdc_2020.csv", index=False)
    final_2022.to_csv(PROCESSED_DIR / "cdc_2022.csv", index=False)
    
    print("Successfully mapped and saved CDC Data!")
    print(f"CDC 2020 Shape: {final_2020.shape}")
    print(f"CDC 2022 Shape: {final_2022.shape}")

if __name__ == "__main__":
    process_cdc_data()
