import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

def process_race_data():
    print("Loading 2020 CDC dataset...")
    df = pd.read_csv(RAW_DIR / "heart_2020.csv")
    
    print("Available Race categories:", df['Race'].unique())
    
    # Source Domain = Black and Asian
    df_source = df[df['Race'].isin(['Black', 'Asian'])].copy()
    
    # Target Domain = White
    df_target = df[df['Race'] == 'White'].copy()
    
    print(f"Total Black/Asian patients: {len(df_source)}")
    print(f"Total White patients: {len(df_target)}")
    
    def clean_and_format(data):
        # Target
        target = (data['HeartDisease'] == 'Yes').astype(int)
        
        # Features (We do NOT include Race here, so the model learns purely from health factors)
        features = pd.DataFrame()
        features['numeric__bmi'] = data['BMI']
        features['numeric__physical_health'] = data['PhysicalHealth']
        features['numeric__mental_health'] = data['MentalHealth']
        features['numeric__sleep_time'] = data['SleepTime']
        
        features['categorical__sex_male'] = (data['Sex'] == 'Male').astype(int)
        features['categorical__stroke'] = (data['Stroke'] == 'Yes').astype(int)
        features['categorical__diabetic'] = data['Diabetic'].apply(lambda x: 1 if 'Yes' in str(x) else 0)
        features['categorical__kidney_disease'] = (data['KidneyDisease'] == 'Yes').astype(int)
        features['categorical__skin_cancer'] = (data['SkinCancer'] == 'Yes').astype(int)
        features['categorical__asthma'] = (data['Asthma'] == 'Yes').astype(int)
        features['categorical__diff_walking'] = (data['DiffWalking'] == 'Yes').astype(int)
        features['categorical__smoking'] = (data['Smoking'] == 'Yes').astype(int)
        features['categorical__alcohol'] = (data['AlcoholDrinking'] == 'Yes').astype(int)
        features['categorical__physical_activity'] = (data['PhysicalActivity'] == 'Yes').astype(int)
        
        # Encode AgeCategory as a numeric/ordinal feature
        age_mapping = {
            '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, 
            '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, 
            '70-74': 10, '75-79': 11, '80 or older': 12
        }
        features['numeric__age_group'] = data['AgeCategory'].map(age_mapping).fillna(0).astype(int)
        
        return pd.concat([features, target.rename('target')], axis=1)

    print("\nCleaning and formatting features (removing Race variable to prevent direct bias)...")
    source_clean = clean_and_format(df_source)
    target_clean = clean_and_format(df_target)
    
    # We must balance and downsample for the SFDA KNN to prevent memory issues
    def sample_data(df, n_samples=10000):
        pos = df[df['target'] == 1]
        neg = df[df['target'] == 0]
        n_pos = min(len(pos), n_samples // 2)
        n_neg = n_samples - n_pos
        pos_sampled = pos.sample(n=n_pos, random_state=42)
        neg_sampled = neg.sample(n=n_neg, random_state=42)
        return pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    print("Sampling 10,000 patients for SFDA testing...")
    final_source = sample_data(source_clean, 10000)
    final_target = sample_data(target_clean, 10000)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_source.to_csv(PROCESSED_DIR / "cdc_source_minority.csv", index=False)
    final_target.to_csv(PROCESSED_DIR / "cdc_target_white.csv", index=False)
    
    print("\nData processing complete!")
    print(f"Source (Minority - Black/Asian) Shape: {final_source.shape}")
    print(f"Target (White) Shape: {final_target.shape}")

if __name__ == "__main__":
    process_race_data()
