#!/usr/bin/env python3
"""
Train and Save Production Model
Creates: production_retention_model.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Survival Model for Production")
print("=" * 70)

# 1. Load data
print("\n📂 Step 1: Loading data...")
df = pd.read_csv('employee_retention_data1.csv')

# Add Tenure_Years if needed
if 'Tenure_Years' not in df.columns and 'Tenure_Days' in df.columns:
    df['Tenure_Years'] = df['Tenure_Days'] / 365.25

print(f"✅ Loaded {len(df)} employees")
print(f"   Terminated: {df['Has_Left'].sum()} ({df['Has_Left'].mean()*100:.1f}%)")

# 2. Prepare features
print("\n🔧 Step 2: Preparing features...")

exclude_columns = ['Employee_ID', 'Supervisor_ID', 'Tenure_Days', 'Has_Left',
                   'Days_Since_Last_Raise', 'Hire_Year', 'Hire_Month', 
                   'Hire_Quarter', 'Hire_Day_Of_Week', 'Department_Name',
                   'Position_Title', 'Division_Name', 'Location_Name']

features = [col for col in df.columns 
           if col not in exclude_columns 
           and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

print(f"✅ Using {len(features)} features")
print(f"   Sample: {features[:5]}")

# Prepare X and y
X = df[features].fillna(0)
y = df['Has_Left']
y_time = df['Tenure_Years']

# 3. Split data
print("\n✂️  Step 3: Splitting data (70/30)...")
X_train, X_test, y_train, y_test, y_time_train, y_time_test = train_test_split(
    X, y, y_time, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# 4. Scale features
print("\n📊 Step 4: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# 5. Train model
print("\n🧬 Step 5: Training Random Survival Forest...")
print("   This takes 2-3 minutes...")

# Create survival targets
y_train_surv = Surv.from_arrays(y_train.astype(bool).values, y_time_train.values)
y_test_surv = Surv.from_arrays(y_test.astype(bool).values, y_time_test.values)

# Train model
model = RandomSurvivalForest(
    n_estimators=100,
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_surv)

# Evaluate
train_c_index = model.score(X_train_scaled, y_train_surv)
test_c_index = model.score(X_test_scaled, y_test_surv)

print(f"✅ Model trained successfully!")
print(f"\n📊 Performance:")
print(f"   Train C-index: {train_c_index:.4f}")
print(f"   Test C-index: {test_c_index:.4f}")

if test_c_index > 0.85:
    print("   ✅ Excellent performance!")
elif test_c_index > 0.75:
    print("   ✅ Good performance!")
else:
    print("   ⚠️  Performance could be improved")

# 6. Test predictions (verify it works)
print("\n🧪 Step 6: Testing predictions...")
test_predictions = model.predict(X_test_scaled[:5])
print("Sample risk scores:")
for i, pred in enumerate(test_predictions):
    actual = "Left" if y_test.iloc[i] == 1 else "Active"
    risk_cat = "High" if pred > 2000 else "Medium" if pred > 1500 else "Low"
    print(f"   Sample {i+1}: {pred:.0f} ({risk_cat}) - Actual: {actual}")

# 7. SAVE THE MODEL
print("\n💾 Step 7: Saving model package...")

model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': features,
    'metadata': {
        'model_type': 'RandomSurvivalForest',
        'train_c_index': train_c_index,
        'test_c_index': test_c_index,
        'n_features': len(features),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'trained_date': pd.Timestamp.now().isoformat()
    }
}

# Save to file
output_path = 'production_retention_model.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(model_package, f)

file_size_mb = len(pickle.dumps(model_package)) / 1024 / 1024

print(f"✅ Model saved: {output_path}")
print(f"   File size: {file_size_mb:.1f} MB")

# 8. Verify saved model works
print("\n🔍 Step 8: Verifying saved model...")
with open(output_path, 'rb') as f:
    loaded_package = pickle.load(f)

loaded_model = loaded_package['model']
loaded_scaler = loaded_package['scaler']

# Test loaded model
test_pred_loaded = loaded_model.predict(X_test_scaled[:3])
print(f"✅ Loaded model works! Sample predictions: {test_pred_loaded}")

# Summary
print("\n" + "=" * 70)
print("✅ SUCCESS! MODEL READY FOR DEPLOYMENT")
print("=" * 70)
print(f"""
📦 Model Package Contents:
   ✅ Trained Random Survival Forest
   ✅ StandardScaler (fitted)
   ✅ Feature names ({len(features)} features)
   ✅ Metadata (C-index: {test_c_index:.4f})

📊 Performance:
   • Test C-index: {test_c_index:.4f}
   • Training samples: {len(X_train):,}
   • Test samples: {len(X_test):,}

📁 File:
   • Name: {output_path}
   • Size: {file_size_mb:.1f} MB
   • Location: Current directory

🚀 Next Steps:
   1. Copy production_retention_model.pkl to your demo folder
   2. Put it next to demo_retencion_simple.py
   3. Run: streamlit run demo_retencion_simple.py
   4. Your demo is ready!

💡 To deploy:
   Upload these 4 files to Streamlit Cloud:
   - demo_retencion_simple.py
   - production_retention_model.pkl  ← This file
   - employee_retention_data1.csv
   - requirements.txt
""")

print("🎉 Done! Model is ready for your presentation.")