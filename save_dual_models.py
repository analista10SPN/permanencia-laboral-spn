#!/usr/bin/env python3
"""
Train and Save BOTH Production Models
Creates: production_retention_model.pkl (with BOTH models inside)

Models:
1. Random Survival Forest - For time-to-event predictions (risk scores, probabilities over time)
2. Random Forest Classifier - For binary classification (will leave / won't leave)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training DUAL MODEL SYSTEM for Production")
print("=" * 70)
print("Building complementary prediction system:")
print("  1. Survival Analysis → Time-based risk predictions")
print("  2. Random Forest → Binary classification")
print("=" * 70)

# 1. Load data
print("\n📂 Step 1: Loading data...")
df = pd.read_csv('employee_retention_data1.csv')

# Add Tenure_Years if needed
if 'Tenure_Years' not in df.columns and 'Tenure_Days' in df.columns:
    df['Tenure_Years'] = df['Tenure_Days'] / 365.25

print(f"✅ Loaded {len(df)} employees")
print(f"   Terminated: {df['Has_Left'].sum()} ({df['Has_Left'].mean()*100:.1f}%)")
print(f"   Active: {(df['Has_Left']==0).sum()} ({(1-df['Has_Left'].mean())*100:.1f}%)")

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

# 3. Split data (SAME split for both models)
print("\n✂️  Step 3: Splitting data (70/30) - SAME split for both models...")
X_train, X_test, y_train, y_test, y_time_train, y_time_test = train_test_split(
    X, y, y_time, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)} samples ({y_train.sum()} terminated)")
print(f"✅ Test: {len(X_test)} samples ({y_test.sum()} terminated)")

# 4. Scale features (SAME scaler for both models)
print("\n📊 Step 4: Scaling features (shared scaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# 5. Train SURVIVAL MODEL
print("\n🧬 Step 5: Training SURVIVAL ANALYSIS MODEL...")
print("   (Random Survival Forest for time-to-event prediction)")
print("   This takes 2-3 minutes...")

# Create survival targets
y_train_surv = Surv.from_arrays(y_train.astype(bool).values, y_time_train.values)
y_test_surv = Surv.from_arrays(y_test.astype(bool).values, y_time_test.values)

# Train survival model
survival_model = RandomSurvivalForest(
    n_estimators=100,
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

survival_model.fit(X_train_scaled, y_train_surv)

# Evaluate survival model
surv_train_c_index = survival_model.score(X_train_scaled, y_train_surv)
surv_test_c_index = survival_model.score(X_test_scaled, y_test_surv)

print(f"✅ Survival Model trained!")
print(f"   Train C-index: {surv_train_c_index:.4f}")
print(f"   Test C-index: {surv_test_c_index:.4f}")

# 6. Train RANDOM FOREST CLASSIFIER
print("\n🌲 Step 6: Training RANDOM FOREST CLASSIFIER...")
print("   (Binary classification: will leave / won't leave)")

# Train RF classifier with same parameters for consistency
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_scaled, y_train)

# Evaluate RF model
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

rf_train_acc = rf_model.score(X_train_scaled, y_train)
rf_test_acc = rf_model.score(X_test_scaled, y_test)

# Get predictions for AUC
rf_test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_test_auc = roc_auc_score(y_test, rf_test_probs)

print(f"✅ Random Forest trained!")
print(f"   Train Accuracy: {rf_train_acc:.4f}")
print(f"   Test Accuracy: {rf_test_acc:.4f}")
print(f"   Test AUC: {rf_test_auc:.4f}")

# 7. Test BOTH models together
print("\n🧪 Step 7: Testing DUAL PREDICTIONS...")
print("Comparing predictions from both models:")
print("-" * 60)

test_samples = X_test_scaled[:5]
surv_predictions = survival_model.predict(test_samples)
rf_predictions = rf_model.predict(test_samples)
rf_probabilities = rf_model.predict_proba(test_samples)[:, 1]

for i in range(5):
    actual = "Left" if y_test.iloc[i] == 1 else "Active"
    surv_risk = surv_predictions[i]
    rf_prob = rf_probabilities[i]
    rf_pred = "Will Leave" if rf_predictions[i] == 1 else "Will Stay"
    
    surv_cat = "High" if surv_risk > 2000 else "Medium" if surv_risk > 1500 else "Low"
    
    print(f"\nEmployee {i+1} (Actual: {actual}):")
    print(f"  Survival Risk Score: {surv_risk:.0f} ({surv_cat})")
    print(f"  RF Probability: {rf_prob:.1%}")
    print(f"  RF Prediction: {rf_pred}")

# 8. SAVE BOTH MODELS
print("\n💾 Step 8: Saving DUAL MODEL PACKAGE...")

model_package = {
    # BOTH MODELS
    'survival_model': survival_model,
    'rf_model': rf_model,
    
    # SHARED COMPONENTS
    'scaler': scaler,
    'feature_names': features,
    
    # METADATA
    'metadata': {
        'package_version': '2.0_dual_models',
        'models_included': ['RandomSurvivalForest', 'RandomForestClassifier'],
        'trained_date': pd.Timestamp.now().isoformat(),
        
        # Survival model metrics
        'survival_train_c_index': surv_train_c_index,
        'survival_test_c_index': surv_test_c_index,
        
        # RF model metrics
        'rf_train_accuracy': rf_train_acc,
        'rf_test_accuracy': rf_test_acc,
        'rf_test_auc': rf_test_auc,
        
        # Data info
        'n_features': len(features),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'train_terminated_pct': y_train.mean(),
        'test_terminated_pct': y_test.mean(),
    }
}

# Save to file
output_path = 'production_retention_model.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(model_package, f)

file_size_mb = len(pickle.dumps(model_package)) / 1024 / 1024

print(f"✅ Model package saved: {output_path}")
print(f"   File size: {file_size_mb:.1f} MB")

# 9. Verify saved package works
print("\n🔍 Step 9: Verifying saved package...")
with open(output_path, 'rb') as f:
    loaded_package = pickle.load(f)

# Test both loaded models
loaded_surv = loaded_package['survival_model']
loaded_rf = loaded_package['rf_model']
loaded_scaler = loaded_package['scaler']

test_data = X_test_scaled[:2]
surv_test = loaded_surv.predict(test_data)
rf_test = loaded_rf.predict_proba(test_data)[:, 1]

print(f"✅ Both models work after loading!")
print(f"   Survival predictions: {surv_test}")
print(f"   RF probabilities: {rf_test}")

# 10. Summary
print("\n" + "=" * 70)
print("✅ SUCCESS! DUAL MODEL SYSTEM READY FOR DEPLOYMENT")
print("=" * 70)
print(f"""
📦 Model Package Contents:
   ✅ Random Survival Forest (time-to-event predictions)
   ✅ Random Forest Classifier (binary classification)
   ✅ StandardScaler (fitted on training data)
   ✅ Feature names ({len(features)} features)
   ✅ Complete metadata

📊 Performance Summary:
   
   Survival Model:
   • Test C-index: {surv_test_c_index:.4f}
   • Purpose: Risk scores + temporal probabilities
   
   Random Forest Classifier:
   • Test Accuracy: {rf_test_acc:.4f}
   • Test AUC: {rf_test_auc:.4f}
   • Purpose: Binary predictions (complementary)
   
   Training Data:
   • Training samples: {len(X_train):,}
   • Test samples: {len(X_test):,}
   • Features: {len(features)}

📁 File Information:
   • Name: {output_path}
   • Size: {file_size_mb:.1f} MB
   • Location: Current directory
   • Contains: BOTH models + scaler + metadata

🎯 Why DUAL Models?
   • Survival Model → "WHEN will they leave?" (1 month, 3 months, 6 months, 1 year)
   • Random Forest → "WILL they leave?" (binary yes/no with probability)
   • Together = Complete picture for HR decision-making

🚀 Next Steps:
   1. Copy production_retention_model.pkl to your demo folder
   2. Put it next to demo_retencion_simple.py
   3. The demo app will load BOTH models automatically
   4. Run: streamlit run demo_retencion_simple.py
   5. Your dual-model demo is ready!

💡 What the demo will show:
   • Risk score (from Survival Model)
   • Probability in 1/3/6/12 months (from Survival Model)
   • Binary prediction + probability (from Random Forest)
   • Risk category (Bajo/Medio/Urgente/Inminente)
   • All predictions are complementary!

📤 To deploy on Streamlit Cloud:
   Upload these 4 files:
   - demo_retencion_simple.py
   - production_retention_model.pkl  ← This file (with BOTH models)
   - employee_retention_data1.csv
   - requirements.txt
""")

print("\n🎉 Done! Dual model system ready for your presentation.")
print("   Both models work together for comprehensive predictions!")