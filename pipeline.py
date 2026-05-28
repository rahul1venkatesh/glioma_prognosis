import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Explainability
import shap

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

def parse_age(age_str):
    """Parses age string like '51 years 108 days' or '87 years' into a float number of years."""
    if pd.isna(age_str) or age_str == '--' or age_str == 'not reported':
        return None
    val = str(age_str).strip()
    years = 0.0
    days = 0.0
    
    # Try regex matches
    y_match = re.search(r'(\d+)\s*years?', val)
    d_match = re.search(r'(\d+)\s*days?', val)
    
    if y_match:
        years = float(y_match.group(1))
    if d_match:
        days = float(d_match.group(1))
        
    if not y_match and not d_match:
        try:
            return float(val)
        except ValueError:
            return None
            
    return years + (days / 365.25)

def preprocess_data(data_path):
    """Loads and preprocesses the TCGA Glioma dataset."""
    print("Loading dataset...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # Drop completely irrelevant columns or target leaks
    cols_to_drop = ['Case_ID', 'Project', 'Primary_Diagnosis']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"Dropped columns: {cols_to_drop}")
    
    # Clean target
    df['Grade'] = df['Grade'].map({'LGG': 0, 'GBM': 1})
    
    # Preprocess mutations: MUTATED -> 1, NOT_MUTATED -> 0
    mutation_cols = [
        'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 
        'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 
        'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
    ]
    for col in mutation_cols:
        if col in df.columns:
            df[col] = df[col].map({'MUTATED': 1, 'NOT_MUTATED': 0}).fillna(0).astype(int)
            
    # Clean age column
    df['Age_at_diagnosis'] = df['Age_at_diagnosis'].apply(parse_age)
    
    # Replace '--' and 'not reported' in categorical columns with None/NaN
    categorical_cols = ['Gender', 'Race']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace({'--': None, 'not reported': None})
            
    # Separate features and target
    X = df.drop(columns=['Grade'])
    y = df['Grade']
    
    # Split into train/test (80/20) stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Impute missing values using pandas (fit on train, transform on both)
    # 1. Age (numeric) -> median
    age_median = X_train['Age_at_diagnosis'].median()
    X_train_age = X_train['Age_at_diagnosis'].fillna(age_median)
    X_test_age = X_test['Age_at_diagnosis'].fillna(age_median)
    
    # 2. Gender (categorical) -> mode
    gender_mode = X_train['Gender'].mode()[0]
    X_train_gender = X_train['Gender'].fillna(gender_mode)
    X_test_gender = X_test['Gender'].fillna(gender_mode)
    
    # 3. Race (categorical) -> mode
    race_mode = X_train['Race'].mode()[0]
    X_train_race = X_train['Race'].fillna(race_mode)
    X_test_race = X_test['Race'].fillna(race_mode)
    
    # Binary encode Gender (Male = 1, Female = 0)
    X_train_gender_enc = X_train_gender.map({'Male': 1, 'Female': 0}).astype(int)
    X_test_gender_enc = X_test_gender.map({'Male': 1, 'Female': 0}).astype(int)
    
    # One-hot encode Race
    ohe_race = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Reshape for OHE
    race_train_reshaped = X_train_race.values.reshape(-1, 1)
    race_test_reshaped = X_test_race.values.reshape(-1, 1)
    
    race_train_encoded = ohe_race.fit_transform(race_train_reshaped)
    race_test_encoded = ohe_race.transform(race_test_reshaped)
    
    race_cols = [f"Race_{cat}" for cat in ohe_race.categories_[0]]
    race_train_df = pd.DataFrame(race_train_encoded, columns=race_cols, index=X_train.index)
    race_test_df = pd.DataFrame(race_test_encoded, columns=race_cols, index=X_test.index)
    
    # Scale Age
    scaler = StandardScaler()
    X_train_age_scaled = scaler.fit_transform(X_train_age.values.reshape(-1, 1))
    X_test_age_scaled = scaler.transform(X_test_age.values.reshape(-1, 1))
    
    # Reassemble train and test sets
    X_train_clean = pd.DataFrame(index=X_train.index)
    X_train_clean['Age_at_diagnosis'] = X_train_age_scaled.flatten()
    X_train_clean['Gender'] = X_train_gender_enc
    X_train_clean = pd.concat([X_train_clean, race_train_df], axis=1)
    
    X_test_clean = pd.DataFrame(index=X_test.index)
    X_test_clean['Age_at_diagnosis'] = X_test_age_scaled.flatten()
    X_test_clean['Gender'] = X_test_gender_enc
    X_test_clean = pd.concat([X_test_clean, race_test_df], axis=1)
    
    # Add mutation columns
    for col in mutation_cols:
        X_train_clean[col] = X_train[col]
        X_test_clean[col] = X_test[col]
        
    print(f"Preprocessed features shape: X_train = {X_train_clean.shape}, X_test = {X_test_clean.shape}")
    
    # Apply SMOTE to balance classes on training set
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
    
    # Save artifacts for inference
    artifacts = {
        'age_median': age_median,
        'gender_mode': gender_mode,
        'race_mode': race_mode,
        'ohe_race': ohe_race,
        'scaler': scaler,
        'feature_names': list(X_train_clean.columns)
    }
    
    return X_train_res, X_test_clean, y_train_res, y_test, artifacts

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains 11 different ML models and benchmarks their performance."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'MLP Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    print("\n--- Training and Evaluating 11 Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['LGG', 'GBM']))
        print("-" * 50)
        
    return results, fitted_models

def plot_and_save_visualizations(results, y_test, fitted_models, X_test, output_dir):
    """Generates and saves the required evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Model Comparison Bar Chart
    print("Generating model comparison bar chart...")
    metrics_data = []
    for name, metrics in results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'F1-Score': metrics['F1-Score'],
            'ROC-AUC': metrics['ROC-AUC']
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    # Melt dataframe for easy plotting with seaborn
    df_melted = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 8))
    # Elegant custom HSL palette translated to hex
    palette = ['#4F46E5', '#06B6D4', '#10B981'] # Indigo, Cyan, Emerald
    ax = sns.barplot(x='Score', y='Model', hue='Metric', data=df_melted, palette=palette)
    plt.title('Performance Comparison of 11 ML Models for Glioma Prognosis', pad=20, weight='bold', size=16)
    plt.xlabel('Metric Score (higher is better)', weight='semibold', labelpad=10)
    plt.ylabel('Model Name', weight='semibold', labelpad=10)
    plt.xlim(0, 1.05)
    
    # Add value annotations
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                    va='center', ha='left', fontsize=9, color='#374151', weight='medium')
                    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Combined ROC Curves Plot
    print("Generating ROC curves...")
    plt.figure(figsize=(12, 9))
    
    # Curated palette for 11 models
    color_map = plt.colormaps.get_cmap('tab20')
    
    for idx, (name, metrics) in enumerate(results.items()):
        if metrics['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
            auc_val = metrics['ROC-AUC']
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color_map(idx), linewidth=2)
            
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guess')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', weight='semibold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', weight='semibold', labelpad=10)
    plt.title('ROC Curves Comparison of 11 Classifiers', pad=20, weight='bold', size=15)
    plt.legend(loc='lower right', fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 3. Confusion Matrix for best model (XGBoost)
    print("Generating Confusion Matrix for XGBoost...")
    xgb_metrics = results['XGBoost']
    y_pred_xgb = xgb_metrics['predictions']
    cm = confusion_matrix(y_test, y_pred_xgb)
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['LGG (Class 0)', 'GBM (Class 1)'],
                yticklabels=['LGG (Class 0)', 'GBM (Class 1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.xlabel('Predicted Grade', weight='semibold', labelpad=10)
    plt.ylabel('True Grade', weight='semibold', labelpad=10)
    plt.title('XGBoost Confusion Matrix', pad=20, weight='bold', size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def run_shap_interpretability(xgb_model, X_test, output_dir):
    """Executes SHAP interpretability on the XGBoost model and saves visualization plots."""
    print("\nRunning SHAP interpretability on XGBoost...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)
    
    # 1. SHAP Summary Beeswarm Plot
    print("Saving SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    # Render plot
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title('SHAP Feature Importance (XGBoost Glioma Grade Classifier)', pad=25, weight='bold', size=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Patient-Specific SHAP Waterfall Plot
    print("Saving SHAP waterfall plot for single patient prediction...")
    plt.figure(figsize=(12, 7.5))
    # Pick patient 0 (index 0)
    shap.plots.waterfall(shap_values[0], show=False, max_display=12)
    plt.title('SHAP Local Prediction Explanation (Patient #1 / index 0)', pad=25, weight='bold', size=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP explainability completed successfully!")
    
    return explainer, shap_values

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'tcga_glioma.csv')
    model_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'outputs')
    
    # 1. Preprocess Data
    X_train, X_test, y_train, y_test, artifacts = preprocess_data(data_path)
    
    # 2. Train and Evaluate
    results, fitted_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 3. Generate Evaluation Plots
    plot_and_save_visualizations(results, y_test, fitted_models, X_test, output_dir)
    
    # 4. SHAP Explainability on XGBoost
    xgb_model = fitted_models['XGBoost']
    explainer, shap_values = run_shap_interpretability(xgb_model, X_test, output_dir)
    
    # 5. Save best model and metadata artifacts
    print("\nSaving best model and pipeline artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(model_dir, 'glioma_model.pkl'))
    joblib.dump(artifacts, os.path.join(model_dir, 'model_artifacts.joblib'))
    
    # Print comparison table of all 11 models
    print("\n--- Summary Performance Table ---")
    metrics_summary = []
    for name, metrics in results.items():
        metrics_summary.append({
            'Model': name,
            'Accuracy': f"{metrics['Accuracy']*100:.2f}%",
            'Precision': f"{metrics['Precision']*100:.2f}%",
            'Recall': f"{metrics['Recall']*100:.2f}%",
            'F1-Score': f"{metrics['F1-Score']*100:.2f}%",
            'ROC-AUC': f"{metrics['ROC-AUC']*100:.2f}%" if not np.isnan(metrics['ROC-AUC']) else "N/A"
        })
    df_summary = pd.DataFrame(metrics_summary)
    print(df_summary.to_string(index=False))
    
    print("\nPipeline executed successfully! All models built, evaluated, and saved.")

if __name__ == "__main__":
    main()
