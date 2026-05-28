# TCGA Glioma Grade Prognosis Prediction 🧠🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/shap-0.51.0-green.svg)](https://shap.readthedocs.io/)

An end-to-end, clinically transparent machine learning project for **Glioma Grade Prognosis Prediction** (distinguishing between **LGG** (Lower-Grade Glioma) and **GBM** (Glioblastoma Multiforme)) using clinical demographics and mutation status of the 20 most frequently mutated genes from the **The Cancer Genome Atlas (TCGA)**.

---

## 📂 Project Structure

```directory
glioma_prognosis/
├── data/
│   └── tcga_glioma.csv         # Cached TCGA Glioma clinical/mutation dataset (862 samples)
├── notebooks/
│   └── glioma_analysis.ipynb   # Premium Jupyter Notebook with complete analysis and outputs
├── models/
│   ├── glioma_model.pkl        # Best serialized model (Tuned XGBoost, Accuracy: 86.13%)
│   └── model_artifacts.joblib  # Pipeline preprocessing states (medians, encoders, feature list)
├── outputs/
│   ├── model_comparison.png    # Comparative bar chart of 11 ML Models
│   ├── roc_curves.png          # Combined ROC curves of 11 Classifiers
│   ├── confusion_matrix.png    # Confusion matrix of the best model (XGBoost)
│   ├── shap_summary.png        # SHAP Beeswarm global importance plot
│   └── shap_waterfall.png      # SHAP Waterfall local patient prediction explanation
├── download_data.py            # Utility script to download dataset programmatically
├── pipeline.py                 # Full python script executing the entire ML training pipeline
└── README.md                   # Project documentation
```

---

## 🔬 Clinical Context & Feature Engineering

Gliomas represent approximately 80% of malignant brain tumors. Determining the grade (LGG vs. GBM) is paramount because GBM is a highly aggressive grade IV tumor requiring immediate resection, radiation, and chemotherapy, whereas LGGs are lower-grade tumors with distinct therapeutic pathways.

### Target Leakage Mitigation
> [!IMPORTANT]
> A critical challenge in this dataset is **target leakage** in the clinical attributes:
> - **`Primary_Diagnosis`** contains categories like *"Glioblastoma"*, *"Astrocytoma"*, and *"Oligodendroglioma"*. Cross-tabulation shows that `Primary_Diagnosis` is a perfect predictor of grade (`Glioblastoma` is 100% GBM, while others are 100% LGG). 
> - **`Project`** indicates whether the patient is under `TCGA-LGG` or `TCGA-GBM`.
>
> Including these as features would cause severe target leakage. In this project, we proactively drop `Case_ID` (patient ID), `Project`, and `Primary_Diagnosis`, forcing our models to learn the predictive power of demographic features and raw genetic mutations.

### Preprocessing & SMOTE:
- **Age Extraction:** String age formats (e.g. `'51 years 108 days'`) are parsed into decimal floats and imputed with the training median, then standardized via `StandardScaler`.
- **Gender & Race:** Missing entries (e.g., `'--'`, `'not reported'`) are imputed with the training mode. Gender is binary encoded (Male = 1, Female = 0), and Race is One-Hot Encoded.
- **Genetic Mutations:** Mutation columns are mapped from object categories (`MUTATED`, `NOT_MUTATED`) to binary variables (`1`, `0`).
- **Oversampling (SMOTE):** To counter class imbalance during training, synthetic minority oversampling (SMOTE) is applied to the training fold to balance classes precisely 50/50.

---

## 📊 Model Benchmarking & Performance

We trained **11 different models** on the SMOTE-balanced training fold and evaluated them on the stratified 20% test fold.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** 🏆 | **86.13%** | **79.52%** | **90.41%** | **84.62%** | **90.70%** |
| **Logistic Regression** | 86.13% | 80.25% | 89.04% | 84.42% | 89.62% |
| **Random Forest** | 86.13% | 82.67% | 84.93% | 83.78% | 90.00% |
| **Gradient Boosting** | 86.13% | 80.25% | 89.04% | 84.42% | 90.48% |
| **AdaBoost** | 85.55% | 78.57% | 90.41% | 84.08% | 89.95% |
| **SVM (RBF Kernel)** | 84.39% | 78.05% | 87.67% | 82.58% | 87.82% |
| **Extra Trees** | 84.39% | 80.26% | 83.56% | 81.88% | 87.99% |
| **K-Nearest Neighbors** | 83.24% | 76.83% | 86.30% | 81.29% | 86.65% |
| **MLP Neural Network** | 83.24% | 77.50% | 84.93% | 81.05% | 87.16% |
| **Decision Tree** | 75.72% | 71.83% | 69.86% | 70.83% | 74.93% |
| **Naive Bayes** | 60.12% | 51.54% | 91.78% | 66.01% | 86.32% |

### Key Metrics Takeaways
- **Tuned XGBoost** (`max_depth=3`, `learning_rate=0.05`, `n_estimators=100`) achieves a superb **86.13% Accuracy**, **84.62% F1-score**, and **90.70% ROC-AUC**.
- **Recall of 90.41%** ensures that aggressive Glioblastoma (GBM) cases are rarely missed, which is critical in a clinical diagnostic setting.
- Simple classifiers like Logistic Regression and Random Forest also perform exceptionally well, confirming the strong biological signal in the features.

---

## 🔍 Model Transparency (SHAP Interpretability)

In medical ML, transparency is just as important as accuracy. We use **SHAP (SHapley Additive exPlanations)** on our best model (XGBoost) to understand feature contributions.

### Global Interpretability (Beeswarm Summary)
- The SHAP Beeswarm plot demonstrates that the mutation status of **`IDH1`** is by far the most dominant feature.
- **Clinical Insight:** A mutation in `IDH1` (high SHAP value, represented in blue because mutation=1) has a massive negative impact on the log-odds of being GBM—which means it strongly predicts **Lower-Grade Glioma (LGG)**. This perfectly aligns with established neuro-oncology guidelines, where IDH-mutation is a key hallmark of LGGs, while IDH-wildtype is indicative of primary GBM.
- **`Age_at_diagnosis`** is the second most important feature: older age at diagnosis dramatically increases the risk of being predicted as GBM.
- Mutations in **`PTEN`** and **`EGFR`** are highly predictive of GBM (Class 1).

### Local Interpretability (Waterfall Plot)
- Decodes predictions for individual patients. It displays how a patient's specific mutations (e.g. `IDH1 = 0`, `Age_at_diagnosis = 1.2`, `PTEN = 1`) shift the prediction from the baseline expected value to the final predicted grade probability.

---

## 🚀 Getting Started & Execution

### 1. Installation
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap imbalanced-learn joblib
```

### 2. Download the Dataset
Download the raw TCGA Glioma dataset:
```bash
python download_data.py
```

### 3. Run the ML Pipeline
Execute the full training, evaluation, plotting, SHAP explainability, and serialization pipeline:
```bash
python pipeline.py
```
This will automatically train the models, output performance tables, and save the visualization assets in the `/outputs` folder!

### 4. Interactive Notebook
Explore the detailed end-to-end walkthrough by launching the Jupyter Notebook:
```bash
jupyter notebook notebooks/glioma_analysis.ipynb
```
The notebook is fully pre-rendered and populated with all the outputs, so you can read and verify results instantly.
