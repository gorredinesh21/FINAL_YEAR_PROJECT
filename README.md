# 🧠 GAN - Based Data Augmentation for Imbalanced Datasets 

This repository presents an end-to-end pipeline for handling class imbalance in tabular datasets using **Custom GAN** and **SMOTE**, followed by **training multiple machine learning models** and **evaluating their performance with advanced metrics and statistical tests**.

---

## 📂 Repository Structure and File Overview

### 🏗️ Model Training and Oversampling

- **`custom_gan.py`**  
  Custom GAN implementation for generating synthetic tabular data.

- **`smote_oversampler.py`**  
  Script implementing the SMOTE oversampling technique.

- **`model_trainer.py`**  
  Trains various classifiers (Random Forest, XGBoost, LightGBM, Gradient Boosting) using original and augmented data. Performs 5-fold cross-validation.

- **`gan_model.pkl`**  
  Saved GAN model after training for reuse.

---

### 📊 Model Results (5-Fold Cross-Validation)

| Model | File |
|-------|------|
| Random Forest | `random_forest_5fold_results.csv` |
| XGBoost       | `xgboost_5fold_results.csv` |
| LightGBM      | `lightgbm_5fold_results.csv` |
| Gradient Boosting | `gradient_boosting_5fold_results.csv` |

Each file contains metrics like Accuracy, Precision, Recall, F1-score, and AUC averaged over 5 folds.

---

### 📈 Evaluation Utilities

- **`evaluate_metrics.py`**  
  Computes performance metrics for trained models.

- **`evaluate_confusion.py`**  
  Plots and analyzes confusion matrices.

- **`evaluate_mcnemar.py`**  
  Runs McNemar’s test for statistical comparison between classifiers.

- **`plot_pr_curve.py`**  
  Plots precision-recall curves to visualize model performance.

---

### 📓 Notebooks

- **`smote_gan.ipynb`**  
  Walkthrough for data augmentation using SMOTE and GAN, and class distribution visualization.

- **`evaluation.ipynb`** ✅ **(Main Evaluation Notebook)**  
  **🔥 This notebook is the core evaluation module.**  
  It:
  - Loads results from model training (CSV files).
  - Plots **confusion matrices** and **PR curves**.
  - Computes all standard metrics (Accuracy, F1, AUC, etc.).
  - Runs **McNemar's test** to statistically compare models.
  
  📌 **Run this notebook to evaluate all models and compare performance in a centralized manner.**

---

## 🔁 Code Flow Summary

1. **Data Augmentation**
   - Use `custom_gan.py` and `smote_oversampler.py` to generate balanced datasets.

2. **Model Training**
   - Run `model_trainer.py` to train 4 classifiers (Random Forest, XGBoost, LightGBM, Gradient Boosting) with 5-fold CV.

3. **Model Evaluation**
   - Launch `evaluation.ipynb` to:
     - Import CSV results
     - Compare metrics
     - Plot visualizations
     - Perform statistical tests

---

## 📌 Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, XGBoost, LightGBM, SciPy

```bash
pip install -r requirements.txt
