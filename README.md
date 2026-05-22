# Scania-APS-Failure-Detection-System
A predictive maintenance Streamlit dashboard powered by scikit-learn and LightGBM to detect Air Production System (APS) failures in Scania heavy trucks while minimizing inspection costs.


# 🚛 Scania APS Failure Detection & Predictive Maintenance System

An end-to-end Machine Learning framework designed for predictive maintenance to diagnose and predict failures in the **Air Production System (APS)** of Scania heavy trucks. This system targets minimizing fleet operational costs by accurately classifying specific component bugs while preventing catastrophic road failures, wrapped beautifully in an interactive web dashboard built with **Streamlit**.

---

## ✨ Features

- **Industrial Preprocessing Pipeline**: Custom data clearing workflows including dynamic Median Imputation, Variance Threshold Selection, Highly-Correlated Feature Dropping, and Standard Scaling (compiled tightly inside `preprocess_artifacts.pkl`).
- **Cost-Optimized Architecture**: The classification boundary focuses on reducing the total cost of maintenance, optimizing the tradeoff between False Negatives (unpredicted total breakdowns on the road) and False Positives (unnecessary workshop checkups).
- **Multi-Model Benchmark Ecosystem**:
  - **Logistic Regression**: Interpretable linear diagnostic baseline.
  - **Random Forest**: Robust ensemble boundary maps.
  - **LightGBM (Light Gradient Boosting Machine)**: High-performance gradient boosting optimized for imbalanced sensor data layouts.
- **Production Dashboard Experience**: Multi-view tabs supporting dynamic CSV file batch uploading, automated column validation, runtime missing values distribution plots, and granular performance metric reports.

---

## 📂 Repository Structure

```bash
├── models/
│   ├── preprocess_artifacts.pkl     # Imputer, Scaler, Variance Selector, and Dropped Cols map
│   ├── logistic_regression_model.pkl # Trained Logistic Regression weights
│   ├── random_forest_model.pkl     # Trained Random Forest classifier weights
│   ├── lightgbm_model.pkl          # Trained LightGBM booster model weights
│   └── training_metrics.json       # Consolidated Accuracy, Recall, and ROC scores map
├── Final_updated.ipynb             # Research notebook containing extraction, EDA, and model training
├── merge.py                        # Utility script to compile separate model outputs into training_metrics.json
├── app.py                          # Multi-page Streamlit analytical dashboard entry point
└── README.md                       # System documentation manual
