# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# st.title("📊 Model Information & Performance")

# # ── Known training metrics from the notebook ─────────────────────────────────
# # Fill in the actual values from your training run here.
# # These are placeholders — replace with your real outputs.

# METRICS = {
#     "Logistic Regression": {
#         "accuracy": 0.9736,   # e.g. 0.9712
#         "recall":   0.9147,
#         "f1":       0.6191,
#         "cm":       [[15235, 390]
#                     ,[32, 343]]   # 2x2 list [[TN,FP],[FN,TP]]
#     },
#     "Random Forest": {
#         "accuracy": 0.9896,
#         "recall":   0.8187,
#         "f1":       0.7862,
#         "cm":       [[15526, 99]
#                     ,[68, 307]]
#     },
#     "LightGBM": {
#         "accuracy": 0.9915,
#         "recall":   0.8533,
#         "f1":       0.8247,
#         "cm":       [[15544, 81]
#                     ,[55, 320]]
#     },
# }

# # ── Allow user to paste their metrics ────────────────────────────────────────
# st.markdown("""
# > **How to use this page:**  
# > Paste your real model metrics from the training notebook output below.  
# > The charts will update automatically.
# """)

# with st.expander("✏️ Enter model metrics (from your training notebook)", expanded=True):
#     for model_name in METRICS:
#         st.markdown(f"**{model_name}**")
#         cols = st.columns(3)
#         acc = cols[0].number_input(f"Accuracy",  key=f"{model_name}_acc", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")
#         rec = cols[1].number_input(f"Recall",    key=f"{model_name}_rec", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")
#         f1  = cols[2].number_input(f"F1 Score",  key=f"{model_name}_f1",  min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")

#         METRICS[model_name]["accuracy"] = acc if acc > 0 else None
#         METRICS[model_name]["recall"]   = rec if rec > 0 else None
#         METRICS[model_name]["f1"]       = f1  if f1  > 0 else None

#         cm_cols = st.columns(4)
#         tn = cm_cols[0].number_input("TN", key=f"{model_name}_tn", min_value=0, value=0)
#         fp = cm_cols[1].number_input("FP", key=f"{model_name}_fp", min_value=0, value=0)
#         fn = cm_cols[2].number_input("FN", key=f"{model_name}_fn", min_value=0, value=0)
#         tp = cm_cols[3].number_input("TP", key=f"{model_name}_tp", min_value=0, value=0)

#         if tn + fp + fn + tp > 0:
#             METRICS[model_name]["cm"] = [[tn, fp], [fn, tp]]

#         st.markdown("---")

# # ── Comparison table ──────────────────────────────────────────────────────────
# has_any_metric = any(v["accuracy"] is not None for v in METRICS.values())

# if has_any_metric:
#     st.markdown("### 📋 Performance Comparison")
#     rows = []
#     for name, m in METRICS.items():
#         rows.append({
#             "Model":    name,
#             "Accuracy": f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "—",
#             "Recall":   f"{m['recall']:.4f}"   if m["recall"]   is not None else "—",
#             "F1 Score": f"{m['f1']:.4f}"       if m["f1"]       is not None else "—",
#         })
#     st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

#     # ── Bar chart comparison ──────────────────────────────────────────────────
#     st.markdown("### 📈 Metric Comparison Chart")
#     model_names = [n for n, m in METRICS.items() if m["accuracy"] is not None]
#     accs = [METRICS[n]["accuracy"] or 0 for n in model_names]
#     recs = [METRICS[n]["recall"]   or 0 for n in model_names]
#     f1s  = [METRICS[n]["f1"]       or 0 for n in model_names]

#     x = np.arange(len(model_names))
#     width = 0.25

#     fig, ax = plt.subplots(figsize=(10, 5))
#     b1 = ax.bar(x - width,     accs, width, label="Accuracy", color="#2196F3", alpha=0.9)
#     b2 = ax.bar(x,             recs, width, label="Recall",   color="#4CAF50", alpha=0.9)
#     b3 = ax.bar(x + width,     f1s,  width, label="F1 Score", color="#FF9800", alpha=0.9)

#     ax.set_xticks(x)
#     ax.set_xticklabels(model_names, fontweight="bold")
#     ax.set_ylim(0, 1.1)
#     ax.set_ylabel("Score")
#     ax.set_title("Model Performance Metrics", fontweight="bold")
#     ax.legend()
#     ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
#     ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
#     ax.bar_label(b3, fmt="%.3f", padding=3, fontsize=8)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close()

#     # ── Confusion matrices ────────────────────────────────────────────────────
#     models_with_cm = [(n, m) for n, m in METRICS.items() if m["cm"] is not None]
#     if models_with_cm:
#         st.markdown("### 🔲 Confusion Matrices")
#         cm_tabs = st.tabs([n for n, _ in models_with_cm])

#         for tab, (name, m) in zip(cm_tabs, models_with_cm):
#             with tab:
#                 cm = np.array(m["cm"])
#                 fig, ax = plt.subplots(figsize=(5, 4))
#                 im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
#                 plt.colorbar(im, ax=ax)
#                 labels = ["Normal (neg)", "APS Failure (pos)"]
#                 ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
#                 ax.set_xticklabels(labels, fontsize=9)
#                 ax.set_yticklabels(labels, fontsize=9)
#                 ax.set_xlabel("Predicted", fontweight="bold")
#                 ax.set_ylabel("Actual", fontweight="bold")
#                 ax.set_title(f"{name} — Confusion Matrix", fontweight="bold")

#                 thresh = cm.max() / 2
#                 for i in range(2):
#                     for j in range(2):
#                         ax.text(j, i, f"{cm[i, j]:,}",
#                                 ha="center", va="center", fontweight="bold",
#                                 color="white" if cm[i, j] > thresh else "black")
#                 plt.tight_layout()
#                 st.pyplot(fig)
#                 plt.close()

#                 tn, fp = cm[0]
#                 fn, tp = cm[1]
#                 total = tn + fp + fn + tp
#                 st.markdown(f"""
#                 | Metric | Value |
#                 |--------|-------|
#                 | True Negatives (TN) | {tn:,} |
#                 | False Positives (FP) | {fp:,} |
#                 | False Negatives (FN) | {fn:,} — missed APS failures |
#                 | True Positives (TP) | {tp:,} |
#                 | **Cost (FN×500 + FP×10)** | **{fn*500 + fp*10:,}** |
#                 """)
#                 st.info("💡 FN cost = 500, FP cost = 10 (from dataset cost matrix). "
#                         "Recall is prioritized to minimize missed failures.")

# else:
#     st.info("👆 Enter model metrics in the form above to see the comparison charts.")

# # ── Model architecture info ───────────────────────────────────────────────────
# st.markdown("---")
# st.markdown("### 🏗️ Model Architecture Details")

# arch_tabs = st.tabs(["Logistic Regression", "Random Forest", "LightGBM"])

# with arch_tabs[0]:
#     st.markdown("""
#     **Logistic Regression** — Linear baseline  
#     - Solver: `lbfgs`  
#     - Max iterations: 1000  
#     - Trained on SMOTE-balanced data  
#     - Good for interpretability; struggles with non-linear boundaries in high-dim sensor data
#     """)

# with arch_tabs[1]:
#     st.markdown("""
#     **Random Forest** — Ensemble of decision trees  
#     - 100 estimators (default)  
#     - Trained on SMOTE-balanced data  
#     - Robust to noise and outliers  
#     - Provides feature importances
#     """)

# with arch_tabs[2]:
#     st.markdown("""
#     **LightGBM** — Leaf-wise gradient boosting  
#     - `n_estimators=300`, `learning_rate=0.05`, `num_leaves=63`  
#     - Trained on SMOTE-balanced data  
#     - Best suited for high-dimensional tabular data  
#     - Fastest of the three for inference
#     """)

# st.markdown("---")
# st.markdown("### ⚙️ Preprocessing Pipeline (applied to all models)")
# st.markdown("""
# 1. **Drop high-missing columns** — threshold > 60%  
# 2. **Median imputation** — `SimpleImputer(strategy='median')`  
# 3. **Variance filtering** — `VarianceThreshold(0.01)` removes near-constant features  
# 4. **StandardScaler** — zero mean, unit variance  
# 5. **Correlation-based pruning** — drop one of each pair with |r| > 0.95 (keep higher target correlation)  
# 6. **Combined feature selection** — top 50 features via MI (40%) + RF importance (40%) + ReliefF (20%)  
# 7. **SMOTE** — synthetic oversampling on training set only
# """)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

st.title("📊 Model Information & Performance")

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_training_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Training benchmark metrics file not found.")
        return None
    except Exception as e:
        st.error(f"Failed to load training metrics: {e}")
        return None


def plot_confusion_matrix(cm, title):
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Normal (0)", "APS Failure (1)"]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    thresh = cm.max() / 2

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def calculate_cost(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (fn * 500) + (fp * 10)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
training_metrics = load_training_metrics()
live_results = st.session_state.evaluation_results


# ==================================================
# SECTION A — TRAINING BENCHMARK METRICS
# ==================================================
st.markdown("## 🏆 Training Benchmark Performance")

if training_metrics:
    rows = []

    for model_name, metrics in training_metrics.items():
        cm = metrics["cm"]
        cost = calculate_cost(cm)

        rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "Cost": cost
        })

    benchmark_df = pd.DataFrame(rows)

    st.dataframe(
        benchmark_df.set_index("Model"),
        use_container_width=True
    )

    # --------------------------------------------------
    # BENCHMARK BAR CHART
    # --------------------------------------------------
    st.markdown("### 📈 Benchmark Metric Comparison")

    model_names = benchmark_df["Model"].tolist()
    accs = benchmark_df["Accuracy"].tolist()
    recs = benchmark_df["Recall"].tolist()
    f1s = benchmark_df["F1"].tolist()

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - width, accs, width, label="Accuracy")
    b2 = ax.bar(x, recs, width, label="Recall")
    b3 = ax.bar(x + width, f1s, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Training Benchmark Metrics", fontweight="bold")
    ax.legend()

    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    ax.bar_label(b3, fmt="%.3f", padding=3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --------------------------------------------------
    # BENCHMARK CONFUSION MATRICES
    # --------------------------------------------------
    st.markdown("### 🔲 Training Confusion Matrices")

    tabs = st.tabs(list(training_metrics.keys()))

    for tab, (model_name, metrics) in zip(tabs, training_metrics.items()):
        with tab:
            cm = metrics["cm"]

            plot_confusion_matrix(
                cm,
                f"{model_name} — Training Benchmark"
            )

            tn, fp = cm[0]
            fn, tp = cm[1]

            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | TN | {tn:,} |
            | FP | {fp:,} |
            | FN | {fn:,} |
            | TP | {tp:,} |
            | **APS Cost** | **{calculate_cost(cm):,}** |
            """)

else:
    st.info("Training benchmark metrics not available.")

# ==================================================
# SECTION B — LIVE EVALUATION
# ==================================================
st.markdown("---")
st.markdown("## 🧪 Live Uploaded Dataset Evaluation")

if live_results:
    live_rows = []

    for model_name, result in live_results.items():
        if "metrics" not in result:
            continue

        metrics = result["metrics"]

        live_rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "Cost": metrics["cost"]
        })

    if live_rows:
        live_df = pd.DataFrame(live_rows)

        st.dataframe(
            live_df.set_index("Model"),
            use_container_width=True
        )

        st.markdown("### 🔲 Live Confusion Matrices")

        live_tabs = st.tabs([row["Model"] for row in live_rows])

        for tab, row in zip(live_tabs, live_rows):
            with tab:
                metrics = live_results[row["Model"]]["metrics"]

                plot_confusion_matrix(
                    metrics["cm"],
                    f"{row['Model']} — Live Evaluation"
                )

    else:
        st.info("Uploaded dataset has no labels. Live evaluation unavailable.")

else:
    st.info("Run predictions first to see live evaluation metrics.")

# ==================================================
# SECTION C — MODEL DETAILS
# ==================================================
st.markdown("---")
st.markdown("## 🏗️ Model Architecture Details")

arch_tabs = st.tabs([
    "Logistic Regression",
    "Random Forest",
    "LightGBM"
])

with arch_tabs[0]:
    st.markdown("""
    **Logistic Regression**
    
    - Linear baseline classifier
    - Solver: `lbfgs`
    - Max iterations: 1000
    - Trained on SMOTE-balanced APS data
    - Fast and interpretable
    - Less effective for complex nonlinear patterns
    """)

with arch_tabs[1]:
    st.markdown("""
    **Random Forest**
    
    - Ensemble of decision trees
    - Trained on SMOTE-balanced APS data
    - Robust to noise and outliers
    - Good feature importance analysis
    - Strong general-purpose tabular model
    """)

with arch_tabs[2]:
    st.markdown("""
    **LightGBM**
    
    - Gradient boosting decision trees
    - `n_estimators=300`
    - `learning_rate=0.05`
    - `num_leaves=63`
    - Excellent for high-dimensional tabular data
    - Fast inference
    """)

# ==================================================
# SECTION D — PREPROCESSING PIPELINE
# ==================================================
st.markdown("---")
st.markdown("## ⚙️ Preprocessing Pipeline")

st.markdown("""
1. **Drop high-missing columns**
   - Remove columns with >60% missing values

2. **Median imputation**
   - Fill missing values with median

3. **VarianceThreshold**
   - Remove near-constant features

4. **StandardScaler**
   - Normalize feature distributions

5. **Correlation pruning**
   - Remove highly correlated redundant features

6. **Feature selection**
   - Keep top 50 informative features

7. **SMOTE (training only)**
   - Balance APS failure class
""")