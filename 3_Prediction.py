# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from utils.model_loader import list_model_labels
# from utils.predictor import predict, predict_all

# st.title("🤖 Prediction & Model Comparison")

# if st.session_state.uploaded_df is None:
#     st.warning("⚠️ Please upload a dataset first (Page 1).")
#     st.stop()

# df = st.session_state.uploaded_df

# # ── Mode selection ─────────────────────────────────────────────────────────
# mode = st.radio(
#     "Prediction mode:",
#     ["Compare All 3 Models", "Single Model"],
#     horizontal=True
# )

# # ─────────────────────────────────────────
# # MODE A: Compare All 3 Models
# # ─────────────────────────────────────────
# if mode == "Compare All 3 Models":
#     st.markdown("Run **all available models** on the uploaded dataset and compare their predictions side-by-side.")

#     if st.button("🚀 Run All Models", type="primary"):
#         with st.spinner("Running all models... this may take a moment."):
#             try:
#                 all_results = predict_all(df)
#                 st.session_state.all_model_results = all_results
#                 st.success(f"✅ Predictions completed for: {', '.join(all_results.keys())}")
#             except FileNotFoundError as e:
#                 st.error(f"Model or artifact file not found: {e}")
#                 st.info("Make sure the `models/` directory contains `.pkl` files and `preprocess_artifacts.pkl`.")
#                 st.stop()
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")
#                 st.stop()

#     all_results = st.session_state.all_model_results

#     if all_results:
#         model_names = list(all_results.keys())

#         # ── Summary comparison table ──────────────────────────
#         st.markdown("### 📊 Summary Comparison")

#         summary_rows = []
#         for name, res in all_results.items():
#             failures = (res["Prediction"] == "APS Failure").sum()
#             normals  = (res["Prediction"] == "Normal").sum()
#             fail_pct = failures / len(res) * 100
#             summary_rows.append({
#                 "Model":         name,
#                 "Total Samples": len(res),
#                 "APS Failures":  failures,
#                 "Normal":        normals,
#                 "Failure Rate %": round(fail_pct, 2),
#             })

#         summary_df = pd.DataFrame(summary_rows)
#         st.dataframe(summary_df.set_index("Model"), use_container_width=True)

#         # ── Side-by-side bar chart ─────────────────────────────
#         st.markdown("### 📈 Failure Count per Model")
#         fig, ax = plt.subplots(figsize=(9, 4))
#         x = np.arange(len(model_names))
#         width = 0.35
#         failures_list = [summary_rows[i]["APS Failures"] for i in range(len(model_names))]
#         normals_list  = [summary_rows[i]["Normal"]       for i in range(len(model_names))]

#         bars1 = ax.bar(x - width / 2, failures_list, width, label="APS Failure", color="#F44336", alpha=0.85)
#         bars2 = ax.bar(x + width / 2, normals_list,  width, label="Normal",      color="#2196F3", alpha=0.85)

#         ax.set_xticks(x)
#         ax.set_xticklabels(model_names, fontweight="bold")
#         ax.set_ylabel("Sample Count")
#         ax.set_title("Prediction Counts per Model", fontweight="bold")
#         ax.legend()
#         ax.bar_label(bars1, fmt="%d", padding=3)
#         ax.bar_label(bars2, fmt="%d", padding=3)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()

#         # ── Agreement analysis ────────────────────────────────
#         st.markdown("### 🤝 Model Agreement")
#         pred_matrix = pd.DataFrame({
#             name: (res["Prediction"] == "APS Failure").astype(int).values
#             for name, res in all_results.items()
#         })

#         agree_all    = (pred_matrix.sum(axis=1) == len(model_names)).sum()
#         disagree_all = (pred_matrix.sum(axis=1) == 0).sum()
#         mixed        = len(pred_matrix) - agree_all - disagree_all

#         c1, c2, c3 = st.columns(3)
#         c1.metric("All agree: Failure", f"{agree_all:,}")
#         c2.metric("All agree: Normal",  f"{disagree_all:,}")
#         c3.metric("Models disagree",    f"{mixed:,}")

#         if mixed > 0:
#             st.info(f"ℹ️ {mixed} sample(s) have conflicting predictions across models. "
#                     "Check the individual model tabs below for details.")

#         # ── Per-model result tabs ─────────────────────────────
#         st.markdown("### 🔎 Per-Model Results")
#         model_tabs = st.tabs([f"{n}" for n in model_names])

#         for tab, (name, res) in zip(model_tabs, all_results.items()):
#             with tab:
#                 failures = (res["Prediction"] == "APS Failure").sum()
#                 normals  = (res["Prediction"] == "Normal").sum()

#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Total", len(res))
#                 col2.metric("Failures 🔴", failures)
#                 col3.metric("Normal 🟢",   normals)

#                 st.dataframe(res, use_container_width=True)

#                 csv = res.to_csv(index=False)
#                 st.download_button(
#                     f"⬇️ Download {name} results",
#                     csv,
#                     file_name=f"aps_{name.lower().replace(' ', '_')}_predictions.csv",
#                     mime="text/csv"
#                 )

# # ─────────────────────────────────────────
# # MODE B: Single Model
# # ─────────────────────────────────────────
# else:
#     available = list_model_labels()

#     if not available:
#         st.error("No trained models found in the `models/` directory.")
#         st.stop()

#     selected_model = st.selectbox("Choose model:", available)

#     if st.button("▶️ Run Prediction", type="primary"):
#         with st.spinner(f"Running {selected_model}..."):
#             try:
#                 results = predict(df, selected_model)
#                 st.session_state.prediction_results = results
#                 st.session_state.selected_model     = selected_model
#                 st.success("✅ Prediction complete.")
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")
#                 st.stop()

#     if st.session_state.prediction_results is not None and \
#        st.session_state.selected_model == selected_model:
#         results  = st.session_state.prediction_results
#         failures = (results["Prediction"] == "APS Failure").sum()
#         normals  = (results["Prediction"] == "Normal").sum()

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total Samples", len(results))
#         c2.metric("APS Failures 🔴", failures)
#         c3.metric("Normal 🟢", normals)

#         # Donut chart
#         fig, ax = plt.subplots(figsize=(5, 4))
#         ax.pie(
#             [failures, normals],
#             labels=["APS Failure", "Normal"],
#             colors=["#F44336", "#2196F3"],
#             autopct="%1.1f%%",
#             startangle=90,
#             wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.6}
#         )
#         ax.set_title(f"{selected_model} — Prediction Distribution", fontweight="bold")
#         st.pyplot(fig)
#         plt.close()

#         st.dataframe(results, use_container_width=True)

#         csv = results.to_csv(index=False)
#         st.download_button(
#             "⬇️ Download Results CSV",
#             csv,
#             file_name=f"aps_{selected_model.lower().replace(' ', '_')}_predictions.csv",
#             mime="text/csv"
#         )



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.model_loader import list_model_labels
from utils.predictor import predict, predict_all

st.title("🤖 Prediction & Model Comparison")

# --------------------------------------------------
# DATA CHECK
# --------------------------------------------------
if st.session_state.uploaded_df is None:
    st.warning("⚠️ Please upload a dataset first (Page 1).")
    st.stop()

df = st.session_state.uploaded_df
true_labels = st.session_state.true_labels


def display_prediction_labels(pred_df):
    """
    Convert numeric predictions to UI-friendly labels.
    Backend remains numeric.
    """
    df_display = pred_df.copy()
    df_display["Prediction"] = df_display["Prediction"].map({
        0: "Normal",
        1: "APS Failure"
    })
    return df_display


# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
mode = st.radio(
    "Prediction mode:",
    ["Compare All 3 Models", "Single Model"],
    horizontal=True
)

# ==================================================
# MODE A — COMPARE ALL
# ==================================================
if mode == "Compare All 3 Models":
    st.markdown(
        "Run **all available models** on the uploaded dataset "
        "and compare predictions side-by-side."
    )

    if st.button("🚀 Run All Models", type="primary"):
        with st.spinner("Running all models..."):
            try:
                all_results = predict_all(df, true_labels)
                st.session_state.evaluation_results = all_results
                st.session_state.selected_model = None

                st.success(
                    f"✅ Completed predictions for: {', '.join(all_results.keys())}"
                )

            except FileNotFoundError as e:
                st.error(f"Model or artifact file not found: {e}")
                st.stop()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    all_results = st.session_state.evaluation_results

    if all_results:
        model_names = list(all_results.keys())

        # --------------------------------------------------
        # SUMMARY TABLE
        # --------------------------------------------------
        st.markdown("### 📊 Summary Comparison")

        summary_rows = []

        for model_name, result in all_results.items():
            y_pred = result["y_pred"]

            failures = int((y_pred == 1).sum())
            normals = int((y_pred == 0).sum())
            fail_pct = (failures / len(y_pred)) * 100

            row = {
                "Model": model_name,
                "Total Samples": len(y_pred),
                "APS Failures": failures,
                "Normal": normals,
                "Failure Rate %": round(fail_pct, 2),
            }

            if "metrics" in result:
                row["Accuracy"] = round(result["metrics"]["accuracy"], 4)
                row["Recall"] = round(result["metrics"]["recall"], 4)
                row["F1"] = round(result["metrics"]["f1"], 4)
                row["Cost"] = result["metrics"]["cost"]

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df.set_index("Model"), use_container_width=True)

        # --------------------------------------------------
        # BAR CHART
        # --------------------------------------------------
        st.markdown("### 📈 Failure Count per Model")

        fig, ax = plt.subplots(figsize=(9, 4))

        x = np.arange(len(model_names))
        width = 0.35

        failures_list = [row["APS Failures"] for row in summary_rows]
        normals_list = [row["Normal"] for row in summary_rows]

        bars1 = ax.bar(
            x - width / 2,
            failures_list,
            width,
            label="APS Failure",
            alpha=0.85
        )

        bars2 = ax.bar(
            x + width / 2,
            normals_list,
            width,
            label="Normal",
            alpha=0.85
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontweight="bold")
        ax.set_ylabel("Sample Count")
        ax.set_title("Prediction Counts per Model", fontweight="bold")
        ax.legend()

        ax.bar_label(bars1, fmt="%d", padding=3)
        ax.bar_label(bars2, fmt="%d", padding=3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --------------------------------------------------
        # AGREEMENT ANALYSIS
        # --------------------------------------------------
        st.markdown("### 🤝 Model Agreement")

        pred_matrix = pd.DataFrame({
            model_name: result["y_pred"]
            for model_name, result in all_results.items()
        })

        all_failure = int((pred_matrix.sum(axis=1) == len(model_names)).sum())
        all_normal = int((pred_matrix.sum(axis=1) == 0).sum())
        mixed = len(pred_matrix) - all_failure - all_normal

        c1, c2, c3 = st.columns(3)

        c1.metric("All agree: Failure", f"{all_failure:,}")
        c2.metric("All agree: Normal", f"{all_normal:,}")
        c3.metric("Models disagree", f"{mixed:,}")

        # --------------------------------------------------
        # PER MODEL TABS
        # --------------------------------------------------
        st.markdown("### 🔎 Per-Model Results")

        tabs = st.tabs(model_names)

        for tab, (model_name, result) in zip(tabs, all_results.items()):
            with tab:
                y_pred = result["y_pred"]

                failures = int((y_pred == 1).sum())
                normals = int((y_pred == 0).sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(y_pred))
                c2.metric("Failures 🔴", failures)
                c3.metric("Normal 🟢", normals)

                if "metrics" in result:
                    m = result["metrics"]

                    st.markdown("#### Live Evaluation Metrics")
                    mc1, mc2, mc3, mc4 = st.columns(4)

                    mc1.metric("Accuracy", f"{m['accuracy']:.4f}")
                    mc2.metric("Recall", f"{m['recall']:.4f}")
                    mc3.metric("F1", f"{m['f1']:.4f}")
                    mc4.metric("Cost", f"{m['cost']:,}")

                display_df = display_prediction_labels(result["prediction_df"])
                st.dataframe(display_df, use_container_width=True)

                csv = display_df.to_csv(index=False)

                st.download_button(
                    f"⬇️ Download {model_name} Results",
                    csv,
                    file_name=f"aps_{model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

# ==================================================
# MODE B — SINGLE MODEL
# ==================================================
else:
    available_models = list_model_labels()

    if not available_models:
        st.error("No trained models found.")
        st.stop()

    selected_model = st.selectbox("Choose model:", available_models)

    if st.button("▶️ Run Prediction", type="primary"):
        with st.spinner(f"Running {selected_model}..."):
            try:
                result = predict(df, selected_model, true_labels)

                st.session_state.evaluation_results = {
                    selected_model: result
                }

                st.session_state.selected_model = selected_model

                st.success("✅ Prediction complete.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    eval_results = st.session_state.evaluation_results

    if (
        eval_results
        and selected_model in eval_results
        and st.session_state.selected_model == selected_model
    ):
        result = eval_results[selected_model]
        y_pred = result["y_pred"]

        failures = int((y_pred == 1).sum())
        normals = int((y_pred == 0).sum())

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Samples", len(y_pred))
        c2.metric("APS Failures 🔴", failures)
        c3.metric("Normal 🟢", normals)

        if "metrics" in result:
            st.markdown("### Live Evaluation Metrics")

            m = result["metrics"]

            mc1, mc2, mc3, mc4 = st.columns(4)

            mc1.metric("Accuracy", f"{m['accuracy']:.4f}")
            mc2.metric("Recall", f"{m['recall']:.4f}")
            mc3.metric("F1", f"{m['f1']:.4f}")
            mc4.metric("Cost", f"{m['cost']:,}")

        # Donut chart
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.pie(
            [failures, normals],
            labels=["APS Failure", "Normal"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.6}
        )

        ax.set_title(
            f"{selected_model} — Prediction Distribution",
            fontweight="bold"
        )

        st.pyplot(fig)
        plt.close()

        display_df = display_prediction_labels(result["prediction_df"])
        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False)

        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            file_name=f"aps_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )