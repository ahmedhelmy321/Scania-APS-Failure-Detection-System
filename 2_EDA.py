import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 Exploratory Data Analysis")

if st.session_state.uploaded_df is None:
    st.warning("⚠️ Please upload a dataset first (Page 1).")
    st.stop()

df = st.session_state.uploaded_df.copy()
true_labels = st.session_state.get("true_labels")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "📊 Class Distribution",
    "❓ Missing Values",
    "📈 Feature Distributions",
    "🔗 Correlation Heatmap",
])

# ─────────────────────────────────────────
# TAB 1: Overview
# ─────────────────────────────────────────
with tabs[0]:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df.shape[0]:,}")
    c2.metric("Feature Columns", df.shape[1])
    total_missing = df.isnull().sum().sum()
    c3.metric("Total Missing Cells", f"{total_missing:,}")
    miss_pct = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2)
    c4.metric("Overall Missing %", f"{miss_pct}%")

    st.markdown("#### Data Types")
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "count"]
    st.dataframe(dtype_counts, use_container_width=True)

    st.markdown("#### Statistical Summary (numeric columns)")
    st.dataframe(df.describe().T.style.format("{:.4f}"), use_container_width=True)

# ─────────────────────────────────────────
# TAB 2: Class Distribution
# ─────────────────────────────────────────
with tabs[1]:
    st.subheader("Class Distribution")

    if true_labels is None:
        st.info("No `class` column found in the uploaded file. Class distribution is unavailable.")
    else:
        vc = true_labels.value_counts()
        vc_pct = true_labels.value_counts(normalize=True) * 100

        c1, c2 = st.columns(2)
        c1.metric("Negative (Normal)", f"{vc.get('neg', 0):,}", f"{vc_pct.get('neg', 0):.1f}%")
        c2.metric("Positive (APS Failure)", f"{vc.get('pos', 0):,}", f"{vc_pct.get('pos', 0):.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        colors = ["#2196F3", "#F44336"]
        axes[0].bar(["Normal (neg)", "APS Failure (pos)"],
                    [vc.get("neg", 0), vc.get("pos", 0)],
                    color=colors, edgecolor="white", linewidth=1.5)
        axes[0].set_title("Class Counts", fontweight="bold")
        axes[0].set_ylabel("Count")
        for bar in axes[0].patches:
            axes[0].annotate(f"{int(bar.get_height()):,}",
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontweight="bold")

        # Pie chart
        axes[1].pie(
            [vc.get("neg", 0), vc.get("pos", 0)],
            labels=["Normal", "APS Failure"],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        axes[1].set_title("Class Proportions", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ratio = vc.get("neg", 1) / max(vc.get("pos", 1), 1)
        st.info(f"⚖️ Class imbalance ratio (neg:pos) = **{ratio:.1f}:1** — SMOTE was used during training to handle this.")

# ─────────────────────────────────────────
# TAB 3: Missing Values
# ─────────────────────────────────────────
with tabs[2]:
    st.subheader("Missing Values Analysis")

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    miss_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct
    }).sort_values("Missing %", ascending=False)

    cols_with_missing = miss_df[miss_df["Missing Count"] > 0]
    cols_no_missing   = miss_df[miss_df["Missing Count"] == 0]

    c1, c2 = st.columns(2)
    c1.metric("Columns WITH missing values", len(cols_with_missing))
    c2.metric("Complete columns", len(cols_no_missing))

    if not cols_with_missing.empty:
        st.markdown("#### Top 30 columns by missing %")
        top30 = cols_with_missing.head(30)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_bar = ["#F44336" if v > 60 else "#FF9800" if v > 20 else "#2196F3"
                      for v in top30["Missing %"]]
        ax.barh(top30.index[::-1], top30["Missing %"][::-1], color=colors_bar[::-1])
        ax.axvline(60, color="red", linestyle="--", alpha=0.7, label="60% threshold (drop)")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values per Column (top 30)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        high_miss = cols_with_missing[cols_with_missing["Missing %"] > 60]
        if not high_miss.empty:
            st.warning(f"🗑️ {len(high_miss)} columns exceed 60% missing — they were **dropped** during preprocessing.")

        st.markdown("#### Full missing value table")
        st.dataframe(cols_with_missing.style.background_gradient(
            subset=["Missing %"], cmap="Reds"), use_container_width=True)
    else:
        st.success("✅ No missing values found in this dataset.")

# ─────────────────────────────────────────
# TAB 4: Feature Distributions
# ─────────────────────────────────────────
with tabs[3]:
    st.subheader("Feature Distributions")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found.")
    else:
        selected_cols = st.multiselect(
            "Select features to visualize (max 12):",
            options=numeric_cols,
            default=numeric_cols[:6],
            max_selections=12
        )

        if selected_cols:
            n = len(selected_cols)
            ncols = 3
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
            axes = axes.flatten() if n > 1 else [axes]

            for i, col in enumerate(selected_cols):
                data = df[col].dropna()
                axes[i].hist(data, bins=40, color="#0f3460", edgecolor="white", alpha=0.85)
                axes[i].set_title(col, fontweight="bold", fontsize=9)
                axes[i].set_xlabel("Value", fontsize=8)
                axes[i].set_ylabel("Frequency", fontsize=8)
                axes[i].tick_params(labelsize=7)

                mean_val = data.mean()
                axes[i].axvline(mean_val, color="#F44336", linestyle="--",
                                linewidth=1.2, label=f"Mean: {mean_val:.2f}")
                axes[i].legend(fontsize=7)

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle("Feature Distributions", fontweight="bold", fontsize=13, y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Box Plots (outlier view)")
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
            axes2 = axes2.flatten() if n > 1 else [axes2]

            for i, col in enumerate(selected_cols):
                data = df[col].dropna()
                axes2[i].boxplot(data, patch_artist=True,
                                 boxprops=dict(facecolor="#0f3460", alpha=0.7),
                                 medianprops=dict(color="#F44336", linewidth=2))
                axes2[i].set_title(col, fontweight="bold", fontsize=9)
                axes2[i].tick_params(labelsize=7)

            for j in range(i + 1, len(axes2)):
                axes2[j].set_visible(False)

            plt.suptitle("Box Plots (Outlier Detection)", fontweight="bold", fontsize=13, y=1.01)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────
# TAB 5: Correlation Heatmap
# ─────────────────────────────────────────
with tabs[4]:
    st.subheader("Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis.")
    else:
        max_cols = st.slider(
            "Number of columns to include (sorted by variance):",
            min_value=5, max_value=min(60, len(numeric_cols)), value=30, step=5
        )

        # Pick top-variance columns to keep the heatmap readable
        top_var_cols = (
            df[numeric_cols]
            .var()
            .sort_values(ascending=False)
            .head(max_cols)
            .index.tolist()
        )

        corr = df[top_var_cols].corr()

        # Highlight highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > 0.95:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

        if high_corr_pairs:
            st.warning(
                f"⚠️ {len(high_corr_pairs)} feature pair(s) have |correlation| > 0.95 "
                f"— these were removed during preprocessing."
            )

        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            center=0,
            annot=False,
            linewidths=0.3,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"Correlation Matrix (top {max_cols} features by variance)",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if high_corr_pairs:
            st.markdown("#### Highly Correlated Pairs (|r| > 0.95)")
            st.dataframe(
                pd.DataFrame(high_corr_pairs, columns=["Feature A", "Feature B", "|Correlation|"]),
                use_container_width=True
            )