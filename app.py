"""
NIErnaya 2026 | AI/ML Preliminary Diabetes Risk Assessment & Clinical Escalation Suite
======================================================================================
Upgraded End-to-End Analytics, Leak-Free ML Benchmark, Exact Shapley XAI & Clinical Dashboard

How to Run:
-----------
Option A (Terminal - Recommended):
    streamlit run "niernaya_2026_phase_2_streamlit_dashboard (1).py" --server.port 8501
Option B (Direct Python Execution):
    python "niernaya_2026_phase_2_streamlit_dashboard (1).py"
"""

import itertools
import math
import os
import subprocess
import sys
import warnings
from datetime import datetime

# -----------------------------------------------------------------------------
# 0. SAFE ENVIRONMENT BOOTSTRAPPER (ONE-CLICK IDE RUN & AUTO-BROWSER LAUNCH)
# -----------------------------------------------------------------------------
def _boot_streamlit_if_direct():
    """Launches Streamlit server cleanly (without blocking on first-run email prompt) if run via IDE Run button."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None and os.environ.get("NIERNAYA_STREAMLIT_BOOTED") != "1":
            os.environ["NIERNAYA_STREAMLIT_BOOTED"] = "1"
            # Ensure ~/.streamlit/credentials.toml exists so Streamlit never hangs on stdin email prompt
            cred_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
            cred_file = os.path.join(cred_dir, "credentials.toml")
            if not os.path.exists(cred_file):
                os.makedirs(cred_dir, exist_ok=True)
                with open(cred_file, "w", encoding="utf-8") as f:
                    f.write('[general]\nemail = ""\n')

            import threading
            import webbrowser
            print("\n🚀 Launching NIErnaya 2026 Streamlit Analytics Suite on http://localhost:8501 ...")
            threading.Timer(2.0, lambda: webbrowser.open("http://localhost:8501")).start()
            cmd = [
                sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
                "--server.headless=true",
                "--server.address=localhost",
                "--server.port=8501",
                "--browser.gatherUsageStats=false",
            ]
            sys.exit(subprocess.call(cmd, stdin=subprocess.DEVNULL))
    except ImportError:
        pass


_boot_streamlit_if_direct()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ADAPTIVE DARK/LIGHT PREMIUM UI THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NIErnaya 2026 | AI/ML Diabetes Risk & Explainable Clinical Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #0f2942 52%, #0d9488 100%);
        padding: 1.65rem 2.1rem;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 1.35rem;
        box-shadow: 0 10px 28px rgba(13, 148, 136, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.35rem 0;
    }
    .hero-banner p {
        color: #e2e8f0 !important;
        font-size: 0.96rem;
        margin: 0;
        line-height: 1.5;
    }
    .hero-badges {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(6px);
        color: #f8fafc !important;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Custom KPI Cards (Adaptive to Light & Dark Themes) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-left: 5px solid #14b8a6;
        border-radius: 13px;
        padding: 1.05rem 1.25rem;
        color: #f8fafc !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    .kpi-card.accent-blue { border-left-color: #3b82f6; }
    .kpi-card.accent-amber { border-left-color: #f59e0b; }
    .kpi-card.accent-rose { border-left-color: #f43f5e; }
    .kpi-card.accent-violet { border-left-color: #8b5cf6; }
    .kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #5eead4 !important;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* High-Contrast Clinical Insight Callout (Safe in Dark & Light Mode) */
    .insight-box {
        background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
        color: #f1f5f9 !important;
        border-left: 5px solid #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left-width: 5px;
        padding: 1.05rem 1.35rem;
        border-radius: 11px;
        margin: 0.75rem 0 1.1rem 0;
        font-size: 0.93rem;
        line-height: 1.55;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .insight-box b, .insight-box strong {
        color: #7dd3fc !important;
    }
    .insight-box code {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #fde047 !important;
        padding: 0.12rem 0.38rem;
        border-radius: 4px;
    }

    /* Architecture Stage Cards */
    .arch-card {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-top: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.15rem 1.3rem;
        color: #e2e8f0 !important;
        height: 100%;
    }
    .arch-card h4 {
        color: #ffffff !important;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. LEAK-FREE CUSTOM SCIKIT-LEARN CLINICAL PREPROCESSING & FE TRANSFORMER
# -----------------------------------------------------------------------------
class ClinicalFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end leak-free scikit-learn transformer that:
      1. Converts biologically impossible 0s to NaN in Glucose, BP, SkinThickness, Insulin, BMI.
      2. Fits Multivariate MICE (IterativeImputer) strictly on training fold X (zero target leakage).
      3. Applies 1st-99th percentile clinical Winsorization learned from the training fold.
      4. Engineers 6 clinical domain features (HOMA_IR, ICMR_BMI_Risk_Code, Metabolic_Risk_Score,
         Glucose_BMI_Ratio, Visceral_Adiposity_Proxy, Age_Glycemic_Interaction).
    """

    def __init__(self, winsorize: bool = True):
        self.winsorize = winsorize
        self.bio_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        self.base_cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ]

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.base_cols).copy()
        df[self.bio_zero_cols] = df[self.bio_zero_cols].replace(0, np.nan)
        self.imputer_ = IterativeImputer(max_iter=25, random_state=42)
        X_imp = pd.DataFrame(self.imputer_.fit_transform(df), columns=self.base_cols)
        if self.winsorize:
            self.lower_bounds_ = X_imp.quantile(0.01)
            self.upper_bounds_ = X_imp.quantile(0.99)
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.base_cols).copy()
        df[self.bio_zero_cols] = df[self.bio_zero_cols].replace(0, np.nan)
        X_imp = pd.DataFrame(
            self.imputer_.transform(df),
            columns=self.base_cols,
            index=df.index,
        )
        # Enforce physiological lower floors
        X_imp["Glucose"] = X_imp["Glucose"].clip(lower=44.0)
        X_imp["BloodPressure"] = X_imp["BloodPressure"].clip(lower=38.0)
        X_imp["SkinThickness"] = X_imp["SkinThickness"].clip(lower=7.0)
        X_imp["Insulin"] = X_imp["Insulin"].clip(lower=14.0)
        X_imp["BMI"] = X_imp["BMI"].clip(lower=15.0)

        if self.winsorize:
            X_imp = X_imp.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

        # Engineered Domain Features
        X_imp["HOMA_IR"] = (X_imp["Glucose"] * X_imp["Insulin"]) / 405.0
        X_imp["ICMR_BMI_Risk_Code"] = pd.cut(
            X_imp["BMI"],
            bins=[-np.inf, 22.99, 24.99, np.inf],
            labels=[0, 1, 2],
        ).astype(int)
        X_imp["Metabolic_Risk_Score"] = (
            (X_imp["Glucose"] >= 100).astype(int)
            + (X_imp["BMI"] >= 23.0).astype(int)
            + (X_imp["BloodPressure"] >= 80).astype(int)
            + (X_imp["HOMA_IR"] >= 2.5).astype(int)
        )
        X_imp["Glucose_BMI_Ratio"] = X_imp["Glucose"] / X_imp["BMI"]
        X_imp["Visceral_Adiposity_Proxy"] = (X_imp["SkinThickness"] * X_imp["BMI"]) / 100.0
        X_imp["Age_Glycemic_Interaction"] = (X_imp["Age"] * X_imp["Glucose"]) / 100.0
        return X_imp


# -----------------------------------------------------------------------------
# 3. DATA LOADING, CLEANING, WINSORIZATION & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def resolve_dataset_path(filename: str) -> str:
    """Finds dataset across workspace root or subdirectories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(script_dir, "datasets", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return filename


@st.cache_data(show_spinner="Running Stage 1 Clinical Audit, MICE Imputation, Winsorization & Statistical Tests...")
def load_and_process_stage1(filepath: str = "diabetes2.csv"):
    filepath = resolve_dataset_path(filepath)
    if not os.path.exists(filepath):
        return None

    df_raw = pd.read_csv(filepath)
    bio_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    feature_cols = [c for c in df_raw.columns if c != "Outcome"]

    # 1. Audit Disguised Missingness (Biologically Impossible Zeros)
    zero_audit = pd.DataFrame({
        "Feature": bio_zero_cols,
        "Disguised_Zeros_Count": [(df_raw[c] == 0).sum() for c in bio_zero_cols],
        "Disguised_Zeros_Pct": [round((df_raw[c] == 0).mean() * 100, 2) for c in bio_zero_cols],
        "Raw_Min": [df_raw[c].min() for c in bio_zero_cols],
        "Clinical_Valid_Min": [df_raw.loc[df_raw[c] > 0, c].min() for c in bio_zero_cols],
        "Raw_Corr_with_Outcome": [round(df_raw[c].corr(df_raw["Outcome"]), 4) for c in bio_zero_cols],
    })

    # 2. Convert biological zeros to NaN
    df_nan = df_raw.copy()
    df_nan[bio_zero_cols] = df_nan[bio_zero_cols].replace(0, np.nan)
    X_nan = df_nan[feature_cols]
    y = df_raw["Outcome"]

    # 3. Compare Imputation Methods (Mean vs KNN vs MICE)
    X_mean = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_nan), columns=feature_cols)
    X_knn = pd.DataFrame(KNNImputer(n_neighbors=7).fit_transform(X_nan), columns=feature_cols)
    mice_imputer = IterativeImputer(max_iter=25, random_state=42)
    X_mice = pd.DataFrame(mice_imputer.fit_transform(X_nan), columns=feature_cols)
    X_mice["Glucose"] = X_mice["Glucose"].clip(lower=44.0)
    X_mice["BloodPressure"] = X_mice["BloodPressure"].clip(lower=38.0)
    X_mice["SkinThickness"] = X_mice["SkinThickness"].clip(lower=7.0)
    X_mice["Insulin"] = X_mice["Insulin"].clip(lower=14.0)
    X_mice["BMI"] = X_mice["BMI"].clip(lower=15.0)

    # 4. Implement 1st-99th Percentile Winsorization & IQR Outlier Audit
    lower_1 = X_mice.quantile(0.01)
    upper_99 = X_mice.quantile(0.99)
    X_winsor = X_mice.clip(lower=lower_1, upper=upper_99, axis=1)

    outlier_rows = []
    for col in feature_cols:
        s_obs = df_nan[col].dropna()
        s_mice = X_mice[col]
        s_win = X_winsor[col]
        q1, q3 = s_obs.quantile(0.25), s_obs.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s_obs < lower) | (s_obs > upper)).sum())
        outlier_rows.append({
            "Feature": col,
            "Q1 (25%)": round(q1, 2),
            "Median (50%)": round(s_obs.median(), 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "IQR_Outliers": n_out,
            "Outlier_Pct": round((n_out / len(s_obs)) * 100, 2),
            "Winsor_1st_99th_Bounds": f"[{lower_1[col]:.1f}, {upper_99[col]:.1f}]",
            "Skew_Pre_Winsor": round(s_mice.skew(), 2),
            "Skew_Post_Winsor": round(s_win.skew(), 2),
        })
    outlier_audit = pd.DataFrame(outlier_rows)

    zero_audit["Cleaned_MICE_Corr_with_Outcome"] = [
        round(X_winsor[c].corr(y), 4) for c in bio_zero_cols
    ]
    zero_audit["Signal_Boost_Pct"] = (
        (zero_audit["Cleaned_MICE_Corr_with_Outcome"] - zero_audit["Raw_Corr_with_Outcome"])
        / zero_audit["Raw_Corr_with_Outcome"]
        * 100
    ).round(1)

    # 5. Build Full Cleaned + Winsorized + Feature-Engineered Dataset
    df_clean = X_winsor.copy()
    df_clean["Outcome"] = y.values
    df_clean["Outcome_Label"] = df_clean["Outcome"].map(
        {0: "Non-Diabetic (0)", 1: "Diabetic / High-Risk (1)"}
    )

    # 6 Domain-Engineered Features
    df_clean["HOMA_IR"] = ((df_clean["Glucose"] * df_clean["Insulin"]) / 405.0).round(3)
    df_clean["ICMR_BMI_Category"] = pd.cut(
        df_clean["BMI"],
        bins=[0, 22.99, 24.99, 29.99, 100],
        labels=[
            "1. Normal (<23.0)",
            "2. Indian Overweight (23.0-24.9)",
            "3. Indian Obese / WHO Overweight (25.0-29.9)",
            "4. WHO Obese (>=30.0)",
        ],
    )
    df_clean["ICMR_BMI_Risk_Code"] = pd.cut(
        df_clean["BMI"], bins=[0, 22.99, 24.99, 100], labels=[0, 1, 2]
    ).astype(int)
    df_clean["Metabolic_Risk_Score"] = (
        (df_clean["Glucose"] >= 100).astype(int)
        + (df_clean["BMI"] >= 23.0).astype(int)
        + (df_clean["BloodPressure"] >= 80).astype(int)
        + (df_clean["HOMA_IR"] >= 2.5).astype(int)
    )
    df_clean["Glucose_BMI_Ratio"] = (df_clean["Glucose"] / df_clean["BMI"]).round(3)
    df_clean["Visceral_Adiposity_Proxy"] = ((df_clean["SkinThickness"] * df_clean["BMI"]) / 100.0).round(3)
    df_clean["Age_Glycemic_Interaction"] = ((df_clean["Age"] * df_clean["Glucose"]) / 100.0).round(3)
    df_clean["Age_Bracket"] = pd.cut(
        df_clean["Age"],
        bins=[20, 30, 45, 60, 100],
        labels=["Young Adult (21-30)", "Early Middle (31-45)", "Late Middle (46-60)", "Senior (60+)"],
    )

    # 7. Statistical Hypothesis Testing (Mann-Whitney U, Welch's t-test, Cohen's d)
    stat_cols = feature_cols + [
        "HOMA_IR", "ICMR_BMI_Risk_Code", "Metabolic_Risk_Score",
        "Glucose_BMI_Ratio", "Visceral_Adiposity_Proxy", "Age_Glycemic_Interaction",
    ]
    stat_rows = []
    g0 = df_clean[df_clean["Outcome"] == 0]
    g1 = df_clean[df_clean["Outcome"] == 1]
    for col in stat_cols:
        x0, x1 = g0[col].dropna(), g1[col].dropna()
        u_stat, u_pval = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        t_stat, t_pval = stats.ttest_ind(x1, x0, equal_var=False)
        pooled_sd = np.sqrt((x0.std() ** 2 + x1.std() ** 2) / 2.0)
        cohens_d = (x1.mean() - x0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        stat_rows.append({
            "Feature": col,
            "Type": "Engineered (Domain)" if col not in feature_cols else "Baseline Biometric",
            "Non_Diabetic_Median": round(x0.median(), 2),
            "Diabetic_Median": round(x1.median(), 2),
            "Cohens_d_Effect": round(cohens_d, 3),
            "Effect_Tier": "Large (>=0.8)" if abs(cohens_d) >= 0.8 else "Medium (0.5-0.8)" if abs(cohens_d) >= 0.5 else "Small (<0.5)",
            "Welch_t_Stat": round(t_stat, 2),
            "Mann_Whitney_p": f"{u_pval:.2e}" if u_pval < 0.0001 else f"{u_pval:.5f}",
        })
    stat_tests_df = pd.DataFrame(stat_rows).sort_values("Cohens_d_Effect", ascending=False)

    return {
        "df_raw": df_raw,
        "df_nan": df_nan,
        "df_clean": df_clean,
        "X_mean": X_mean,
        "X_knn": X_knn,
        "X_mice": X_mice,
        "X_winsor": X_winsor,
        "zero_audit": zero_audit,
        "outlier_audit": outlier_audit,
        "stat_tests_df": stat_tests_df,
        "feature_cols": feature_cols,
        "healthy_medians": g0[feature_cols].median().to_dict(),
        "cohort_medians": df_clean[feature_cols].median().to_dict(),
    }


def map_icd9_to_chapter(code) -> str:
    """Maps 700+ raw ICD-9 diagnosis strings into 9 standardized clinical disease chapters."""
    if pd.isna(code) or str(code).strip() == "?":
        return "Unrecorded"
    s = str(code).strip()
    if s.startswith(("V", "E")):
        return "Supplemental / External"
    try:
        val = float(s)
        if (390 <= val <= 459) or val == 785:
            return "Circulatory / Cardiovascular"
        elif (460 <= val <= 519) or val == 786:
            return "Respiratory"
        elif (520 <= val <= 579) or val == 787:
            return "Digestive"
        elif 250 <= val < 251:
            return "Uncontrolled Diabetes (250.xx)"
        elif 800 <= val <= 999:
            return "Injury & Poisoning"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif (580 <= val <= 629) or val == 788:
            return "Renal / Genitourinary"
        elif 140 <= val <= 239:
            return "Neoplasms"
        else:
            return "Other Systemic"
    except ValueError:
        return "Other Systemic"


def _summarize_stage2_cohort(df_sub: pd.DataFrame) -> dict:
    """Computes aggregated clinical tables and Sankey flow data for a given Stage 2 cohort."""
    diag_summary = (
        df_sub.groupby("Primary_Diagnosis_Chapter", observed=False)
        .agg(
            Encounters=("encounter_id", "count"),
            Acute_30d_Readmit_Pct=("Acute_30d_Readmission", lambda x: round(x.mean() * 100, 2)),
            Total_Readmit_Pct=("Any_Readmission", lambda x: round(x.mean() * 100, 2)),
            Avg_Time_in_Hospital=("time_in_hospital", lambda x: round(x.mean(), 2)),
            Avg_Medications=("num_medications", lambda x: round(x.mean(), 2)),
        )
        .reset_index()
        .sort_values("Encounters", ascending=False)
    )

    a1c_summary = (
        df_sub.groupby("HbA1c_Screening_Status", observed=False)
        .agg(
            Encounters=("encounter_id", "count"),
            Acute_30d_Readmit_Pct=("Acute_30d_Readmission", lambda x: round(x.mean() * 100, 2)),
            Total_Readmit_Pct=("Any_Readmission", lambda x: round(x.mean() * 100, 2)),
            Avg_LOS_Days=("time_in_hospital", lambda x: round(x.mean(), 2)),
        )
        .reset_index()
    )

    age_readmit = (
        df_sub.groupby("age", observed=False)
        .agg(
            Encounters=("encounter_id", "count"),
            Acute_30d_Readmit_Pct=("Acute_30d_Readmission", lambda x: round(x.mean() * 100, 2)),
            Total_Readmit_Pct=("Any_Readmission", lambda x: round(x.mean() * 100, 2)),
            Unscreened_HbA1c_Pct=("A1Cresult", lambda x: round(x.isna().mean() * 100, 2)),
            Avg_Medications=("num_medications", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
    )

    # Sankey flow: Age Tier -> HbA1c Screening -> Readmission Outcome
    sankey_grp = (
        df_sub.groupby(["Age_Macro_Group", "HbA1c_Screened_Binary", "Readmit_Outcome_Label"], observed=False)
        .size()
        .reset_index(name="Count")
    )

    return {
        "diag_summary": diag_summary,
        "a1c_summary": a1c_summary,
        "age_readmit": age_readmit,
        "sankey_grp": sankey_grp,
        "acute_30d_pct": round(df_sub["Acute_30d_Readmission"].mean() * 100, 2),
        "any_readmit_pct": round(df_sub["Any_Readmission"].mean() * 100, 2),
        "unscreened_pct": round(df_sub["A1Cresult"].isna().mean() * 100, 2),
    }


@st.cache_data(show_spinner="Running Stage 2 Hospital Cohort Wrangling & Longitudinal Patient Deduplication...")
def load_and_process_stage2(filepath: str = "diabetic_data2.csv"):
    filepath = resolve_dataset_path(filepath)
    if not os.path.exists(filepath):
        return None

    df_uci = pd.read_csv(filepath, low_memory=False)
    total_raw_rows = len(df_uci)
    unique_patients_raw = df_uci["patient_nbr"].nunique()
    repeat_encounters_raw = total_raw_rows - unique_patients_raw

    # 1. Audit '?' and standard NaNs
    q_counts = (df_uci == "?").sum()
    na_counts = df_uci.isna().sum()
    total_miss = q_counts + na_counts
    miss_df = pd.DataFrame({
        "Feature": df_uci.columns,
        "Question_Mark_Count": q_counts.values,
        "Standard_NaN_Count": na_counts.values,
        "Total_Missing": total_miss.values,
        "Missing_Pct": ((total_miss / total_raw_rows) * 100).round(2).values,
    })
    miss_df = miss_df[miss_df["Total_Missing"] > 0].sort_values("Missing_Pct", ascending=False)

    # 2. Identify Zero-Variance features
    zero_var_cols = [c for c in df_uci.columns if df_uci[c].nunique() <= 1]

    # 3. Filter out Expired / Hospice encounters [11, 13, 14, 19, 20, 21]
    expired_ids = [11, 13, 14, 19, 20, 21]
    expired_count = int(df_uci["discharge_disposition_id"].isin(expired_ids).sum())

    df_valid = df_uci[~df_uci["discharge_disposition_id"].isin(expired_ids)].copy()
    df_valid = df_valid[df_valid["gender"] != "Unknown/Invalid"].copy()

    # 4. Feature Engineering on Stage 2
    df_valid["HbA1c_Screening_Status"] = df_valid["A1Cresult"].fillna("Not Screened (Barrier)").replace({
        "Norm": "Screened: Normal (<7%)",
        ">7": "Screened: Elevated (>7%)",
        ">8": "Screened: Severe (>8%)",
    })
    df_valid["HbA1c_Screened_Binary"] = np.where(
        df_valid["A1Cresult"].isna(), "HbA1c Unscreened (Barrier)", "HbA1c Screened"
    )
    df_valid["Primary_Diagnosis_Chapter"] = df_valid["diag_1"].apply(map_icd9_to_chapter)
    df_valid["Total_Prior_Visits"] = (
        df_valid["number_outpatient"] + df_valid["number_emergency"] + df_valid["number_inpatient"]
    )

    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    df_valid["Active_Diabetes_Meds"] = (df_valid[med_cols] != "No").sum(axis=1)
    df_valid["Acute_30d_Readmission"] = (df_valid["readmitted"] == "<30").astype(int)
    df_valid["Any_Readmission"] = (df_valid["readmitted"] != "NO").astype(int)
    df_valid["Readmit_Outcome_Label"] = df_valid["readmitted"].map({
        "<30": "Acute Readmission (<30d)",
        ">30": "Delayed Readmission (>30d)",
        "NO": "No Readmission",
    })
    df_valid["Age_Macro_Group"] = df_valid["age"].map({
        "[0-10)": "Pediatric/Young (<40)", "[10-20)": "Pediatric/Young (<40)",
        "[20-30)": "Pediatric/Young (<40)", "[30-40)": "Pediatric/Young (<40)",
        "[40-50)": "Middle Aged (40-60)", "[50-60)": "Middle Aged (40-60)",
        "[60-70)": "Senior (60-80)", "[70-80)": "Senior (60-80)",
        "[80-90)": "Elderly (80+)", "[90-100)": "Elderly (80+)",
    })

    # 5. Longitudinal Patient Deduplication (First Index Encounter per patient_nbr)
    df_dedup = df_valid.sort_values("encounter_id").drop_duplicates(subset=["patient_nbr"], keep="first").copy()

    all_encounters_summary = _summarize_stage2_cohort(df_valid)
    dedup_patients_summary = _summarize_stage2_cohort(df_dedup)

    return {
        "total_raw_rows": total_raw_rows,
        "unique_patients_raw": unique_patients_raw,
        "repeat_encounters_raw": repeat_encounters_raw,
        "expired_count": expired_count,
        "valid_rows": len(df_valid),
        "dedup_rows": len(df_dedup),
        "zero_var_cols": zero_var_cols,
        "miss_df": miss_df,
        "all_encounters": all_encounters_summary,
        "dedup_patients": dedup_patients_summary,
        "df_valid_sample": df_valid[
            [
                "encounter_id", "patient_nbr", "age", "gender", "time_in_hospital",
                "num_lab_procedures", "num_medications", "Total_Prior_Visits",
                "Active_Diabetes_Meds", "HbA1c_Screening_Status",
                "Primary_Diagnosis_Chapter", "readmitted",
            ]
        ].head(500),
    }


# -----------------------------------------------------------------------------
# 4. LEAK-FREE ML MODEL LAB, 5-FOLD CV, PCA & EXACT SHAPLEY ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Leak-Free 5-Model Ensemble, Running 5-Fold CV, PCA & Exact Shapley Engine...")
def compute_models_pca_and_xai(df_raw: pd.DataFrame, df_clean: pd.DataFrame, feature_cols: list):
    engineered_cols = [
        "Glucose", "BMI", "Age", "Pregnancies", "BloodPressure",
        "SkinThickness", "Insulin", "DiabetesPedigreeFunction",
        "HOMA_IR", "ICMR_BMI_Risk_Code", "Metabolic_Risk_Score",
        "Glucose_BMI_Ratio", "Visceral_Adiposity_Proxy", "Age_Glycemic_Interaction",
    ]
    X_all_clean = df_clean[engineered_cols]
    y = df_raw["Outcome"]
    X_raw_base = df_raw[feature_cols]

    # 1. Feature Selection Ranking (RF Gini + Mutual Info + Pearson r + Cohen's d)
    rf_fs = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
    rf_fs.fit(X_all_clean, y)
    mi_scores = mutual_info_classif(X_all_clean, y, random_state=42)

    fe_ranking = pd.DataFrame({
        "Feature": engineered_cols,
        "Feature_Type": [
            "Engineered (Domain)" if c not in feature_cols else "Baseline Biometric"
            for c in engineered_cols
        ],
        "RF_Gini_Importance_Pct": (rf_fs.feature_importances_ * 100).round(2),
        "Mutual_Information": np.round(mi_scores, 4),
        "Pearson_Corr_with_Outcome": [round(X_all_clean[c].corr(y), 4) for c in engineered_cols],
    }).sort_values("RF_Gini_Importance_Pct", ascending=False)

    # 2. PCA on Standardized Baseline Features + Biplot Loadings
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(df_clean[feature_cols])
    scaler_rob = RobustScaler()
    X_rob = scaler_rob.fit_transform(df_clean[feature_cols])

    pca = PCA()
    X_pca = pca.fit_transform(X_std)

    pca_df = pd.DataFrame(
        X_pca[:, :3],
        columns=["PC1 (Metabolic/Adiposity)", "PC2 (Chronological/BP)", "PC3 (Glycemic/Genetic)"],
    )
    pca_df["Outcome_Label"] = df_clean["Outcome_Label"].values
    pca_df["Glucose"] = df_clean["Glucose"].values
    pca_df["BMI"] = df_clean["BMI"].values
    pca_df["HOMA_IR"] = df_clean["HOMA_IR"].values

    loadings = pd.DataFrame(
        pca.components_.T[:, :5],
        index=feature_cols,
        columns=[f"PC{i}" for i in range(1, 6)],
    ).round(3)

    # 3. Leak-Free 5-Model Benchmark using Stratified 5-Fold Cross-Validation
    def _make_models():
        lr_clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
        rf_clf = RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42
        )
        et_clf = ExtraTreesClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42
        )
        gb_clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.85, random_state=42
        )
        ens_clf = VotingClassifier(
            estimators=[("lr", lr_clf), ("rf", rf_clf), ("et", et_clf), ("gb", gb_clf)],
            voting="soft",
            weights=[1.2, 1.5, 1.3, 1.0],
        )
        return {
            "Champion Soft-Voting Ensemble": Pipeline([
                ("fe", ClinicalFeaturePipeline(winsorize=True)),
                ("scaler", RobustScaler()),
                ("clf", ens_clf),
            ]),
            "Extra Trees (Balanced)": Pipeline([
                ("fe", ClinicalFeaturePipeline(winsorize=True)),
                ("scaler", RobustScaler()),
                ("clf", et_clf),
            ]),
            "Random Forest (Balanced)": Pipeline([
                ("fe", ClinicalFeaturePipeline(winsorize=True)),
                ("scaler", RobustScaler()),
                ("clf", rf_clf),
            ]),
            "Logistic Regression (L2)": Pipeline([
                ("fe", ClinicalFeaturePipeline(winsorize=True)),
                ("scaler", RobustScaler()),
                ("clf", lr_clf),
            ]),
            "Gradient Boosting": Pipeline([
                ("fe", ClinicalFeaturePipeline(winsorize=True)),
                ("scaler", RobustScaler()),
                ("clf", gb_clf),
            ]),
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_dict = _make_models()
    oof_probs = {}
    roc_curves = {}
    pr_curves = {}
    benchmark_rows = []

    for name, pipe in models_dict.items():
        probs = cross_val_predict(pipe, X_raw_base, y, cv=skf, method="predict_proba")[:, 1]
        oof_probs[name] = probs
        preds_default = (probs >= 0.42).astype(int)

        auc_val = roc_auc_score(y, probs)
        ap_val = average_precision_score(y, probs)
        brier_val = brier_score_loss(y, probs)
        acc_val = accuracy_score(y, preds_default)
        sens_val = recall_score(y, preds_default)
        spec_val = recall_score(y, preds_default, pos_label=0)
        prec_val = precision_score(y, preds_default)
        f1_val = f1_score(y, preds_default)

        fpr, tpr, _ = roc_curve(y, probs)
        prec_c, rec_c, _ = precision_recall_curve(y, probs)
        roc_curves[name] = (fpr, tpr, auc_val)
        pr_curves[name] = (rec_c, prec_c, ap_val)

        benchmark_rows.append({
            "Model": name,
            "OOF_ROC_AUC": round(auc_val, 4),
            "OOF_PR_AUC": round(ap_val, 4),
            "Sensitivity_Recall_Pct": round(sens_val * 100, 2),
            "Specificity_Pct": round(spec_val * 100, 2),
            "Precision_PPV_Pct": round(prec_val * 100, 2),
            "F1_Score": round(f1_val, 4),
            "Accuracy_Pct": round(acc_val * 100, 2),
            "Brier_Calibration_Loss": round(brier_val, 4),
        })

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values("OOF_ROC_AUC", ascending=False)

    # Fit Champion Pipeline & all models on full dataset for Live Inference
    fitted_models = {}
    for name, pipe in models_dict.items():
        pipe.fit(X_raw_base, y)
        fitted_models[name] = pipe

    champion_pipe = fitted_models["Champion Soft-Voting Ensemble"]

    # 4. Precompute Exact Game-Theoretic Shapley Coalition Tables (2^8 = 256 coalitions)
    n_feat = len(feature_cols)
    bits = np.array(list(itertools.product([0, 1], repeat=n_feat)))  # (256, 8)
    fact = [math.factorial(i) for i in range(n_feat + 1)]
    powers = 2 ** np.arange(n_feat - 1, -1, -1)

    shap_lookup = []
    for i in range(n_feat):
        idx_0 = np.where(bits[:, i] == 0)[0]
        bits_1 = bits[idx_0].copy()
        bits_1[:, i] = 1
        idx_1 = bits_1.dot(powers)
        s_sizes = bits[idx_0].sum(axis=1)
        weights = np.array([fact[s] * fact[n_feat - s - 1] / fact[n_feat] for s in s_sizes])
        shap_lookup.append((idx_0, idx_1, weights))

    baseline_ref = df_clean[feature_cols].median().values  # Population median reference vector

    return {
        "fe_ranking": fe_ranking,
        "pca": pca,
        "pca_df": pca_df,
        "loadings": loadings,
        "X_std": pd.DataFrame(X_std, columns=feature_cols),
        "X_rob": pd.DataFrame(X_rob, columns=feature_cols),
        "benchmark_df": benchmark_df,
        "oof_probs": oof_probs,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "y_true": y.values,
        "fitted_models": fitted_models,
        "champion_pipe": champion_pipe,
        "shap_bits": bits,
        "shap_lookup": shap_lookup,
        "baseline_ref": baseline_ref,
    }


def compute_exact_shapley_Attribution(
    pipe,
    patient_vec: np.ndarray,
    baseline_vec: np.ndarray,
    feature_cols: list,
    bits: np.ndarray,
    shap_lookup: list,
) -> tuple[float, float, pd.DataFrame]:
    """
    Computes Exact Game-Theoretic Shapley Values (phi_i) across all 2^8 = 256 coalitions
    in a single vectorized batch call (~12 ms).
    Guarantees Exact Additivity: base_prob + sum(phi_i) == patient_prob.
    """
    # Build 256 coalition rows: where bit == 1 use patient value, where bit == 0 use baseline median
    coalition_matrix = np.where(bits == 1, patient_vec, baseline_vec)
    coalition_df = pd.DataFrame(coalition_matrix, columns=feature_cols)
    probs = pipe.predict_proba(coalition_df)[:, 1] * 100.0  # in percentage points (0..100)

    base_prob = float(probs[0])       # bits = (0,0,0,0,0,0,0,0)
    patient_prob = float(probs[-1])   # bits = (1,1,1,1,1,1,1,1)

    phis = []
    for i, (idx_0, idx_1, weights) in enumerate(shap_lookup):
        marginal_diffs = probs[idx_1] - probs[idx_0]
        phi_val = float(np.dot(weights, marginal_diffs))
        phis.append(phi_val)

    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "Patient_Value": np.round(patient_vec, 2),
        "Population_Median": np.round(baseline_vec, 2),
        "Shapley_Delta_Pct": np.round(phis, 2),
        "Abs_Impact": np.abs(phis),
    }).sort_values("Abs_Impact", ascending=True)

    return base_prob, patient_prob, shap_df


# -----------------------------------------------------------------------------
# 5. INITIALIZE DATA & SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
s1_data = load_and_process_stage1("diabetes2.csv")
s2_data = load_and_process_stage2("diabetic_data2.csv")

if s1_data is None or s2_data is None:
    st.warning(
        "📂 **Dataset Setup Required:** `diabetes2.csv` and/or `diabetic_data2.csv` were not found in the working directory. "
        "Upload them below to save them directly into your workspace."
    )
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        up_s1 = st.file_uploader("Upload `diabetes2.csv` (Stage 1 Pima Dataset)", type=["csv"])
        if up_s1 is not None:
            with open("diabetes2.csv", "wb") as f:
                f.write(up_s1.getbuffer())
            st.success("Saved `diabetes2.csv`!")
    with up_col2:
        up_s2 = st.file_uploader("Upload `diabetic_data2.csv` (Stage 2 Hospital Dataset)", type=["csv"])
        if up_s2 is not None:
            with open("diabetic_data2.csv", "wb") as f:
                f.write(up_s2.getbuffer())
            st.success("Saved `diabetic_data2.csv`!")

    if os.path.exists("diabetes2.csv") and os.path.exists("diabetic_data2.csv"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

ml_bundle = compute_models_pca_and_xai(
    s1_data["df_raw"], s1_data["df_clean"], s1_data["feature_cols"]
)

with st.sidebar:
    st.markdown("## 🩺 NIErnaya 2026 Suite")
    st.caption("**AI/ML Diabetes Risk & Clinical Escalation**")
    st.markdown("---")
    selected_module = st.radio(
        "📌 Navigate Analytics & ML Modules:",
        [
            "1. Executive Architecture & Cohort Overview",
            "2. Data Cleaning, MICE & Winsorization",
            "3. Deep EDA & Hypothesis Testing",
            "4. India-Focused Feature Engineering",
            "5. PCA & Biplot Dimensionality Reduction",
            "6. AI/ML Model Lab & CV Calibration",
            "7. Stage 2: Hospital Screening Gap (101.7k)",
            "8. Live AI/ML Risk, Shapley XAI & Simulator",
        ],
    )
    st.markdown("---")
    st.markdown("### 🏆 Champion Model Snapshot")
    champ_row = ml_bundle["benchmark_df"].iloc[0]
    st.metric("5-Fold OOF ROC-AUC", f"{champ_row['OOF_ROC_AUC']:.4f}", "Leak-Free Pipeline")
    st.metric("Screening Sensitivity", f"{champ_row['Sensitivity_Recall_Pct']:.1f}%", "Threshold = 0.42")

    st.markdown("---")
    st.markdown("### 📥 Export Cleaned Artifacts")
    csv_stage1 = s1_data["df_clean"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Stage 1 Cleaned + Engineered CSV",
        data=csv_stage1,
        file_name="stage1_pima_cleaned_engineered.csv",
        mime="text/csv",
        width="stretch",
    )
    csv_bench = ml_bundle["benchmark_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ 5-Fold CV Model Benchmark CSV",
        data=csv_bench,
        file_name="stage1_model_benchmark_5fold_cv.csv",
        mime="text/csv",
        width="stretch",
    )


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🩺 AI/ML Preliminary Diabetes Risk Assessment & Clinical Escalation Suite</h1>
        <p><b>NIErnaya 2026 Complete Solution:</b> Leak-Free MICE Imputation, Winsorization, Indian Consensus (ICMR) Feature Engineering,
        5-Fold Cross-Validated Ensemble Modeling, Exact Game-Theoretic Shapley XAI & 101,766-Encounter Hospital Screening Analytics</p>
        <div class="hero-badges">
            <span class="hero-badge">✅ Zero Data Leakage Pipeline</span>
            <span class="hero-badge">🇮🇳 ICMR Indian BMI Phenotype</span>
            <span class="hero-badge">🧠 Exact 2⁸ Shapley Attribution</span>
            <span class="hero-badge">🏠 Non-Invasive & Lab Modes</span>
            <span class="hero-badge">🏥 101,766 Hospital Encounters</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODULE 1: EXECUTIVE ARCHITECTURE & COHORT OVERVIEW
# =============================================================================
if selected_module == "1. Executive Architecture & Cohort Overview":
    total_records = len(s1_data["df_raw"]) + s2_data["total_raw_rows"]
    champ_auc = ml_bundle["benchmark_df"].iloc[0]["OOF_ROC_AUC"]
    unscreened_pct = s2_data["all_encounters"]["unscreened_pct"]
    a1c_df = s2_data["all_encounters"]["a1c_summary"]
    unscreened_readmit = float(
        a1c_df.loc[a1c_df["HbA1c_Screening_Status"] == "Not Screened (Barrier)", "Acute_30d_Readmit_Pct"].values[0]
    )
    normal_readmit = float(
        a1c_df.loc[a1c_df["HbA1c_Screening_Status"] == "Screened: Normal (<7%)", "Acute_30d_Readmit_Pct"].values[0]
    )
    readmit_gap = round(unscreened_readmit - normal_readmit, 2)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Total Clinical Records</div>
                <div class="kpi-value">{total_records:,}</div>
                <div class="kpi-sub">Dual-Stage Screening + Hospital Suite</div>
            </div>
            <div class="kpi-card accent-blue">
                <div class="kpi-title">Stage 1 Screening Cohort</div>
                <div class="kpi-value">{len(s1_data['df_raw']):,} Patients</div>
                <div class="kpi-sub">8 Baseline + 6 Engineered Biomarkers</div>
            </div>
            <div class="kpi-card accent-violet">
                <div class="kpi-title">Champion Ensemble ROC-AUC</div>
                <div class="kpi-value">{champ_auc:.4f}</div>
                <div class="kpi-sub">5-Fold Stratified Out-of-Fold CV</div>
            </div>
            <div class="kpi-card accent-amber">
                <div class="kpi-title">Stage 2 Index Patients</div>
                <div class="kpi-value">{s2_data['dedup_rows']:,} Unique</div>
                <div class="kpi-sub">From {s2_data['valid_rows']:,} Valid Encounters</div>
            </div>
            <div class="kpi-card accent-rose">
                <div class="kpi-title">Hospital HbA1c Screening Gap</div>
                <div class="kpi-value">{unscreened_pct:.2f}%</div>
                <div class="kpi-sub">+{readmit_gap:.2f}% Higher &lt;30d Readmission</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box">
        <b>🎯 End-to-End Problem Statement Alignment:</b> Our system connects <b>Stage 1 Community Preliminary Risk Screening (<code>diabetes2.csv</code>)</b>
        with <b>Stage 2 Clinical Escalation & Screening Barrier Evidence (<code>diabetic_data2.csv</code>)</b>.
        Stage 1 uses a leak-free <b>Soft-Voting Ensemble + Exact Shapley Explainability</b> tailored to <b>Indian Consensus (ICMR) BMI thresholds</b> to identify
        at-risk individuals (even without invasive lab tests via MICE imputation), while Stage 2 proves across <b>101,766 hospital visits</b> why overcoming
        glycemic screening barriers (`HbA1c`) and timely physician consultation prevent acute 30-day hospital readmissions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🏗️ Three-Pillar Clinical AI Architecture")
    arch1, arch2, arch3 = st.columns(3)
    with arch1:
        st.markdown(
            """
            <div class="arch-card">
                <h4>1️⃣ Stage 1: Data Hygiene & Indian Phenotype FE</h4>
                <p>• Unmasks <b>652 biologically impossible zeros</b> across Insulin, SkinThickness, BP, BMI, and Glucose.<br>
                • Applies <b>Leak-Free MICE Imputation</b> & <b>1st–99th Percentile Winsorization</b> inside each CV fold.<br>
                • Engineers 6 clinical indices including <code>HOMA_IR</code> and <code>ICMR_BMI_Category</code> (capturing the <i>Thin-Fat Indian</i> phenotype at BMI 23–24.9).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch2:
        st.markdown(
            """
            <div class="arch-card" style="border-top-color: #14b8a6;">
                <h4>2️⃣ Explainable AI/ML Screening Engine</h4>
                <p>• Benchmarks 5 classifiers via <b>Stratified 5-Fold Cross-Validation</b>.<br>
                • Supports both <b>Non-Invasive Home Screening</b> (auto-imputing unknown lab values) and <b>Full Lab Screening</b>.<br>
                • Computes <b>Exact 2⁸ Coalition Shapley Values</b> in real time + interactive <b>What-If Lifestyle Counterfactuals</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch3:
        st.markdown(
            """
            <div class="arch-card" style="border-top-color: #f43f5e;">
                <h4>3️⃣ Stage 2: Clinical Escalation & Readmission Proof</h4>
                <p>• Audits <b>101,766 hospital encounters</b> (deduplicated to <b>69,987 unique index patients</b>).<br>
                • Maps <b>717 ICD-9 codes</b> into 9 clinical disease chapters.<br>
                • Quantifies how unscreened <code>HbA1c</code> patients face significantly higher acute <code>&lt;30-day</code> hospital readmissions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔹 Stage 1 Cohort: Primary Risk Screening (`diabetes2.csv`)")
        outcome_counts = s1_data["df_clean"]["Outcome_Label"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome_Label", "Count"]
        fig_s1_pie = px.pie(
            outcome_counts,
            names="Outcome_Label",
            values="Count",
            hole=0.52,
            color="Outcome_Label",
            color_discrete_map={"Non-Diabetic (0)": "#10b981", "Diabetic / High-Risk (1)": "#ef4444"},
            title="Stage 1 Target Stratification (n = 768 Patients)",
            height=300,
        )
        fig_s1_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_s1_pie, width="stretch")
        st.dataframe(s1_data["df_clean"].head(6), width="stretch", hide_index=True)

    with col_b:
        st.subheader("🔸 Stage 2 Cohort: Hospital Readmission (`diabetic_data2.csv`)")
        readm_counts = pd.DataFrame({
            "Readmission Status": ["No Readmission (NO)", "Delayed Readmission (>30d)", "Acute Readmission (<30d)"],
            "Encounters": [54864, 35545, 11357],
        })
        fig_s2_pie = px.pie(
            readm_counts,
            names="Readmission Status",
            values="Encounters",
            hole=0.52,
            color="Readmission Status",
            color_discrete_map={
                "No Readmission (NO)": "#10b981",
                "Delayed Readmission (>30d)": "#f59e0b",
                "Acute Readmission (<30d)": "#ef4444",
            },
            title="Stage 2 Readmission Outcomes (n = 101,766 Encounters)",
            height=300,
        )
        fig_s2_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_s2_pie, width="stretch")
        st.dataframe(s2_data["df_valid_sample"].head(6), width="stretch", hide_index=True)


# =============================================================================
# MODULE 2: DATA CLEANING, MICE IMPUTATION & WINSORIZATION PIPELINE
# =============================================================================
elif selected_module == "2. Data Cleaning, MICE & Winsorization":
    st.subheader("🧹 Rigorous Clinical Data Hygiene: Disguised Zeros, Leak-Free MICE, Winsorization & Deduplication")

    tab_m1, tab_m2, tab_m3 = st.tabs([
        "1️⃣ Biological Zeros & MICE Imputation (Stage 1)",
        "2️⃣ IQR Outliers, 1st/99th Winsorization & Scaling (Stage 1)",
        "3️⃣ '?' Tokens, Attrition Funnel & Patient Deduplication (Stage 2)",
    ])

    with tab_m1:
        st.markdown(
            """
            <div class="insight-box">
            <b>💡 Unmasking Disguised Missingness:</b> Standard <code>df.isna().sum()</code> falsely reports <code>0</code> missing values in <code>diabetes2.csv</code>.
            Clinical range validation exposes <b>652 biologically impossible zeros</b> across <b>Serum Insulin (374 zeros | 48.7%)</b>,
            <b>Skin Thickness (227 zeros | 29.6%)</b>, <b>Blood Pressure (35 zeros | 4.6%)</b>, <b>BMI (11 zeros | 1.4%)</b>, and <b>Glucose (5 zeros | 0.7%)</b>.
            Replacing these with <code>NaN</code> and applying <b>Multivariate MICE (IterativeImputer)</b> boosts Insulin's correlation with Diabetes Outcome by <b>+135.8%</b>!
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.15, 1])
        with col1:
            st.markdown("##### 📊 Disguised Zero Audit & Correlation Signal Recovery")
            st.dataframe(s1_data["zero_audit"], width="stretch", hide_index=True)

        with col2:
            fig_boost = go.Figure()
            fig_boost.add_trace(go.Bar(
                y=s1_data["zero_audit"]["Feature"],
                x=s1_data["zero_audit"]["Raw_Corr_with_Outcome"],
                name="Raw Data (Corrupted by 0s)",
                orientation="h",
                marker_color="#f59e0b",
                text=s1_data["zero_audit"]["Raw_Corr_with_Outcome"],
                textposition="auto",
            ))
            fig_boost.add_trace(go.Bar(
                y=s1_data["zero_audit"]["Feature"],
                x=s1_data["zero_audit"]["Cleaned_MICE_Corr_with_Outcome"],
                name="Post-MICE + Winsorized",
                orientation="h",
                marker_color="#10b981",
                text=s1_data["zero_audit"]["Cleaned_MICE_Corr_with_Outcome"],
                textposition="auto",
            ))
            fig_boost.update_layout(
                title="Pearson Correlation (r) with Outcome: Raw vs. Cleaned",
                barmode="group",
                xaxis_title="Pearson Correlation (r)",
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_boost, width="stretch")

        st.markdown("##### 🔬 Distribution Fidelity Comparison: Raw (With 0s) vs. Mean vs. KNN vs. MICE Imputation")
        imp_col = st.selectbox(
            "Select Biomarker to Inspect Imputation Fidelity:",
            ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"],
            index=0,
        )

        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram(
            x=s1_data["df_raw"][imp_col], name="1. Raw Data (Distorted by 0s)",
            opacity=0.45, marker_color="#ef4444", histnorm="probability density"
        ))
        fig_kde.add_trace(go.Histogram(
            x=s1_data["X_mean"][imp_col], name="2. Simple Mean Imputation (Artificial Variance Collapse)",
            opacity=0.45, marker_color="#f59e0b", histnorm="probability density"
        ))
        fig_kde.add_trace(go.Histogram(
            x=s1_data["X_knn"][imp_col], name="3. KNN Imputer (k=7)",
            opacity=0.55, marker_color="#3b82f6", histnorm="probability density"
        ))
        fig_kde.add_trace(go.Histogram(
            x=s1_data["X_winsor"][imp_col], name="4. MICE + Winsorized (Preserves Natural Variance)",
            opacity=0.65, marker_color="#10b981", histnorm="probability density"
        ))
        fig_kde.update_layout(
            barmode="overlay",
            title=f"Probability Density Comparison Across Imputation Strategies ({imp_col})",
            xaxis_title=imp_col,
            yaxis_title="Density",
            height=380,
        )
        st.plotly_chart(fig_kde, width="stretch")

    with tab_m2:
        st.markdown(
            """
            <div class="insight-box">
            <b>⚠️ Clinical Outlier & Winsorization Policy:</b> Extreme glycemic or adiposity measurements represent critically ill patients,
            so dropping rows causes severe clinical selection bias. Instead, we apply <b>1st–99th Percentile Winsorization</b> to clamp extreme tails
            (reducing skewness in <code>Insulin</code> and <code>DiabetesPedigreeFunction</code>) followed by <b>RobustScaler (median/IQR scaling)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_out1, c_out2 = st.columns([1.18, 1])
        with c_out1:
            st.markdown("##### 📋 IQR Outlier & Winsorization Skewness Audit")
            st.dataframe(s1_data["outlier_audit"], width="stretch", hide_index=True)
        with c_out2:
            scale_choice = st.radio(
                "Compare Feature Distributions Across Transformations:",
                [
                    "Pre-Winsorization (MICE Raw)",
                    "Post-Winsorization (1st-99th Clamped)",
                    "StandardScaler (Z-Score)",
                    "RobustScaler (Median/IQR Resistant)",
                ],
                horizontal=True,
            )
            if scale_choice == "Pre-Winsorization (MICE Raw)":
                plot_df = s1_data["X_mice"]
            elif scale_choice == "Post-Winsorization (1st-99th Clamped)":
                plot_df = s1_data["X_winsor"]
            elif scale_choice == "StandardScaler (Z-Score)":
                plot_df = ml_bundle["X_std"]
            else:
                plot_df = ml_bundle["X_rob"]

            fig_box = px.box(
                plot_df.melt(var_name="Feature", value_name="Value"),
                x="Value",
                y="Feature",
                color="Feature",
                title=f"Box & Whisker Outlier Audit ({scale_choice})",
                height=390,
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, width="stretch")

    with tab_m3:
        st.markdown(
            """
            <div class="insight-box">
            <b>🏥 Stage 2 Longitudinal Deduplication Discovery:</b> Out of <b>101,766 raw hospital encounters</b>, filtering 2,423 expired/hospice visits
            and 3 invalid gender records yields <b>99,340 valid encounters</b>. However, because repeat hospital "super-utilizers" visit multiple times,
            deduplicating by <code>patient_nbr</code> (retaining each patient's index encounter) yields <b>69,987 unique patients</b> and reveals that repeat
            visitors inflate raw 30-day readmission rates from <b>8.98% (Index Patients)</b> to <b>11.39% (All Encounters)</b>!
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_s2a, col_s2b = st.columns([1.05, 1])
        with col_s2a:
            st.markdown("##### 📋 Stage 2 Missing & `?` Token Audit (`diabetic_data2.csv`)")
            st.dataframe(s2_data["miss_df"], width="stretch", hide_index=True)
            st.caption(f"Dropped Zero-Variance Medication Columns: `{', '.join(s2_data['zero_var_cols'])}`")
        with col_s2b:
            fig_funnel = go.Figure(go.Funnel(
                y=[
                    "1. Raw Hospital Encounters",
                    "2. Valid Non-Expired Encounters",
                    "3. Unique Deduplicated Patients (Index Visit)",
                    "4. Patients Receiving HbA1c Screening",
                ],
                x=[
                    s2_data["total_raw_rows"],
                    s2_data["valid_rows"],
                    s2_data["dedup_rows"],
                    int(s2_data["dedup_rows"] * (1 - s2_data["dedup_patients"]["unscreened_pct"] / 100.0)),
                ],
                textinfo="value+percent initial",
                marker=dict(color=["#3b82f6", "#0ea5e9", "#14b8a6", "#f59e0b"]),
            ))
            fig_funnel.update_layout(
                title="Stage 2 Cohort Filtering & Longitudinal Deduplication Funnel",
                height=370,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_funnel, width="stretch")


# =============================================================================
# MODULE 3: DEEP EDA & STATISTICAL HYPOTHESIS TESTING
# =============================================================================
elif selected_module == "3. Deep EDA & Hypothesis Testing":
    st.subheader("📊 Multi-Dimensional Exploratory Data Analysis & Non-Parametric Hypothesis Testing")

    df_c = s1_data["df_clean"]

    st.markdown("##### 🧪 Statistical Significance Audit (Mann-Whitney U Test, Welch's t-Test & Cohen's d Effect Size)")
    st.dataframe(s1_data["stat_tests_df"], width="stretch", hide_index=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        feat_dist = st.selectbox(
            "Select Biomarker for Stratified Violin + Box + Jitter Distribution:",
            [
                "Glucose", "HOMA_IR", "BMI", "Metabolic_Risk_Score", "Insulin",
                "Age_Glycemic_Interaction", "Visceral_Adiposity_Proxy", "Age",
                "BloodPressure", "SkinThickness", "DiabetesPedigreeFunction",
            ],
            index=0,
        )
        fig_violin = px.violin(
            df_c,
            x="Outcome_Label",
            y=feat_dist,
            color="Outcome_Label",
            box=True,
            points="all",
            color_discrete_map={"Non-Diabetic (0)": "#10b981", "Diabetic / High-Risk (1)": "#ef4444"},
            title=f"Distribution & Median/IQR Stratification of {feat_dist}",
            height=430,
        )
        st.plotly_chart(fig_violin, width="stretch")

    with col_e2:
        corr_method = st.radio(
            "Select Correlation Metric:",
            ["Pearson (Linear r)", "Spearman (Rank Monotonic ρ)"],
            horizontal=True,
        )
        corr_cols = [
            "Glucose", "HOMA_IR", "Metabolic_Risk_Score", "Insulin", "BMI",
            "Visceral_Adiposity_Proxy", "Age_Glycemic_Interaction",
            "SkinThickness", "Age", "Pregnancies", "BloodPressure",
            "DiabetesPedigreeFunction", "Outcome",
        ]
        method_key = "pearson" if "Pearson" in corr_method else "spearman"
        corr_matrix = df_c[corr_cols].corr(method=method_key).round(2)
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Multivariate {corr_method} Matrix (Cleaned + Engineered)",
            height=430,
        )
        st.plotly_chart(fig_heat, width="stretch")

    st.markdown("##### 🔍 Bivariate Glycemic–Adiposity Risk Quadrant (`Glucose × BMI × HOMA-IR`)")
    fig_scatter = px.scatter(
        df_c,
        x="Glucose",
        y="BMI",
        color="Outcome_Label",
        size="HOMA_IR",
        hover_data=["Age", "Insulin", "BloodPressure", "HOMA_IR", "ICMR_BMI_Category"],
        color_discrete_map={"Non-Diabetic (0)": "#10b981", "Diabetic / High-Risk (1)": "#ef4444"},
        opacity=0.78,
        title="Glycemic–Adiposity Risk Quadrant (Bubble Size = HOMA-IR Insulin Resistance)",
        height=470,
    )
    fig_scatter.add_vline(x=100, line_dash="dot", line_color="#f59e0b", annotation_text="Pre-Diabetes (100 mg/dL)")
    fig_scatter.add_vline(x=126, line_dash="dash", line_color="#ef4444", annotation_text="Diabetic Cutoff (126 mg/dL)")
    fig_scatter.add_hline(y=23.0, line_dash="dot", line_color="#f59e0b", annotation_text="ICMR Indian Overweight (BMI 23)")
    fig_scatter.add_hline(y=25.0, line_dash="dash", line_color="#8b5cf6", annotation_text="ICMR Indian Obese / WHO Overweight (BMI 25)")
    st.plotly_chart(fig_scatter, width="stretch")


# =============================================================================
# MODULE 4: INDIA-FOCUSED FEATURE ENGINEERING & SELECTION
# =============================================================================
elif selected_module == "4. India-Focused Feature Engineering":
    st.subheader("🇮🇳 India-Focused Clinical Feature Engineering & Multi-Metric Feature Selection")

    df_c = s1_data["df_clean"]

    st.markdown(
        """
        <div class="insight-box">
        <b>🏆 Key Clinical Discovery — The "Thin-Fat Indian" Phenotype:</b> Standard Western WHO guidelines classify BMI <code>&lt;25.0 kg/m²</code> as "Normal Weight".
        However, South Asian / Indian populations exhibit higher visceral adiposity and insulin resistance at lower BMI. Applying the
        <b>ICMR / Indian Consensus BMI Cutoffs</b> (Normal <code>&lt;23.0</code>, Overweight <code>23.0–24.9</code>, Obese <code>&ge;25.0</code>)
        uncovers a <b>5.3× surge in diabetes prevalence (from 2.0% to 10.7%)</b> inside the <code>BMI 23.0–24.9</code> zone that Western standards completely miss!
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        icmr_grp = (
            df_c.groupby("ICMR_BMI_Category", observed=False)
            .agg(
                Patients=("Outcome", "count"),
                Diabetes_Rate_Pct=("Outcome", lambda x: round(x.mean() * 100, 2)),
                Avg_HOMA_IR=("HOMA_IR", lambda x: round(x.mean(), 2)),
            )
            .reset_index()
        )
        fig_icmr = px.bar(
            icmr_grp,
            x="ICMR_BMI_Category",
            y="Diabetes_Rate_Pct",
            color="ICMR_BMI_Category",
            text="Diabetes_Rate_Pct",
            hover_data=["Patients", "Avg_HOMA_IR"],
            color_discrete_sequence=["#10b981", "#f59e0b", "#f97316", "#ef4444"],
            title="Diabetes Prevalence (%) Across ICMR Indian Consensus vs. WHO BMI Tiers",
            height=420,
        )
        fig_icmr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_icmr.update_layout(showlegend=False, yaxis_title="Diabetes Prevalence (%)")
        st.plotly_chart(fig_icmr, width="stretch")

    with col_f2:
        fe_rk = ml_bundle["fe_ranking"].sort_values("RF_Gini_Importance_Pct", ascending=True)
        fig_fe = px.bar(
            fe_rk,
            x="RF_Gini_Importance_Pct",
            y="Feature",
            color="Feature_Type",
            orientation="h",
            text="RF_Gini_Importance_Pct",
            color_discrete_map={"Engineered (Domain)": "#14b8a6", "Baseline Biometric": "#3b82f6"},
            title="Feature Importance Leaderboard (6 Engineered vs. 8 Baseline Features)",
            height=420,
        )
        fig_fe.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_fe, width="stretch")

    st.markdown("##### 📈 Diabetes Risk Progression Across Engineered `Metabolic_Risk_Score` (0 to 4 Syndrome Factors)")
    met_grp = (
        df_c.groupby("Metabolic_Risk_Score", observed=False)
        .agg(
            Patients=("Outcome", "count"),
            Diabetes_Prevalence_Pct=("Outcome", lambda x: round(x.mean() * 100, 2)),
            Median_HOMA_IR=("HOMA_IR", "median"),
        )
        .reset_index()
    )
    mc1, mc2 = st.columns([1.2, 1])
    with mc1:
        fig_met = px.bar(
            met_grp,
            x="Metabolic_Risk_Score",
            y="Diabetes_Prevalence_Pct",
            text="Diabetes_Prevalence_Pct",
            color="Diabetes_Prevalence_Pct",
            color_continuous_scale="Tealrose",
            title="Monotonic Surge in Diabetes Rate by Composite Metabolic Risk Score (0 to 4)",
            height=340,
        )
        fig_met.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_met, width="stretch")
    with mc2:
        st.markdown("##### 📑 Tri-Metric Feature Selection Leaderboard")
        st.dataframe(ml_bundle["fe_ranking"], width="stretch", hide_index=True, height=310)


# =============================================================================
# MODULE 5: PCA & BIPLOT DIMENSIONALITY REDUCTION
# =============================================================================
elif selected_module == "5. PCA & Biplot Dimensionality Reduction":
    st.subheader("🧭 Principal Component Analysis (PCA), Eigenvector Biplot & 3D Manifold Projection")

    pca = ml_bundle["pca"]
    pca_df = ml_bundle["pca_df"]
    loadings = ml_bundle["loadings"]

    ev_pct = np.round(pca.explained_variance_ratio_ * 100, 2)
    cum_pct = np.round(np.cumsum(ev_pct), 2)

    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    p_c1.metric("PC1 Variance Explained", f"{ev_pct[0]:.2f}%", "Metabolic & Adiposity Axis")
    p_c2.metric("PC2 Variance Explained", f"{ev_pct[1]:.2f}%", "Chronological & BP Axis")
    p_c3.metric("PC1 + PC2 + PC3", f"{cum_pct[2]:.2f}%", "3D Visual Manifold")
    p_c4.metric("Top 5 PCs Cumulative", f"{cum_pct[4]:.2f}%", "37.5% Feature Compression")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pcs_labels = [f"PC{i}" for i in range(1, 9)]
        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_trace(
            go.Bar(x=pcs_labels, y=ev_pct, name="Individual Variance (%)", marker_color="#3b82f6"),
            secondary_y=False,
        )
        fig_scree.add_trace(
            go.Scatter(
                x=pcs_labels, y=cum_pct, name="Cumulative Variance (%)",
                mode="lines+markers", line=dict(color="#14b8a6", width=3),
            ),
            secondary_y=True,
        )
        fig_scree.add_hline(
            y=float(cum_pct[4]), line_dash="dot", line_color="#f59e0b",
            secondary_y=True, annotation_text=f"5 PCs = {cum_pct[4]:.1f}% Retained",
        )
        fig_scree.update_layout(title="PCA Scree Plot & Cumulative Explained Variance", height=400)
        st.plotly_chart(fig_scree, width="stretch")

    with col_p2:
        fig_load = px.imshow(
            loadings,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-0.8,
            zmax=0.8,
            title="PCA Eigenvector Loadings Matrix (PC1 to PC5)",
            height=400,
        )
        st.plotly_chart(fig_load, width="stretch")

    tab_pca2d, tab_pca3d = st.tabs([
        "📐 2D PCA Biplot with Eigenvector Loading Vectors",
        "🌐 Interactive 3D PCA Phenotype Cluster (PC1 × PC2 × PC3)",
    ])

    with tab_pca2d:
        fig_biplot = px.scatter(
            pca_df,
            x="PC1 (Metabolic/Adiposity)",
            y="PC2 (Chronological/BP)",
            color="Outcome_Label",
            color_discrete_map={"Non-Diabetic (0)": "#10b981", "Diabetic / High-Risk (1)": "#ef4444"},
            opacity=0.65,
            hover_data=["Glucose", "BMI", "HOMA_IR"],
            title="2D PCA Biplot: Patient Scores + Scaled Eigenvector Loading Arrows",
            height=500,
        )
        scale_arrow = 4.2
        for feat in s1_data["feature_cols"]:
            lx = float(loadings.loc[feat, "PC1"]) * scale_arrow
            ly = float(loadings.loc[feat, "PC2"]) * scale_arrow
            fig_biplot.add_annotation(
                ax=0, ay=0, axref="x", ayref="y",
                x=lx, y=ly, xref="x", yref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.2,
                arrowcolor="#fde047",
            )
            fig_biplot.add_annotation(
                x=lx * 1.12, y=ly * 1.12,
                text=f"<b>{feat}</b>",
                showarrow=False,
                font=dict(size=11, color="#f59e0b"),
            )
        st.plotly_chart(fig_biplot, width="stretch")

    with tab_pca3d:
        fig_3d = px.scatter_3d(
            pca_df,
            x="PC1 (Metabolic/Adiposity)",
            y="PC2 (Chronological/BP)",
            z="PC3 (Glycemic/Genetic)",
            color="Outcome_Label",
            size="HOMA_IR",
            hover_data=["Glucose", "BMI", "HOMA_IR"],
            color_discrete_map={"Non-Diabetic (0)": "#10b981", "Diabetic / High-Risk (1)": "#ef4444"},
            opacity=0.8,
            height=540,
            title=f"3D PCA Phenotype Manifold (Captures {cum_pct[2]:.2f}% Total Variance)",
        )
        st.plotly_chart(fig_3d, width="stretch")


# =============================================================================
# MODULE 6: LEAK-FREE AI/ML MODEL LAB, CROSS-VALIDATION & CALIBRATION
# =============================================================================
elif selected_module == "6. AI/ML Model Lab & CV Calibration":
    st.subheader("🤖 Leak-Free 5-Fold Cross-Validation Benchmark, ROC/PR Curves & Threshold Calibration")

    st.markdown(
        """
        <div class="insight-box">
        <b>🛡️ Zero Data Leakage Guarantee:</b> Every model below is wrapped in our custom <code>ClinicalFeaturePipeline</code>.
        During <b>5-Fold Stratified Cross-Validation</b>, biological-zero masking, <b>MICE IterativeImputer</b>, <b>1st–99th Percentile Winsorization</b>,
        <b>Domain Feature Engineering</b>, and <b>RobustScaler</b> are fitted <i>strictly inside each training fold</i> before predicting out-of-fold validation probabilities.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 🏆 Out-of-Fold Stratified 5-Fold Cross-Validation Leaderboard (Screening Threshold = 0.42)")
    st.dataframe(ml_bundle["benchmark_df"], width="stretch", hide_index=True)

    col_roc, col_pr = st.columns(2)
    palette = ["#14b8a6", "#3b82f6", "#f59e0b", "#8b5cf6", "#f43f5e"]

    with col_roc:
        fig_roc = go.Figure()
        for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(ml_bundle["roc_curves"].items()):
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{m_name} (AUC = {auc_val:.4f})",
                line=dict(width=3 if idx == 0 else 2, color=palette[idx % len(palette)]),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline (0.5000)",
            line=dict(dash="dash", color="#64748b"),
        ))
        fig_roc.update_layout(
            title="Out-of-Fold ROC Curves (5-Fold Stratified CV)",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=420,
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        )
        st.plotly_chart(fig_roc, width="stretch")

    with col_pr:
        fig_pr = go.Figure()
        for idx, (m_name, (rec_c, prec_c, ap_val)) in enumerate(ml_bundle["pr_curves"].items()):
            fig_pr.add_trace(go.Scatter(
                x=rec_c, y=prec_c, mode="lines",
                name=f"{m_name} (PR-AUC = {ap_val:.4f})",
                line=dict(width=3 if idx == 0 else 2, color=palette[idx % len(palette)]),
            ))
        fig_pr.update_layout(
            title="Out-of-Fold Precision-Recall Curves (5-Fold Stratified CV)",
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision (Positive Predictive Value)",
            height=420,
            legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        )
        st.plotly_chart(fig_pr, width="stretch")

    st.markdown("---")
    st.markdown("##### 🎚️ Interactive Clinical Screening Threshold Optimizer & Confusion Matrix")
    t_col1, t_col2 = st.columns([1, 1.35])

    with t_col1:
        sel_model_eval = st.selectbox(
            "Select Model for Threshold & Calibration Inspection:",
            list(ml_bundle["oof_probs"].keys()),
            index=0,
        )
        thresh_val = st.slider(
            "Clinical Decision Threshold (Lower = Higher Sensitivity for Preliminary Screening):",
            min_value=0.15,
            max_value=0.85,
            value=0.42,
            step=0.01,
        )
        probs_sel = ml_bundle["oof_probs"][sel_model_eval]
        y_true = ml_bundle["y_true"]
        preds_sel = (probs_sel >= thresh_val).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, preds_sel).ravel()
        sens = tp / (tp + fn) * 100.0
        spec = tn / (tn + fp) * 100.0
        ppv = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
        f1_cur = f1_score(y_true, preds_sel)

        m1, m2 = st.columns(2)
        m1.metric("Sensitivity (Recall)", f"{sens:.1f}%", f"{tp}/{tp+fn} Diabetics Caught")
        m2.metric("Specificity", f"{spec:.1f}%", f"{tn}/{tn+fp} Healthy Cleared")
        m3, m4 = st.columns(2)
        m3.metric("Negative Predictive Value (NPV)", f"{npv:.1f}%", "Rule-Out Confidence")
        m4.metric("F1-Score / PPV", f"{f1_cur:.3f}", f"PPV = {ppv:.1f}%")

    with t_col2:
        cm_sub1, cm_sub2 = st.columns(2)
        with cm_sub1:
            cm_matrix = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                cm_matrix,
                x=["Predicted Healthy (0)", "Predicted High-Risk (1)"],
                y=["Actual Healthy (0)", "Actual Diabetic (1)"],
                text_auto=True,
                color_continuous_scale="Teal",
                title=f"OOF Confusion Matrix (@ Threshold = {thresh_val:.2f})",
                height=340,
            )
            fig_cm.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_cm, width="stretch")

        with cm_sub2:
            prob_true, prob_pred = calibration_curve(y_true, probs_sel, n_bins=8)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers",
                name=sel_model_eval, line=dict(color="#14b8a6", width=3),
            ))
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Perfect Calibration", line=dict(dash="dash", color="#94a3b8"),
            ))
            fig_cal.update_layout(
                title="Probability Calibration Curve (Reliability)",
                xaxis_title="Mean Predicted Risk Probability",
                yaxis_title="Observed Empirical Diabetes Fraction",
                height=340,
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            )
            st.plotly_chart(fig_cal, width="stretch")


# =============================================================================
# MODULE 7: STAGE 2 - HOSPITAL SCREENING GAP & ESCALATION (101.7K ROWS)
# =============================================================================
elif selected_module == "7. Stage 2: Hospital Screening Gap (101.7k)":
    st.subheader("🏥 Stage 2: Clinical Screening Barriers, Longitudinal Deduplication & Readmission Escalation")

    cohort_mode = st.radio(
        "🔄 Select Stage 2 Cohort Analysis Mode:",
        [
            f"All Valid Hospital Encounters (n = {s2_data['valid_rows']:,} Visits)",
            f"Deduplicated Unique Index Patients (n = {s2_data['dedup_rows']:,} Patients — 1st Encounter Only)",
        ],
        horizontal=True,
    )
    active_s2 = s2_data["all_encounters"] if "All Valid" in cohort_mode else s2_data["dedup_patients"]

    s2_k1, s2_k2, s2_k3, s2_k4 = st.columns(4)
    s2_k1.metric("Unscreened HbA1c Rate", f"{active_s2['unscreened_pct']:.2f}%", "Clinical Screening Barrier")
    s2_k2.metric("Acute <30d Readmission", f"{active_s2['acute_30d_pct']:.2f}%", "High-Acuity Relapse")
    s2_k3.metric("Any Readmission (<30d or >30d)", f"{active_s2['any_readmit_pct']:.2f}%", "Longitudinal Burden")
    s2_k4.metric("Repeat Visitor Bias Delta", "+2.41%", "11.39% All vs 8.98% Index")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_a1c = px.bar(
            active_s2["a1c_summary"],
            x="HbA1c_Screening_Status",
            y="Acute_30d_Readmit_Pct",
            color="HbA1c_Screening_Status",
            text="Acute_30d_Readmit_Pct",
            hover_data=["Encounters", "Total_Readmit_Pct", "Avg_LOS_Days"],
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6", "#10b981"],
            title="Acute <30-Day Readmission Rate by HbA1c Screening Status",
            height=410,
        )
        fig_a1c.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_a1c.update_layout(showlegend=False, yaxis_title="<30 Day Readmission Rate (%)")
        st.plotly_chart(fig_a1c, width="stretch")

    with col_h2:
        diag_df = active_s2["diag_summary"][
            active_s2["diag_summary"]["Primary_Diagnosis_Chapter"] != "Unrecorded"
        ].sort_values("Acute_30d_Readmit_Pct", ascending=True)
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Bar(
            y=diag_df["Primary_Diagnosis_Chapter"],
            x=diag_df["Total_Readmit_Pct"],
            name="Total Readmission (<30d + >30d %)",
            orientation="h",
            marker_color="#3b82f6",
        ))
        fig_diag.add_trace(go.Bar(
            y=diag_df["Primary_Diagnosis_Chapter"],
            x=diag_df["Acute_30d_Readmit_Pct"],
            name="Acute <30 Day Readmission (%)",
            orientation="h",
            marker_color="#ef4444",
        ))
        fig_diag.update_layout(
            barmode="group",
            title="717 ICD-9 Codes Mapped to 9 Clinical Chapters vs. Readmission (%)",
            xaxis_title="Hospital Readmission Rate (%)",
            height=410,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_diag, width="stretch")

    st.markdown("##### 🌊 Multi-Stage Clinical Escalation Sankey Flow: Age Tier → HbA1c Screening → Readmission Outcome")
    sk_df = active_s2["sankey_grp"]
    age_nodes = list(sk_df["Age_Macro_Group"].unique())
    scr_nodes = list(sk_df["HbA1c_Screened_Binary"].unique())
    out_nodes = list(sk_df["Readmit_Outcome_Label"].unique())
    all_nodes = age_nodes + scr_nodes + out_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    sources, targets, values = [], [], []
    # Link 1: Age -> Screening
    l1 = sk_df.groupby(["Age_Macro_Group", "HbA1c_Screened_Binary"])["Count"].sum().reset_index()
    for _, row in l1.iterrows():
        sources.append(node_idx[row["Age_Macro_Group"]])
        targets.append(node_idx[row["HbA1c_Screened_Binary"]])
        values.append(int(row["Count"]))
    # Link 2: Screening -> Readmission Outcome
    l2 = sk_df.groupby(["HbA1c_Screened_Binary", "Readmit_Outcome_Label"])["Count"].sum().reset_index()
    for _, row in l2.iterrows():
        sources.append(node_idx[row["HbA1c_Screened_Binary"]])
        targets.append(node_idx[row["Readmit_Outcome_Label"]])
        values.append(int(row["Count"]))

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#f59e0b", "#14b8a6", "#ef4444", "#fb923c", "#10b981"][:len(all_nodes)],
        ),
        link=dict(source=sources, target=targets, value=values),
    )])
    fig_sankey.update_layout(
        title="Longitudinal Patient Pathway: Demographics → Screening Barrier → Readmission Escalation",
        height=420,
    )
    st.plotly_chart(fig_sankey, width="stretch")

    st.markdown("##### 📈 Age Trajectory: Hospital Volume, Polypharmacy & Acute 30-Day Readmission Risk")
    fig_age = make_subplots(specs=[[{"secondary_y": True}]])
    fig_age.add_trace(
        go.Bar(
            x=active_s2["age_readmit"]["age"],
            y=active_s2["age_readmit"]["Encounters"],
            name="Hospital Encounters",
            marker_color="#475569",
        ),
        secondary_y=False,
    )
    fig_age.add_trace(
        go.Scatter(
            x=active_s2["age_readmit"]["age"],
            y=active_s2["age_readmit"]["Acute_30d_Readmit_Pct"],
            name="Acute <30d Readmission (%)",
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
        ),
        secondary_y=True,
    )
    fig_age.update_layout(title="Hospitalization Volume & 30-Day Readmission Risk Across Age Decades", height=380)
    st.plotly_chart(fig_age, width="stretch")


# =============================================================================
# MODULE 8: LIVE AI/ML RISK CALCULATOR, EXACT SHAPLEY XAI & SIMULATOR
# =============================================================================
elif selected_module == "8. Live AI/ML Risk, Shapley XAI & Simulator":
    st.subheader("🩺 Interactive AI/ML Preliminary Diabetes Risk Assessment, Exact Shapley XAI & Lifestyle Simulator")

    # Quick Clinical Presets & Screening Mode
    top_c1, top_c2, top_c3 = st.columns([1.3, 1.3, 1])
    with top_c1:
        screening_mode = st.radio(
            "🔬 Screening Input Mode:",
            [
                "🏠 Non-Invasive Home Mode (Auto-Imputes Lab Insulin & Skinfold)",
                "🧪 Full Clinical Lab Mode (All 8 Biomarkers Known)",
            ],
            index=0,
        )
    with top_c2:
        preset_profile = st.selectbox(
            "⚡ Load Clinical Patient Preset (or Customize Below):",
            [
                "Custom Patient Profile (Interactive Sliders)",
                "Preset 1: Healthy Young Adult (Low Risk)",
                "Preset 2: 'Thin-Fat' Indian Phenotype (BMI 24.1, Pre-Diabetic)",
                "Preset 3: Unscreened High-Risk Metabolic Syndrome",
            ],
            index=2,
        )
    with top_c3:
        gender_sel = st.radio("👤 Patient Biological Sex:", ["Female", "Male"], horizontal=True)

    # Set default values based on preset
    if "Preset 1" in preset_profile:
        d_age, d_preg, d_glu, d_bp, d_bmi, d_skin, d_ins, d_dpf = 26, 0, 92, 70, 21.4, 20, 75, 0.28
    elif "Preset 2" in preset_profile:
        d_age, d_preg, d_glu, d_bp, d_bmi, d_skin, d_ins, d_dpf = 39, 2, 119, 82, 24.1, 29, 145, 0.68
    elif "Preset 3" in preset_profile:
        d_age, d_preg, d_glu, d_bp, d_bmi, d_skin, d_ins, d_dpf = 54, 3, 158, 90, 28.6, 35, 210, 0.95
    else:
        d_age, d_preg, d_glu, d_bp, d_bmi, d_skin, d_ins, d_dpf = 38, 1, 118, 78, 24.2, 26, 120, 0.52

    with st.expander("📏 Optional: Calculate BMI from Height (cm) & Weight (kg)", expanded=False):
        hw1, hw2, hw3 = st.columns(3)
        h_cm = hw1.number_input("Height (cm)", 120.0, 220.0, 165.0, step=0.5)
        w_kg = hw2.number_input("Weight (kg)", 30.0, 160.0, 66.0, step=0.5)
        calc_bmi = round(w_kg / ((h_cm / 100.0) ** 2), 1)
        hw3.metric("Calculated BMI", f"{calc_bmi} kg/m²", "Use in BMI slider below")

    in_c1, in_c2, in_c3, in_c4 = st.columns(4)
    with in_c1:
        u_age = st.slider("Age (Years)", 21, 85, d_age)
        if gender_sel == "Male":
            u_preg = st.number_input("Pregnancies (Locked to 0 for Male)", 0, 0, 0, disabled=True)
        else:
            u_preg = st.number_input("Number of Pregnancies", 0, 17, d_preg)
    with in_c2:
        u_glu = st.slider("Fasting / Oral Glucose (mg/dL)", 50, 220, d_glu)
        u_bp = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 125, d_bp)
    with in_c3:
        u_bmi = st.slider("Body Mass Index (BMI kg/m²)", 15.0, 50.0, float(d_bmi), step=0.1)
        if "Non-Invasive" in screening_mode:
            st.info("🔒 **Triceps Skinfold:** Auto-imputed via MICE Pipeline")
            u_skin_raw = 0.0
        else:
            u_skin_raw = float(st.slider("Triceps Skin Thickness (mm)", 7, 60, d_skin))
    with in_c4:
        u_dpf = st.slider("Family History / Pedigree Score", 0.08, 2.50, float(d_dpf), step=0.02)
        if "Non-Invasive" in screening_mode:
            st.info("🔒 **Serum Insulin:** Auto-imputed via MICE Pipeline")
            u_ins_raw = 0.0
        else:
            u_ins_raw = float(st.slider("Serum Insulin (mu U/ml)", 15, 400, d_ins))

    # Pass through Champion Leak-Free Pipeline
    champion_pipe = ml_bundle["champion_pipe"]
    fe_step: ClinicalFeaturePipeline = champion_pipe.named_steps["fe"]

    raw_input_df = pd.DataFrame(
        [[u_preg, u_glu, u_bp, u_skin_raw, u_ins_raw, u_bmi, u_dpf, u_age]],
        columns=s1_data["feature_cols"],
    )
    engineered_user_df = fe_step.transform(raw_input_df)
    imputed_base_vec = engineered_user_df[s1_data["feature_cols"]].iloc[0].values

    # Compute Exact 2^8 Shapley Values & Calibrated Prediction
    base_prob, prob_diab, shap_df = compute_exact_shapley_Attribution(
        pipe=champion_pipe,
        patient_vec=imputed_base_vec,
        baseline_vec=ml_bundle["baseline_ref"],
        feature_cols=s1_data["feature_cols"],
        bits=ml_bundle["shap_bits"],
        shap_lookup=ml_bundle["shap_lookup"],
    )

    u_homa = float(engineered_user_df["HOMA_IR"].iloc[0])
    u_met_score = int(engineered_user_df["Metabolic_Risk_Score"].iloc[0])
    u_ins_effective = float(engineered_user_df["Insulin"].iloc[0])
    u_skin_effective = float(engineered_user_df["SkinThickness"].iloc[0])

    if u_bmi < 23.0:
        icmr_status = "Normal (<23.0 kg/m²)"
    elif u_bmi < 25.0:
        icmr_status = "Indian Overweight (23.0–24.9 kg/m² | Missed by WHO)"
    elif u_bmi < 30.0:
        icmr_status = "Indian Obese (25.0–29.9 kg/m²)"
    else:
        icmr_status = "Severe Obese (>=30.0 kg/m²)"

    risk_tier = (
        "HIGH RISK (Immediate Clinical Escalation)"
        if prob_diab >= 55 or u_glu >= 126
        else "MODERATE / PRE-DIABETIC RISK"
        if prob_diab >= 32 or u_bmi >= 23.0 or u_glu >= 100
        else "LOW PRELIMINARY RISK"
    )

    # Top KPI Row for Patient
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Champion Ensemble Risk Score", f"{prob_diab:.1f}%", risk_tier.split("(")[0].strip())
    r2.metric(
        "Engineered HOMA-IR Index",
        f"{u_homa:.2f}",
        "Insulin Resistant (>=2.5)" if u_homa >= 2.5 else "Normal Sensitivity (<2.5)",
    )
    r3.metric("ICMR Indian BMI Tier", icmr_status.split("(")[0].strip(), f"BMI = {u_bmi:.1f} kg/m²")
    r4.metric(
        "Metabolic Syndrome Score",
        f"{u_met_score} / 4 Factors",
        f"Imputed Insulin: {u_ins_effective:.0f} μU/mL" if "Non-Invasive" in screening_mode else "Lab Measured",
    )

    # Visual Row: Gauge Chart + Exact Shapley Waterfall + Radar Chart
    v_col1, v_col2, v_col3 = st.columns([1, 1.35, 1.05])

    with v_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_diab,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": base_prob, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            title={"text": "AI/ML Diabetes Risk Probability<br><span style='font-size:0.8em;color:gray'>vs. Cohort Median Baseline</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f172a"},
                "steps": [
                    {"range": [0, 32], "color": "#10b981"},
                    {"range": [32, 55], "color": "#f59e0b"},
                    {"range": [55, 100], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": prob_diab,
                },
            },
        ))
        fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=15))
        st.plotly_chart(fig_gauge, width="stretch")

    with v_col2:
        # Exact Shapley Waterfall Chart
        shap_sorted = shap_df.sort_values("Abs_Impact", ascending=False)
        wf_x = ["Cohort Baseline"] + [
            f"{r['Feature']} ({r['Patient_Value']:.1f})" for _, r in shap_sorted.iterrows()
        ] + ["Final Patient Risk"]
        wf_y = [base_prob] + list(shap_sorted["Shapley_Delta_Pct"].values) + [0]
        wf_measure = ["absolute"] + ["relative"] * len(shap_sorted) + ["total"]

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=wf_measure,
            x=wf_x,
            y=wf_y,
            text=[f"{v:+.1f}%" if m == "relative" else f"{prob_diab:.1f}%" if m == "total" else f"{base_prob:.1f}%"
                  for v, m in zip(wf_y, wf_measure)],
            textposition="outside",
            connector={"line": {"color": "#64748b"}},
            increasing={"marker": {"color": "#ef4444"}},
            decreasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#3b82f6"}},
        ))
        fig_wf.update_layout(
            title="Exact 2⁸ Coalition Shapley Attribution Waterfall (Local XAI)",
            yaxis_title="Risk Probability (%)",
            height=360,
            margin=dict(l=15, r=15, t=45, b=65),
        )
        st.plotly_chart(fig_wf, width="stretch")

    with v_col3:
        # Normalized Radar Comparison: Patient vs Healthy Indian Cohort Median
        radar_Axes = ["Glucose", "BMI", "HOMA_IR", "BloodPressure", "Age", "Pedigree"]
        max_norms = [200.0, 45.0, 6.0, 110.0, 75.0, 1.5]
        pat_vals = [u_glu, u_bmi, min(u_homa, 6.0), u_bp, u_age, min(u_dpf, 1.5)]
        healthy_med = s1_data["healthy_medians"]
        healthy_homa = (healthy_med["Glucose"] * healthy_med["Insulin"]) / 405.0
        hlth_vals = [
            healthy_med["Glucose"], healthy_med["BMI"], healthy_homa,
            healthy_med["BloodPressure"], healthy_med["Age"], healthy_med["DiabetesPedigreeFunction"],
        ]

        pat_pct = [min(round(p / m * 100, 1), 115) for p, m in zip(pat_vals, max_norms)]
        hlth_pct = [round(h / m * 100, 1) for h, m in zip(hlth_vals, max_norms)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=hlth_pct + [hlth_pct[0]],
            theta=radar_Axes + [radar_Axes[0]],
            fill="toself",
            name="Healthy Cohort Median",
            line_color="#10b981",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=pat_pct + [pat_pct[0]],
            theta=radar_Axes + [radar_Axes[0]],
            fill="toself",
            name="Current Patient Profile",
            line_color="#ef4444",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 115])),
            title="Patient vs. Healthy Baseline Radar",
            height=360,
            margin=dict(l=30, r=30, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")
    # Interactive Counterfactual "What-If" Simulator & Stage 2 Escalation
    sim_col1, sim_col2 = st.columns([1.15, 1])

    with sim_col1:
        st.markdown("#### 🔄 Interactive 'What-If' Counterfactual Lifestyle & Treatment Simulator")
        st.caption("Simulate how targeted lifestyle modifications (weight loss, glycemic control, blood pressure reduction) lower your AI risk score:")
        sc1, sc2, sc3 = st.columns(3)
        delta_bmi = sc1.slider("Target BMI Change (kg/m²)", -6.0, 2.0, -2.0, step=0.2)
        delta_glu = sc2.slider("Target Glucose Change (mg/dL)", -50, 10, -18, step=2)
        delta_bp = sc3.slider("Target BP Change (mm Hg)", -20, 5, -6, step=1)

        cf_bmi = max(16.0, u_bmi + delta_bmi)
        cf_glu = max(65, u_glu + delta_glu)
        cf_bp = max(50, u_bp + delta_bp)

        cf_raw_df = pd.DataFrame(
            [[u_preg, cf_glu, cf_bp, u_skin_raw, u_ins_raw, cf_bmi, u_dpf, u_age]],
            columns=s1_data["feature_cols"],
        )
        cf_prob = float(champion_pipe.predict_proba(cf_raw_df)[0, 1]) * 100.0
        risk_drop = prob_diab - cf_prob

        cf_k1, cf_k2, cf_k3 = st.columns(3)
        cf_k1.metric("Current AI Risk Score", f"{prob_diab:.1f}%")
        cf_k2.metric("Post-Intervention Risk", f"{cf_prob:.1f}%", f"{-risk_drop:+.1f}% Risk Change", delta_color="inverse")
        cf_k3.metric("Projected Post-Change BMI", f"{cf_bmi:.1f} kg/m²", f"Glucose: {cf_glu} mg/dL")

    with sim_col2:
        st.markdown("#### 🏥 Stage 2 Empirical Hospital Escalation & Consultation Guidance")
        # Match age bracket to Stage 2 empirical readmission data
        age_decade = f"[{(u_age // 10) * 10}-{((u_age // 10) + 1) * 10})"
        s2_age_df = s2_data["all_encounters"]["age_readmit"]
        matched_row = s2_age_df[s2_age_df["age"] == age_decade]
        emp_readmit_30d = float(matched_row["Acute_30d_Readmit_Pct"].values[0]) if len(matched_row) > 0 else 11.39
        emp_unscreened = float(matched_row["Unscreened_HbA1c_Pct"].values[0]) if len(matched_row) > 0 else 83.28

        if prob_diab >= 55 or u_glu >= 126 or u_homa >= 3.0:
            st.error(
                f"**🚨 Action Required — Schedule Immediate Physician Consultation:**\n"
                f"- **Preliminary Risk:** **{prob_diab:.1f}%** (`Glucose = {u_glu} mg/dL`, `HOMA-IR = {u_homa:.2f}`).\n"
                f"- **Stage 2 Hospital Cohort Evidence (Age {age_decade}):** In our 101,766-visit hospital dataset, **{emp_unscreened:.1f}%** of patients in your age bracket lacked routine `HbA1c` screening, facing an acute `<30-day` hospital readmission rate of **{emp_readmit_30d:.2f}%**.\n"
                f"- **Clinical Checklist:** Order a diagnostic **HbA1c panel**, **Fasting Lipid Profile**, and **HOMA-IR confirmation**."
            )
        elif prob_diab >= 32 or u_bmi >= 23.0 or u_glu >= 100:
            st.warning(
                f"**⚠️ Early Metabolic / Pre-Diabetic Alert (Risk Score: {prob_diab:.1f}%):**\n"
                f"- **ICMR Indian BMI Insight:** `{icmr_status}`.\n"
                f"- **Counterfactual Target:** Reducing BMI to `{cf_bmi:.1f} kg/m²` and fasting glucose to `{cf_glu} mg/dL` lowers your predicted risk to **{cf_prob:.1f}%**.\n"
                f"- **Preventive Screening:** Schedule a baseline **HbA1c test** within 30 days."
            )
        else:
            st.success(
                f"**✅ Low Preliminary Risk ({prob_diab:.1f}%):**\n"
                f"- Your biomarkers (`Glucose = {u_glu} mg/dL`, `BMI = {u_bmi:.1f} kg/m²`, `HOMA-IR = {u_homa:.2f}`) are within healthy Indian Consensus thresholds.\n"
                f"- Maintain 150 mins/week physical activity and annual preventive screening."
            )

        # Downloadable Clinical Report
        top_drivers_str = ", ".join(
            [f"{r['Feature']} ({r['Shapley_Delta_Pct']:+.1f}%)" for _, r in shap_sorted.head(4).iterrows()]
        )
        report_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NIErnaya 2026 Clinical Diabetes Risk Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; line-height: 1.6; }}
.header {{ background: #0f172a; color: white; padding: 20px 24px; border-radius: 10px; }}
.grid {{ display: flex; gap: 16px; margin-top: 20px; }}
.box {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; flex: 1; background: #f8fafc; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
th {{ background: #e2e8f0; }}
</style>
</head>
<body>
<div class="header">
  <h2>🩺 NIErnaya 2026 — AI/ML Preliminary Diabetes Risk & Clinical Consultation Summary</h2>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Screening Mode: {screening_mode}</p>
</div>
<div class="grid">
  <div class="box"><b>AI/ML Ensemble Risk Score:</b><br><span style="font-size: 24px; color: #dc2626;"><b>{prob_diab:.1f}%</b></span><br>{risk_tier}</div>
  <div class="box"><b>ICMR Indian BMI Assessment:</b><br><b>BMI = {u_bmi:.1f} kg/m²</b><br>{icmr_status}</div>
  <div class="box"><b>Insulin Resistance (HOMA-IR):</b><br><b>{u_homa:.2f}</b><br>Metabolic Score: {u_met_score} / 4</div>
</div>
<h3>1. Exact Game-Theoretic Shapley Risk Drivers (Local XAI)</h3>
<p><b>Cohort Baseline Risk:</b> {base_prob:.1f}% &rarr; <b>Top Patient Drivers:</b> {top_drivers_str}</p>
<table>
  <tr><th>Biomarker</th><th>Patient Value</th><th>Cohort Median</th><th>Exact Shapley Contribution (&Delta;%)</th></tr>
  {''.join([f"<tr><td>{r['Feature']}</td><td>{r['Patient_Value']:.2f}</td><td>{r['Population_Median']:.2f}</td><td>{r['Shapley_Delta_Pct']:+.2f}%</td></tr>" for _, r in shap_sorted.iterrows()])}
</table>
<h3>2. Counterfactual Lifestyle Target & Stage 2 Clinical Escalation</h3>
<ul>
  <li><b>Counterfactual Projection:</b> Adjusting BMI to <b>{cf_bmi:.1f} kg/m²</b> and Fasting Glucose to <b>{cf_glu} mg/dL</b> reduces predicted risk from <b>{prob_diab:.1f}%</b> to <b>{cf_prob:.1f}%</b>.</li>
  <li><b>Stage 2 Hospital Screening Evidence (Age Bracket {age_decade}):</b> Unscreened HbA1c rate is <b>{emp_unscreened:.1f}%</b> with an acute 30-day hospital readmission rate of <b>{emp_readmit_30d:.2f}%</b>. Request a laboratory <b>HbA1c test</b> during your physician consultation.</li>
</ul>
<p style="font-size: 12px; color: #64748b; margin-top: 28px;"><i>Disclaimer: This report is generated by an AI/ML preliminary screening decision-support tool and does not replace a formal medical diagnosis by a licensed physician.</i></p>
</body>
</html>"""

        st.download_button(
            label="📄 Download Printable Clinical Consultation Report (.HTML)",
            data=report_html.encode("utf-8"),
            file_name="patient_diabetes_risk_clinical_report.html",
            mime="text/html",
            width="stretch",
        )
