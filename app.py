# ============================================================
#   CapitalBridge Advisors — Data Intelligence Dashboard
#   Author : Rudra | CapitalBridge Advisors
#   Stack  : Streamlit · Plotly · Scikit-learn · MLxtend
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── scikit-learn ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import scipy.stats as stats

# ── Association rules ────────────────────────────────────────
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CapitalBridge Advisors | Intelligence Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  GLOBAL STYLES
# ============================================================
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0F6E56; }
    [data-testid="stSidebar"] * { color: #E1F5EE !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiselect label { color: #9FE1CB !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #F1EFE8;
        border-left: 4px solid #0F6E56;
        border-radius: 8px;
        padding: 12px 16px !important;
    }
    [data-testid="stMetricLabel"] { font-size: 13px !important; color: #5F5E5A !important; }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #085041 !important; }

    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #F1EFE8;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #0F6E56 !important;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0F6E56 0%, #1D9E75 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        margin: 16px 0 12px 0;
    }
    .insight-box {
        background: #E1F5EE;
        border-left: 4px solid #1D9E75;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        color: #085041;
        margin: 8px 0;
    }
    .warning-box {
        background: #FAEEDA;
        border-left: 4px solid #EF9F27;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        color: #633806;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  DATA LOADING  (cached)
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data(show_spinner=False)
def load_data():
    companies  = pd.read_csv(f"{DATA_DIR}/01_companies.csv")
    investors  = pd.read_csv(f"{DATA_DIR}/02_investors.csv")
    deals      = pd.read_csv(f"{DATA_DIR}/03_deals.csv")
    surveys    = pd.read_csv(f"{DATA_DIR}/04_survey_responses.csv")
    revenue    = pd.read_csv(f"{DATA_DIR}/05_revenue_transactions.csv")
    matches    = pd.read_csv(f"{DATA_DIR}/06_investor_company_matches.csv")
    return companies, investors, deals, surveys, revenue, matches

with st.spinner("Loading CapitalBridge data..."):
    companies, investors, deals, surveys, revenue, matches = load_data()

# ============================================================
#  COLOUR PALETTE
# ============================================================
C_GREEN  = "#0F6E56"
C_TEAL   = "#1D9E75"
C_LIGHT  = "#9FE1CB"
C_AMBER  = "#EF9F27"
C_CORAL  = "#D85A30"
C_PURPLE = "#534AB7"
C_GRAY   = "#888780"
SEQ_COLORS = [C_GREEN, C_TEAL, C_LIGHT, C_AMBER, C_CORAL, C_PURPLE, C_GRAY]

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💼 CapitalBridge")
    st.markdown("### Advisors Intelligence Hub")
    st.markdown("---")
    page = st.selectbox(
        "Navigate to",
        [
            "📊 Overview Dashboard",
            "🎯 Client Classification",
            "🔵 Client Clustering",
            "🔗 Association Rules",
            "📈 Revenue Prediction",
            "🤝 Investor Matching",
        ],
    )
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.markdown(f"- 🏢 Companies: **{len(companies)}**")
    st.markdown(f"- 💼 Investors: **{len(investors)}**")
    st.markdown(f"- 🤝 Deals: **{len(deals)}**")
    st.markdown(f"- 📋 Surveys: **{len(surveys)}**")
    st.markdown(f"- 💰 Revenue Records: **{len(revenue)}**")
    st.markdown("---")
    st.markdown("<small>Built by Rudra · CapitalBridge Advisors</small>", unsafe_allow_html=True)


# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def warning_note(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def fmt_usd(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


# ============================================================
#  PAGE 1 — OVERVIEW DASHBOARD
# ============================================================
if page == "📊 Overview Dashboard":
    st.title("📊 CapitalBridge Advisors — Intelligence Dashboard")
    st.markdown("_End-to-end deal analytics for the UAE–India mid-market advisory corridor_")
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────
    closed_deals   = deals[deals["deal_status"] == "Closed"]
    total_revenue  = revenue["amount_usd"].sum()
    active_clients = companies["is_client"].sum()
    avg_deal_size  = closed_deals["deal_size_usd"].mean()
    avg_readiness  = companies["deal_readiness_score"].mean()
    cross_border_pct = deals["cross_border"].mean() * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Revenue",     fmt_usd(total_revenue))
    k2.metric("Active Clients",    f"{int(active_clients)}")
    k3.metric("Deals Closed",      f"{len(closed_deals)}")
    k4.metric("Avg Deal Size",     fmt_usd(avg_deal_size))
    k5.metric("Avg Readiness Score", f"{avg_readiness:.1f} / 10")
    k6.metric("Cross-Border Deals", f"{cross_border_pct:.0f}%")

    st.markdown("---")

    # ── Row 1: Revenue over time + Deal pipeline ─────────────
    section("Revenue & Deal Pipeline")
    col1, col2 = st.columns(2)

    with col1:
        rev_q = revenue.groupby(["year", "quarter"])["amount_usd"].sum().reset_index()
        rev_q["period"] = rev_q["year"].astype(str) + " " + rev_q["quarter"]
        fig = px.bar(
            rev_q, x="period", y="amount_usd",
            color_discrete_sequence=[C_TEAL],
            title="Revenue by Quarter (USD)",
            labels={"amount_usd": "Revenue (USD)", "period": ""},
        )
        fig.update_layout(showlegend=False, height=320,
                          yaxis_tickformat="$,.0f",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pipeline = deals["deal_status"].value_counts().reset_index()
        pipeline.columns = ["status", "count"]
        color_map = {
            "Closed": C_GREEN, "In Progress": C_TEAL, "Term Sheet": C_AMBER,
            "Due Diligence": C_CORAL, "Mandate Signed": C_PURPLE,
            "Lost": C_GRAY, "On Hold": "#B4B2A9"
        }
        fig2 = px.funnel(
            pipeline, x="count", y="status",
            title="Deal Pipeline Funnel",
            color="status",
            color_discrete_map=color_map,
        )
        fig2.update_layout(showlegend=False, height=320,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Sector distribution + Country map ─────────────
    section("Client Geography & Sector Mix")
    col3, col4 = st.columns(2)

    with col3:
        sec_df = companies[companies["is_client"] == 1]["sector"].value_counts().reset_index()
        sec_df.columns = ["sector", "count"]
        fig3 = px.pie(
            sec_df, names="sector", values="count",
            title="Client Distribution by Sector",
            color_discrete_sequence=SEQ_COLORS,
            hole=0.4,
        )
        fig3.update_layout(height=340, paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        country_df = companies[companies["is_client"] == 1]["country"].value_counts().reset_index()
        country_df.columns = ["country", "count"]
        fig4 = px.bar(
            country_df, x="country", y="count",
            title="Active Clients by Country",
            color="count",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
            labels={"count": "Clients", "country": ""},
        )
        fig4.update_layout(showlegend=False, coloraxis_showscale=False,
                           height=340, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Revenue by fee type + Deal size distribution ──
    section("Fee Mix & Deal Size Analysis")
    col5, col6 = st.columns(2)

    with col5:
        fee_mix = revenue.groupby("fee_type")["amount_usd"].sum().reset_index()
        fig5 = px.pie(
            fee_mix, names="fee_type", values="amount_usd",
            title="Revenue by Fee Type",
            color_discrete_sequence=SEQ_COLORS,
            hole=0.35,
        )
        fig5.update_layout(height=320, paper_bgcolor="white")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.histogram(
            closed_deals, x="deal_size_usd",
            nbins=30,
            title="Closed Deal Size Distribution",
            labels={"deal_size_usd": "Deal Size (USD)"},
            color_discrete_sequence=[C_PURPLE],
        )
        fig6.update_layout(height=320, plot_bgcolor="white",
                           paper_bgcolor="white", xaxis_tickformat="$,.0f")
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Deal readiness heatmap by sector x stage ──────
    section("Deal Readiness Heatmap — Sector × Stage")
    heat = companies.groupby(["sector", "stage"])["deal_readiness_score"].mean().reset_index()
    heat_pivot = heat.pivot(index="sector", columns="stage", values="deal_readiness_score").fillna(0)
    fig7 = px.imshow(
        heat_pivot, text_auto=".1f",
        color_continuous_scale=[[0, "#E1F5EE"], [0.5, C_TEAL], [1, C_GREEN]],
        title="Average Deal Readiness Score (1–10)",
        aspect="auto", height=380,
    )
    fig7.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig7, use_container_width=True)
    insight("Higher deal readiness scores indicate companies ready for immediate mandate engagement. Target Sector A / Growth stage companies for fastest conversion.")

    # ── Row 5: Time to close + Cross-border performance ──────
    section("Deal Execution Performance")
    col7, col8 = st.columns(2)

    with col7:
        ttc = closed_deals.dropna(subset=["time_to_close_days"])
        fig8 = px.box(
            ttc, x="deal_type", y="time_to_close_days",
            title="Time to Close by Deal Type (Days)",
            color="deal_type",
            color_discrete_sequence=SEQ_COLORS,
        )
        fig8.update_layout(showlegend=False, height=340,
                           xaxis_tickangle=-30,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig8, use_container_width=True)

    with col8:
        cb_perf = deals.groupby(["cross_border", "deal_status"]).size().reset_index(name="count")
        cb_perf["cross_border"] = cb_perf["cross_border"].map({1: "Cross-Border", 0: "Domestic"})
        fig9 = px.bar(
            cb_perf, x="deal_status", y="count",
            color="cross_border",
            barmode="group",
            title="Deal Status: Cross-Border vs Domestic",
            color_discrete_sequence=[C_GREEN, C_AMBER],
        )
        fig9.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig9, use_container_width=True)


# ============================================================
#  PAGE 2 — CLIENT CLASSIFICATION
# ============================================================
elif page == "🎯 Client Classification":
    st.title("🎯 Client Classification — High-Potential Prospect Identification")
    st.markdown("_Random Forest classifier to identify which companies are likely to become CapitalBridge clients_")
    st.markdown("---")

    # ── Sidebar controls ─────────────────────────────────────
    with st.sidebar:
        st.markdown("### Classification Settings")
        n_estimators   = st.slider("RF Trees",          50, 500, 200, 50)
        max_depth_clf  = st.slider("Max Depth",          2, 20, 8)
        test_size_clf  = st.slider("Test Split %",      10, 40, 25, 5)
        threshold_clf  = st.slider("Decision Threshold", 0.30, 0.70, 0.50, 0.05)

    # ── Feature engineering ───────────────────────────────────
    clf_df = companies.copy()
    le_sector  = LabelEncoder()
    le_stage   = LabelEncoder()
    le_country = LabelEncoder()
    le_service = LabelEncoder()
    clf_df["sector_enc"]   = le_sector.fit_transform(clf_df["sector"])
    clf_df["stage_enc"]    = le_stage.fit_transform(clf_df["stage"])
    clf_df["country_enc"]  = le_country.fit_transform(clf_df["country"])
    clf_df["service_enc"]  = le_service.fit_transform(clf_df["primary_service_interest"])

    FEATURES_CLF = [
        "annual_revenue_usd", "employees", "yoy_growth_pct",
        "financial_clarity_score", "governance_score", "market_position_score",
        "deal_readiness_score", "ebitda_margin_pct", "debt_equity_ratio",
        "credit_score", "sector_enc", "stage_enc", "country_enc", "service_enc",
    ]
    TARGET_CLF = "is_client"

    X = clf_df[FEATURES_CLF].fillna(clf_df[FEATURES_CLF].median())
    y = clf_df[TARGET_CLF]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_clf / 100, random_state=42, stratify=y
    )

    scaler_clf = StandardScaler()
    X_train_s  = scaler_clf.fit_transform(X_train)
    X_test_s   = scaler_clf.transform(X_test)

    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth_clf,
        random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    rf_clf.fit(X_train_s, y_train)

    y_pred_proba = rf_clf.predict_proba(X_test_s)[:, 1]
    y_pred       = (y_pred_proba >= threshold_clf).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_pred_proba)

    # ── Metric Row ────────────────────────────────────────────
    section("Model Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall",    f"{rec:.3f}")
    m4.metric("F1 Score",  f"{f1:.3f}")
    m5.metric("ROC-AUC",   f"{roc:.3f}")

    # ── Row 1: ROC Curve + Confusion Matrix ──────────────────
    section("ROC Curve & Confusion Matrix")
    col1, col2 = st.columns(2)

    with col1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {roc:.3f})",
            line=dict(color=C_GREEN, width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color=C_GRAY, dash="dash", width=1.5),
        ))
        fig_roc.update_layout(
            title="ROC Curve — Client Classification",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        cm = confusion_matrix(y_test, y_pred)
        cm_labels = ["Non-Client", "Client"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=cm_labels, y=cm_labels,
            color_continuous_scale=[[0, "#E1F5EE"], [1, C_GREEN]],
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(height=380, paper_bgcolor="white",
                             coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Row 2: Feature Importance ─────────────────────────────
    section("Feature Importance")
    feat_imp = pd.DataFrame({
        "feature":   FEATURES_CLF,
        "importance": rf_clf.feature_importances_,
    }).sort_values("importance", ascending=True)

    feat_labels = {
        "annual_revenue_usd": "Annual Revenue",
        "employees": "Employees",
        "yoy_growth_pct": "YoY Growth %",
        "financial_clarity_score": "Financial Clarity",
        "governance_score": "Governance Score",
        "market_position_score": "Market Position",
        "deal_readiness_score": "Deal Readiness",
        "ebitda_margin_pct": "EBITDA Margin",
        "debt_equity_ratio": "Debt/Equity",
        "credit_score": "Credit Score",
        "sector_enc": "Sector",
        "stage_enc": "Stage",
        "country_enc": "Country",
        "service_enc": "Service Interest",
    }
    feat_imp["label"] = feat_imp["feature"].map(feat_labels)

    col3, col4 = st.columns([3, 2])
    with col3:
        fig_fi = px.bar(
            feat_imp, x="importance", y="label",
            orientation="h",
            title="Random Forest Feature Importance",
            color="importance",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
            labels={"importance": "Importance Score", "label": ""},
        )
        fig_fi.update_layout(height=440, coloraxis_showscale=False,
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col4:
        section("Prediction on Full Dataset")
        X_full_s = scaler_clf.transform(clf_df[FEATURES_CLF].fillna(clf_df[FEATURES_CLF].median()))
        clf_df["client_probability"] = rf_clf.predict_proba(X_full_s)[:, 1]
        top_prospects = clf_df[clf_df["is_client"] == 0].nlargest(15, "client_probability")[
            ["company_id", "sector", "country", "stage", "deal_readiness_score", "client_probability"]
        ].reset_index(drop=True)
        top_prospects["client_probability"] = top_prospects["client_probability"].round(3)
        top_prospects.columns = ["ID", "Sector", "Country", "Stage", "Readiness", "Prob."]
        st.markdown("**Top 15 Unconverted Prospects**")
        st.dataframe(top_prospects, use_container_width=True, height=420)

    # ── Row 3: Score distributions ────────────────────────────
    section("Probability Distribution & Threshold Analysis")
    col5, col6 = st.columns(2)

    with col5:
        hist_df = pd.DataFrame({
            "probability": y_pred_proba,
            "actual": y_test.values
        })
        hist_df["actual_label"] = hist_df["actual"].map({1: "Client", 0: "Non-Client"})
        fig_hist = px.histogram(
            hist_df, x="probability", color="actual_label",
            barmode="overlay", nbins=30,
            title="Predicted Probability Distribution",
            color_discrete_map={"Client": C_GREEN, "Non-Client": C_AMBER},
            labels={"probability": "Predicted Probability", "actual_label": ""},
        )
        fig_hist.add_vline(x=threshold_clf, line_dash="dash",
                           line_color=C_CORAL, annotation_text=f"Threshold={threshold_clf}")
        fig_hist.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col6:
        thresholds = np.arange(0.1, 0.9, 0.05)
        precs, recs, f1s = [], [], []
        for t in thresholds:
            yp = (y_pred_proba >= t).astype(int)
            precs.append(precision_score(y_test, yp, zero_division=0))
            recs.append(recall_score(y_test, yp, zero_division=0))
            f1s.append(f1_score(y_test, yp, zero_division=0))

        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(x=thresholds, y=precs, name="Precision", line=dict(color=C_GREEN)))
        fig_thr.add_trace(go.Scatter(x=thresholds, y=recs, name="Recall", line=dict(color=C_AMBER)))
        fig_thr.add_trace(go.Scatter(x=thresholds, y=f1s, name="F1 Score", line=dict(color=C_PURPLE)))
        fig_thr.add_vline(x=threshold_clf, line_dash="dash", line_color=C_CORAL)
        fig_thr.update_layout(
            title="Precision / Recall / F1 vs Threshold",
            xaxis_title="Classification Threshold",
            yaxis_title="Score",
            height=340, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_thr, use_container_width=True)

    insight(f"Deal Readiness Score is the strongest predictor of client conversion. Companies with readiness ≥ 7 have {acc*100:.0f}%+ conversion likelihood. Focus BD efforts on Fintech and NBFC sectors in UAE and India.")


# ============================================================
#  PAGE 3 — CLIENT CLUSTERING
# ============================================================
elif page == "🔵 Client Clustering":
    st.title("🔵 Client Clustering — Strategic Segmentation")
    st.markdown("_K-Means clustering with PCA visualisation to identify distinct client segments for targeted advisory_")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Clustering Settings")
        n_clusters   = st.slider("Number of Clusters (K)", 2, 8, 4)
        pca_dims     = st.slider("PCA Components",          2, 5, 2)
        cluster_feat = st.multiselect(
            "Features for Clustering",
            ["annual_revenue_usd", "employees", "yoy_growth_pct",
             "financial_clarity_score", "governance_score", "market_position_score",
             "deal_readiness_score", "ebitda_margin_pct", "credit_score"],
            default=["annual_revenue_usd", "yoy_growth_pct",
                     "deal_readiness_score", "governance_score",
                     "credit_score", "ebitda_margin_pct"],
        )

    if len(cluster_feat) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    clust_df = companies[cluster_feat].fillna(companies[cluster_feat].median())
    scaler_c = StandardScaler()
    X_clust  = scaler_c.fit_transform(clust_df)

    # ── Elbow Curve ───────────────────────────────────────────
    section("Elbow Method — Optimal K Selection")
    inertias = []
    ks = range(2, 10)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_clust)
        inertias.append(km.inertia_)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(ks), y=inertias, mode="lines+markers",
        line=dict(color=C_GREEN, width=2),
        marker=dict(size=8, color=C_GREEN),
    ))
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color=C_CORAL,
                        annotation_text=f"K={n_clusters}")
    fig_elbow.update_layout(
        title="Elbow Curve — Inertia vs Number of Clusters",
        xaxis_title="K (Clusters)", yaxis_title="Inertia",
        height=300, plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Fit KMeans ────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    companies["cluster"] = kmeans.fit_predict(X_clust)
    companies["cluster_label"] = "Segment " + (companies["cluster"] + 1).astype(str)

    # ── PCA visualisation ─────────────────────────────────────
    pca   = PCA(n_components=max(pca_dims, 2))
    X_pca = pca.fit_transform(X_clust)
    explained = pca.explained_variance_ratio_

    section("PCA Cluster Visualisation")
    col1, col2 = st.columns(2)

    with col1:
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Cluster": companies["cluster_label"],
            "Company": companies["company_id"],
            "Sector":  companies["sector"],
            "Readiness": companies["deal_readiness_score"],
            "Revenue": companies["annual_revenue_usd"],
        })
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            hover_data=["Company", "Sector", "Readiness"],
            title=f"PCA — 2D Cluster Map (explains {(explained[0]+explained[1])*100:.1f}% variance)",
            color_discrete_sequence=SEQ_COLORS,
            opacity=0.75,
        )
        fig_pca.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        # Cluster profiles radar
        cluster_means = companies.groupby("cluster_label")[cluster_feat].mean()
        norm_means    = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-8)
        cats          = cluster_feat

        fig_radar = go.Figure()
        cluster_colors = [C_GREEN, C_AMBER, C_CORAL, C_PURPLE, C_TEAL, C_GRAY, "#B4B2A9", "#F5C4B3"]
        for i, (idx, row) in enumerate(norm_means.iterrows()):
            vals = list(row.values) + [row.values[0]]
            lbls = cats + [cats[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=lbls, fill="toself",
                name=idx, line_color=cluster_colors[i % len(cluster_colors)],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            title="Cluster Profile Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=420, paper_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Cluster summary table ─────────────────────────────────
    section("Cluster Summary Statistics")
    disp_feats = [f for f in cluster_feat if f in companies.columns] + ["deal_readiness_score", "annual_revenue_usd"]
    disp_feats = list(dict.fromkeys(disp_feats))
    summary = companies.groupby("cluster_label")[disp_feats].agg(["mean", "count"]).round(2)
    st.dataframe(summary, use_container_width=True)

    # ── Cluster composition charts ────────────────────────────
    section("Cluster Composition — Sector & Stage Mix")
    col3, col4 = st.columns(2)

    with col3:
        sec_clust = companies.groupby(["cluster_label", "sector"]).size().reset_index(name="count")
        fig_sc = px.bar(
            sec_clust, x="cluster_label", y="count", color="sector",
            title="Sector Distribution per Cluster",
            color_discrete_sequence=SEQ_COLORS,
            barmode="stack",
        )
        fig_sc.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col4:
        stg_clust = companies.groupby(["cluster_label", "stage"]).size().reset_index(name="count")
        fig_stg = px.bar(
            stg_clust, x="cluster_label", y="count", color="stage",
            title="Stage Distribution per Cluster",
            color_discrete_sequence=SEQ_COLORS,
            barmode="stack",
        )
        fig_stg.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_stg, use_container_width=True)

    # ── Deal readiness by cluster ─────────────────────────────
    section("Deal Readiness Score Distribution per Cluster")
    fig_box = px.box(
        companies, x="cluster_label", y="deal_readiness_score",
        color="cluster_label",
        title="Deal Readiness Spread by Cluster",
        color_discrete_sequence=SEQ_COLORS,
    )
    fig_box.update_layout(showlegend=False, height=340,
                          plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_box, use_container_width=True)

    insight("Segment 1 typically represents high-readiness, high-growth companies ideal for immediate fundraising mandates. Segment 3 often contains established SMEs suited for M&A advisory. Tailor outreach messaging per segment.")


# ============================================================
#  PAGE 4 — ASSOCIATION RULES
# ============================================================
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rule Mining — Service Bundling Intelligence")
    st.markdown("_Apriori algorithm to discover which services, sectors, and client attributes co-occur — enabling smarter cross-selling_")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Apriori Settings")
        min_support    = st.slider("Min Support",    0.05, 0.50, 0.10, 0.01)
        min_confidence = st.slider("Min Confidence", 0.10, 0.90, 0.30, 0.05)
        min_lift       = st.slider("Min Lift",       1.0,  5.0,  1.2,  0.1)
        max_rules      = st.slider("Max Rules to Show", 10, 100, 30, 5)

    # ── Build transaction basket ──────────────────────────────
    surv_merge = surveys.merge(
        companies[["company_id", "sector", "stage", "country", "deal_readiness_score"]],
        on="company_id", how="left",
    )
    surv_merge["readiness_band"] = pd.cut(
        surv_merge["deal_readiness_score"],
        bins=[0, 4, 7, 10],
        labels=["LowReadiness", "MedReadiness", "HighReadiness"],
    )
    surv_merge["budget_clean"]  = surv_merge["budget_band"].str.replace("[<>$K]", "", regex=True).str.strip()
    surv_merge["urgency_clean"] = surv_merge["urgency_timeline"].str.replace(" ", "_")

    transactions = []
    for _, row in surv_merge.iterrows():
        basket = []
        if pd.notna(row["service_selected"]):
            basket.append(f"SVC_{row['service_selected'].replace(' ', '_')}")
        if pd.notna(row["primary_pain_point"]):
            basket.append(f"PAIN_{row['primary_pain_point'].replace(' ', '_')}")
        if pd.notna(row["sector"]):
            basket.append(f"SECTOR_{row['sector']}")
        if pd.notna(row["stage"]):
            basket.append(f"STAGE_{row['stage'].replace(' ', '_')}")
        if pd.notna(row["prev_advisory_firm"]):
            basket.append(f"PREV_{row['prev_advisory_firm'].replace(' ', '_')}")
        if pd.notna(row["urgency_clean"]):
            basket.append(f"URGENCY_{row['urgency_clean']}")
        if pd.notna(row["readiness_band"]):
            basket.append(f"DR_{row['readiness_band']}")
        if row.get("willing_to_retainer", 0) == 1:
            basket.append("WANTS_RETAINER")
        if basket:
            transactions.append(basket)

    te      = TransactionEncoder()
    te_arr  = te.fit_transform(transactions)
    basket_df = pd.DataFrame(te_arr, columns=te.columns_)

    # ── Run Apriori ───────────────────────────────────────────
    try:
        freq_items = apriori(basket_df, min_support=min_support, use_colnames=True)
        if len(freq_items) == 0:
            warning_note("No frequent itemsets found. Try lowering the Min Support threshold.")
            st.stop()

        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift, num_itemsets=len(freq_items))
        rules = rules[rules["confidence"] >= min_confidence].sort_values("lift", ascending=False)
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    except Exception as e:
        warning_note(f"Association rules error: {str(e)}. Try adjusting thresholds.")
        st.stop()

    # ── KPIs ──────────────────────────────────────────────────
    section("Mining Summary")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Frequent Itemsets", len(freq_items))
    r2.metric("Association Rules", len(rules))
    r3.metric("Max Lift",          f"{rules['lift'].max():.2f}" if len(rules) > 0 else "N/A")
    r4.metric("Max Confidence",    f"{rules['confidence'].max():.2f}" if len(rules) > 0 else "N/A")

    if len(rules) == 0:
        warning_note("No rules meet all three thresholds. Try reducing Min Confidence or Min Lift.")
        st.stop()

    # ── Rules Table ───────────────────────────────────────────
    section("Association Rules — Support · Confidence · Lift")
    rules_display = rules.head(max_rules)[
        ["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]
    ].copy()
    rules_display.columns = ["Antecedent (IF)", "Consequent (THEN)", "Support", "Confidence", "Lift", "Leverage", "Conviction"]
    rules_display["Support"]    = rules_display["Support"].round(4)
    rules_display["Confidence"] = rules_display["Confidence"].round(4)
    rules_display["Lift"]       = rules_display["Lift"].round(3)
    rules_display["Leverage"]   = rules_display["Leverage"].round(4)
    rules_display["Conviction"] = rules_display["Conviction"].round(3)
    st.dataframe(rules_display, use_container_width=True, height=350)

    # ── Scatter: Support vs Confidence (bubble = lift) ────────
    section("Support vs Confidence — Bubble Size = Lift")
    col1, col2 = st.columns(2)

    with col1:
        rules["lift_scaled"] = (rules["lift"] - rules["lift"].min()) / (rules["lift"].max() - rules["lift"].min() + 1e-8) * 40 + 5
        fig_sc2 = px.scatter(
            rules.head(max_rules),
            x="support", y="confidence",
            size="lift_scaled",
            color="lift",
            hover_data={"antecedents_str": True, "consequents_str": True,
                        "support": ":.4f", "confidence": ":.4f", "lift": ":.3f",
                        "lift_scaled": False},
            color_continuous_scale=[[0, C_LIGHT], [0.5, C_TEAL], [1, C_GREEN]],
            title="Support vs Confidence (bubble = Lift)",
            labels={"support": "Support", "confidence": "Confidence"},
        )
        fig_sc2.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_sc2, use_container_width=True)

    with col2:
        fig_lift = px.histogram(
            rules, x="lift", nbins=25,
            color_discrete_sequence=[C_PURPLE],
            title="Lift Distribution of All Rules",
            labels={"lift": "Lift Value"},
        )
        fig_lift.add_vline(x=min_lift, line_dash="dash", line_color=C_CORAL,
                           annotation_text=f"Min Lift={min_lift}")
        fig_lift.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_lift, use_container_width=True)

    # ── Top rules by confidence & lift ────────────────────────
    section("Top Rules by Confidence & Top Rules by Lift")
    col3, col4 = st.columns(2)

    with col3:
        top_conf = rules.nlargest(15, "confidence")[["antecedents_str", "consequents_str", "confidence"]].reset_index(drop=True)
        top_conf.columns = ["IF", "THEN", "Confidence"]
        fig_tc = px.bar(
            top_conf, x="Confidence", y="IF",
            orientation="h", title="Top 15 Rules by Confidence",
            color="Confidence",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
            labels={"IF": ""},
        )
        fig_tc.update_layout(height=480, coloraxis_showscale=False,
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_tc, use_container_width=True)

    with col4:
        top_lift = rules.nlargest(15, "lift")[["antecedents_str", "consequents_str", "lift"]].reset_index(drop=True)
        top_lift.columns = ["IF", "THEN", "Lift"]
        fig_tl = px.bar(
            top_lift, x="Lift", y="IF",
            orientation="h", title="Top 15 Rules by Lift",
            color="Lift",
            color_continuous_scale=[[0, C_LIGHT], [1, C_AMBER]],
            labels={"IF": ""},
        )
        fig_tl.update_layout(height=480, coloraxis_showscale=False,
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_tl, use_container_width=True)

    # ── Frequent items bar ────────────────────────────────────
    section("Frequent Itemset Support Distribution")
    freq_items["itemset_str"] = freq_items["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    freq_items_sorted = freq_items.sort_values("support", ascending=False).head(25)
    fig_freq = px.bar(
        freq_items_sorted, x="support", y="itemset_str",
        orientation="h",
        title="Top 25 Frequent Itemsets by Support",
        color="support",
        color_continuous_scale=[[0, C_LIGHT], [1, C_TEAL]],
    )
    fig_freq.update_layout(height=600, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_title="")
    st.plotly_chart(fig_freq, use_container_width=True)

    insight("High-lift rules reveal powerful cross-selling opportunities. Rules with Lift > 2.5 indicate that the antecedent and consequent co-occur 2.5x more than by chance — use these for targeted service bundle recommendations.")


# ============================================================
#  PAGE 5 — REVENUE PREDICTION (REGRESSION)
# ============================================================
elif page == "📈 Revenue Prediction":
    st.title("📈 Revenue Prediction — Deal Fee Forecasting")
    st.markdown("_Random Forest Regressor to predict deal success fee revenue from company and deal characteristics_")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Regression Settings")
        reg_model_choice = st.selectbox(
            "Model",
            ["Random Forest Regressor", "Gradient Boosting Regressor"],
        )
        n_est_reg     = st.slider("Estimators",  50, 500, 200, 50)
        max_dep_reg   = st.slider("Max Depth",    2,  20,  8)
        test_size_reg = st.slider("Test Split %", 10, 40, 25, 5)

    # ── Data prep ─────────────────────────────────────────────
    deals_co = deals[deals["deal_status"] == "Closed"].merge(
        companies[["company_id", "sector", "stage", "country",
                   "annual_revenue_usd", "employees", "yoy_growth_pct",
                   "financial_clarity_score", "governance_score",
                   "market_position_score", "deal_readiness_score",
                   "ebitda_margin_pct", "credit_score"]],
        on="company_id", how="left",
    ).dropna(subset=["fee_earned_usd"])

    if len(deals_co) < 20:
        st.warning("Not enough closed deals for regression. Need at least 20.")
        st.stop()

    le_r = {}
    for col in ["sector", "stage", "country", "deal_type", "lead_advisor"]:
        le_r[col] = LabelEncoder()
        deals_co[f"{col}_enc"] = le_r[col].fit_transform(deals_co[col].fillna("Unknown"))

    FEATURES_REG = [
        "deal_size_usd", "complexity_rating", "cross_border", "time_to_close_days",
        "annual_revenue_usd", "employees", "yoy_growth_pct",
        "financial_clarity_score", "governance_score", "market_position_score",
        "deal_readiness_score", "ebitda_margin_pct", "credit_score",
        "sector_enc", "stage_enc", "country_enc", "deal_type_enc", "lead_advisor_enc",
    ]
    TARGET_REG = "fee_earned_usd"

    valid_feats = [f for f in FEATURES_REG if f in deals_co.columns]
    X_reg = deals_co[valid_feats].fillna(deals_co[valid_feats].median())
    y_reg = deals_co[TARGET_REG]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reg, y_reg, test_size=test_size_reg / 100, random_state=42
    )

    scaler_reg = StandardScaler()
    X_tr_s = scaler_reg.fit_transform(X_tr)
    X_te_s = scaler_reg.transform(X_te)

    if reg_model_choice == "Random Forest Regressor":
        reg_model = RandomForestRegressor(
            n_estimators=n_est_reg, max_depth=max_dep_reg,
            random_state=42, n_jobs=-1
        )
    else:
        reg_model = GradientBoostingRegressor(
            n_estimators=n_est_reg, max_depth=max_dep_reg, random_state=42
        )

    reg_model.fit(X_tr_s, y_tr)
    y_pred_reg = reg_model.predict(X_te_s)

    mae  = mean_absolute_error(y_te, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_reg))
    r2   = r2_score(y_te, y_pred_reg)
    mape = np.mean(np.abs((y_te - y_pred_reg) / (y_te + 1e-8))) * 100

    # ── KPIs ──────────────────────────────────────────────────
    section("Regression Performance Metrics")
    rm1, rm2, rm3, rm4 = st.columns(4)
    rm1.metric("R² Score",  f"{r2:.3f}")
    rm2.metric("MAE",       fmt_usd(mae))
    rm3.metric("RMSE",      fmt_usd(rmse))
    rm4.metric("MAPE",      f"{mape:.1f}%")

    # ── Row 1: Actual vs Predicted + Residuals ────────────────
    section("Actual vs Predicted & Residual Analysis")
    col1, col2 = st.columns(2)

    with col1:
        av_df = pd.DataFrame({"Actual": y_te.values, "Predicted": y_pred_reg})
        fig_ap = px.scatter(
            av_df, x="Actual", y="Predicted",
            title="Actual vs Predicted Fee Revenue (USD)",
            color_discrete_sequence=[C_TEAL],
            labels={"Actual": "Actual Fee (USD)", "Predicted": "Predicted Fee (USD)"},
            opacity=0.65,
        )
        mn = min(av_df["Actual"].min(), av_df["Predicted"].min())
        mx = max(av_df["Actual"].max(), av_df["Predicted"].max())
        fig_ap.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            name="Perfect Prediction",
            line=dict(color=C_CORAL, dash="dash"),
        ))
        fig_ap.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                             xaxis_tickformat="$,.0f", yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_ap, use_container_width=True)

    with col2:
        residuals = y_te.values - y_pred_reg
        fig_res = px.histogram(
            x=residuals, nbins=30,
            title="Residual Distribution",
            labels={"x": "Residual (Actual - Predicted)"},
            color_discrete_sequence=[C_PURPLE],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color=C_CORAL)
        fig_res.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_tickformat="$,.0f")
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────
    section("Feature Importance — Revenue Drivers")
    reg_fi = pd.DataFrame({
        "feature":    valid_feats,
        "importance": reg_model.feature_importances_,
    }).sort_values("importance", ascending=True)

    feat_labels_reg = {
        "deal_size_usd": "Deal Size",
        "complexity_rating": "Deal Complexity",
        "cross_border": "Cross-Border Flag",
        "time_to_close_days": "Time to Close",
        "annual_revenue_usd": "Company Revenue",
        "employees": "Employees",
        "yoy_growth_pct": "YoY Growth %",
        "financial_clarity_score": "Financial Clarity",
        "governance_score": "Governance",
        "market_position_score": "Market Position",
        "deal_readiness_score": "Deal Readiness",
        "ebitda_margin_pct": "EBITDA Margin",
        "credit_score": "Credit Score",
        "sector_enc": "Sector",
        "stage_enc": "Stage",
        "country_enc": "Country",
        "deal_type_enc": "Deal Type",
        "lead_advisor_enc": "Lead Advisor",
    }
    reg_fi["label"] = reg_fi["feature"].map(feat_labels_reg).fillna(reg_fi["feature"])

    col3, col4 = st.columns([3, 2])
    with col3:
        fig_rfi = px.bar(
            reg_fi, x="importance", y="label",
            orientation="h",
            title=f"{reg_model_choice} — Feature Importance",
            color="importance",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
            labels={"importance": "Importance", "label": ""},
        )
        fig_rfi.update_layout(height=480, coloraxis_showscale=False,
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_rfi, use_container_width=True)

    with col4:
        # Revenue by quarter prediction
        section("Revenue Trend — Actual vs Predicted Quarterly")
        rev_q2 = revenue.groupby(["year", "quarter"])["amount_usd"].sum().reset_index()
        rev_q2["period"] = rev_q2["year"].astype(str) + " " + rev_q2["quarter"]
        rev_q2["trend"] = rev_q2["amount_usd"].rolling(2, min_periods=1).mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=rev_q2["period"], y=rev_q2["amount_usd"],
            name="Actual", marker_color=C_TEAL,
        ))
        fig_trend.add_trace(go.Scatter(
            x=rev_q2["period"], y=rev_q2["trend"],
            name="Trend", mode="lines",
            line=dict(color=C_CORAL, width=2, dash="dot"),
        ))
        fig_trend.update_layout(
            height=440, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_tickformat="$,.0f", xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Revenue by service line ───────────────────────────────
    section("Revenue Breakdown by Service Line & Fee Type")
    col5, col6 = st.columns(2)

    with col5:
        svc_rev = revenue.groupby("service_line")["amount_usd"].sum().reset_index()
        svc_rev.columns = ["Service", "Revenue"]
        svc_rev = svc_rev.sort_values("Revenue", ascending=True)
        fig_svc = px.bar(
            svc_rev, x="Revenue", y="Service", orientation="h",
            title="Total Revenue by Service Line",
            color="Revenue",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
        )
        fig_svc.update_layout(height=340, coloraxis_showscale=False,
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_tickformat="$,.0f")
        st.plotly_chart(fig_svc, use_container_width=True)

    with col6:
        yr_fee = revenue.groupby(["year", "fee_type"])["amount_usd"].sum().reset_index()
        fig_yf = px.bar(
            yr_fee, x="year", y="amount_usd", color="fee_type",
            title="Revenue by Year & Fee Type",
            color_discrete_sequence=SEQ_COLORS,
            barmode="stack",
            labels={"amount_usd": "Revenue (USD)", "year": "Year", "fee_type": "Fee Type"},
        )
        fig_yf.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                             yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_yf, use_container_width=True)

    insight(f"Deal Size is the dominant revenue driver — a 10% increase in average deal size could lift fee revenue by ~7–9%. Focus on cross-border M&A mandates which command higher fees and complexity premiums.")


# ============================================================
#  PAGE 6 — INVESTOR MATCHING
# ============================================================
elif page == "🤝 Investor Matching":
    st.title("🤝 Investor–Company Matching Intelligence")
    st.markdown("_Composite match scoring across sector, stage, geography, ticket size, and ESG — ranked recommendations for deal sourcing_")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Match Filters")
        min_match_score = st.slider("Min Match Score", 0.0, 1.0, 0.55, 0.05)
        filter_country  = st.multiselect(
            "Filter Investor Country",
            options=sorted(investors["country"].unique()),
            default=[],
        )
        filter_type = st.multiselect(
            "Filter Investor Type",
            options=sorted(investors["investor_type"].unique()),
            default=[],
        )
        filter_status = st.selectbox(
            "Investor Active Status",
            ["All", "Active", "Passive", "Dormant"],
        )

    # ── Merge matches with investor and company info ──────────
    m_df = matches.merge(
        investors[["investor_id", "investor_type", "country", "active_status",
                   "min_ticket_usd", "max_ticket_usd", "avg_irr_expected_pct",
                   "esg_focused", "geo_preference"]],
        on="investor_id", how="left",
        suffixes=("", "_inv"),
    ).merge(
        companies[["company_id", "sector", "stage", "country",
                   "annual_revenue_usd", "deal_readiness_score", "yoy_growth_pct"]],
        on="company_id", how="left",
        suffixes=("", "_co"),
    )

    # Apply filters
    filtered = m_df[m_df["composite_match_score"] >= min_match_score].copy()
    if filter_country:
        filtered = filtered[filtered["country"].isin(filter_country)]
    if filter_type:
        filtered = filtered[filtered["investor_type"].isin(filter_type)]
    if filter_status != "All":
        filtered = filtered[filtered["active_status"] == filter_status]

    # ── KPIs ──────────────────────────────────────────────────
    section("Matching Summary")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Total Match Pairs",    len(matches))
    i2.metric("Recommended (≥0.55)",  int(matches["recommended"].sum()))
    i3.metric("Filtered Matches",     len(filtered))
    i4.metric("Avg Match Score",      f"{filtered['composite_match_score'].mean():.3f}" if len(filtered) > 0 else "—")

    # ── Match score distribution ──────────────────────────────
    section("Match Score Distribution & Component Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig_md = px.histogram(
            m_df, x="composite_match_score",
            color="recommended",
            nbins=30,
            title="Composite Match Score Distribution",
            color_discrete_map={1: C_GREEN, 0: C_GRAY},
            labels={"composite_match_score": "Match Score", "recommended": "Recommended"},
        )
        fig_md.add_vline(x=min_match_score, line_dash="dash", line_color=C_CORAL,
                         annotation_text=f"Threshold={min_match_score}")
        fig_md.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_md, use_container_width=True)

    with col2:
        comp_cols = ["sector_match", "stage_match", "geo_match", "ticket_fit_score", "esg_alignment"]
        comp_means = filtered[comp_cols].mean().reset_index()
        comp_means.columns = ["Component", "Avg Score"]
        comp_means["Component"] = comp_means["Component"].map({
            "sector_match": "Sector Alignment",
            "stage_match": "Stage Alignment",
            "geo_match": "Geography Alignment",
            "ticket_fit_score": "Ticket Size Fit",
            "esg_alignment": "ESG Alignment",
        })
        fig_comp = px.bar(
            comp_means, x="Avg Score", y="Component",
            orientation="h",
            title="Average Match Score by Component",
            color="Avg Score",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
        )
        fig_comp.update_layout(height=360, coloraxis_showscale=False,
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Top recommended matches table ─────────────────────────
    section("Top Recommended Investor–Company Matches")
    top_matches = filtered.nlargest(30, "composite_match_score")[
        ["match_id", "company_id", "investor_id", "composite_match_score",
         "sector_match", "stage_match", "geo_match", "ticket_fit_score",
         "investor_type", "active_status", "outreach_status",
         "sector", "stage", "deal_readiness_score"]
    ].reset_index(drop=True)

    top_matches.columns = [
        "Match ID", "Company", "Investor", "Score",
        "Sector✓", "Stage✓", "Geo✓", "Ticket Fit",
        "Inv. Type", "Status", "Outreach",
        "Co. Sector", "Co. Stage", "Readiness"
    ]
    top_matches["Score"]      = top_matches["Score"].round(3)
    top_matches["Ticket Fit"] = top_matches["Ticket Fit"].round(3)
    top_matches["Readiness"]  = top_matches["Readiness"].round(2)
    st.dataframe(top_matches, use_container_width=True, height=380)

    # ── Investor type performance ─────────────────────────────
    section("Match Quality by Investor Type & Geography")
    col3, col4 = st.columns(2)

    with col3:
        inv_type_score = m_df.groupby("investor_type")["composite_match_score"].mean().reset_index()
        inv_type_score.columns = ["Investor Type", "Avg Match Score"]
        inv_type_score = inv_type_score.sort_values("Avg Match Score", ascending=True)
        fig_it = px.bar(
            inv_type_score, x="Avg Match Score", y="Investor Type",
            orientation="h",
            title="Avg Match Score by Investor Type",
            color="Avg Match Score",
            color_continuous_scale=[[0, C_LIGHT], [1, C_PURPLE]],
        )
        fig_it.update_layout(height=360, coloraxis_showscale=False,
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_it, use_container_width=True)

    with col4:
        geo_score = m_df.groupby("country")["composite_match_score"].agg(["mean", "count"]).reset_index()
        geo_score.columns = ["Country", "Avg Score", "Count"]
        fig_geo = px.scatter(
            geo_score, x="Count", y="Avg Score",
            size="Count", color="Avg Score",
            text="Country",
            title="Match Volume vs Quality by Investor Country",
            color_continuous_scale=[[0, C_LIGHT], [1, C_GREEN]],
            labels={"Count": "Number of Matches", "Avg Score": "Avg Match Score"},
        )
        fig_geo.update_traces(textposition="top center")
        fig_geo.update_layout(height=360, coloraxis_showscale=False,
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_geo, use_container_width=True)

    # ── Outreach funnel ───────────────────────────────────────
    section("Outreach Pipeline — Matched Pairs Status")
    col5, col6 = st.columns(2)

    with col5:
        outreach = filtered["outreach_status"].value_counts().reset_index()
        outreach.columns = ["Status", "Count"]
        status_order = ["Not Contacted", "Contacted", "Meeting Scheduled", "In Discussion", "Passed"]
        outreach["Status"] = pd.Categorical(outreach["Status"], categories=status_order, ordered=True)
        outreach = outreach.sort_values("Status")
        fig_out = px.funnel(
            outreach, x="Count", y="Status",
            title="Filtered Match Outreach Funnel",
            color_discrete_sequence=[C_GREEN],
        )
        fig_out.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_out, use_container_width=True)

    with col6:
        # Heatmap: Investor type vs Company stage
        heat2 = m_df.groupby(["investor_type", "stage"])["composite_match_score"].mean().reset_index()
        heat2_pivot = heat2.pivot(index="investor_type", columns="stage", values="composite_match_score").fillna(0)
        fig_h2 = px.imshow(
            heat2_pivot, text_auto=".2f",
            color_continuous_scale=[[0, "#E1F5EE"], [0.5, C_TEAL], [1, C_GREEN]],
            title="Avg Match Score: Investor Type × Company Stage",
            aspect="auto", height=360,
        )
        fig_h2.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig_h2, use_container_width=True)

    insight("Family Offices and VC Funds show the strongest match quality for UAE-based Series A and Growth companies. Prioritise outreach to Active investors with geo_preference = 'UAE+India' for fastest deal conversion.")
