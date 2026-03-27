# CapitalBridge Advisors — Data Intelligence Dashboard

> AI-powered deal sourcing, investor matching, and revenue prediction for a boutique investment banking advisory firm operating in the UAE–India corridor.

## Live App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Modules

| Module | Technique | Key Output |
|--------|-----------|-----------|
| **Overview** | Descriptive analytics | KPIs, deal pipeline, revenue trends |
| **Client Classification** | Random Forest | High-potential client identification (Accuracy, Precision, Recall, F1, ROC) |
| **Client Clustering** | K-Means + PCA | Client segmentation for targeted advisory |
| **Association Rules** | Apriori (mlxtend) | Service bundling with Support, Confidence, Lift |
| **Revenue Prediction** | Random Forest Regressor | Deal fee forecasting with feature importance |
| **Investor Matching** | Composite scoring | Ranked investor–company match recommendations |

---

## Deployment on Streamlit Cloud

1. Fork / upload this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
CapitalBridge_App/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml        ← Theme configuration
└── data/
    ├── 01_companies.csv
    ├── 02_investors.csv
    ├── 03_deals.csv
    ├── 04_survey_responses.csv
    ├── 05_revenue_transactions.csv
    └── 06_investor_company_matches.csv
```

---

Built by **Rudra | CapitalBridge Advisors**
