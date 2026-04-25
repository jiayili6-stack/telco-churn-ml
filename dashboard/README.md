# Telco Customer Churn Prediction Dashboard

An end-to-end machine learning system that predicts customer churn for a telecom operator and surfaces the predictions through an interactive Streamlit dashboard.

Built as a final project for a Machine Learning course (Spring 2026, Group 8).

---

## What it does

- Predicts whether a telecom customer is likely to churn in the next billing cycle, using a Logistic Regression model trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 30 features after encoding).
- Provides a **Live Prediction** view for single-customer real-time inference, with feature-level explanations of why the model flagged the customer.
- Provides a **Batch Dashboard** view with portfolio-level metrics, a churn probability distribution, and a ranked list of the top-20 highest-risk customers.

The model uses a **business-tuned classification threshold of 0.307** (lowered from the default 0.5) to lift recall on the churn class from 55% to 76% — prioritizing catching real churners over avoiding false alarms.

---

## Architecture

```
┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐
│   Amazon S3    │ → │  Amazon SageMaker│ → │  RDS MySQL or    │ ← │  Streamlit   │
│ (model + data) │   │ (batch inference)│   │  SQLite (fallback)│   │ (local UI)   │
└────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────┘
```

- **Original deployment** uses S3, SageMaker, and RDS MySQL on AWS.
- **This repo** is configured so anyone can clone and run it locally without an AWS account — by default it uses a local SQLite database that is auto-seeded with batch predictions on first run.

---

## Quickstart (no AWS required)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/telco-churn-dashboard.git
cd telco-churn-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(Recommended: do this inside a virtual environment.)

### 3. Run the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

On first run, the app creates a local SQLite database (`predictions.db`) and seeds it with batch predictions for all 7,043 customers in the dataset. This takes a few seconds.

---

## Folder structure

```
telco-churn-dashboard/
├── app.py                        # main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md
├── .env.example                  # template for optional RDS credentials
├── .gitignore
├── model/
│   ├── final_model_lr.pkl        # trained Logistic Regression
│   ├── final_scaler.pkl          # fitted StandardScaler
│   └── model_metadata.json       # threshold + feature schema
└── data/
    └── Telco-Customer-Churn.xlsx # raw dataset (7,043 customers)
```

---

## Optional: connecting to AWS RDS

If you have your own RDS MySQL instance and want to use it instead of the local SQLite fallback:

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Fill in your RDS credentials in `.env`:
   ```
   RDS_HOST=your-endpoint.us-east-1.rds.amazonaws.com
   RDS_USER=admin
   RDS_PASSWORD=your-password
   RDS_DATABASE=telco_churn
   ```

3. Make sure the `predictions` table exists in your RDS database. You can create it with:
   ```sql
   CREATE TABLE predictions (
     id INT AUTO_INCREMENT PRIMARY KEY,
     customerID VARCHAR(50),
     churn_probability FLOAT,
     churn_prediction TINYINT,
     actual_churn TINYINT,
     threshold_used FLOAT,
     timestamp DATETIME,
     source VARCHAR(20)
   );
   ```

4. Run the app — it will detect the `.env` file and use RDS automatically.

`.env` is gitignored, so your credentials stay local.

---

## Using the dashboard

### Live Prediction tab

- Pick a profile from the sidebar dropdown:
  - **Custom input** — fill in any combination of features manually.
  - **Demo personas** — four pre-built customer archetypes (high-risk fiber, borderline, loyal long-term, new low-risk).
  - **Existing customer** — pick any of the 7,043 customers in the dataset (typeable search).
- The right panel updates in real time as you change any sidebar input.
- Expand "Which features are driving this prediction?" to see the top 8 features ranked by their per-prediction contribution.
- Click "Save this prediction" to append the result to the database (visible in the Batch Dashboard tab).

### Batch Dashboard tab

- KPI cards summarizing total customers scored, high-risk count, flag rate, and average churn probability.
- Churn probability distribution histogram across all customers.
- Top 20 highest-risk customers ranked by predicted probability.
- Recent live predictions submitted from the Live Prediction tab.

---

## Model details

- **Model**: Logistic Regression (scikit-learn, lbfgs solver, max_iter=2000)
- **Training data**: 5,282 customers (75% of dataset, stratified split)
- **Test data**: 1,761 customers (25%, stratified split)
- **Features**: 30 (after one-hot encoding of categorical variables, with drop_first=True)
- **Scaling**: StandardScaler fit on training set only

### Test-set performance

| Metric | Default threshold (0.5) | Tuned threshold (0.307) |
|---|---|---|
| Accuracy | 0.806 | 0.762 |
| Precision (churn) | 0.661 | 0.536 |
| Recall (churn) | 0.555 | **0.756** |
| F1 (churn) | 0.603 | **0.628** |
| AUC-ROC | 0.846 | 0.846 |

We compared Logistic Regression against Random Forest. LR won on every metric (Test AUC 0.846 vs. 0.826) and was selected for its superior interpretability and lighter deployment footprint. The Logistic Regression model serializes to ~2 KB versus a much heavier 100-tree Random Forest.

---

## Tech stack

- **Python 3.11+**
- **scikit-learn** for modeling
- **pandas, NumPy** for data preparation
- **joblib** for model serialization
- **Streamlit** for the dashboard UI
- **pymysql** for RDS MySQL connectivity (optional)
- **sqlite3** (Python stdlib) for the local database fallback

---

## License

MIT
