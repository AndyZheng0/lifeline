# LifeLine — B2C Streamlit Dashboard

An editable, data-grounded dashboard shell based on the supplied **Customer_Churn_Dataset_Expanded (1).xlsx**. It uses a slate/violet design with responsive KPI cards, a customer signal carousel, chart tabs, and searchable customer accounts.

## Run in VS Code

1. Extract this project and open the `LifeLine_B2C_Dashboard` folder in VS Code.
2. Create a virtual environment (recommended): `python -m venv .venv` and activate it.
3. Install: `pip install -r requirements.txt`
4. Start: `streamlit run app.py`

The dataset is already included under `data/customer_churn.xlsx`.

## Connect the account assistant

Run Streamlit on a machine that can reach the Unsloth Decision API and the TrueNAS Open WebUI server over Tailscale. The browser talks to Streamlit, so the machine running Python needs this access even if the browser can already open TrueNAS.

1. Open `.env` beside `app.py`. If your checkout has only `.env.example`, copy it to `.env` first. Fill in the two base URLs and API keys. Include the port if applicable. Use the base address without `/v1` or `/api/chat/completions`. Git ignores `.env`.
2. Enter the exact model IDs. If you only know the keys and URLs, open **Model + Settings**. The model-list buttons show which IDs each key can access. Use the Unsloth Decision Model trained on the main workbook and a regular Open WebUI chat model.
3. Open **Accounts**, select a customer in **Account assistant**, and ask “Will this account churn?” Then try “Why did it give that score?” or “What if they had two fewer complaints?”

Unsloth returns the prediction and score for the selected account. Open WebUI interprets chat requests and helps explain comparisons. A what-if changes one input and shows the difference between two scores in percentage points. These tests describe how the model responds to an input change. Both services receive only the relevant account fields and neither receives `CustomerID` or the recorded `Churn` label from the workbook. Open WebUI also receives the questions typed in the account chat.

The app reads `.env` on each Streamlit rerun. Server environment variables take precedence. Existing Streamlit secrets still work as a fallback.

## Optional integrations

Edit **`integrations.py`** (and not the main dashboard layout):

- `get_predictions(customers)` → `None`, or DataFrame with `CustomerID`, `churn_probability` (0–1). Never use the actual `Churn` label as an inference feature.
- `get_ai_recommendations(segment_name, segment_stats)` → `None`, or list of `{'title': ..., 'reason': ...}` actions.
- `save_action(customer_id, action, status)` → `False` or your database write. *Currently the carousel saves a segment-level action and passes the segment name as the first argument; adapt this if you implement per-customer outreach.*

If you add batch predictions, the Accounts table gains a prediction column and is ordered by it. The account chat itself makes on-demand requests for the selected record.

## What is and is not real

- Dataset: **10,500 customers**, **1,286 recorded churn cases** (12.2%).
- Supported fields: CustomerID, Age, Gender, Region, ContractType, MonthlyCharges, SignupDate, AgeMissing, AvgCallMinutes, AvgDataUsageGB, AvgSMS, TotalComplaints, MonthsOfData, Churn.
- Four homepage KPIs are computed directly from the workbook, not invented. The fourth KPI sums monthly charges among customers marked churned; it is **not** a forecast of future revenue loss.
- Five **overlapping** customer groups use quartile cutoffs or Monthly contract type. They are descriptive cohorts, **not verified causes** or SHAP attributions.
- The churn ring shows the **observed historical churn rate** of the selected group, **not** prediction probability.
- Suggested actions are labeled rule-based examples until an AI agent is connected.
- No invented time trends or 'past 30 days' comparisons: the dataset is customer-level and does not provide repeated daily observations.
- The workflow is a UI starter, not yet a production CRM.

## Files

```
app.py                       Streamlit UI, filtering, charts, styles
integrations.py              <-- ML MODEL / AI AGENT / DATABASE hooks
.streamlit/config.toml       Native Streamlit theme
data/customer_churn.xlsx     Uploaded dataset
requirements.txt             Dependencies
README.md                    Instructions
```
# LifelineCustomerChurn
