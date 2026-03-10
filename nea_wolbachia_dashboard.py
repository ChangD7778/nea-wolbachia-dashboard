import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="NEA Wolbachia Intervention Dashboard", layout="wide")

DATASET_ID = "d_ca168b2cb763640d72c4600a68f9909e"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"
CACHE_FILE = Path("singapore_dengue_raw_records.csv")

SEASONAL_PERIOD = 52
FIXED_ORDER = (1, 1, 1)
FIXED_SEASONAL_ORDER = (0, 1, 1, SEASONAL_PERIOD)
FORECAST_ALPHA = 0.2  # 80% interval
RISK_Z = 1.2816

WATCH_MONTHS = {4, 5, 6, 7}  # Apr-Jul


def _headers() -> dict:
    api_key = os.getenv("DATA_GOV_SG_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def request_json(url: str, params: dict | None = None, max_attempts: int = 10, timeout: int = 30) -> dict:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code == 429:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "").strip()
                wait_s = float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 120)
                time.sleep(wait_s)
                continue
            if resp.status_code in (500, 502, 503, 504):
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(min(2 ** attempt, 60))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_attempts:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"Request failed: {last_err}")


def fetch_all_records(resource_id: str) -> list[dict]:
    payload = request_json(DATASTORE_URL, params={"resource_id": resource_id, "limit": 50000, "offset": 0})
    if not payload.get("success"):
        raise RuntimeError("data.gov.sg datastore_search returned success=false")

    result = payload.get("result", {})
    total = int(result.get("total", 0))
    records = list(result.get("records", []))
    if len(records) >= total:
        return records

    offset = len(records)
    page_size = 5000
    while offset < total:
        payload = request_json(DATASTORE_URL, params={"resource_id": resource_id, "limit": page_size, "offset": offset})
        if not payload.get("success"):
            raise RuntimeError("Paged datastore_search returned success=false")
        page = payload.get("result", {}).get("records", [])
        if not page:
            break
        records.extend(page)
        offset += len(page)
        time.sleep(0.5)

    if not records:
        raise RuntimeError("No records returned from datastore_search")
    return records


def parse_epi_week_to_sunday(epi_week: str):
    match = re.fullmatch(r"(\d{4})-W(\d{1,2})", str(epi_week).strip())
    if not match:
        return pd.NaT
    year = int(match.group(1))
    week = int(match.group(2))
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    first_week_sunday = jan1 - pd.Timedelta(days=(jan1.weekday() + 1) % 7)
    return first_week_sunday + pd.Timedelta(weeks=week - 1)


def load_records() -> pd.DataFrame:
    # Keep fresh API-first with local fallback.
    try:
        records = fetch_all_records(DATASET_ID)
        raw_df = pd.DataFrame(records)
        raw_df.to_csv(CACHE_FILE, index=False)
        return raw_df
    except Exception:
        if CACHE_FILE.exists():
            return pd.read_csv(CACHE_FILE)
        raise


@st.cache_data(show_spinner=False)
def build_weekly_dengue_series(records_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"epi_week", "disease", "no._of_cases"}
    missing = required_cols.difference(records_df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")

    work = records_df.copy()
    work["no._of_cases"] = pd.to_numeric(work["no._of_cases"], errors="coerce").fillna(0)
    work = work[work["disease"].isin(["Dengue Fever", "Dengue Haemorrhagic Fever"])].copy()
    if work.empty:
        raise RuntimeError("No dengue rows found in source data")

    weekly = (
        work.groupby(["epi_week", "disease"], as_index=False)["no._of_cases"]
        .sum()
        .pivot(index="epi_week", columns="disease", values="no._of_cases")
        .fillna(0)
        .reset_index()
    )

    if "Dengue Fever" not in weekly.columns:
        weekly["Dengue Fever"] = 0
    if "Dengue Haemorrhagic Fever" not in weekly.columns:
        weekly["Dengue Haemorrhagic Fever"] = 0

    weekly["week_start"] = weekly["epi_week"].map(parse_epi_week_to_sunday)
    weekly = weekly.dropna(subset=["week_start"]).sort_values("week_start").reset_index(drop=True)

    weekly["Total Dengue Cases"] = weekly["Dengue Fever"] + weekly["Dengue Haemorrhagic Fever"]
    weekly["moving_avg_12w"] = weekly["Total Dengue Cases"].rolling(12, min_periods=1).mean()

    checks = []
    checks.append({"check": "non_empty", "value": int(len(weekly)), "status": "PASS" if len(weekly) > 0 else "FAIL"})
    checks.append({"check": "duplicate_week_start", "value": int(weekly["week_start"].duplicated().sum()), "status": "PASS"})
    checks.append({"check": "negative_cases", "value": int((weekly["Total Dengue Cases"] < 0).sum()), "status": "PASS"})

    return weekly[["week_start", "Dengue Fever", "Dengue Haemorrhagic Fever", "Total Dengue Cases", "moving_avg_12w"]], pd.DataFrame(checks)


@st.cache_data(show_spinner=False)
def stl_diagnostics(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    ts = pd.Series(weekly_df["Total Dengue Cases"].values, index=weekly_df["week_start"])
    stl = STL(ts, period=SEASONAL_PERIOD, robust=True).fit()

    seasonal_strength = max(0.0, 1.0 - (np.var(stl.resid) / np.var(stl.seasonal + stl.resid)))

    trend_series = pd.Series(stl.trend, index=weekly_df["week_start"]).astype(float)
    recent_trend = trend_series.dropna().tail(6)
    trend_slope = float((recent_trend.iloc[-1] - recent_trend.iloc[0]) / max(1, len(recent_trend) - 1))

    stl_df = pd.DataFrame(
        {
            "week_start": weekly_df["week_start"].to_numpy(),
            "observed": ts.to_numpy(),
            "trend": np.asarray(stl.trend),
            "seasonal": np.asarray(stl.seasonal),
            "resid": np.asarray(stl.resid),
        }
    )

    diag_df = pd.DataFrame(
        [
            {"metric": "seasonal_period_weeks", "value": SEASONAL_PERIOD},
            {"metric": "seasonal_strength", "value": round(float(seasonal_strength), 4)},
            {"metric": "recent_trend_slope_per_week", "value": round(trend_slope, 4)},
        ]
    )
    return stl_df, diag_df, seasonal_strength, trend_slope


@st.cache_data(show_spinner=False)
def fit_fixed_sarima_and_forecast(weekly_df: pd.DataFrame, steps: int, alpha: float = FORECAST_ALPHA):
    y = weekly_df["Total Dengue Cases"].astype(float)
    sm_model = SARIMAX(
        y,
        order=FIXED_ORDER,
        seasonal_order=FIXED_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="lbfgs", maxiter=30, disp=False)

    forecast = sm_model.get_forecast(steps=steps)
    conf = forecast.conf_int(alpha=alpha)

    last_date = weekly_df["week_start"].iloc[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-SUN")

    fcst_df = pd.DataFrame(
        {
            "week_start": idx,
            "predicted_cases": forecast.predicted_mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }
    )
    return fcst_df


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return norm.cdf(x)


@st.cache_data(show_spinner=False)
def assign_risk_bands(fcst_df: pd.DataFrame, threshold: float, low_cut: float, high_cut: float) -> pd.DataFrame:
    out = fcst_df.copy()
    sigma = (out["upper"] - out["lower"]).abs() / (2 * RISK_Z)
    sigma = sigma.replace(0, np.nan).fillna(1.0)
    z = (threshold - out["predicted_cases"]) / sigma
    p_exceed = 1 - normal_cdf(z.to_numpy())
    out["p_exceed_threshold"] = np.clip(p_exceed, 0.0, 1.0)

    out["risk_band"] = np.select(
        [out["p_exceed_threshold"] < low_cut, out["p_exceed_threshold"] < high_cut],
        ["Low", "Medium"],
        default="High",
    )
    return out


def intervention_recommendation(risk_df: pd.DataFrame, trend_slope: float, horizon_mode: int, watch_enabled: bool, threshold: float) -> dict:
    decision_window = risk_df.head(min(4, horizon_mode)).copy()
    high_mask = decision_window["risk_band"].eq("High")

    month_now = pd.Timestamp.today().month
    in_watch_window = month_now in WATCH_MONTHS if watch_enabled else False

    baseline = bool(high_mask.any() and trend_slope > 0)
    medium_or_high = decision_window["risk_band"].isin(["Medium", "High"]).any()
    adjusted = bool(baseline or (in_watch_window and medium_or_high and trend_slope > 0))

    if adjusted and not decision_window.empty:
        week = decision_window[decision_window["risk_band"].isin(["High", "Medium"])]["week_start"].iloc[0]
        conf = float(decision_window["p_exceed_threshold"].max())
        reason = "Intervene: risk signal + positive trend"
        if in_watch_window and not baseline:
            reason = "Pre-season readiness trigger: Medium/High risk + positive trend"
        action = "Intervene"
    else:
        week = pd.NaT
        conf = float(decision_window["p_exceed_threshold"].max()) if not decision_window.empty else 0.0
        reason = "Monitor: no qualifying trigger in next 2-4 weeks"
        action = "Monitor"

    return {
        "action": action,
        "recommended_week": week,
        "confidence": conf,
        "reason": reason,
        "current_risk": decision_window["risk_band"].iloc[0] if not decision_window.empty else "Low",
        "in_watch_window": in_watch_window,
        "threshold": threshold,
    }


def plot_history(weekly_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["Total Dengue Cases"], name="Weekly cases", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=weekly_df["week_start"], y=weekly_df["moving_avg_12w"], name="12-week MA", line=dict(width=2.8)))
    fig.update_layout(title="Historical Dengue Cases", xaxis_title="Week", yaxis_title="Cases", height=420)
    return fig


def plot_stl(stl_df: pd.DataFrame):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["observed"], name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["trend"], name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["seasonal"], name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=stl_df["week_start"], y=stl_df["resid"], name="Residual"), row=4, col=1)
    fig.update_layout(height=760, title="STL Decomposition")
    return fig


def plot_future_forecast(weekly_df: pd.DataFrame, risk_df: pd.DataFrame, threshold: float):
    hist = weekly_df.tail(52).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["Total Dengue Cases"], name="Actual (last 52 weeks)", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=risk_df["week_start"], y=risk_df["predicted_cases"], name="Forecast", line=dict(color="firebrick", dash="dash"), mode="lines+markers"))
    fig.add_trace(
        go.Scatter(
            x=list(risk_df["week_start"]) + list(risk_df["week_start"])[::-1],
            y=list(risk_df["upper"]) + list(risk_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(255,0,0,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast interval (80%)",
        )
    )
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}")

    color_map = {"Low": "rgba(0,180,0,0.10)", "Medium": "rgba(255,165,0,0.12)", "High": "rgba(255,0,0,0.12)"}
    for _, row in risk_df.iterrows():
        fig.add_vrect(
            x0=row["week_start"] - pd.Timedelta(days=3),
            x1=row["week_start"] + pd.Timedelta(days=3),
            fillcolor=color_map[row["risk_band"]],
            line_width=0,
            layer="below",
        )

    fig.update_layout(title="SARIMA Forecast (last 1 year context)", xaxis_title="Week", yaxis_title="Cases", height=480)
    return fig


def build_seasonal_profile(stl_df: pd.DataFrame) -> pd.DataFrame:
    tmp = stl_df.copy()
    tmp["month"] = pd.to_datetime(tmp["week_start"]).dt.month
    prof = tmp.groupby("month", as_index=False)["seasonal"].mean()
    prof["month_name"] = pd.to_datetime(prof["month"], format="%m").dt.strftime("%b")
    prof["is_peak_window"] = prof["month"].isin([5, 6, 7])
    return prof


def plot_seasonal_profile(profile_df: pd.DataFrame):
    colors = np.where(profile_df["is_peak_window"], "crimson", "steelblue")
    fig = go.Figure(go.Bar(x=profile_df["month_name"], y=profile_df["seasonal"], marker_color=colors, name="Avg STL seasonal"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Month-of-Year Seasonal Profile (STL)", xaxis_title="Month", yaxis_title="Average seasonal component", height=380)
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("NEA Controls")
horizon = st.sidebar.selectbox("Future forecast horizon (weeks)", [4, 8, 12], index=2)
threshold = st.sidebar.number_input("Outbreak threshold", min_value=50, max_value=1000, value=400, step=10)

sensitivity = st.sidebar.selectbox("Risk sensitivity", ["Conservative", "Balanced", "Aggressive"], index=1)
if sensitivity == "Conservative":
    low_cut, high_cut = 0.40, 0.70
elif sensitivity == "Aggressive":
    low_cut, high_cut = 0.20, 0.50
else:
    low_cut, high_cut = 0.30, 0.60

preseason_adjust = st.sidebar.toggle("Enable pre-season adjusted policy (Apr-Jul)", value=True)


# -----------------------------
# Pipeline
# -----------------------------
st.title("NEA Wolbachia Intervention Dashboard")
st.caption("SARIMA + STL dashboard for Wolbachia intervention planning. Data source: Singapore MOH weekly dengue cases.")

try:
    with st.spinner("Loading latest dengue data and running SARIMA. This may take a while on first load..."):
        if "raw_df_session" not in st.session_state:
            st.session_state["raw_df_session"] = load_records()
        raw_df = st.session_state["raw_df_session"]

        weekly_df, qa_df = build_weekly_dengue_series(raw_df)
        stl_df, stl_diag_df, seasonal_strength, trend_slope_now = stl_diagnostics(weekly_df)

        forecast_df = fit_fixed_sarima_and_forecast(weekly_df, steps=horizon, alpha=FORECAST_ALPHA)
        risk_df = assign_risk_bands(forecast_df, threshold=threshold, low_cut=low_cut, high_cut=high_cut)

        rec = intervention_recommendation(
            risk_df,
            trend_slope=trend_slope_now,
            horizon_mode=horizon,
            watch_enabled=preseason_adjust,
            threshold=threshold,
        )

        seasonal_profile_df = build_seasonal_profile(stl_df)
except Exception as exc:
    st.error(f"Failed to load dashboard data/model pipeline: {exc}")
    st.stop()


# -----------------------------
# Header cards
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current risk band", rec["current_risk"])
col2.metric("Recommended action", rec["action"])
col3.metric("Model confidence", f"{rec['confidence']:.2f}")

st.info(
    f"STL seasonal strength={seasonal_strength:.2f} | "
    f"Current trend slope={trend_slope_now:.2f} | SARIMA={FIXED_ORDER}x{FIXED_SEASONAL_ORDER}"
)
st.write(rec["reason"])


# -----------------------------
# Presentation flow
# -----------------------------
st.markdown("## 1) Weekly Dengue Cases")
st.plotly_chart(plot_history(weekly_df), use_container_width=True)

st.markdown("## 2) STL Decomposition")
st.plotly_chart(plot_stl(stl_df), use_container_width=True)

st.markdown("## 3) SARIMA Forecast (Last 1 Year Context)")
st.plotly_chart(plot_future_forecast(weekly_df, risk_df, threshold=threshold), use_container_width=True)

st.markdown("## 4) Seasonal Profile (Operational Insight)")
st.plotly_chart(plot_seasonal_profile(seasonal_profile_df), use_container_width=True)

with st.expander("QA + Diagnostics"):
    st.write("### STL diagnostics")
    st.dataframe(stl_diag_df, use_container_width=True)
    st.write("### QA checks")
    st.dataframe(qa_df, use_container_width=True)

with st.expander("Appendix: How to Read This Dashboard"):
    st.markdown("""
### 1) What is a risk signal?
A risk signal is the model-estimated probability that weekly dengue cases will exceed the selected threshold.

### 2) How risk is quantified
- `p_exceed_threshold`: probability that forecasted weekly cases exceed the threshold (default 150).
- Risk bands:
  - Low: probability < lower cutoff
  - Medium: lower cutoff <= probability < upper cutoff
  - High: probability >= upper cutoff
- Default (Balanced) cutoffs:
  - Low < 0.30
  - Medium 0.30 to < 0.60
  - High >= 0.60

### 3) Recommendation logic shown in the header
The recommendation combines:
- forecast risk in the next few weeks, and
- STL trend slope direction.

### 4) How to read each chart
- Weekly Dengue Cases: historical level and 12-week moving average.
- STL Decomposition: observed, trend, seasonal, and residual components.
- SARIMA Forecast: next-week projections with confidence interval and threshold line.
- Seasonal Profile: average STL seasonal component by month (May/June highlighted).

### 5) Important note on dates
If your source data is historical (for example ending in 2022), all recommendations are relative to that last available week.
""")

# Run:
# & C:/Users/davin/anaconda3/python.exe -m streamlit run "C:/Users/davin/OneDrive/Documents/Python stuffs/nea_wolbachia_dashboard.py"



