import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import timedelta
from pathlib import Path

st.set_page_config(page_title="CRNS Soil Moisture Dashboard (A.5.3/A.5.4)", layout="wide")

# Utils

def absolute_humidity_gm3(temp_C, rh_pct):
    """Approx absolute humidity (g/m^3) from T (°C) and RH (%)."""
    try:
        temp_C = pd.to_numeric(temp_C, errors="coerce")
        rh_pct = pd.to_numeric(rh_pct, errors="coerce")
        es = 6.112 * np.exp((17.67*temp_C) / (temp_C + 243.5))  # hPa
        e = (rh_pct/100.0) * es
        return 216.7 * e / (temp_C + 273.15)
    except Exception:
        return pd.Series(np.nan, index=temp_C.index if hasattr(temp_C, "index") else None)

def apply_crns_corrections(df, counts_col, P_col=None, T_col=None, RH_col=None,
                           nmdb_col=None, P_ref=None, beta_inv_kpa=0.735,
                           alpha=0.0054, v_ref=0.0, N_ref=None):
    """
    Appendix A.5.3: N = N' * fp * fv * fi
      fp = exp((P - P_ref)/(1/β))   with (1/β)=beta_inv_kpa (kPa)
      fv = 1 + α*(v - v_ref)        v = absolute humidity (g/m^3)
      fi = N_ref / NMDB(t)          cosmic-ray intensity normalization (optional)
    """
    out = df.copy()
    Np = pd.to_numeric(out[counts_col], errors="coerce")

    # Pressure factor
    if P_col and P_col in out.columns:
        P = pd.to_numeric(out[P_col], errors="coerce")
        if P_ref is None or not np.isfinite(P_ref):
            P_ref = np.nanmean(P)
        fp = np.exp((P - P_ref) / beta_inv_kpa)
    else:
        fp = 1.0

    # Water-vapor factor
    if T_col and T_col in out.columns and RH_col and RH_col in out.columns:
        v = absolute_humidity_gm3(out[T_col], out[RH_col])
        fv = 1.0 + alpha * (v - v_ref)
    else:
        fv = 1.0

    # Cosmic intensity factor
    if nmdb_col and nmdb_col in out.columns:
        Nm = pd.to_numeric(out[nmdb_col], errors="coerce")
        if N_ref is None or not np.isfinite(N_ref):
            N_ref = np.nanmedian(Nm)
        fi = N_ref / Nm.replace(0, np.nan)
    else:
        fi = 1.0

    out["counts_corr"] = Np * fp * fv * fi
    return out

def smooth_series(x, points):
    """Savitzky–Golay if SciPy available; else centered rolling median."""
    try:
        from scipy.signal import savgol_filter
        w = int(points)
        if w % 2 == 0:
            w += 1
        if w < 3:
            w = 3
        y = savgol_filter(pd.to_numeric(x, errors="coerce").values, w, 2)
        return pd.Series(y, index=x.index)
    except Exception:
        w = max(3, int(points))
        return pd.to_numeric(x, errors="coerce").rolling(w, center=True, min_periods=w//2).median()

def theta_total_from_counts(N, N0, a0=0.0808, a1=0.372, a2=0.115):
    """Appendix shape function -> total water equivalent (treated as volumetric-like)."""
    ratio = pd.to_numeric(N, errors="coerce") / max(float(N0), 1e-9)
    # avoid divide-by-zero
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    return (a0 / ratio) - a1 - a2

def clip_by_porosity(vwc, porosity_frac=0.5):
    v = pd.to_numeric(vwc, errors="coerce")
    return np.minimum(v, porosity_frac)

def estimate_cadence_minutes(ts):
    if ts is None or len(ts) < 3:
        return 5.0
    dt = pd.Series(ts).sort_values().diff().median()
    try:
        return float(dt.total_seconds() / 60.0)
    except Exception:
        return 5.0

def choose_timestamp_column(df):
    for cand in ["Timestamp", "timestamp_parsed", "timestamp", "time", "ts", "date"]:
        if cand in df.columns:
            return cand
    return None

# ====================== Data loading ======================

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Custom file uploaded and loaded.")
else:
    # Try to load a default combined file if present
    default_path = Path("combined_field_station_data.csv")
    if default_path.exists():
        df = pd.read_csv(default_path)
        st.info(f"Loaded default file: `{default_path}` (upload a CSV to override).")
    else:
        st.warning("Please upload a CSV to begin.")
        st.stop()

# Timestamp parsing
ts_col = choose_timestamp_column(df)
if ts_col is None:
    st.error("No timestamp column found. Expected one of: Timestamp, timestamp_parsed, timestamp, time, ts, date.")
    st.stop()

df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

# Standardize numeric columns (keep originals)
for c in ["temperature", "humidity", "pressure", "counts_0", "counts_1", "counts", "counts_sum",
          "soil_moisture", "soil_moisture_pct", "soil_temp",
          "sm_1in", "sm_3in", "SM_1in", "SM_3in"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute counts_sum / counts_avg
if "counts_sum" not in df.columns:
    if "counts_0" in df.columns and "counts_1" in df.columns:
        df["counts_sum"] = df["counts_0"] + df["counts_1"]
    elif "counts" in df.columns:
        df["counts_sum"] = df["counts"]

if "counts_avg" not in df.columns and "counts_sum" in df.columns:
    if "counts_0" in df.columns and "counts_1" in df.columns:
        df["counts_avg"] = df["counts_sum"] / 2.0

# Map potential in-situ columns into a consistent set for plotting
in_situ_cols = []
for cand in ["soil_moisture", "soil_moisture_pct", "sm_1in", "sm_3in", "SM_1in", "SM_3in"]:
    if cand in df.columns:
        in_situ_cols.append(cand)

# Sidebar Controls

st.sidebar.header("Filters")
min_date = df[ts_col].min().date()
max_date = df[ts_col].max().date()
if min_date == max_date:
    min_date, max_date = min_date - timedelta(days=1), max_date + timedelta(days=1)

date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df[ts_col].dt.date >= start_date) & (df[ts_col].dt.date <= end_date)
    df = df[mask].reset_index(drop=True)
else:
    st.error("Please select a valid start and end date.")
    st.stop()

threshold = st.sidebar.slider("Max value to include (per channel)", min_value=0, max_value=200, value=150)
if "counts_0" in df.columns and "counts_1" in df.columns:
    df = df[(df["counts_0"] <= threshold) & (df["counts_1"] <= threshold)].reset_index(drop=True)

# Corrections + smoothing

st.sidebar.header("A.5.3 Corrections")
apply_corr = st.sidebar.checkbox("Apply corrections (fp·fv·fi)", value=True)
P_ref_default = float(df["pressure"].mean()) if "pressure" in df.columns else 101.3
P_ref = st.sidebar.number_input("P_ref (kPa)", value=P_ref_default, help="Reference pressure for fp")
beta_inv = st.sidebar.number_input("1/β (kPa)", value=0.735, help="Pressure factor parameter")
alpha = st.sidebar.number_input("α (water-vapor factor)", value=0.0054)
nmdb_options = ["(none)"] + [c for c in df.columns if "nmdb" in c.lower()]
use_nmdb = st.sidebar.selectbox("Cosmic intensity (NMDB) column", nmdb_options, index=0)
nmdb_col = None if use_nmdb == "(none)" else use_nmdb

st.sidebar.header("A.5.4 Smoothing")
cad_min = estimate_cadence_minutes(df[ts_col])
sg_hours = st.sidebar.slider("SG window (hours)", 3, 15, 11, help="If SciPy unavailable, uses centered rolling median")
points = int(np.ceil((sg_hours*60) / max(cad_min, 1e-6)))

# CRNS → VWC

st.sidebar.header("CRNS → Soil Moisture")
N0_mode = st.sidebar.radio("N0 selection", ["Slider", "Auto-fit to in-situ (if available)"], index=0)
N0_slider_default = float(df["counts_sum"].quantile(0.9)) if "counts_sum" in df.columns else 60.0
N0_slider = st.sidebar.number_input("N0 (dry-soil count rate)", value=N0_slider_default, help="Start near a high-quantile of counts")
porosity = st.sidebar.slider("Porosity cap (fraction)", 0.1, 0.8, 0.51)

# Core dataset with corrections

work = df.copy()
if "counts_sum" not in work.columns:
    st.error("No counts available (counts_sum / counts_0+counts_1 / counts).")
    st.stop()

# Corrections
if apply_corr:
    work = apply_crns_corrections(
        work, counts_col="counts_sum",
        P_col="pressure" if "pressure" in work.columns else None,
        T_col="temperature" if "temperature" in work.columns else None,
        RH_col="humidity" if "humidity" in work.columns else None,
        nmdb_col=nmdb_col,
        P_ref=P_ref, beta_inv_kpa=beta_inv, alpha=alpha
    )
    work["counts_for_moisture"] = work["counts_corr"]
else:
    work["counts_for_moisture"] = work["counts_sum"]

# Smoothing (SG or rolling)
work["counts_sg"] = smooth_series(work["counts_for_moisture"], max(points, 3))

# Auto-fit N0 if possible
N0_used = N0_slider
if N0_mode == "Auto-fit to in-situ (if available)":
    # pick first available in-situ column as target
    target_col = None
    for cand in ["soil_moisture", "soil_moisture_pct", "sm_1in", "sm_3in", "SM_1in", "SM_3in"]:
        if cand in work.columns:
            target_col = cand
            break
    if target_col:
        target = work[target_col]
        ok = target.notna() & work["counts_sg"].notna()
        if ok.sum() > 10:
            qs = np.linspace(0.7, 0.98, 12)
            candidates = np.quantile(work.loc[ok, "counts_sg"], qs)
            best = None
            best_mse = np.inf
            for n0 in candidates:
                theta = theta_total_from_counts(work.loc[ok, "counts_sg"], n0)
                # Treat theta_total as volumetric-like for a first look; if target is %, convert to fraction
                tgt = target.loc[ok]
                if target_col.endswith("_pct"):
                    tgt = tgt / 100.0
                mse = np.nanmean((theta - tgt)**2)
                if mse < best_mse:
                    best_mse = mse
                    best = n0
            if best is not None and np.isfinite(best):
                N0_used = float(best)

# Moisture retrieval + porosity clip
work["theta_total"] = theta_total_from_counts(work["counts_sg"], N0_used)
work["vwc_crns"] = work["theta_total"]
work["vwc_crns_clipped"] = clip_by_porosity(work["vwc_crns"], porosity)

# Dashboard

st.title("Neutron Sensor Dashboard — A.5.3/A.5.4 Enhancements")
st.caption("Raw vs corrected counts, Savitzky–Golay smoothing, and CRNS→VWC retrieval with validation.")

# Choose which counts series to use elsewhere
series_choice = st.radio("Counts series for general plots:", ["Raw (counts_sum)", "Corrected (counts_corr)", "Corrected + SG (counts_sg)"], index=2, horizontal=True)
if series_choice == "Raw (counts_sum)":
    series_name = "counts_sum"
elif series_choice == "Corrected (counts_corr)":
    series_name = "counts_corr" if "counts_corr" in work.columns else "counts_sum"
else:
    series_name = "counts_sg"

# Variables to plot (include derived series)
base_vars = [v for v in [series_name, "counts_sum", "counts_corr", "counts_sg", "temperature", "humidity", "pressure"] if v in work.columns]
extra_vars = [v for v in ["soil_moisture", "soil_moisture_pct", "soil_temp", "sm_1in", "sm_3in", "SM_1in", "SM_3in"] if v in work.columns]
variables = st.sidebar.multiselect("Variables to plot", base_vars + extra_vars, default=base_vars[:1])

# Anomaly detection on selected series
detect_anom = st.sidebar.checkbox("Detect anomalies (IsolationForest on selected counts series)", value=False)
anomalies = pd.DataFrame()
if detect_anom and series_name in work.columns:
    s = pd.to_numeric(work[series_name], errors="coerce")
    s_ok = s.dropna()
    if len(s_ok) > 20:
        model = IsolationForest(contamination=0.02, random_state=42)
        pred = model.fit_predict(s_ok.to_frame())
        anomalies = work.loc[s_ok.index].copy()
        anomalies["anomaly"] = pred
        anomalies = anomalies[anomalies["anomaly"] == -1]
    else:
        st.info("Not enough samples for anomaly detection; skipping.")

# ===== Figure 1: Counts raw vs corrected vs SG =====
st.subheader("Counts time series")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=work[ts_col], y=work["counts_sum"], mode="lines", name="Counts (raw)"))
if "counts_corr" in work.columns:
    fig1.add_trace(go.Scatter(x=work[ts_col], y=work["counts_corr"], mode="lines", name="Counts (corrected)"))
fig1.add_trace(go.Scatter(x=work[ts_col], y=work["counts_sg"], mode="lines", name="Counts (SG smoothed)"))
if not anomalies.empty and series_name in ["counts_sum", "counts_corr", "counts_sg"]:
    fig1.add_trace(go.Scatter(x=anomalies[ts_col], y=anomalies[series_name], mode="markers", name="Anomaly", marker=dict(symbol="x", size=8)))
fig1.update_layout(height=420, xaxis_title="Time", yaxis_title="Counts", legend_title="Series")
st.plotly_chart(fig1, use_container_width=True)

# ===== Figure 2: VWC (CRNS) vs in-situ time series =====
if len(in_situ_cols) > 0:
    st.subheader("Soil moisture comparison (CRNS vs in-situ)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=work[ts_col], y=work["vwc_crns_clipped"], mode="lines", name="VWC (CRNS, clipped)"))
    for col in in_situ_cols:
        y = work[col]
        name = col
        if col.endswith("_pct"):
            y = y / 100.0
            name = f"{col} (as fraction)"
        fig2.add_trace(go.Scatter(x=work[ts_col], y=y, mode="lines", name=name))
    fig2.update_layout(height=380, xaxis_title="Time", yaxis_title="VWC (m³/m³)")
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter + fit vs first in-situ column
    tgt_col = in_situ_cols[0]
    tgt = work[tgt_col]
    if tgt_col.endswith("_pct"):
        tgt = tgt / 100.0
    ok = pd.concat([work["vwc_crns_clipped"], tgt], axis=1).dropna()
    if len(ok) > 2:
        m, b = np.polyfit(ok["vwc_crns_clipped"], ok[tgt_col if not tgt_col.endswith("_pct") else 0], 1) if isinstance(ok.columns[1], int) else np.polyfit(ok["vwc_crns_clipped"], ok[tgt_col], 1)
        xs = np.linspace(ok["vwc_crns_clipped"].min(), ok["vwc_crns_clipped"].max(), 100)
        ys = m*xs + b
        r = np.corrcoef(ok["vwc_crns_clipped"], ok[ ok.columns[1] ])[0,1]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ok["vwc_crns_clipped"], y=ok[ ok.columns[1] ], mode="markers", name="Samples"))
        fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (R²={r**2:.2f})"))
        fig3.update_layout(height=380, xaxis_title="VWC (CRNS, clipped)", yaxis_title=f"VWC ({tgt_col})")
        st.plotly_chart(fig3, use_container_width=True)

# ===== Additional: Counts0 vs Counts1 + Bland–Altman =====
with st.expander("Channel agreement (counts_0 vs counts_1)"):
    if "counts_0" in work.columns and "counts_1" in work.columns:
        c0 = pd.to_numeric(work["counts_0"], errors="coerce")
        c1 = pd.to_numeric(work["counts_1"], errors="coerce")
        ok = c0.notna() & c1.notna()
        if ok.sum() > 2:
            r = np.corrcoef(c0[ok], c1[ok])[0,1]
        else:
            r = np.nan
        fig4 = px.scatter(x=c0, y=c1, labels={"x": "counts_0", "y": "counts_1"}, title=f"counts_0 vs counts_1 (r={r:.3f})")
        mn, mx = float(np.nanmin([c0.min(), c1.min()])), float(np.nanmax([c0.max(), c1.max()]))
        fig4.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash"), name="1:1"))
        st.plotly_chart(fig4, use_container_width=True)

        mean_vals = (c0 + c1) / 2.0
        diff_vals = (c1 - c0)
        md = float(np.nanmean(diff_vals)); sd = float(np.nanstd(diff_vals, ddof=1))
        fig5 = px.scatter(x=mean_vals, y=diff_vals, labels={"x":"Mean of channels", "y":"counts_1 - counts_0"},
                          title="Bland–Altman: agreement between channels")
        fig5.add_hline(md, line_dash="dash")
        fig5.add_hline(md + 1.96*sd, line_dash="dot")
        fig5.add_hline(md - 1.96*sd, line_dash="dot")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("counts_0/counts_1 not available.")

# ===== Summary & Downloads =====
st.subheader("Summary statistics")
summary_cols = [c for c in ["counts_sum", "counts_corr", "counts_sg", "temperature", "humidity", "pressure",
                            "vwc_crns", "vwc_crns_clipped"] + in_situ_cols if c in work.columns]
if summary_cols:
    st.dataframe(work[summary_cols].describe().T)

st.subheader("Download processed data")
csv_bytes = work.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV (with corrections & VWC)", data=csv_bytes, file_name="crns_processed.csv", mime="text/csv")
