import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import timedelta
from pathlib import Path

st.set_page_config(page_title="Sensor Data Dashboard", layout="wide")

# ====================== Helpers for A.5.3 / A.5.4 / CRNS→VWC ======================

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

def apply_crns_corrections_avg(df, counts_avg_col="counts_avg",
                               P_col="pressure", T_col="temperature", RH_col="humidity",
                               nmdb_col=None, P_ref=None, beta_inv_kpa=0.735, alpha=0.0054, v_ref=0.0, N_ref=None):
    """
    A.5.3: N = N' * fp * fv * fi   (applied to counts_avg scale for dashboard continuity)
      fp = exp((P - P_ref)/(1/β))       with (1/β)=beta_inv_kpa (kPa)
      fv = 1 + α*(v - v_ref)            v = absolute humidity (g/m^3)
      fi = N_ref / NMDB(t)               (optional)
    """
    out = df.copy()
    if counts_avg_col not in out.columns:
        return out
    base = pd.to_numeric(out[counts_avg_col], errors="coerce")

    # Pressure factor
    if P_col in out.columns:
        P = pd.to_numeric(out[P_col], errors="coerce")
        if P_ref is None or not np.isfinite(P_ref):
            P_ref = np.nanmean(P)
        fp = np.exp((P - P_ref) / beta_inv_kpa)
    else:
        fp = 1.0

    # Water-vapor factor
    if (T_col in out.columns) and (RH_col in out.columns):
        v = absolute_humidity_gm3(out[T_col], out[RH_col])
        fv = 1.0 + alpha * (v - v_ref)
    else:
        fv = 1.0

    # Cosmic factor
    if nmdb_col and nmdb_col in out.columns:
        Nm = pd.to_numeric(out[nmdb_col], errors="coerce")
        if N_ref is None or not np.isfinite(N_ref):
            N_ref = np.nanmedian(Nm)
        fi = N_ref / Nm.replace(0, np.nan)
    else:
        fi = 1.0

    out["counts_corr"] = base * fp * fv * fi
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

def estimate_cadence_minutes(ts):
    if ts is None or len(ts) < 3:
        return 5.0
    dt = pd.Series(ts).sort_values().diff().median()
    try:
        return float(dt.total_seconds() / 60.0)
    except Exception:
        return 5.0

def theta_total_from_counts(N, N0, a0=0.0808, a1=0.372, a2=0.115):
    """Appendix shape function → total water equivalent (treated as volumetric-like)."""
    ratio = pd.to_numeric(N, errors="coerce") / max(float(N0), 1e-9)
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    return (a0 / ratio) - a1 - a2

def clip_by_porosity(vwc, porosity_frac=0.5):
    v = pd.to_numeric(vwc, errors="coerce")
    return np.minimum(v, porosity_frac)

def choose_timestamp_column(df):
    for cand in ["Timestamp", "timestamp_parsed", "timestamp", "time", "ts", "date"]:
        if cand in df.columns:
            return cand
    return None

# ====================== Sidebar: File Selection (kept) ======================

st.sidebar.header("Options")
file_option = st.sidebar.selectbox("Select built-in data file:", [
    "LiFo_05_13_2025__13_17.csv", "LiFo_06_20_2025__17_17.csv"
])
uploaded = st.sidebar.file_uploader("Or upload your own CSV", type="csv")

# ====================== Load Data (kept) ======================

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Custom file uploaded and loaded!")
else:
    # allow reading built-in files from local dir if present
    fpath = Path(file_option)
    df = pd.read_csv(fpath if fpath.exists() else file_option)
    st.info(f"Loaded built-in file: `{file_option}`")

# ====================== Preprocessing (kept + tiny hardening) ======================

ts_col = choose_timestamp_column(df) or "Timestamp"
df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
df = df.dropna(subset=[ts_col])

# Clean and standardize known columns
for col in ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1', 'counts', 'counts_sum']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'pressure' in df.columns:
    df['pressure'] = df['pressure'].interpolate()

# Compute counts_avg (kept) and counts_sum if not present
if 'counts_0' in df.columns and 'counts_1' in df.columns:
    df = df.dropna(subset=['counts_0', 'counts_1'])
    df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2
    if 'counts_sum' not in df.columns:
        df['counts_sum'] = df['counts_0'] + df['counts_1']
elif 'counts_sum' in df.columns:
    df['counts_avg'] = df['counts_sum'] / 2
elif 'counts' in df.columns:
    df['counts_avg'] = df['counts']

# Detect Optional Soil Columns (kept)
soil_cols = {
    'Soil Moisture Value': 'soil_moisture_value',
    'Soil Moisture (%)': 'soil_moisture_pct',
    'Soil Temperature (°C)': 'soil_temp'
}
for old_col, new_col in soil_cols.items():
    if old_col in df.columns:
        df[new_col] = pd.to_numeric(df[old_col], errors='coerce')

# ====================== Sidebar Filters (kept) ======================

min_date = df[ts_col].min().date()
max_date = df[ts_col].max().date()
if min_date == max_date:
    min_date -= timedelta(days=1)
    max_date += timedelta(days=1)

date_range = st.sidebar.date_input("Select date range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df[ts_col].dt.date >= start_date) & (df[ts_col].dt.date <= end_date)]
else:
    st.error("Please select a valid start and end date.")
    st.stop()

# ====================== Variable & Graph Settings (kept, extended) ======================

base_vars = ['counts_avg', 'temperature', 'humidity', 'pressure']
# will inject new series after corrections below
extra_vars = [v for v in ['soil_moisture_value', 'soil_moisture_pct', 'soil_temp'] if v in df.columns]

threshold = st.sidebar.slider("Maximum count value to include:", min_value=0, max_value=200, value=150)
window = st.sidebar.slider("Smoothing window (in points):", min_value=1, max_value=300, value=12)
plot_mode = st.sidebar.selectbox("Graph Mode", ["% Change (Smoothed)", "Raw Counts (Smoothed)", "Both"])
detect_anomalies = st.sidebar.checkbox("Detect Anomalies in Counts Avg")
show_corr = st.sidebar.checkbox("Show Correlation Matrix")

# Filter outliers (kept)
if 'counts_0' in df.columns and 'counts_1' in df.columns:
    df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]
if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ====================== NEW: A.5.3 Corrections & A.5.4 SG Smoothing ======================

st.sidebar.markdown("---")
st.sidebar.subheader("Appendix A.5 Options")
apply_corr = st.sidebar.checkbox("Apply corrections (fp·fv·fi)", value=True)
P_ref_default = float(df['pressure'].mean()) if 'pressure' in df.columns else 101.3
P_ref = st.sidebar.number_input("P_ref (kPa)", value=P_ref_default, help="Pressure reference for fp")
beta_inv = st.sidebar.number_input("1/β (kPa)", value=0.735, help="Pressure factor parameter")
alpha = st.sidebar.number_input("α (water-vapor factor)", value=0.0054)
nmdb_choices = ["(none)"] + [c for c in df.columns if "nmdb" in c.lower()]
nmdb_col = st.sidebar.selectbox("Cosmic intensity (NMDB) column", nmdb_choices, index=0)
nmdb_col = None if nmdb_col == "(none)" else nmdb_col

# Apply corrections to counts_avg scale for dashboard continuity
work = df.copy()
if apply_corr and 'counts_avg' in work.columns:
    work = apply_crns_corrections_avg(
        work, counts_avg_col="counts_avg",
        P_col="pressure", T_col="temperature", RH_col="humidity",
        nmdb_col=nmdb_col, P_ref=P_ref, beta_inv_kpa=beta_inv, alpha=alpha
    )
else:
    work['counts_corr'] = work.get('counts_avg', pd.Series(dtype=float))

# A.5.4: SG window in HOURS → points via cadence
cad_min = estimate_cadence_minutes(work[ts_col])
sg_hours = st.sidebar.slider("SG window (hours)", 3, 15, 11, help="If SciPy not available, uses centered rolling median")
sg_points = max(3, int(np.ceil((sg_hours*60) / max(cad_min, 1e-6))))
work['counts_sg'] = smooth_series(work['counts_corr'] if apply_corr else work['counts_avg'], sg_points)

# Extend variables with the new series
if 'counts_corr' in work.columns:
    base_vars = ['counts_avg', 'counts_corr', 'counts_sg', 'temperature', 'humidity', 'pressure']
else:
    base_vars = ['counts_avg', 'counts_sg', 'temperature', 'humidity', 'pressure']

all_vars = base_vars + extra_vars
variables = st.sidebar.multiselect("Select variables to plot:", all_vars, default=['counts_avg'])

# Scale factors (kept)
scale_factors = {}
for var in variables:
    scale_factors[var] = st.sidebar.number_input(f"Scale factor for % Change – {var}", min_value=0.1, max_value=20.0, value=1.0, step=0.1)

# ====================== Anomaly Detection (kept) ======================

if detect_anomalies and 'counts_avg' in work.columns:
    model = IsolationForest(contamination=0.02, random_state=42)
    work['anomaly'] = model.fit_predict(work[['counts_avg']].dropna())
    anomalies = work[work['anomaly'] == -1]
else:
    anomalies = pd.DataFrame(columns=work.columns)

# ====================== Process Variables (% change + raw smooth) (kept) ======================

plot_df = pd.DataFrame({ts_col: work[ts_col]})
for col in variables:
    if col not in work.columns:
        continue
    mean_val = work[col].mean()
    scale = scale_factors[col]
    work[f'{col}_pct'] = ((work[col] - mean_val) / mean_val * 100) * scale
    work[f'{col}_pct_smooth'] = work[f'{col}_pct'].rolling(window=window).mean()
    work[f'{col}_smooth'] = work[col].rolling(window=window).mean()
    plot_df[f'{col}_pct_smooth'] = work[f'{col}_pct_smooth']
    plot_df[f'{col}_smooth'] = work[f'{col}_smooth']

# ====================== Dashboard Title (kept) ======================

st.title("Neutron Sensor Dashboard")
st.markdown("Visualize neutron counts, environment, and soil data with anomaly detection — now with Appendix A.5 corrections, SG smoothing, and CRNS→VWC retrieval.")

# ====================== Plots: % Change & Raw (kept) ======================

if plot_mode in ["% Change (Smoothed)", "Both"]:
    st.subheader("Smoothed % Change Over Time")
    fig = px.line()
    for col in variables:
        fig.add_scatter(x=plot_df[ts_col], y=plot_df[f'{col}_pct_smooth'],
                        mode='lines', name=col.replace('_', ' ').title())
    fig.update_layout(title="Smoothed % Change (%)", xaxis_title="Time", yaxis_title="% Change")
    st.plotly_chart(fig, use_container_width=True)

if plot_mode in ["Raw Counts (Smoothed)", "Both"]:
    st.subheader("Raw Counts (Smoothed)")
    fig2 = px.line()
    for col in variables:
        fig2.add_scatter(x=plot_df[ts_col], y=plot_df[f'{col}_smooth'],
                         mode='lines', name=col.replace('_', ' ').title())
    if detect_anomalies and 'counts_avg' in variables and not anomalies.empty:
        fig2.add_scatter(x=anomalies[ts_col], y=anomalies['counts_avg'],
                         mode='markers', marker=dict(color='red', size=8, symbol='x'),
                         name="Anomaly")
    fig2.update_layout(title="Smoothed Raw Values", xaxis_title="Time", yaxis_title="Raw Values")
    st.plotly_chart(fig2, use_container_width=True)

# ====================== NEW: Corrections Overview Plot ======================

with st.expander("Appendix A.5 Corrections & Smoothing (Overview)"):
    figc = go.Figure()
    if 'counts_avg' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_avg'], mode="lines", name="Counts (avg, raw)"))
    if 'counts_corr' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_corr'], mode="lines", name="Counts (avg, corrected)"))
    if 'counts_sg' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_sg'], mode="lines", name="Counts (avg, corrected + SG)"))
    figc.update_layout(height=420, xaxis_title="Time", yaxis_title="Counts (avg scale)")
    st.plotly_chart(figc, use_container_width=True)

# ====================== NEW: CRNS → Soil Moisture (VWC) ======================

st.sidebar.markdown("---")
st.sidebar.subheader("CRNS → VWC (Shape Function)")
N0_mode = st.sidebar.radio("N0 selection", ["Slider", "Auto-fit to in-situ (if available)"], index=0)
N0_default = float(work['counts_sg'].quantile(0.9)) if 'counts_sg' in work.columns else float(work['counts_avg'].quantile(0.9))
N0_slider = st.sidebar.number_input("N0 (dry-soil count rate)", value=N0_default)
porosity = st.sidebar.slider("Porosity cap (fraction)", 0.1, 0.8, 0.51)

# Pick counts series to convert → VWC: corrected+SG if available, else counts_avg
counts_for_vwc = work['counts_sg'] if 'counts_sg' in work.columns else work['counts_avg']

# Auto-fit N0 if asked and we have in-situ series
in_situ_candidates = [c for c in ['soil_moisture_value', 'soil_moisture_pct'] if c in work.columns]
N0_used = N0_slider
if N0_mode == "Auto-fit to in-situ (if available)" and len(in_situ_candidates) > 0:
    tgt_col = in_situ_candidates[0]
    tgt = work[tgt_col].copy()
    if tgt_col.endswith("_pct"):
        tgt = tgt / 100.0
    ok = counts_for_vwc.notna() & tgt.notna()
    if ok.sum() > 10:
        qs = np.linspace(0.7, 0.98, 12)
        candidates = np.quantile(counts_for_vwc.loc[ok], qs)
        best, best_mse = None, np.inf
        for n0 in candidates:
            theta = theta_total_from_counts(counts_for_vwc.loc[ok], n0)
            mse = np.nanmean((theta - tgt.loc[ok])**2)
            if mse < best_mse:
                best, best_mse = n0, mse
        if best is not None and np.isfinite(best):
            N0_used = float(best)

work['theta_total'] = theta_total_from_counts(counts_for_vwc, N0_used)
work['vwc_crns'] = work['theta_total']
work['vwc_crns_clipped'] = clip_by_porosity(work['vwc_crns'], porosity)

# Time-series overlay with in-situ if present
if len(in_situ_candidates) > 0:
    st.subheader("Soil Moisture (CRNS vs In-situ)")
    figv = go.Figure()
    figv.add_trace(go.Scatter(x=work[ts_col], y=work['vwc_crns_clipped'], mode="lines", name="VWC (CRNS, clipped)"))
    for col in in_situ_candidates:
        y = work[col] if not col.endswith("_pct") else work[col]/100.0
        figv.add_trace(go.Scatter(x=work[ts_col], y=y, mode="lines", name=f"{col} (as fraction)" if col.endswith("_pct") else col))
    figv.update_layout(height=380, xaxis_title="Time", yaxis_title="VWC (m³/m³)")
    st.plotly_chart(figv, use_container_width=True)

    # Scatter + fit vs first in-situ
    tgt_col = in_situ_candidates[0]
    tgt = work[tgt_col] if not tgt_col.endswith("_pct") else work[tgt_col]/100.0
    ok2 = pd.concat([work['vwc_crns_clipped'], tgt], axis=1).dropna()
    if len(ok2) > 2:
        m, b = np.polyfit(ok2['vwc_crns_clipped'], ok2.iloc[:,1], 1)
        xs = np.linspace(ok2['vwc_crns_clipped'].min(), ok2['vwc_crns_clipped'].max(), 100)
        ys = m*xs + b
        r = np.corrcoef(ok2['vwc_crns_clipped'], ok2.iloc[:,1])[0,1]
        figs = go.Figure()
        figs.add_trace(go.Scatter(x=ok2['vwc_crns_clipped'], y=ok2.iloc[:,1], mode="markers", name="Samples"))
        figs.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (R²={r**2:.2f})"))
        figs.update_layout(height=360, xaxis_title="VWC (CRNS, clipped)", yaxis_title=f"VWC ({tgt_col})")
        st.plotly_chart(figs, use_container_width=True)

# ====================== Summary Stats (kept) ======================

st.subheader("Summary Statistics")
summary_cols = [c for c in variables if c in work.columns]
st.dataframe(work[summary_cols].describe())

# ====================== Correlation Matrix (kept) ======================

if show_corr:
    st.subheader("Correlation Matrix")
    corr_fig, ax = plt.subplots(figsize=(6, 4))
    # use only selected variables that exist
    sel = [c for c in variables if c in work.columns]
    if len(sel) >= 2:
        sns.heatmap(work[sel].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(corr_fig)
    else:
        st.info("Select at least two variables to show a correlation matrix.")

# ====================== Download processed data (new) ======================

st.subheader("Download Processed Data")
out_cols = list(dict.fromkeys([ts_col] + all_vars + [f"{v}_pct_smooth" for v in variables] + [f"{v}_smooth" for v in variables]
                              + ["counts_corr", "counts_sg", "theta_total", "vwc_crns", "vwc_crns_clipped"]))
csv_bytes = work[out_cols].to_csv(index=False).encode("utf-8", errors="ignore")
st.download_button("Download CSV (with corrections, smoothing, VWC)", data=csv_bytes, file_name="processed_crns.csv", mime="text/csv")
