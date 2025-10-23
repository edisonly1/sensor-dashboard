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
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Neutron Sensor Dashboard (A.5)", layout="wide")

# ====================== Utilities (A.5.3 / A.5.4 / CRNS→VWC) ======================

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
    A.5.3: N = N' * fp * fv * fi  (applied on counts_avg scale so the rest of the app still works)
      fp = exp((P - P_ref)/(1/β))   with (1/β)=beta_inv_kpa (kPa)
      fv = 1 + α*(v - v_ref)        v = absolute humidity (g/m^3)
      fi = N_ref / NMDB(t)          (optional)
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
    """Appendix shape function → total water equivalent (treated as volumetric-like here)."""
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

# ====================== Sidebar: Data (kept) ======================

st.sidebar.header("Data")
file_option = st.sidebar.selectbox(
    "Select built-in data file:",
    ["LiFo_05_13_2025__13_17.csv", "LiFo_06_20_2025__17_17.csv"]
)
uploaded = st.sidebar.file_uploader("Or upload your CSV", type=["csv"])

# ====================== Load Data (kept) ======================

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Custom file uploaded and loaded.")
else:
    path = Path(file_option)
    df = pd.read_csv(path if path.exists() else file_option)
    st.info(f"Loaded built-in file: `{file_option}`")

# ====================== Preprocess (kept + hardened) ======================

ts_col = choose_timestamp_column(df) or "Timestamp"
df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

# Numeric casts
for col in ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1', 'counts', 'counts_sum']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'pressure' in df.columns:
    df['pressure'] = df['pressure'].interpolate()

# counts_avg / counts_sum
if 'counts_0' in df.columns and 'counts_1' in df.columns:
    df = df.dropna(subset=['counts_0', 'counts_1'])
    df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2
    if 'counts_sum' not in df.columns:
        df['counts_sum'] = df['counts_0'] + df['counts_1']
elif 'counts_sum' in df.columns:
    df['counts_avg'] = df['counts_sum'] / 2
elif 'counts' in df.columns:
    df['counts_avg'] = df['counts']

# Optional in-situ columns
soil_map = {
    'Soil Moisture Value': 'soil_moisture_value',
    'Soil Moisture (%)': 'soil_moisture_pct',
    'Soil Temperature (°C)': 'soil_temp'
}
for old, new in soil_map.items():
    if old in df.columns:
        df[new] = pd.to_numeric(df[old], errors='coerce')

# ====================== Sidebar Filters (kept) ======================

min_date = df[ts_col].min().date()
max_date = df[ts_col].max().date()
if min_date == max_date:
    min_date -= timedelta(days=1)
    max_date += timedelta(days=1)

date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df[ts_col].dt.date >= start_date) & (df[ts_col].dt.date <= end_date)].reset_index(drop=True)
else:
    st.error("Please select a valid start and end date.")
    st.stop()

# ====================== Graph controls (kept) ======================

st.sidebar.markdown("### Smoothing")
window = st.sidebar.slider("Smoothing window (in points):", min_value=1, max_value=300, value=12)

plot_mode = st.sidebar.selectbox("Graph Mode", ["% Change (Smoothed)", "Raw Counts (Smoothed)", "Both"])
detect_anomalies = st.sidebar.checkbox("Detect Anomalies in Counts Avg")
show_corr = st.sidebar.checkbox("Show Correlation Matrix")

# ====================== Appendix A.5 (new) ======================

st.sidebar.markdown("---")
st.sidebar.subheader("Appendix A.5 Options")

# A.5.3 corrections
apply_corr = st.sidebar.checkbox("Apply corrections (fp·fv·fi)", value=True)
P_ref_default = float(df['pressure'].mean()) if 'pressure' in df.columns else 101.3
P_ref = st.sidebar.number_input("P_ref (kPa)", value=P_ref_default, help="Reference pressure for fp")
beta_inv = st.sidebar.number_input("1/β (kPa)", value=0.735, help="Pressure factor parameter")
alpha = st.sidebar.number_input("α (water-vapor factor)", value=0.0054)
nmdb_choices = ["(none)"] + [c for c in df.columns if "nmdb" in c.lower()]
nmdb_col = st.sidebar.selectbox("Cosmic intensity (NMDB) column", nmdb_choices, index=0)
nmdb_col = None if nmdb_col == "(none)" else nmdb_col

# A.5.4 smoothing
cad_min = estimate_cadence_minutes(df[ts_col])
sg_hours = st.sidebar.slider("SG window (hours)", 3, 15, 11, help="If SciPy unavailable, uses centered rolling median")
sg_points = max(3, int(np.ceil((sg_hours*60) / max(cad_min, 1e-6))))

# Build working copy and apply corrections/smoothing
work = df.copy()
if apply_corr and 'counts_avg' in work.columns:
    work = apply_crns_corrections_avg(
        work, counts_avg_col="counts_avg",
        P_col="pressure", T_col="temperature", RH_col="humidity",
        nmdb_col=nmdb_col, P_ref=P_ref, beta_inv_kpa=beta_inv, alpha=alpha
    )
else:
    work['counts_corr'] = work.get('counts_avg', pd.Series(dtype=float))

work['counts_sg'] = smooth_series(work['counts_corr'] if apply_corr else work['counts_avg'], sg_points)

# ====================== NEW: Max handling (visual control) ======================

st.sidebar.markdown("### Max handling")
y_max = st.sidebar.number_input("Max value:", value=100.0, min_value=0.0, step=1.0)
max_mode = st.sidebar.selectbox(
    "Use this max to…",
    ["Filter per-channel (current behavior)", "Filter plotted series", "Clamp plotted series", "Axis only"],
    index=0
)

# If user wants the *old* behavior (per-channel filtering), do it here (before plotting)
if max_mode == "Filter per-channel (current behavior)" and {"counts_0","counts_1"}.issubset(work.columns):
    work = work[(work["counts_0"] <= y_max) & (work["counts_1"] <= y_max)].reset_index(drop=True)

# ====================== Variables, scaling (kept, extended) ======================

base_vars = [v for v in ["counts_avg", "counts_corr", "counts_sg", "temperature", "humidity", "pressure"] if v in work.columns]
extra_vars = [v for v in ["soil_moisture_value", "soil_moisture_pct", "soil_temp"] if v in work.columns]
all_vars = base_vars + extra_vars

variables = st.sidebar.multiselect("Variables to plot", all_vars, default=[base_vars[0]] if base_vars else [])

scale_factors = {var: st.sidebar.number_input(f"Scale factor for % Change – {var}",
                                              min_value=0.1, max_value=20.0, value=1.0, step=0.1)
                 for var in variables}

# Pick which counts series drives general plots
series_choice = st.radio("Counts series for general plots:",
                         ["Raw (counts_avg)", "Corrected (counts_corr)", "Corrected + SG (counts_sg)"],
                         index=2, horizontal=True)
series_name = {"Raw (counts_avg)":"counts_avg", "Corrected (counts_corr)":"counts_corr",
               "Corrected + SG (counts_sg)":"counts_sg"}[series_choice]
series_name = series_name if series_name in work.columns else base_vars[0] if base_vars else None

# Apply the other three max modes *after* series selection/corrections
yaxis_range = None
if series_name is not None:
    s = work[series_name].astype(float)
    if max_mode == "Filter plotted series":
        keep = s.le(y_max) | s.isna()
        work = work.loc[keep].reset_index(drop=True)
    elif max_mode == "Clamp plotted series":
        work[series_name] = np.minimum(s, y_max)
    elif max_mode == "Axis only":
        yaxis_range = [0, y_max]

st.caption(f"Plotted series max (after handling): {work[series_name].max():.1f}" if series_name else "")

# ====================== Anomaly detection (kept) ======================

if detect_anomalies and series_name in work.columns:
    arr = pd.to_numeric(work[series_name], errors="coerce").dropna()
    if len(arr) > 20:
        model = IsolationForest(contamination=0.02, random_state=42)
        pred = model.fit_predict(arr.to_frame())
        anomalies = work.loc[arr.index].copy()
        anomalies["anomaly"] = pred
        anomalies = anomalies[anomalies["anomaly"] == -1]
    else:
        anomalies = pd.DataFrame(columns=work.columns)
else:
    anomalies = pd.DataFrame(columns=work.columns)

# ====================== % change & raw smoothed (kept) ======================

plot_df = pd.DataFrame({ts_col: work[ts_col]})
for col in variables:
    if col not in work.columns: 
        continue
    mean_val = work[col].mean()
    scale = scale_factors.get(col, 1.0)
    work[f'{col}_pct'] = ((work[col] - mean_val) / mean_val * 100) * scale
    work[f'{col}_pct_smooth'] = work[f'{col}_pct'].rolling(window=window).mean()
    work[f'{col}_smooth'] = work[col].rolling(window=window).mean()
    plot_df[f'{col}_pct_smooth'] = work[f'{col}_pct_smooth']
    plot_df[f'{col}_smooth'] = work[f'{col}_smooth']

st.title("Neutron Sensor Dashboard")
st.markdown("Visualize counts, environment, and soil data with anomalies — now with Appendix A.5 corrections, SG smoothing, CRNS→VWC, and robust y-axis control.")

if plot_mode in ["% Change (Smoothed)", "Both"]:
    st.subheader("Smoothed % Change Over Time")
    fig = px.line()
    for col in variables:
        y = plot_df.get(f"{col}_pct_smooth")
        if y is not None:
            fig.add_scatter(x=plot_df[ts_col], y=y, mode='lines', name=col.replace('_',' ').title())
    fig.update_layout(title="Smoothed % Change (%)", xaxis_title="Time", yaxis_title="% Change")
    st.plotly_chart(fig, use_container_width=True)

if plot_mode in ["Raw Counts (Smoothed)", "Both"]:
    st.subheader("Smoothed Raw Values")
    fig2 = px.line()
    for col in variables:
        y = plot_df.get(f"{col}_smooth")
        if y is not None:
            fig2.add_scatter(x=plot_df[ts_col], y=y, mode='lines', name=col.replace('_',' ').title())
    if not anomalies.empty and series_name in variables:
        fig2.add_scatter(x=anomalies[ts_col], y=anomalies[series_name],
                         mode='markers', marker=dict(symbol='x', size=8), name="Anomaly")
    fig2.update_layout(title="Smoothed Raw Values", xaxis_title="Time", yaxis_title="Raw Values",
                       yaxis=dict(range=yaxis_range) if yaxis_range else dict(autorange=True))
    st.plotly_chart(fig2, use_container_width=True)

# ====================== Corrections overview (new) ======================

with st.expander("Appendix A.5 Corrections & Smoothing (overview)"):
    figc = go.Figure()
    if 'counts_avg' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_avg'], mode="lines", name="Counts (avg, raw)"))
    if 'counts_corr' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_corr'], mode="lines", name="Counts (avg, corrected)"))
    if 'counts_sg' in work.columns:
        figc.add_trace(go.Scatter(x=work[ts_col], y=work['counts_sg'], mode="lines", name="Counts (avg, corrected + SG)"))
    figc.update_layout(height=420, xaxis_title="Time", yaxis_title="Counts (avg scale)",
                       yaxis=dict(range=yaxis_range) if yaxis_range else dict(autorange=True))
    st.plotly_chart(figc, use_container_width=True)

# ====================== CRNS → VWC (new) ======================

st.sidebar.markdown("---")
st.sidebar.subheader("CRNS → VWC (Shape Function)")
N0_mode = st.sidebar.radio("N0 selection", ["Slider", "Auto-fit to in-situ (if available)"], index=0)
N0_default = float(work['counts_sg'].quantile(0.9)) if 'counts_sg' in work.columns else float(work['counts_avg'].quantile(0.9))
N0_slider = st.sidebar.number_input("N0 (dry-soil count rate)", value=N0_default)
porosity = st.sidebar.slider("Porosity cap (fraction)", 0.1, 0.8, 0.51)

counts_for_vwc = work['counts_sg'] if 'counts_sg' in work.columns else work['counts_avg']

# pick first in-situ target if present
in_situ_candidates = [c for c in ['soil_moisture_value', 'soil_moisture_pct'] if c in work.columns]
N0_used = N0_slider
if N0_mode == "Auto-fit to in-situ (if available)" and in_situ_candidates:
    tgt_col = in_situ_candidates[0]
    tgt = work[tgt_col] if not tgt_col.endswith("_pct") else work[tgt_col]/100.0
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

if in_situ_candidates:
    st.subheader("Soil Moisture (CRNS vs In-situ)")
    figv = go.Figure()
    figv.add_trace(go.Scatter(x=work[ts_col], y=work['vwc_crns_clipped'], mode="lines", name="VWC (CRNS, clipped)"))
    for col in in_situ_candidates:
        yy = work[col] if not col.endswith("_pct") else work[col]/100.0
        figv.add_trace(go.Scatter(x=work[ts_col], y=yy, mode="lines", name=f"{col} (fraction)" if col.endswith("_pct") else col))
    figv.update_layout(height=380, xaxis_title="Time", yaxis_title="VWC (m³/m³)")
    st.plotly_chart(figv, use_container_width=True)

    # Scatter + fit vs first in-situ
    tgt_col = in_situ_candidates[0]
    tgt = work[tgt_col] if not tgt_col.endswith("_pct") else work[tgt_col]/100.0
    ok2 = pd.concat([work['vwc_crns_clipped'], tgt], axis=1).dropna()
    if len(ok2) > 2:
        m, b = np.polyfit(ok2.iloc[:,0], ok2.iloc[:,1], 1)
        xs = np.linspace(ok2.iloc[:,0].min(), ok2.iloc[:,0].max(), 100)
        ys = m*xs + b
        r = np.corrcoef(ok2.iloc[:,0], ok2.iloc[:,1])[0,1]
        figs = go.Figure()
        figs.add_trace(go.Scatter(x=ok2.iloc[:,0], y=ok2.iloc[:,1], mode="markers", name="Samples"))
        figs.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (R²={r**2:.2f})"))
        figs.update_layout(height=360, xaxis_title="VWC (CRNS, clipped)", yaxis_title=f"VWC ({tgt_col})")
        st.plotly_chart(figs, use_container_width=True)

# ====================== Summary / Correlation (kept) ======================

st.subheader("Summary Statistics")
if variables:
    st.dataframe(work[[v for v in variables if v in work.columns]].describe())

if show_corr:
    st.subheader("Correlation Matrix")
    sel = [c for c in variables if c in work.columns]
    if len(sel) >= 2:
        corr_fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(work[sel].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(corr_fig)
    else:
        st.info("Select at least two variables to show a correlation matrix.")

# ====================== Download processed data (new) ======================

st.subheader("Download Processed Data")
out_cols = list(dict.fromkeys(
    [ts_col] + all_vars +
    [f"{v}_pct_smooth" for v in variables] +
    [f"{v}_smooth" for v in variables] +
    ["counts_corr", "counts_sg", "theta_total", "vwc_crns", "vwc_crns_clipped"]
))
csv_bytes = work[[c for c in out_cols if c in work.columns]].to_csv(index=False).encode("utf-8", errors="ignore")
st.download_button("Download CSV (with corrections, smoothing, VWC)", data=csv_bytes,
                   file_name="processed_crns.csv", mime="text/csv")





# === Seasonal analysis
import numpy as np, pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL

with st.expander("Seasonal analysis (trend, climatology, lags)"):
    # 1) Choose base series
    base_series_name = st.selectbox(
        "Series to analyze seasonally",
        [c for c in ["counts_sg","counts_corr","counts_avg","counts_sum"] if c in work.columns],
        index=0
    )
    y = pd.to_numeric(work[base_series_name], errors="coerce")
    t = pd.to_datetime(work[ts_col])

    # Mask the known data gap to avoid step artifacts (you can tune dates)
    # If you have a mask vector already, use it instead of this example.
    gap_mask = (t >= pd.Timestamp("2025-08-18")) & (t < pd.Timestamp("2025-08-30"))
    y_gapfree = y.mask(gap_mask)

    # 2) STL decomposition at diurnal period
    # Downsample to 15 min before STL to speed up (optional)
    df_stl = pd.DataFrame({"y": y_gapfree.values}, index=t).dropna()
    df_15 = df_stl.resample("15min").median()
    period_pts = int((24*60)/15)  # 96 for 15-min
    stl = STL(df_15["y"], period=period_pts, robust=True)
    res = stl.fit()

    # Plot: trend vs met variables
    figT = go.Figure()
    figT.add_scatter(x=df_15.index, y=res.trend, mode="lines", name=f"{base_series_name} trend (STL)")
    if "humidity" in work.columns:
        # smooth humidity to the same 15-min grid for comparison
        H = pd.Series(pd.to_numeric(work["humidity"], errors="coerce").values, index=t).resample("15min").median()
        figT.add_scatter(x=H.index, y=(H - H.median()), mode="lines", name="Humidity (centered)")
    if ("soil_moisture_value" in work.columns) or ("soil_moisture_pct" in work.columns):
        if "soil_moisture_value" in work.columns:
            SM = pd.Series(pd.to_numeric(work["soil_moisture_value"], errors="coerce").values, index=t).resample("15min").median()
        else:
            SM = pd.Series(pd.to_numeric(work["soil_moisture_pct"], errors="coerce").values, index=t).resample("15min").median()/100.0
        figT.add_scatter(x=SM.index, y=(SM - SM.median()), mode="lines", name="Soil moisture (centered)")
    figT.update_layout(height=320, title="STL trend vs met variables (centered)", xaxis_title="Time", yaxis_title="Value")
    st.plotly_chart(figT, use_container_width=True)

    # 3) Day-of-year (DOY) climatology of the smoothed counts
    df_seas = pd.DataFrame({"y": y.values, "t": t}).dropna()
    df_seas["doy"] = df_seas["t"].dt.dayofyear
    clim = df_seas.groupby("doy")["y"].agg(["median","mean","std","count"]).reset_index()
    # Current season curve (bin to DOY)
    figC = go.Figure()
    figC.add_scatter(x=clim["doy"], y=clim["median"], mode="lines", name="Climatology (median)")
    figC.add_scatter(x=clim["doy"], y=clim["median"]+clim["std"], mode="lines", name="+1σ", line=dict(dash="dot"))
    figC.add_scatter(x=clim["doy"], y=clim["median"]-clim["std"], mode="lines", name="-1σ", line=dict(dash="dot"))
    # Overlay the actual (smoothed) season as DOY
    figC.add_scatter(x=df_seas["doy"], y=df_seas["y"], mode="markers", name="This season (points)", opacity=0.25, marker=dict(size=3))
    figC.update_layout(height=300, title=f"{base_series_name}: DOY climatology ±1σ", xaxis_title="Day of year", yaxis_title=base_series_name)
    st.plotly_chart(figC, use_container_width=True)

    # 4) Seasonal segmentation stats (pre-rain / wet / dry-down)
    bins = st.multiselect("Season bins (months)", ["May-Jun (pre)","Jul-Aug (wet)","Sep (dry)"], default=["May-Jun (pre)","Jul-Aug (wet)","Sep (dry)"])
    def season_label(dt):
        m = dt.month
        if m in [5,6]: return "May-Jun (pre)"
        if m in [7,8]: return "Jul-Aug (wet)"
        if m in [9]:   return "Sep (dry)"
        return "Other"
    seg = df_seas.copy()
    seg["season"] = seg["t"].apply(season_label)
    # summarize
    table = seg.groupby("season")["y"].agg(["median","mean","std","count"]).reset_index()
    st.dataframe(table[table["season"].isin(bins)])

    # 5) Lag analysis per season: counts vs humidity and/or soil moisture
    max_lag_hours = st.slider("Max lag for cross-correlation (hours)", 0, 72, 48)
    step = 6  # 6 steps/hour for 10-min; we’ll reindex to 10-min grid for speed/robustness
    grid = df_stl.index.floor("10min").unique()
    # Build 10-min aligned series
    Y10 = pd.Series(y.values, index=t).groupby(pd.Grouper(freq="10min")).median()
    pairs = []
    if "humidity" in work.columns:
        H10 = pd.Series(pd.to_numeric(work["humidity"], errors="coerce").values, index=t).groupby(pd.Grouper(freq="10min")).median()
        pairs.append(("Humidity", H10))
    if ("soil_moisture_value" in work.columns) or ("soil_moisture_pct" in work.columns):
        if "soil_moisture_value" in work.columns:
            S10 = pd.Series(pd.to_numeric(work["soil_moisture_value"], errors="coerce").values, index=t).groupby(pd.Grouper(freq="10min")).median()
        else:
            S10 = (pd.Series(pd.to_numeric(work["soil_moisture_pct"], errors="coerce").values, index=t)
                   .groupby(pd.Grouper(freq="10min")).median()/100.0)
        pairs.append(("Soil moisture", S10))

    for name, X10 in pairs:
        # limit to common window
        jj = Y10.index.intersection(X10.index)
        yy = Y10.loc[jj].astype(float)
        xx = X10.loc[jj].astype(float)
        # remove mean (anomalies)
        yy = yy - yy.mean(); xx = xx - xx.mean()
        # compute lagged correlations
        lags = np.arange(-max_lag_hours*6, max_lag_hours*6+1)  # 10-min steps
        cc = []
        for L in lags:
            if L < 0:
                c = np.corrcoef(yy[-L:].values, xx[:L if L!=0 else None].values)[0,1]
            elif L > 0:
                c = np.corrcoef(yy[:-L].values, xx[L:].values)[0,1]
            else:
                c = np.corrcoef(yy.values, xx.values)[0,1]
            cc.append(c)
        lhr = lags/6.0
        figLag = go.Figure()
        figLag.add_scatter(x=lhr, y=cc, mode="lines", name=f"r({base_series_name}, {name})")
        # annotate best lag
        if len(cc) > 0 and np.isfinite(cc).any():
            i = int(np.nanargmax(np.abs(cc)))
            figLag.add_vline(x=lhr[i], line_dash="dot")
            figLag.update_layout(title=f"Lag correlation: best |r|={cc[i]:.2f} at {lhr[i]:.1f} h",
                                 xaxis_title="Lag (hours, +means X leads)", yaxis_title="Correlation r")
        st.plotly_chart(figLag, use_container_width=True)
