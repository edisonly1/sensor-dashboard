import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import timedelta

st.set_page_config(page_title="Sensor Data Dashboard", layout="wide")

# --- Sidebar: File Selection ---
st.sidebar.header("Options")

file_option = st.sidebar.selectbox(
    "Select built-in data file:",
    ["LiFo_05_13_2025__13_17.csv", "LiFo_06_20_2025__17_17.csv"]
)

uploaded = st.sidebar.file_uploader("Or upload your own CSV", type="csv")

# --- Load Data ---
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Custom file uploaded and loaded!")
else:
    df = pd.read_csv(file_option)
    st.info(f"Loaded built-in file: `{file_option}`")

# --- Preprocessing ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Convert numeric columns
cols = ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=cols)

# Combine neutron counts
df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2

# --- Sidebar: Date Range ---
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

if min_date == max_date:
    min_date -= timedelta(days=1)
    max_date += timedelta(days=1)

date_range = st.sidebar.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter by date
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
else:
    st.error("Please select a valid start and end date.")
    st.stop()

# --- Sidebar: Variable Controls ---
variables = st.sidebar.multiselect(
    "Select variables to plot:",
    ['temperature', 'humidity', 'pressure', 'counts_avg'],
    default=['counts_avg']
)

if not variables:
    st.warning("Please select at least one variable to plot.")
    st.stop()

scaling_factors = {
    var: st.sidebar.number_input(
        f"Scale factor for {var.title()} (%)", min_value=0.1, max_value=20.0, value=1.0, step=0.1
    )
    for var in variables
}

threshold = st.sidebar.slider(
    "Maximum count value to include:",
    min_value=0, max_value=150, value=100
)

window = st.sidebar.slider(
    "Smoothing window (in points):", min_value=1, max_value=50, value=12
)

show_corr = st.sidebar.checkbox("Show Correlation Matrix")
detect_anomalies = st.sidebar.checkbox("Detect Anomalies in Counts Avg")

# --- Filter by count threshold ---
df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]

# Warn if no data remains
if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Compute % change and smoothing ---
for col in variables:
    mean = df[col].mean()
    scale = scaling_factors.get(col, 1.0)
    df[f'{col}_pct'] = ((df[col] - mean) / mean * 100) * scale
    df[f'{col}_smoothed'] = df[f'{col}_pct'].rolling(window=window).mean()

# --- Prepare Plot Data ---
plot_df = pd.melt(
    df,
    id_vars='Timestamp',
    value_vars=[f'{col}_smoothed' for col in variables],
    var_name='Variable',
    value_name='% Change'
)

if plot_df.empty:
    st.warning("No data available to plot after filtering and smoothing.")
    st.stop()

# Clean variable names safely
plot_df['Variable'] = plot_df['Variable'].astype(str)
plot_df['Variable'] = (
    plot_df['Variable']
    .str.replace('_smoothed', '', regex=False)
    .str.replace('_', ' ', regex=False)
    .str.title()
)

# --- Dashboard Title ---
st.title("Sensor Data Dashboard")
st.markdown("""
This dashboard visualizes smoothed percent changes in environmental and neutron count data.
Use the sidebar to filter date range, select variables, adjust smoothing, and upload your own data.
""")

if detect_anomalies:
    st.markdown("""
### Anomaly Detection Using Isolation Forest

Anomalies in neutron count behavior are identified using the **Isolation Forest algorithm**, a machine learning model designed for unsupervised outlier detection. This model is particularly effective for **high-dimensional and noisy time series data**, such as environmental sensor outputs.

#### How It Works:
- Isolation Forest works by randomly selecting features (e.g., `counts_avg`) and thresholds to construct **decision trees**.
- The idea is that **anomalies are "few and different"** — meaning they are more easily isolated than typical data points.
- Each point is assigned an **anomaly score** based on the average path length required to isolate it in the forest.
- Points with short path lengths (i.e., isolated quickly) are likely to be **statistical outliers**.

#### ⚛ Applied to Neutron Counts:
- The model processes the average neutron count (`counts_avg`) over time.
- It flags timestamps where the neutron count **suddenly drops or spikes** relative to the recent baseline distribution.
- These events are marked with a **red ❌ on the graph**.
"")


# --- Plot Line Chart ---
fig = px.line(
    plot_df,
    x='Timestamp',
    y='% Change',
    color='Variable',
    color_discrete_map={
        'Temperature': '#e74c3c',
        'Humidity': '#8e44ad',
        'Pressure': '#3498db',
        'Counts Avg': '#27ae60'
    }
)
fig.update_layout(title='Smoothed % Change Over Time')
fig.update_traces(line=dict(width=1), opacity=0.8)

# --- Anomaly Detection ---
if detect_anomalies:
    if 'counts_avg' in df.columns and not df['counts_avg'].isna().all():
        model = IsolationForest(contamination=0.02, random_state=42)
        df_no_na = df[['counts_avg']].dropna()
        df.loc[df_no_na.index, 'anomaly'] = model.fit_predict(df_no_na)
        anomalies = df[df['anomaly'] == -1]

        if not anomalies.empty:
            fig.add_scatter(
                x=anomalies['Timestamp'],
                y=anomalies['counts_avg'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name='Anomaly'
            )
    else:
        st.warning("Counts Avg data not available for anomaly detection.")

# --- Display Plot ---
st.plotly_chart(fig, use_container_width=True)

# --- Summary Statistics ---
st.subheader("Summary Statistics")
st.dataframe(df[variables].describe())

# --- Correlation Matrix ---
if show_corr:
    st.subheader("Correlation Matrix")
    corr = df[variables].corr()
    fig_corr, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)
