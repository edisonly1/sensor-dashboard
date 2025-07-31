import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Sidebar file selection
st.sidebar.header("Options")
file_option = st.sidebar.selectbox(
    "Select data file:",
    ["LiFo_05_13_2025__13_17.csv", "LiFo_06_20_2025__17_17.csv"]
)

uploaded = st.sidebar.file_uploader("Or upload your own CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Uploaded file loaded!")
else:
    df = pd.read_csv(file_option)
    st.info(f"Loaded built-in file: {file_option}")

# Preprocess
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

numeric_cols = ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)
df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2

# Filter by threshold
threshold = st.sidebar.slider("Maximum count value to include:", 0, 150, 100)
df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]

# Date selection
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
date_range = st.sidebar.date_input("Select date range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
else:
    st.error("Please select a valid start and end date.")

# Variable selection
variables = st.sidebar.multiselect(
    "Select variables to plot:",
    ['temperature', 'humidity', 'pressure', 'counts_avg'],
    default=['counts_avg']
)

scaling_factors = {
    var: st.sidebar.number_input(f"Scale factor for {var.title()} (%)", 0.1, 20.0, 1.0, step=0.1)
    for var in variables
}

window = st.sidebar.slider("Smoothing window (in points):", 1, 50, 12)
plot_mode = st.sidebar.selectbox("Graph Mode", ["% Change (Smoothed)", "Raw Counts (Smoothed)", "Both"])

# Smoothing + percent change
for col in variables:
    mean = df[col].mean()
    scale = scaling_factors.get(col, 1.0)
    df[f'{col}_pct'] = ((df[col] - mean) / mean * 100) * scale
    df[f'{col}_smoothed'] = df[f'{col}_pct'].rolling(window=window).mean()
    df[f'{col}_rawsmooth'] = df[col].rolling(window=window).mean()

# Anomaly detection on counts_avg
anomaly_df = df[['counts_avg']].copy().dropna()
model = IsolationForest(contamination=0.02, random_state=42)
anomaly_df['anomaly'] = model.fit_predict(anomaly_df[['counts_avg']])
df['anomaly_flag'] = -1
if len(anomaly_df) == len(df):
    df['anomaly_flag'] = anomaly_df['anomaly']

# Plot % Change
if plot_mode in ["% Change (Smoothed)", "Both"]:
    st.subheader("% Change Over Time")
    plot_df = pd.melt(
        df,
        id_vars=['Timestamp'],
        value_vars=[f'{col}_smoothed' for col in variables],
        var_name='Variable',
        value_name='% Change'
    )
    plot_df['Variable'] = plot_df['Variable'].str.replace('_smoothed', '').str.replace('_', ' ').str.title()
    fig = px.line(plot_df, x='Timestamp', y='% Change', color='Variable')
    fig.update_layout(title='Smoothed % Change Over Time')

    # Overlay anomalies
    anomaly_times = df[df['anomaly_flag'] == -1]['Timestamp']
    for t in anomaly_times:
        fig.add_vline(x=t, line=dict(color='red', width=1), opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

# Plot raw smoothed values
if plot_mode in ["Raw Counts (Smoothed)", "Both"]:
    st.subheader("Raw Counts (Smoothed)")
    raw_fig = px.line()
    for col in variables:
        raw_fig.add_scatter(
            x=df['Timestamp'],
            y=df[f'{col}_rawsmooth'],
            mode='lines',
            name=col.replace('_', ' ').title()
        )
    # Overlay anomalies
    anomaly_times = df[df['anomaly_flag'] == -1]['Timestamp']
    anomaly_vals = df[df['anomaly_flag'] == -1]['counts_avg']
    raw_fig.add_scatter(
        x=anomaly_times,
        y=anomaly_vals,
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomalies'
    )
    raw_fig.update_layout(title="Smoothed Raw Sensor Readings", xaxis_title="Time", yaxis_title="Value")
    st.plotly_chart(raw_fig, use_container_width=True)

# Stats
if variables:
    st.subheader("Summary Statistics")
    st.dataframe(df[variables].describe())

# Explanation
with st.expander("Anomaly Detection Explanation"):
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
""")

