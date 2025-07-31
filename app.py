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

for col in ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['counts_0', 'counts_1'])
df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2

# --- Sidebar Filters ---
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
if min_date == max_date:
    min_date -= timedelta(days=1)
    max_date += timedelta(days=1)

date_range = st.sidebar.date_input("Select date range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
else:
    st.error("Please select a valid start and end date.")
    st.stop()

# Variable selector
all_vars = ['counts_avg', 'temperature', 'humidity', 'pressure']
variables = st.sidebar.multiselect("Select variables to plot:", all_vars, default=['counts_avg'])

threshold = st.sidebar.slider("Maximum count value to include:", min_value=0, max_value=150, value=100)
window = st.sidebar.slider("Smoothing window (in points):", min_value=1, max_value=50, value=12)
scale = st.sidebar.number_input("Scale factor for % Change", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
plot_mode = st.sidebar.selectbox("Graph Mode", ["% Change (Smoothed)", "Raw Counts (Smoothed)", "Both"])
detect_anomalies = st.sidebar.checkbox("Detect Anomalies in Counts Avg")
show_corr = st.sidebar.checkbox("Show Correlation Matrix")


# --- Filter & Smooth ---
df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]
if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Anomaly Detection (only on counts_avg raw) ---
if detect_anomalies:
    model = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = model.fit_predict(df[['counts_avg']].dropna())
    anomalies = df[df['anomaly'] == -1]
else:
    anomalies = pd.DataFrame(columns=df.columns)

# --- Process selected variables ---
plot_df = pd.DataFrame({'Timestamp': df['Timestamp']})

for col in variables:
    mean_val = df[col].mean()
    df[f'{col}_pct'] = ((df[col] - mean_val) / mean_val * 100) * scale
    df[f'{col}_pct_smooth'] = df[f'{col}_pct'].rolling(window=window).mean()
    df[f'{col}_smooth'] = df[col].rolling(window=window).mean()
    plot_df[f'{col}_pct_smooth'] = df[f'{col}_pct_smooth']
    plot_df[f'{col}_smooth'] = df[f'{col}_smooth']

# --- Dashboard Title ---
st.title("Neutron Sensor Dashboard")
st.markdown("Visualize neutron count & environment data trends, with anomaly detection and optional correlation analysis.")
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
""")

# --- Graph: % Change ---
if plot_mode in ["% Change (Smoothed)", "Both"]:
    st.subheader("Smoothed % Change Over Time")
    fig = px.line()

    for col in variables:
        fig.add_scatter(
            x=plot_df['Timestamp'],
            y=plot_df[f'{col}_pct_smooth'],
            mode='lines',
            name=col.replace('_', ' ').title()
        )
    fig.update_layout(title="Smoothed % Change (%)", xaxis_title="Time", yaxis_title="% Change")
    st.plotly_chart(fig, use_container_width=True)

# --- Graph: Raw Smoothed Counts ---
if plot_mode in ["Raw Counts (Smoothed)", "Both"]:
    st.subheader("Raw Counts (Smoothed)")
    fig2 = px.line()

    for col in variables:
        fig2.add_scatter(
            x=plot_df['Timestamp'],
            y=plot_df[f'{col}_smooth'],
            mode='lines',
            name=col.replace('_', ' ').title()
        )

    # Anomaly markers ONLY on raw counts_avg graph
    if detect_anomalies and 'counts_avg' in variables and not anomalies.empty:
        fig2.add_scatter(
            x=anomalies['Timestamp'],
            y=anomalies['counts_avg'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name="Anomaly"
        )

    fig2.update_layout(title="Smoothed Raw Values", xaxis_title="Time", yaxis_title="Raw Values")
    st.plotly_chart(fig2, use_container_width=True)

# --- Stats ---
st.subheader("Summary Statistics")
st.dataframe(df[variables].describe())

# --- Correlation Matrix ---
if show_corr:
    st.subheader("Correlation Matrix")
    corr_fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[variables].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(corr_fig)
