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

df['counts_0'] = pd.to_numeric(df['counts_0'], errors='coerce')
df['counts_1'] = pd.to_numeric(df['counts_1'], errors='coerce')
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

# Compute % change
mean = df['counts_avg'].mean()
df['counts_avg_pct'] = ((df['counts_avg'] - mean) / mean * 100) * scale
df['counts_avg_smoothed'] = df['counts_avg_pct'].rolling(window=window).mean()
df['counts_avg_raw_smooth'] = df['counts_avg'].rolling(window=window).mean()

# Prepare plot data
plot_df = df[['Timestamp', 'counts_avg_smoothed']].dropna()
plot_df.rename(columns={'counts_avg_smoothed': '% Change'}, inplace=True)

# --- Dashboard Title ---
st.title("Neutron Count Dashboard – `counts_avg` Focused")
st.markdown("""
This dashboard visualizes **smoothed percent change** and **raw counts** of the average neutron count (`counts_avg`) over time.
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
""")



# --- Create % Change Plot ---
fig = px.line(
    plot_df, x='Timestamp', y='% Change',
    title="Smoothed % Change in Counts Avg",
    labels={'% Change': 'Smoothed % Change (%)'}
)
fig.update_traces(line=dict(width=1), opacity=0.8)

# --- Anomaly Detection ---
if detect_anomalies:
    model = IsolationForest(contamination=0.02, random_state=42)
    subset = df[['counts_avg']].dropna()
    df.loc[subset.index, 'anomaly'] = model.fit_predict(subset)
    anomalies = df[df['anomaly'] == -1]
    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies['Timestamp'],
            y=anomalies['counts_avg'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='Anomaly'
        )

# --- Show Graph(s) ---
if plot_mode in ["% Change (Smoothed)", "Both"]:
    st.subheader("% Change Over Time")
    st.plotly_chart(fig, use_container_width=True)

if plot_mode in ["Raw Counts (Smoothed)", "Both"]:
    st.subheader("Raw Counts (Smoothed)")
    fig2 = px.line(
        df,
        x='Timestamp',
        y='counts_avg_raw_smooth',
        title="Smoothed Raw Counts Avg",
        labels={'counts_avg_raw_smooth': 'Smoothed Counts Avg'}
    )
    fig2.update_traces(line=dict(width=1), opacity=0.8)
    st.plotly_chart(fig2, use_container_width=True)

# --- Summary Statistics ---
st.subheader("Summary Statistics")
st.dataframe(df[['counts_avg']].describe())

# --- Correlation Matrix ---
if show_corr:
    st.subheader("Correlation Matrix (Single Variable)")
    corr_fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(df[['counts_avg']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(corr_fig)
