import streamlit as st
import pandas as pd
import plotly.express as px
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
plot_df['Variable'] = plot_df['Variable'].str.replace('_smoothed', '').str_
