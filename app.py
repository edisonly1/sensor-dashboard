# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

st.set_page_config(page_title="Sensor Data Dashboard", layout="wide")

# --- Sidebar: File Selection ---
st.sidebar.header("Options")
file_option = st.sidebar.selectbox(
    "Select data file:",
    ["LiFo_05_13_2025__13_17.csv", "LiFo_06_20_2025__17_17.csv"]
)

# --- Load and Clean Data ---
df = pd.read_csv(file_option)

# Ensure valid timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Show error if no data
if df.empty or df['Timestamp'].isna().all():
    st.error("No valid timestamp data in the selected file.")
    st.stop()

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

# Expand if range is too narrow
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

# --- Sidebar: Variable Selection ---
variables = st.sidebar.multiselect(
    "Select variables to plot:",
    ['temperature', 'humidity', 'pressure', 'counts_avg'],
    default=['counts_avg']
)

# Sidebar: Scale inputs
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
    "Smoothing window (in points):",
    min_value=1, max_value=50, value=12
)

# --- Filter and Calculate ---
df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]

for col in variables:
    mean = df[col].mean()
    scale = scaling_factors.get(col, 1.0)
    df[f'{col}_pct'] = ((df[col] - mean) / mean * 100) * scale
