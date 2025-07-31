# streamlit_app.py

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

# --- Clean and Preprocess ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Convert numeric columns
cols = ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=cols)

# Combine counts
df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2

# --- Date Range Filter ---
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

if min_date == max_date:
    from datetime import timedelta
    min_date -= timedelta(days=1)
    max_date += timedelta(days=1)

date_range = st.sidebar.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
else:
    st.error("Please select a valid start and end date.")
    st.stop()

# --- Variable Sele
