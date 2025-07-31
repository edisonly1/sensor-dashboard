# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# --- File selection in sidebar ---
st.sidebar.header("Options")
file_option = st.sidebar.selectbox(
    "Select data file:",
    ["LiFo_05_13_2025__13_17.csv", "LiFo_06_01_2025__10_00.csv"]
)

# --- Load data based on selection ---
df = pd.read_csv(file_option)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Convert numeric columns
cols = ['temperature', 'humidity', 'pressure', 'counts_0', 'counts_1']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=cols)

# Combine counts
df['counts_avg'] = (df['counts_0'] + df['counts_1']) / 2

# --- Date range filter ---
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

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

# --- Variable selection and scaling ---
variables = st.sidebar.multiselect(
    "Select variables to plot:",
    ['temperature', 'humidity', 'pressure', 'counts_avg'],
    default=['counts_avg']
)

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

# --- Filter and process data ---
df = df[(df['counts_0'] <= threshold) & (df['counts_1'] <= threshold)]

for col in variables:
    mean = df[col].mean()
    scale = scaling_factors.get(col, 1.0)
    df[f'{col}_pct'] = ((df[col] - mean) / mean * 100) * scale
    df[f'{col}_smoothed'] = df[f'{col}_pct'].rolling(window=window).mean()

# --- Prepare long-format data for Plotly ---
plot_df = pd.melt(
    df,
    id_vars='Timestamp',
    value_vars=[f'{col}_smoothed' for col in variables],
    var_name='Variable',
    value_name='% Change'
)
plot_df['Variable'] = plot_df['Variable'].str.replace('_smoothed', '').str.replace('_', ' ').str.title()

# --- Plot ---
st.title("Sensor Data Dashboard")
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
st.plotly_chart(fig)

# --- Stats ---
if variables:
    st.subheader("Summary Statistics")
    st.dataframe(df[variables].describe())
else:
    st.warning("Please select at least one variable to view summary statistics.")
