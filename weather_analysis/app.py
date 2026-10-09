"""
Weather Analysis — Streamlit Browser App
=========================================
Wraps weather_analysis.py pipeline functions into an interactive web UI.

Run:
    streamlit run weather_analysis/app.py
"""

import io
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required before pyplot import

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Re-use all logic from weather_analysis.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from weather_analysis import (
    generate_synthetic_data,
    load_data,
    clean_data,
    engineer_features,
    compute_summary_statistics,
    compute_correlation,
    seasonal_aggregation,
    trend_decomposition,
    generate_insights,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Helper: render matplotlib figure into Streamlit without saving to disk
# ---------------------------------------------------------------------------

def _st_figure() -> plt.Figure:
    """Return the current matplotlib figure and clear it afterwards."""
    fig = plt.gcf()
    return fig


# ---------------------------------------------------------------------------
# Browser-friendly plot functions (return Figure instead of saving to PNG)
# ---------------------------------------------------------------------------

def st_plot_temperature_humidity_trend(df: pd.DataFrame) -> plt.Figure:
    df_ts = df.set_index("Date").sort_index()
    temp_roll  = df_ts["Temperature"].rolling(30, center=True).mean()
    humid_roll = df_ts["Humidity"].rolling(30, center=True).mean()

    fig, ax1 = plt.subplots(figsize=(13, 4))
    color_temp  = "#e05c5c"
    color_humid = "#4a90d9"

    ax1.plot(temp_roll.index, temp_roll.values, color=color_temp,
             linewidth=2, label="Temperature (30-day avg)")
    ax1.set_ylabel("Temperature (°C)", color=color_temp)
    ax1.tick_params(axis="y", labelcolor=color_temp)

    ax2 = ax1.twinx()
    ax2.plot(humid_roll.index, humid_roll.values, color=color_humid,
             linewidth=2, linestyle="--", label="Humidity (30-day avg)")
    ax2.set_ylabel("Humidity (%)", color=color_humid)
    ax2.tick_params(axis="y", labelcolor=color_humid)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Daily Temperature & Humidity — 30-Day Rolling Average",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def st_plot_monthly_rainfall_bar(df: pd.DataFrame) -> plt.Figure:
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rain = df.groupby("Month")["Rainfall"].sum().reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        monthly_rain.index,
        monthly_rain.values,
        color=sns.color_palette("Blues_d", 12),
        edgecolor="white",
        linewidth=0.6,
    )
    peak_month = monthly_rain.idxmax()
    bars[peak_month - 1].set_edgecolor("#e05c5c")
    bars[peak_month - 1].set_linewidth(2.5)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Rainfall (mm)")
    ax.set_title("Monthly Rainfall Distribution (Aggregated)",
                 fontsize=13, fontweight="bold")

    peak_val = monthly_rain[peak_month]
    ax.annotate(
        f"Peak: {MONTH_NAMES[peak_month - 1]}\n{peak_val:.0f} mm",
        xy=(peak_month, peak_val),
        xytext=(peak_month + 1.2, peak_val * 0.95),
        arrowprops=dict(arrowstyle="->", color="#e05c5c"),
        color="#e05c5c",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


def st_plot_seasonal_rainfall_histogram(df: pd.DataFrame) -> plt.Figure:
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    palette = {"Winter": "#4a90d9", "Spring": "#7bc47f",
               "Summer / Monsoon": "#e05c5c", "Autumn": "#f0a500"}

    fig, ax = plt.subplots(figsize=(10, 4))
    for season in season_order:
        subset = df[df["Season"] == season]["Rainfall"].dropna()
        subset = subset[subset > 0]
        if len(subset):
            ax.hist(subset, bins=30, alpha=0.55, label=season,
                    color=palette[season], edgecolor="none")

    ax.set_xlabel("Daily Rainfall (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Seasonal Rainfall Distribution (Rainy Days Only)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Season")
    plt.tight_layout()
    return fig


def st_plot_correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                square=True, ax=ax, annot_kws={"size": 12})
    ax.set_title("Pearson Correlation — Temperature, Rainfall, Humidity",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def st_plot_seasonal_boxplots(df: pd.DataFrame) -> plt.Figure:
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    palette = ["#4a90d9", "#7bc47f", "#e05c5c", "#f0a500"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, ylabel in zip(
        axes,
        ["Temperature", "Humidity"],
        ["Temperature (°C)", "Humidity (%)"],
    ):
        sns.boxplot(data=df, x="Season", y=col, order=season_order,
                    palette=palette, width=0.5,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    ax=ax)
        ax.set_xlabel("Season")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Seasonal {col} Distribution", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Seasonal Box Plots — Temperature & Humidity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def st_plot_trend_decomposition(df: pd.DataFrame) -> plt.Figure | None:
    ts = df.set_index("Date")["Temperature"].resample("ME").mean().dropna()
    if len(ts) < 24:
        return None

    decomp = seasonal_decompose(ts, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Monthly Temperature — Seasonal Decomposition",
                 fontsize=14, fontweight="bold")

    for ax, component, label in zip(
        axes,
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(component.index, component.values, linewidth=1.5)
        ax.set_ylabel(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weather Analysis Dashboard",
    page_icon="🌦️",
    layout="wide",
)

st.title("🌦️ Weather Data Analysis Dashboard")
st.markdown("An end-to-end pipeline for **temperature**, **rainfall**, and **humidity** analysis.")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Data Source")
    data_source = st.radio("Choose data source", ["Synthetic (demo)", "Upload CSV"])

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV with columns: Date, Temperature, Rainfall, Humidity",
            type="csv",
        )
        if uploaded_file is None:
            st.info("Waiting for CSV upload…")

    st.markdown("---")
    st.markdown("**Required CSV columns**")
    st.code("Date, Temperature, Rainfall, Humidity")

# ── Load & process data ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running pipeline…")
def run_pipeline(csv_bytes: bytes | None) -> tuple:
    if csv_bytes is not None:
        tmp = Path("_upload_tmp.csv")
        tmp.write_bytes(csv_bytes)
        df_raw = load_data(str(tmp))
        tmp.unlink(missing_ok=True)
    else:
        df_raw = load_data(None)

    df_clean = clean_data(df_raw)
    df = engineer_features(df_clean)

    cols = ["Temperature", "Rainfall", "Humidity"]
    overall = df[cols].agg(["mean", "median", "min", "max", "std"]).T
    overall.columns = ["Mean", "Median", "Min", "Max", "Std Dev"]
    overall.index.name = "Variable"

    seasonal_stats = (
        df.groupby("Season")[cols]
        .agg(["mean", "median", "min", "max", "std"])
        .round(2)
    )

    monthly = (
        df.groupby("Month")[["Temperature", "Rainfall", "Humidity"]]
        .agg(["mean", "sum", "std"])
        .round(2)
    )

    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    seasonal_mean = (
        df.groupby("Season")[["Temperature", "Rainfall", "Humidity"]]
        .mean()
        .reindex(season_order)
        .round(2)
    )

    corr = df[cols].corr(method="pearson")
    insights = generate_insights(df, corr)

    return df, overall, seasonal_stats, monthly, seasonal_mean, corr, insights


csv_bytes = uploaded_file.read() if uploaded_file else None
if data_source == "Upload CSV" and csv_bytes is None:
    st.stop()

df, overall, seasonal_stats, monthly, seasonal_mean, corr, insights = run_pipeline(csv_bytes)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_stats, tab_plots, tab_insights, tab_raw = st.tabs([
    "📋 Overview", "📊 Statistics", "📈 Plots", "💡 Insights", "🗄️ Raw Data"
])

# ── Tab 1 : Overview ─────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Date Range",
                f"{df['Date'].min().strftime('%d %b %Y')} → {df['Date'].max().strftime('%d %b %Y')}")
    col3.metric("Avg Temperature", f"{df['Temperature'].mean():.1f} °C")
    col4.metric("Total Rainfall", f"{df['Rainfall'].sum():.0f} mm")

    st.markdown("#### Seasonal Mean Values")
    st.dataframe(seasonal_mean.style.format("{:.2f}"), use_container_width=True)

# ── Tab 2 : Statistics ────────────────────────────────────────────────────────
with tab_stats:
    st.subheader("Overall Summary Statistics")
    st.dataframe(overall.round(2).style.format("{:.2f}"), use_container_width=True)

    st.subheader("Seasonal Summary Statistics")
    st.dataframe(seasonal_stats, use_container_width=True)

    st.subheader("Monthly Aggregates (mean / sum / std)")
    MONTH_LABELS = {i: name for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
    )}
    monthly_display = monthly.copy()
    monthly_display.index = [MONTH_LABELS[m] for m in monthly_display.index]
    st.dataframe(monthly_display, use_container_width=True)

    st.subheader("Pearson Correlation Matrix")
    st.dataframe(corr.round(3).style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
                 use_container_width=True)

# ── Tab 3 : Plots ─────────────────────────────────────────────────────────────
with tab_plots:
    st.subheader("Temperature & Humidity Trend")
    st.pyplot(st_plot_temperature_humidity_trend(df))

    st.subheader("Monthly Rainfall")
    st.pyplot(st_plot_monthly_rainfall_bar(df))

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Seasonal Rainfall Distribution")
        st.pyplot(st_plot_seasonal_rainfall_histogram(df))
    with col_r:
        st.subheader("Correlation Heatmap")
        st.pyplot(st_plot_correlation_heatmap(corr))

    st.subheader("Seasonal Box Plots")
    st.pyplot(st_plot_seasonal_boxplots(df))

    st.subheader("Trend Decomposition")
    fig_decomp = st_plot_trend_decomposition(df)
    if fig_decomp:
        st.pyplot(fig_decomp)
    else:
        st.info("Not enough data for decomposition (need ≥ 24 months).")

# ── Tab 4 : Insights ──────────────────────────────────────────────────────────
with tab_insights:
    st.subheader("💡 Key Insights")
    for bullet in insights:
        st.markdown(f"- {bullet}")

# ── Tab 5 : Raw Data ──────────────────────────────────────────────────────────
with tab_raw:
    st.subheader("Raw (Cleaned) Dataset")

    season_filter = st.multiselect(
        "Filter by Season",
        options=df["Season"].unique().tolist(),
        default=df["Season"].unique().tolist(),
    )
    filtered = df[df["Season"].isin(season_filter)]
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    csv_out = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv_out,
                       "weather_filtered.csv", "text/csv")
