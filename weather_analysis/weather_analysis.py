"""
Weather Data Analysis: Temperature, Rainfall, Humidity, and Seasonal Patterns
==============================================================================
A modular, end-to-end Python pipeline for ingesting, cleaning, analysing,
and visualising weather data.

Usage
-----
    python weather_analysis.py                    # runs on synthetic data
    python weather_analysis.py --csv your_data.csv  # runs on real data

Dependencies
------------
    pip install pandas numpy matplotlib seaborn scipy statsmodels
"""

import argparse
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 encoding errors for non-ASCII chars)
import io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = Path("weather_analysis/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATOR
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    start: str = "2022-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate two years of realistic daily weather records.

    Returns a DataFrame with columns:
        Date, Temperature (°C), Rainfall (mm), Humidity (%)
    Seasonal signals are embedded so analysis results are meaningful.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Day-of-year in [0, 2π] for seasonal sine waves — use plain numpy array
    doy = np.array(dates.dayofyear, dtype=float)
    day_angle = 2 * np.pi * doy / 365.0

    # --- Temperature: peaks ~July (summer), troughs ~January ---
    temp_base = 20 + 14 * np.sin(day_angle - np.pi / 2)   # range ≈ 6–34 °C
    temperature = np.array(temp_base + rng.normal(0, 2.5, n), dtype=float)

    # --- Humidity: inversely related to temp + extra monsoon peak (Jul–Sep) ---
    monsoon_boost = 20 * np.exp(-0.5 * ((doy - 210) / 30) ** 2)
    humidity_base = 65 - 0.6 * (temperature - 20) + monsoon_boost
    humidity = np.clip(humidity_base + rng.normal(0, 5, n), 20, 100).astype(float)

    # --- Rainfall: mostly monsoon (Jun–Sep) + light winter/spring rain ---
    rain_signal = (
        12 * np.exp(-0.5 * ((doy - 210) / 40) ** 2)   # monsoon peak
        + 2 * np.exp(-0.5 * ((doy - 60) / 20) ** 2)   # spring shower
    )
    rainfall = np.clip(
        rain_signal * rng.exponential(1, n), 0, None
    ).astype(float)  # exponential ensures right-skewed distribution

    # Inject ~2 % missing values randomly across columns
    for arr in [temperature, humidity, rainfall]:
        missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
        arr[missing_idx] = np.nan

    # Inject a handful of obvious outliers for the cleaner to handle
    for arr, lo, hi in [
        (temperature, -50, 60),
        (humidity, 0, 100),
        (rainfall, 0, 300),
    ]:
        outlier_idx = rng.choice(n, size=3, replace=False)
        arr[outlier_idx] = rng.choice([lo, hi])

    df = pd.DataFrame(
        {
            "Date": dates,
            "Temperature": temperature,
            "Rainfall": rainfall,
            "Humidity": humidity,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. DATA INGESTION
# ---------------------------------------------------------------------------

def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load weather data from a CSV file, or generate synthetic data if no path
    is supplied.

    Expected CSV columns (case-insensitive):
        Date, Temperature, Rainfall, Humidity
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        # Normalise column names to title-case
        df.columns = [c.strip().title() for c in df.columns]
        print(f"[Ingestion] Loaded {len(df):,} rows from '{csv_path}'.")
    else:
        print("[Ingestion] No CSV supplied — generating synthetic data.")
        df = generate_synthetic_data()
        print(f"[Ingestion] Generated {len(df):,} synthetic daily records.")

    return df


# ---------------------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------------------

VALID_RANGES = {
    "Temperature": (-40.0, 55.0),   # °C
    "Rainfall":    (0.0,  300.0),   # mm/day
    "Humidity":    (0.0,  100.0),   # %
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all data-quality steps:
      1. Parse Date column to datetime.
      2. Coerce numeric columns; replace non-numeric values with NaN.
      3. Replace physically impossible values with NaN.
      4. Remove statistical outliers using IQR-fencing per numeric column.
      5. Interpolate remaining NaN values (time-aware linear fill).
      6. Drop any residual NaN rows.
      7. Sort by Date and reset index.
    """
    df = df.copy()

    # --- 3.1 Parse dates ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    before = len(df)
    df.dropna(subset=["Date"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[Cleaning] Dropped {dropped} rows with unparseable dates.")

    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # --- 3.2 Coerce numeric columns ---
    for col in ["Temperature", "Rainfall", "Humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 3.3 Replace values outside physical range with NaN ---
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_bad = mask.sum()
            if n_bad:
                print(f"[Cleaning] {col}: flagged {n_bad} values outside [{lo}, {hi}] -> NaN.")
            df.loc[mask, col] = np.nan

    # --- 3.4 IQR-based outlier removal ---
    for col in ["Temperature", "Rainfall", "Humidity"]:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr   # use 3× for weather (high natural variance)
        upper = q3 + 3.0 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n_out = mask.sum()
        if n_out:
            print(f"[Cleaning] {col}: removed {n_out} IQR-outliers -> NaN.")
        df.loc[mask, col] = np.nan

    # --- 3.5 Time-aware interpolation for short gaps ---
    df[["Temperature", "Rainfall", "Humidity"]] = (
        df[["Temperature", "Rainfall", "Humidity"]]
        .interpolate(method="time", limit=7)
    )

    # --- 3.6 Drop residual NaNs ---
    before = len(df)
    df.dropna(subset=["Temperature", "Rainfall", "Humidity"], inplace=True)
    print(f"[Cleaning] Dropped {before - len(df)} residual NaN rows after interpolation.")

    df.reset_index(inplace=True)
    print(f"[Cleaning] Final dataset: {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def _assign_season(month: int) -> str:
    """Map a calendar month to a meteorological season."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8, 9):
        return "Summer / Monsoon"
    else:  # 10, 11
        return "Autumn"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features and derive the Season column.

    New columns added:
        Year, Month, Day, DayOfYear, WeekOfYear, Season
    """
    df = df.copy()
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["DayOfYear"]  = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Season"]     = df["Month"].map(_assign_season)

    print("[Features] Added: Year, Month, Day, DayOfYear, WeekOfYear, Season.")
    return df


# ---------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS & METRICS
# ---------------------------------------------------------------------------

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, min, max, std dev for Temperature, Rainfall,
    and Humidity — both overall and grouped by Season.
    """
    cols = ["Temperature", "Rainfall", "Humidity"]
    overall = df[cols].agg(["mean", "median", "min", "max", "std"]).T
    overall.columns = ["Mean", "Median", "Min", "Max", "Std Dev"]
    overall.index.name = "Variable"

    seasonal = (
        df.groupby("Season")[cols]
        .agg(["mean", "median", "min", "max", "std"])
        .round(2)
    )

    print("\n=== Overall Summary Statistics ===")
    print(overall.round(2).to_string())
    print("\n=== Seasonal Summary Statistics ===")
    print(seasonal.to_string())

    return overall


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for Temperature, Rainfall, Humidity.
    """
    cols = ["Temperature", "Rainfall", "Humidity"]
    corr = df[cols].corr(method="pearson")
    print("\n=== Pearson Correlation Matrix ===")
    print(corr.round(3).to_string())
    return corr


# ---------------------------------------------------------------------------
# 6. SEASONAL & TREND ANALYSIS
# ---------------------------------------------------------------------------

def seasonal_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Temperature, Rainfall, and Humidity by Month and Season.
    Returns the monthly aggregate DataFrame.
    """
    monthly = (
        df.groupby("Month")[["Temperature", "Rainfall", "Humidity"]]
        .agg(["mean", "sum", "std"])
        .round(2)
    )
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    seasonal = (
        df.groupby("Season")[["Temperature", "Rainfall", "Humidity"]]
        .mean()
        .reindex(season_order)
        .round(2)
    )

    print("\n=== Monthly Aggregates (mean/sum/std) ===")
    print(monthly.to_string())
    print("\n=== Seasonal Mean Values ===")
    print(seasonal.to_string())

    return monthly


def trend_decomposition(df: pd.DataFrame) -> None:
    """
    Perform STL-style seasonal decomposition on the monthly-resampled
    Temperature time series.  Saves decomposition plot to figures/.
    """
    ts = df.set_index("Date")["Temperature"].resample("ME").mean().dropna()

    if len(ts) < 24:
        print("[Trend] Not enough data for full decomposition (need ≥ 24 months).")
        return

    decomp = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Monthly Temperature — Seasonal Decomposition", fontsize=14, fontweight="bold")

    for ax, component, label in zip(
        axes,
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(component.index, component.values, linewidth=1.5)
        ax.set_ylabel(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.tight_layout()
    _save_figure("trend_decomposition")


# ---------------------------------------------------------------------------
# 7. VISUALISATIONS
# ---------------------------------------------------------------------------

def _save_figure(name: str) -> None:
    """Save and close the current matplotlib figure."""
    path = FIGURE_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_temperature_humidity_trend(df: pd.DataFrame) -> None:
    """
    Line chart: 30-day rolling mean of Temperature and Humidity over time.
    Dual y-axes so both series are visible on the same plot.
    """
    df_ts = df.set_index("Date").sort_index()
    temp_roll  = df_ts["Temperature"].rolling(30, center=True).mean()
    humid_roll = df_ts["Humidity"].rolling(30, center=True).mean()

    fig, ax1 = plt.subplots(figsize=(14, 5))
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

    plt.title("Daily Temperature & Humidity — 30-Day Rolling Average", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure("temperature_humidity_trend")


def plot_monthly_rainfall_bar(df: pd.DataFrame) -> None:
    """
    Bar chart: Total monthly rainfall aggregated across all years.
    Annotates the peak month.
    """
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_rain = df.groupby("Month")["Rainfall"].sum().reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        monthly_rain.index,
        monthly_rain.values,
        color=sns.color_palette("Blues_d", 12),
        edgecolor="white",
        linewidth=0.6,
    )

    # Highlight peak month
    peak_month = monthly_rain.idxmax()
    bars[peak_month - 1].set_edgecolor("#e05c5c")
    bars[peak_month - 1].set_linewidth(2.5)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Rainfall (mm)")
    ax.set_title("Monthly Rainfall Distribution (Aggregated)", fontsize=13, fontweight="bold")

    # Annotate peak
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
    _save_figure("monthly_rainfall_bar")


def plot_seasonal_rainfall_histogram(df: pd.DataFrame) -> None:
    """
    Overlapping histograms of daily rainfall, one distribution per season.
    """
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    palette = {"Winter": "#4a90d9", "Spring": "#7bc47f",
               "Summer / Monsoon": "#e05c5c", "Autumn": "#f0a500"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for season in season_order:
        subset = df[df["Season"] == season]["Rainfall"].dropna()
        subset = subset[subset > 0]   # only rainy days
        if len(subset):
            ax.hist(
                subset,
                bins=30,
                alpha=0.55,
                label=season,
                color=palette[season],
                edgecolor="none",
            )

    ax.set_xlabel("Daily Rainfall (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Seasonal Rainfall Distribution (Rainy Days Only)", fontsize=13, fontweight="bold")
    ax.legend(title="Season")
    plt.tight_layout()
    _save_figure("seasonal_rainfall_histogram")


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    """
    Annotated heatmap of the Pearson correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = False   # show full matrix

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_title("Pearson Correlation — Temperature, Rainfall, Humidity",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_figure("correlation_heatmap")


def plot_seasonal_boxplots(df: pd.DataFrame) -> None:
    """
    Side-by-side box plots for Temperature and Humidity, grouped by season.
    Highlights seasonal variance and outliers.
    """
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    palette = ["#4a90d9", "#7bc47f", "#e05c5c", "#f0a500"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, ylabel in zip(
        axes,
        ["Temperature", "Humidity"],
        ["Temperature (°C)", "Humidity (%)"],
    ):
        sns.boxplot(
            data=df,
            x="Season",
            y=col,
            order=season_order,
            palette=palette,
            width=0.5,
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
            ax=ax,
        )
        ax.set_xlabel("Season")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Seasonal {col} Distribution", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Seasonal Box Plots — Temperature & Humidity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure("seasonal_boxplots")


# ---------------------------------------------------------------------------
# 8. INSIGHTS & SUMMARY GENERATOR
# ---------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    """
    Derive automated, data-driven bullet-point insights from the cleaned,
    feature-engineered DataFrame and its correlation matrix.
    """
    insights = []
    MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }

    # --- Temperature insights ---
    hottest_month = df.groupby("Month")["Temperature"].mean().idxmax()
    coldest_month = df.groupby("Month")["Temperature"].mean().idxmin()
    max_temp = df["Temperature"].max()
    min_temp = df["Temperature"].min()
    insights += [
        f"🌡️  Hottest month on average: {MONTH_NAMES[hottest_month]} "
        f"({df.groupby('Month')['Temperature'].mean()[hottest_month]:.1f} °C).",
        f"❄️  Coldest month on average: {MONTH_NAMES[coldest_month]} "
        f"({df.groupby('Month')['Temperature'].mean()[coldest_month]:.1f} °C).",
        f"📈  Temperature range across the dataset: {min_temp:.1f} °C → {max_temp:.1f} °C "
        f"(spread = {max_temp - min_temp:.1f} °C).",
    ]

    # --- Rainfall insights ---
    peak_rain_month  = df.groupby("Month")["Rainfall"].sum().idxmax()
    dry_month        = df.groupby("Month")["Rainfall"].sum().idxmin()
    total_rain       = df["Rainfall"].sum()
    rainy_days       = (df["Rainfall"] > 1).sum()
    insights += [
        f"🌧️  Peak rainfall month: {MONTH_NAMES[peak_rain_month]} "
        f"({df.groupby('Month')['Rainfall'].sum()[peak_rain_month]:.0f} mm total).",
        f"☀️  Driest month: {MONTH_NAMES[dry_month]} "
        f"({df.groupby('Month')['Rainfall'].sum()[dry_month]:.0f} mm total).",
        f"💧  Total rainfall in dataset: {total_rain:.0f} mm "
        f"across {rainy_days} rainy days (> 1 mm).",
    ]

    # --- Humidity insights ---
    high_humid_season = df.groupby("Season")["Humidity"].mean().idxmax()
    low_humid_season  = df.groupby("Season")["Humidity"].mean().idxmin()
    insights += [
        f"💦  Highest average humidity: {high_humid_season} season "
        f"({df.groupby('Season')['Humidity'].mean()[high_humid_season]:.1f} %).",
        f"🏜️  Lowest average humidity: {low_humid_season} season "
        f"({df.groupby('Season')['Humidity'].mean()[low_humid_season]:.1f} %).",
    ]

    # --- Correlation insights ---
    r_temp_humid = corr.loc["Temperature", "Humidity"]
    r_temp_rain  = corr.loc["Temperature", "Rainfall"]
    r_rain_humid = corr.loc["Rainfall", "Humidity"]
    insights += [
        f"📊  Temperature ↔ Humidity correlation: r = {r_temp_humid:.2f} "
        f"({'negative' if r_temp_humid < 0 else 'positive'} — "
        f"{'humidity drops as temperature rises' if r_temp_humid < 0 else 'both tend to rise together'}).",
        f"📊  Temperature ↔ Rainfall correlation: r = {r_temp_rain:.2f}.",
        f"📊  Rainfall ↔ Humidity correlation: r = {r_rain_humid:.2f} "
        f"({'rainfall boosts humidity' if r_rain_humid > 0.3 else 'weak association'}).",
    ]

    # --- Seasonal extremes ---
    season_order = ["Winter", "Spring", "Summer / Monsoon", "Autumn"]
    season_temp  = df.groupby("Season")["Temperature"].mean().reindex(season_order)
    insights.append(
        f"🗓️  Seasonal mean temperatures — "
        + ", ".join(f"{s}: {v:.1f} °C" for s, v in season_temp.items())
        + "."
    )

    # --- Year-on-year change (if > 1 year of data) ---
    years = sorted(df["Year"].unique())
    if len(years) >= 2:
        y1, y2 = years[0], years[-1]
        delta = (
            df[df["Year"] == y2]["Temperature"].mean()
            - df[df["Year"] == y1]["Temperature"].mean()
        )
        direction = "warmer" if delta > 0 else "cooler"
        insights.append(
            f"📅  Year-on-year temperature shift ({y1} → {y2}): "
            f"{abs(delta):.2f} °C {direction}."
        )

    return insights


# ---------------------------------------------------------------------------
# 9. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str | None = None) -> None:
    """
    Orchestrate the full analysis pipeline end-to-end.

    Steps
    -----
    1. Ingest data (CSV or synthetic)
    2. Clean & validate
    3. Engineer features
    4. Summary statistics + correlations
    5. Seasonal & trend analysis
    6. Generate all plots
    7. Print automated insights
    """
    print("\n" + "=" * 60)
    print("  Weather Data Analysis Pipeline")
    print("=" * 60)

    # --- Step 1: Load ---
    df_raw = load_data(csv_path)

    # --- Step 2: Clean ---
    df_clean = clean_data(df_raw)

    # --- Step 3: Features ---
    df = engineer_features(df_clean)

    # --- Step 4: EDA ---
    summary = compute_summary_statistics(df)
    corr    = compute_correlation(df)

    # --- Step 5: Seasonal & Trend ---
    seasonal_aggregation(df)
    trend_decomposition(df)

    # --- Step 6: Visualisations ---
    print("\n[Plots] Generating visualisations...")
    plot_temperature_humidity_trend(df)
    plot_monthly_rainfall_bar(df)
    plot_seasonal_rainfall_histogram(df)
    plot_correlation_heatmap(corr)
    plot_seasonal_boxplots(df)

    # --- Step 7: Insights ---
    insights = generate_insights(df, corr)
    print("\n" + "=" * 60)
    print("  Key Insights & Summary")
    print("=" * 60)
    for bullet in insights:
        print(f"  • {bullet}")

    print(f"\n[Done] All figures saved to: {FIGURE_DIR.resolve()}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weather Data Analysis Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a CSV file with columns: Date, Temperature, Rainfall, Humidity. "
             "If omitted, synthetic data is generated automatically.",
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv)
