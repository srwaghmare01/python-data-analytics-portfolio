"""
air_quality/openaq_analysis.py

Cleans and analyses OpenAQ air quality data (PM2.5 and NO2).
Produces descriptive statistics and interactive Plotly charts.

Why Parquet over CSV at scale:
- Columnar storage: load only the columns you need
- Built-in compression: smaller file sizes
- Schema enforcement: no silent type coercion on read
- For 10TB+: pair with Spark or Dask for distributed processing

Usage:
    python openaq_analysis.py
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px


POLLUTANTS = ["PM2.5", "NO2"]

POLLUTANT_ALIASES = {
    "PM2,5":  "PM2.5",
    "PM 2.5": "PM2.5",
    "PM2. 5": "PM2.5",
    "₂":      "2",
}


def normalise_pollutants(series: pd.Series) -> pd.Series:
    """Standardise pollutant name variants to PM2.5 and NO2."""
    s = series.astype(str).str.upper()
    for old, new in POLLUTANT_ALIASES.items():
        s = s.str.replace(old, new, regex=False)
    return s


def load_and_clean(parquet_path: str):
    """
    Load raw OpenAQ parquet file, clean it, and return
    a wide-format DataFrame with one row per station per timestamp.

    Parameters
    ----------
    parquet_path : str
        Path to the raw OpenAQ .parquet file.

    Returns
    -------
    df_wide : pd.DataFrame
        Wide-format with columns: PM2.5, NO2, City, Country Label.
    summary : dict
        Cleaning stats: negative values removed, missing counts.
    """
    df = pd.read_parquet(parquet_path)
    df = df[[
        "Last Updated", "Location", "City",
        "Country Label", "Pollutant", "Value"
    ]].copy()
    df.rename(columns={"Location": "station_id"}, inplace=True)

    df["Last Updated"] = pd.to_datetime(
        df["Last Updated"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["Last Updated"])

    df["Pollutant"] = normalise_pollutants(df["Pollutant"])
    df = df[df["Pollutant"].isin(POLLUTANTS)]

    neg_count = int((df["Value"] < 0).sum())
    df = df[df["Value"] >= 0]

    wide = (
        df.pivot_table(
            index=["Last Updated", "station_id"],
            columns="Pollutant",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
    )
    for col in POLLUTANTS:
        if col not in wide.columns:
            wide[col] = np.nan

    summary = {
        "negative_removed": neg_count,
        "missing_pm25": int(wide["PM2.5"].isna().sum()),
        "missing_no2":  int(wide["NO2"].isna().sum()),
    }

    meta = df[["station_id", "City", "Country Label"]].drop_duplicates()
    df_wide = wide.merge(meta, on="station_id", how="left")

    return df_wide, summary


def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Compute three descriptive statistics from the cleaned DataFrame.

    Returns
    -------
    dict with keys:
        avg_pm25_by_city        — average PM2.5 per city, descending
        avg_no2_by_month        — monthly average NO2 across all stations
        missing_pm25_by_country — % missing PM2.5 readings per country
    """
    avg_pm25_by_city = (
        df.groupby("City", dropna=False)["PM2.5"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"PM2.5": "avg_PM2.5"})
    )

    df = df.copy()
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df["month"] = df["Last Updated"].dt.to_period("M").astype(str)

    avg_no2_by_month = (
        df.groupby("month")["NO2"]
        .mean()
        .reset_index()
        .rename(columns={"NO2": "avg_NO2"})
    )

    country_stats = (
        df.groupby("Country Label")["PM2.5"]
        .agg(total="count", missing=lambda s: s.isna().sum())
        .reset_index()
    )
    country_stats["pct_missing_PM2.5"] = np.where(
        country_stats["total"] > 0,
        (country_stats["missing"] / country_stats["total"]) * 100,
        np.nan,
    )

    return {
        "avg_pm25_by_city": avg_pm25_by_city,
        "avg_no2_by_month": avg_no2_by_month,
        "missing_pm25_by_country": country_stats[
            ["Country Label", "pct_missing_PM2.5"]
        ],
    }


def plot_top_cities(avg_pm25: pd.DataFrame, top_n: int = 10) -> None:
    """Bar chart — top N cities by average PM2.5."""
    top = avg_pm25.dropna(subset=["avg_PM2.5"]).head(top_n)
    if top.empty:
        print("No PM2.5 data available.")
        return
    fig = px.bar(
        top, x="City", y="avg_PM2.5",
        title=f"Top {top_n} Cities by Average PM2.5",
        labels={"avg_PM2.5": "Average PM2.5 (µg/m³)"},
        color="City",
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    fig.show()


def plot_monthly_no2(avg_no2: pd.DataFrame) -> None:
    """Line chart — monthly average NO2 across all stations."""
    if avg_no2.empty:
        print("No NO2 data available.")
        return
    fig = px.line(
        avg_no2, x="month", y="avg_NO2",
        title="Monthly Average NO2 Across All Stations",
        labels={"avg_NO2": "Average NO2 (µg/m³)", "month": "Month"},
    )
    fig.update_traces(mode="lines+markers")
    fig.show()


def plot_pm25_vs_no2(df: pd.DataFrame, city: str = None) -> None:
    """
    Scatter plot of PM2.5 vs NO2 for a single city
    with a linear regression line overlaid.
    Defaults to the city with the most paired readings.
    """
    paired = df.dropna(subset=["PM2.5", "NO2"])
    if paired.empty:
        print("No paired PM2.5/NO2 data available.")
        return

    if city is None:
        city = paired.groupby("City").size().idxmax()

    cdf = paired[paired["City"] == city]
    if len(cdf) < 3:
        print(f"Not enough data for '{city}'.")
        return

    x, y = cdf["PM2.5"].to_numpy(), cdf["NO2"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)

    fig = px.scatter(
        cdf, x="PM2.5", y="NO2",
        title=f"PM2.5 vs NO2 — {city}",
        labels={"PM2.5": "PM2.5 (µg/m³)", "NO2": "NO2 (µg/m³)"},
    )
    fig.add_scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines", name="Regression Line",
        line={"color": "red"},
    )
    fig.show()


def openaq_analysis(parquet_path: str = "openaq_sample.parquet",
                    city_for_scatter: str = None):
    """
    Full pipeline: load, clean, compute statistics, visualise.

    Parameters
    ----------
    parquet_path : str
        Path to raw OpenAQ parquet file.
    city_for_scatter : str or None
        City for PM2.5 vs NO2 scatter plot.

    Returns
    -------
    tuple: (df_cleaned, avg_pm25_by_city, avg_no2_by_month,
            missing_pm25_by_country)
    """
    df, summary = load_and_clean(parquet_path)

    print(f"Negative readings removed : {summary['negative_removed']}")
    print(f"Missing PM2.5             : {summary['missing_pm25']}")
    print(f"Missing NO2               : {summary['missing_no2']}")

    df.to_parquet("cleaned_openaq.parquet", index=False)
    print("Cleaned data saved to cleaned_openaq.parquet")

    stats = compute_statistics(df)
    plot_top_cities(stats["avg_pm25_by_city"])
    plot_monthly_no2(stats["avg_no2_by_month"])
    plot_pm25_vs_no2(df, city=city_for_scatter)

    return (
        df,
        stats["avg_pm25_by_city"],
        stats["avg_no2_by_month"],
        stats["missing_pm25_by_country"],
    )


if __name__ == "__main__":
    df_c, pm25, no2, missing = openaq_analysis("openaq_sample.parquet")
    print("\nTop 10 cities by avg PM2.5:")
    print(pm25.head(10))
    print("\nMonthly NO2 (first 6 months):")
    print(no2.head(6))
    print("\nMissing PM2.5 by country:")
    print(missing.head(10))
