"""
web_scraping/ballon_dor_analysis.py

Scrapes Ballon d'Or award data from Wikipedia and analyses
winner trends across players, nations, and clubs.

Usage:
    python ballon_dor_analysis.py
"""

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


HEADER_MAP = {
    "year": "Year", "season": "Year",
    "player": "Player", "name": "Player", "footballer": "Player",
    "nationality": "Nationality", "country": "Nationality", "nation": "Nationality",
    "club": "Club", "team": "Club",
    "points": "Points", "votes": "Points", "ballots": "Points",
    "rank": "Rank", "position": "Rank", "place": "Rank",
}

DEFAULT_HEADERS = ["year", "rank", "player", "nationality", "club", "points"]


def clean_text(x) -> str:
    """Strip Wikipedia citation brackets and extra whitespace."""
    if pd.isna(x):
        return x
    s = str(x)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\(\d+\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a webpage and return parsed HTML."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_row(values: list, headers: list) -> dict | None:
    """
    Map a table row to a standardised dictionary.
    Handles both regular and irregular Wikipedia table layouts.
    """
    row = {
        "Year": None, "Rank": None, "Player": None,
        "Nationality": None, "Club": None, "Points": None
    }

    if len(values) == len(headers):
        for i, v in enumerate(values):
            if headers[i] in row:
                row[headers[i]] = v
    else:
        for idx, v in enumerate(values):
            match = re.search(r"\b(19|20)\d{2}\b", v)
            if match:
                row["Year"] = int(match.group(0))
                for field, val in zip(
                    ["Rank", "Player", "Nationality", "Club", "Points"],
                    values[idx + 1:]
                ):
                    row[field] = val
                break

    if row["Year"] is None:
        return None

    if row["Points"] is not None:
        try:
            row["Points"] = float(str(row["Points"]).replace(",", "").strip())
        except (ValueError, TypeError):
            row["Points"] = None

    return row


def scrape_tables(soup: BeautifulSoup, start: int, end: int) -> list:
    """Pull all valid award rows from Wikipedia tables within the year range."""
    rows = []
    for table in soup.find_all("table", {"class": "wikitable"}):
        trs = table.find_all("tr")
        if not trs:
            continue

        raw_headers = [
            clean_text(h.get_text(separator=" ", strip=True)).lower()
            for h in trs[0].find_all(["th", "td"])
        ]
        if len(raw_headers) < 3:
            raw_headers = DEFAULT_HEADERS

        norm_headers = [HEADER_MAP.get(h, h.title()) for h in raw_headers]

        for tr in trs[1:]:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            values = [clean_text(c.get_text(separator=" ", strip=True)) for c in cells]
            row = parse_row(values, norm_headers)
            if row is None:
                continue
            try:
                yr = int(row["Year"])
            except (ValueError, TypeError):
                continue
            if not (start <= yr <= end):
                continue
            row["Year"] = yr
            rows.append(row)

    return rows


def plot_player_wins(df: pd.DataFrame, start: int, end: int) -> None:
    """Bar chart — top 10 players by number of Ballon d'Or wins."""
    wins = df[df["Winner"]]["Player"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    wins.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"Top 10 Ballon d'Or Winners ({start}–{end})")
    ax.set_ylabel("Wins")
    ax.set_xlabel("Player")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_group_scatter(df: pd.DataFrame, group_col: str,
                       start: int, end: int, color: str) -> None:
    """Scatter plot — unique top-3 players vs wins, grouped by nation or club."""
    subset = df[df[group_col].astype(bool)]
    unique_players = subset.groupby(group_col)["Player"].nunique()
    wins = subset[subset["Winner"]].groupby(group_col).size()
    scatter_df = pd.DataFrame({
        "UniqueTop3Players": unique_players,
        "Wins": wins
    }).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(scatter_df["UniqueTop3Players"], scatter_df["Wins"],
               color=color, alpha=0.8, s=100)
    ax.set_title(f"{group_col} Analysis ({start}–{end})")
    ax.set_xlabel("Unique Players in Top 3")
    ax.set_ylabel("Total Wins")
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(scatter_df) < 30:
        for name, row in scatter_df.iterrows():
            ax.annotate(name, (row["UniqueTop3Players"], row["Wins"]),
                        fontsize=8, alpha=0.9)
    plt.tight_layout()
    plt.show()


def ballon_dor_scraper(url: str, start: int, end: int,
                       perf_query: str, viz: str = "None") -> pd.DataFrame:
    """
    Scrape and analyse Ballon d'Or data from Wikipedia.

    Parameters
    ----------
    url : str
        Wikipedia page URL.
    start : int
        Start year (inclusive).
    end : int
        End year (inclusive).
    perf_query : str
        Player name to look up.
    viz : str
        One of 'Player', 'Nation', 'Club', or 'None'.

    Returns
    -------
    pd.DataFrame
        Top-3 finishers per year with Winner flag.
    """
    soup = fetch_page(url)
    raw_rows = scrape_tables(soup, start, end)

    df = pd.DataFrame(raw_rows)
    for col in ["Year", "Player", "Nationality", "Club", "Points", "Winner"]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["Player", "Nationality", "Club"]:
        df[col] = df[col].astype(str).apply(clean_text)

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    df_top3 = df.loc[
        df.groupby("Year")["Points"].nlargest(3).index.get_level_values(1)
    ].copy()
    df_top3["Winner"] = df_top3.groupby("Year")["Points"].transform(
        lambda x: x == x.max()
    )

    query_clean = clean_text(perf_query)
    in_top3 = int((df_top3["Player"] == query_clean).sum())
    wins = int(((df_top3["Player"] == query_clean) & df_top3["Winner"]).sum())
    print(f"{perf_query} — Top 3 finishes: {in_top3} | Wins: {wins} ({start}–{end})")

    if viz == "Player":
        plot_player_wins(df_top3, start, end)
    elif viz == "Nation":
        plot_group_scatter(df_top3, "Nationality", start, end, "#55A868")
    elif viz == "Club":
        plot_group_scatter(df_top3, "Club", start, end, "#C44E52")

    return df_top3[
        ["Year", "Player", "Nationality", "Club", "Points", "Winner"]
    ].reset_index(drop=True)


if __name__ == "__main__":
    URL = "https://en.wikipedia.org/wiki/Ballon_d%27Or"
    df = ballon_dor_scraper(
        url=URL,
        start=2000,
        end=2023,
        perf_query="Lionel Messi",
        viz="Player"
    )
    print(df.head(10))
