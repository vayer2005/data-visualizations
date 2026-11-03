#!/usr/bin/env python3
"""Process raw South Korea career-aspiration and alcohol inputs into tidy CSVs and plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "output"
PLOTS = OUT / "plots"

CAREER_CATEGORIES = [
    "national_agency",
    "public_corporation",
    "large_corporation",
    "self_employment",
    "professional_firm",
    "foreign_affiliate",
    "sme",
    "overseas_employment",
    "venture",
    "other",
]
K = len(CAREER_CATEGORIES)
OECD_BREAK_YEAR = 1988


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_career_shares() -> pd.DataFrame:
    path = RAW / "career_workplace_shares.csv"
    if not path.exists():
        fail(f"Missing {path.relative_to(ROOT)}")
    df = pd.read_csv(path)
    required = {"year", "age_group", "category", "share_pct"}
    if not required.issubset(df.columns):
        fail(f"{path.name} must contain columns: {sorted(required)}")
    df = df[df["category"].isin(CAREER_CATEGORIES)].copy()
    if df.empty:
        fail("No recognized workplace categories in career shares file")
    dup = df.duplicated(subset=["year", "age_group", "category"], keep=False)
    if dup.any():
        fail("Duplicate year/age_group/category rows in career shares")
    df["share"] = df["share_pct"] / 100.0
    df["source_id"] = "youth_stats_social_survey_workplace"
    tidy = df[["year", "category", "share", "age_group", "source_id"]].copy()
    return tidy


def compute_concentration(shares: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, age_group), grp in shares.groupby(["year", "age_group"]):
        if set(grp["category"]) != set(CAREER_CATEGORIES):
            missing = set(CAREER_CATEGORIES) - set(grp["category"])
            fail(f"Year {year} ({age_group}) missing categories: {sorted(missing)}")
        p = grp.set_index("category").loc[CAREER_CATEGORIES, "share"].astype(float)
        total = p.sum()
        if not np.isclose(total, 1.0, atol=0.02):
            fail(
                f"Year {year} ({age_group}) shares sum to {total:.4f}; "
                "expected ~1.0 after published rounding"
            )
        if not np.isclose(total, 1.0):
            p = p / total
        hhi = float((p**2).sum())
        norm = (hhi - 1 / K) / (1 - 1 / K)
        rows.append(
            {
                "year": int(year),
                "hhi": hhi,
                "normalized_hhi": norm,
                "effective_categories": 1 / hhi,
                "age_group": age_group,
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def load_oecd_alcohol() -> pd.DataFrame:
    path = RAW / "oecd_alcohol_kor.csv"
    if not path.exists():
        fail(f"Missing {path.relative_to(ROOT)}")
    df = pd.read_csv(path)
    year_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else "Time period"
    val_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else "Observation value"
    status_col = "OBS_STATUS" if "OBS_STATUS" in df.columns else "Observation status"
    if year_col not in df.columns or val_col not in df.columns:
        fail(f"Unexpected OECD CSV columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
            "alcohol_value": pd.to_numeric(df[val_col], errors="coerce"),
            "unit": "litres_pure_alcohol_per_capita_15plus",
            "age_group": "15+",
            "measure_definition": "OECD sales-based alcohol consumption",
            "source_id": "oecd_health_lvng_ac_kor",
        }
    )
    if status_col in df.columns:
        out["obs_status"] = df[status_col].fillna("")
    out = out.dropna(subset=["year", "alcohol_value"]).copy()
    out["year"] = out["year"].astype(int)
    if out["year"].duplicated().any():
        fail("Duplicate years in OECD alcohol series")
    return out.sort_values("year")


def load_khepi_youth_drinking() -> pd.DataFrame:
    path = RAW / "khepi_monthly_drinking.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_age = "19-29(연령별)"
    if col_age not in df.columns:
        return pd.DataFrame()
    rows = []
    for year, grp in df.groupby("연도"):
        vals = pd.to_numeric(grp[col_age], errors="coerce")
        if vals.isna().all():
            continue
        rows.append(
            {
                "year": int(year),
                "alcohol_value": float(vals.mean()),
                "unit": "percent",
                "age_group": "19-29",
                "measure_definition": "KNHANES monthly drinking prevalence (sex-averaged published rates)",
                "source_id": "khepi_knhanes_monthly_drinking",
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_alcohol_table(oecd: pd.DataFrame, khepi: pd.DataFrame) -> pd.DataFrame:
    primary = oecd[
        ["year", "alcohol_value", "unit", "age_group", "measure_definition", "source_id"]
    ].copy()
    if not khepi.empty:
        primary = pd.concat([primary, khepi], ignore_index=True)
    return primary.sort_values(["source_id", "year"])


def build_analysis(concentration: pd.DataFrame, oecd: pd.DataFrame) -> pd.DataFrame:
    career = concentration[concentration["age_group"] == "13-24"][
        ["year", "normalized_hhi", "hhi", "effective_categories"]
    ].rename(columns={"normalized_hhi": "career_concentration"})
    merged = career.merge(
        oecd[["year", "alcohol_value"]].rename(columns={"alcohol_value": "alcohol_litres_15plus"}),
        on="year",
        how="inner",
    )
    return merged.sort_values("year")


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=1)


def plot_panels(concentration: pd.DataFrame, oecd: pd.DataFrame) -> None:
    career = concentration[concentration["age_group"] == "13-24"].sort_values("year")
    alc = oecd.sort_values("year")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        career["year"],
        career["normalized_hhi"],
        marker="o",
        color="#c5d2b2",
        linewidth=2,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Normalized concentration (0–1)")
    axes[0].set_title("Workplace-aspiration concentration (Social Survey, ages 13–24)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(alc["year"], alc["alcohol_value"], marker="o", color="#e2a84a", linewidth=2)
    if (alc["year"] == OECD_BREAK_YEAR).any():
        axes[1].axvline(OECD_BREAK_YEAR, color="#ff5a36", linestyle="--", alpha=0.7, linewidth=1)
        axes[1].text(
            OECD_BREAK_YEAR + 0.5,
            alc["alcohol_value"].max() * 0.95,
            "OECD series break",
            fontsize=8,
            color="#ff5a36",
        )
    axes[1].set_ylabel("Litres pure alcohol\n(per capita, 15+)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Alcohol consumption (OECD sales-based)")
    axes[1].grid(True, alpha=0.25)

    fig.savefig(PLOTS / "panel_timeseries.png", dpi=160)
    plt.close(fig)


def plot_standardized(analysis: pd.DataFrame) -> None:
    years = analysis["year"]
    z_c = zscore(analysis["career_concentration"])
    z_a = zscore(analysis["alcohol_litres_15plus"])

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(years, z_c, marker="o", label="Career concentration (z)", color="#c5d2b2", linewidth=2)
    ax.plot(years, z_a, marker="s", label="Alcohol consumption (z)", color="#e2a84a", linewidth=2)
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardized value (overlap years only)")
    ax.set_title(
        f"Standardized co-movement ({len(analysis)} paired years: "
        f"{years.min()}–{years.max()})"
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(PLOTS / "standardized_overlay.png", dpi=160)
    plt.close(fig)


def plot_scatter(analysis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    x = analysis["career_concentration"]
    y = analysis["alcohol_litres_15plus"]
    ax.scatter(x, y, s=70, color="#f4f1e8", edgecolors="#10120c", linewidths=0.8, zorder=3)
    for _, row in analysis.iterrows():
        ax.annotate(
            str(int(row["year"])),
            (row["career_concentration"], row["alcohol_litres_15plus"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    if len(analysis) >= 3:
        slope, intercept, r, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#9a9486", linestyle="--", linewidth=1)
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    ax.set_xlabel("Normalized workplace-aspiration concentration")
    ax.set_ylabel("OECD alcohol (litres per capita, 15+)")
    ax.set_title("Career concentration vs alcohol by survey year")
    ax.grid(True, alpha=0.25)
    fig.savefig(PLOTS / "scatter_concentration_alcohol.png", dpi=160)
    plt.close(fig)


def write_sources() -> pd.DataFrame:
    rows = [
        {
            "source_id": "youth_stats_social_survey_workplace",
            "publisher": "Ministry of Gender Equality and Family / KIHYS",
            "dataset": "2024 Youth Statistics press PDF",
            "table_or_indicator_id": "IV-18 preferred workplace (Social Survey)",
            "url": "raw/2024_youth_statistics_press.pdf",
            "access_date": "2025-11-03",
            "observation_years": "2011,2013,2015,2017,2019,2021,2023",
            "population": "ages 13-24",
            "definition": "Single-choice preferred workplace type; shares from published table",
            "notes": "Biennial Social Survey waves; extracted to career_workplace_shares.csv",
        },
        {
            "source_id": "oecd_health_lvng_ac_kor",
            "publisher": "OECD",
            "dataset": "Health living conditions – alcohol consumption",
            "table_or_indicator_id": "DF_HEALTH_LVNG_AC / KOR",
            "url": "raw/oecd_alcohol_kor.csv",
            "access_date": "2025-11-03",
            "observation_years": "1961-2024",
            "population": "15+",
            "definition": "Sales-based litres of pure alcohol per capita",
            "notes": "OBS_STATUS break flagged at 1988 in plots",
        },
        {
            "source_id": "khepi_knhanes_monthly_drinking",
            "publisher": "KDCA / KHEPI",
            "dataset": "KNHANES monthly drinking rates",
            "table_or_indicator_id": "khepi_monthly_drinking.csv",
            "url": "raw/khepi_monthly_drinking.csv",
            "access_date": "2025-11-03",
            "observation_years": "2005-2023",
            "population": "19-29 (sex-averaged from published male/female rows)",
            "definition": "Monthly drinking prevalence percent",
            "notes": "Supplementary only; not merged into primary paired analysis",
        },
        {
            "source_id": "worldbank_sh_alc_pcap_li",
            "publisher": "World Bank",
            "dataset": "Health nutrition and population",
            "table_or_indicator_id": "SH.ALC.PCAP.LI",
            "url": "raw/worldbank_SH_ALC_PCAP_LI_KR.json",
            "access_date": "2025-11-03",
            "observation_years": "2000-2020",
            "population": "15+",
            "definition": "WHO total consumption projected estimates",
            "notes": "Archived raw; primary analysis uses OECD only",
        },
    ]
    return pd.DataFrame(rows)


def write_stats_report(analysis: pd.DataFrame) -> dict:
    c = analysis["career_concentration"]
    a = analysis["alcohol_litres_15plus"]
    pearson = stats.pearsonr(c, a) if len(analysis) >= 2 else (np.nan, np.nan)
    spearman = stats.spearmanr(c, a) if len(analysis) >= 2 else (np.nan, np.nan)
    return {
        "paired_years": len(analysis),
        "years": list(analysis["year"].astype(int)),
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    career_shares = load_career_shares()
    concentration = compute_concentration(career_shares)
    oecd = load_oecd_alcohol()
    khepi = load_khepi_youth_drinking()
    alcohol = build_alcohol_table(oecd, khepi)
    analysis = build_analysis(concentration, oecd)
    sources = write_sources()
    report = write_stats_report(analysis)

    career_shares.to_csv(OUT / "career_shares.csv", index=False)
    concentration.to_csv(OUT / "career_concentration.csv", index=False)
    alcohol.to_csv(OUT / "alcohol.csv", index=False)
    analysis.to_csv(OUT / "analysis.csv", index=False)
    sources.to_csv(OUT / "sources.csv", index=False)

    with open(OUT / "association_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_panels(concentration, oecd)
    plot_standardized(analysis)
    plot_scatter(analysis)

    print(f"Wrote outputs to {OUT.relative_to(ROOT)}/")
    print(f"Paired survey years ({report['paired_years']}): {report['years']}")
    print(f"Pearson r = {report['pearson_r']:.3f}, Spearman rho = {report['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
