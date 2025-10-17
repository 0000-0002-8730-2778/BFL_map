import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.ticker import LogLocator

# ------------------ Configuration ------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ------------------ Constants ------------------
# Country name normalization mapping
COUNTRY_NORMALIZATION = {
    "KOREA, REPUBLIC OF": "South Korea",
    "USA": "United States",
    "UK": "United Kingdom",
    "GERMANY": "Germany",
    "CHINA": "China",
    "INDIA": "India",
    "SPAIN": "Spain",
    "ITALY": "Italy",
    "BRAZIL": "Brazil",
    "CANADA": "Canada",
    "AUSTRALIA": "Australia",
    "PAKISTAN": "Pakistan",
    "EGYPT": "Egypt",
    "NORWAY": "Norway",
    "FINLAND": "Finland",
    "GREECE": "Greece",
    "PORTUGAL": "Portugal",
    "BELGIUM": "Belgium",
    "DENMARK": "Denmark",
    "JAPAN": "Japan",
}

# Raw epidemiological data
RAW_DATA = {
    "Country": [
        "GERMANY", "UNITED STATES", "DENMARK", "JAPAN", "UNITED KINGDOM", "DENMARK", "UNITED KINGDOM", "DENMARK",
        "BELGIUM", "DENMARK", "FINLAND", "GREECE", "NORWAY", "PORTUGAL",
        "KOREA, REPUBLIC OF", "PAKISTAN", "GREECE", "SPAIN", "SPAIN", "INDIA",
        "CHINA", "UNITED STATES", "DENMARK", "INDIA", "UNITED STATES", "ITALY", "BELGIUM",
        "CHINA", "EGYPT", "UNITED STATES", "AUSTRALIA", "BRAZIL", "CANADA", "SPAIN",
        "UNITED KINGDOM", "GERMANY", "CANADA", "INDIA", "INDIA", "INDIA"
    ],
    "Incidence_per_105": [
        2.5, 1.61, 2.0, 2.5, 0.9, 1.16, 2.63, 1.19, 0.5, 1.45,
        1.4, 1.9, 0.55, 1.9, 1.65, None, 4.63, 7.6, None, None,
        None, 32.55, 0.287, None, None, None, None, 0.7536, None,
        None, None, None, None, None, 3.264, None, 1.267875,
        1.77744, 0.56282, 0.57
    ],
    "Prevalence": [
        None, 2.19, 4.0, 6.3, None, None, None, None, 2.75, 4.05, 5.0,
        4.85, 3.35, 8.2, None, 3.22, 17.3, None, 14.41, 1.86, 0.92, 70.3,
        None, 19.71, 6.5, 38.99, 7.34, None, 10.83, 14.25, 12.13, 9.33,
        15.79, 15.05, 31.92, 25.93, None, None, None, 2.46
    ],
}

# ------------------ Utility Functions ------------------
def setup_logging(output_dir: Path):
    """Initialize logging and create log directory."""
    log_file = output_dir / "logs" / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

def normalize_country(name: str) -> str:
    """Normalize country name using COUNTRY_NORMALIZATION mapping."""
    return COUNTRY_NORMALIZATION.get(name, name)

def create_map(df: pd.DataFrame, value_col: str, cmap, title: str, shapefile_path: Path):
    """Create a world map showing the specified value per country."""
    import numpy as np

    if not shapefile_path.exists():
        logging.error(f"Shapefile not found: {shapefile_path}")
        return None

    gdf = gpd.read_file(shapefile_path)
    # Aggregate values by country and normalize names
    country_values = df.groupby("Country")[value_col].mean().dropna().to_dict()
    country_values = {normalize_country(k): v for k, v in country_values.items()}
    gdf['value'] = gdf['NAME'].apply(lambda x: country_values.get(normalize_country(x), np.nan))

    values = list(country_values.values())
    if not values:
        logging.warning(f"No valid data for {value_col}")
        return None
    norm = colors.LogNorm(vmin=min(values), vmax=max(values))

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    # Plot countries with no data (gray)
    gdf[gdf['value'].isna()].plot(ax=ax, transform=ccrs.PlateCarree(),
                                  facecolor='lightgray', edgecolor='black', linewidth=0.3)
    # Plot countries with data (colored)
    gdf[~gdf['value'].isna()].plot(ax=ax, transform=ccrs.PlateCarree(),
                                   facecolor=lambda x: cmap(norm(x['value'])),
                                   edgecolor='black', linewidth=0.3)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, aspect=40)
    cbar.set_label(f"{value_col.replace('_', ' ')} (per 100,000)", fontsize=12, fontweight='bold')
    cbar.locator = LogLocator(subs=(1, 2, 5))
    cbar.update_ticks()

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    return fig

def save_tables(df: pd.DataFrame, output_dir: Path):
    """Save detailed and summary epidemiological tables."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Detailed table
    detail = df.copy()
    detail["Incidence_per_105"] = detail["Incidence_per_105"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detail["Prevalence"] = detail["Prevalence"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detail.to_csv(tables_dir / "epidemiological_data_table.csv", index=False)

    # Summary statistics
    incidence = df["Incidence_per_105"].dropna()
    prevalence = df["Prevalence"].dropna()
    summary = pd.DataFrame({
        "Metric": ["Incidence (per 10⁵)", "Prevalence (per 10⁵)"],
        "Studies": [len(incidence), len(prevalence)],
        "Mean": [f"{incidence.mean():.2f}", f"{prevalence.mean():.2f}"],
        "Median": [f"{incidence.median():.2f}", f"{prevalence.median():.2f}"],
        "SD": [f"{incidence.std():.2f}", f"{prevalence.std():.2f}"],
        "Range": [f"{incidence.min():.2f}-{incidence.max():.2f}", f"{prevalence.min():.2f}-{prevalence.max():.2f}"],
    })
    summary.to_csv(tables_dir / "statistical_summary_table.csv", index=False)
    logging.info("Tables saved")

def main():
    parser = argparse.ArgumentParser(description="Generate epidemiological maps")
    parser.add_argument("--shapefile", required=True, help="Path to world shapefile (.shp)")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    df = pd.DataFrame(RAW_DATA)
    save_tables(df, output_dir)

    shapefile_path = Path(args.shapefile)

    # Generate incidence map
    inc_fig = create_map(df, "Incidence_per_105", plt.cm.Reds, "Global Disease Incidence", shapefile_path)
    if inc_fig:
        (output_dir / "maps").mkdir(parents=True, exist_ok=True)
        inc_fig.savefig(output_dir / "maps" / "disease_incidence.png", dpi=600)
        inc_fig.savefig(output_dir / "maps" / "disease_incidence.pdf")
        plt.close(inc_fig)
        logging.info("Incidence map saved")

    # Generate prevalence map
    prev_fig = create_map(df, "Prevalence", plt.cm.Blues, "Global Disease Prevalence", shapefile_path)
    if prev_fig:
        prev_fig.savefig(output_dir / "maps" / "disease_prevalence.png", dpi=600)
        prev_fig.savefig(output_dir / "maps" / "disease_prevalence.pdf")
        plt.close(prev_fig)
        logging.info("Prevalence map saved")

if __name__ == "__main__":
    main()
