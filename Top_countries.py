import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# === 1. Load data ===
df = pd.read_csv("Pigeons_data.csv")

# === 2. Select top 25 countries by mean abundance and sort ===
top25 = df.sort_values("abundance_mean", ascending=False).head(25).reset_index(drop=True)

# === 3. Set basic style ===
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# === 4. Continent color palette ===
continent_palette = {
    "Asia": "#E07B39",
    "Europe": "#3A7CA5",
    "Africa": "#75954C",
    "North America": "#9C4F96",
    "South America": "#C94C4C",
    "Oceania": "#E1A73C"
}

# === 5. Plotting ===
fig, ax1 = plt.subplots(figsize=(12, 3.5))  # Short-height figure

# Bar chart (range coverage)
colors = top25["continent_name"].map(lambda x: continent_palette.get(x, "gray"))
bars = ax1.bar(
    top25["region_name"],
    top25["range_occupied_percent"] * 100,
    color=colors,
    alpha=0.8,
    width=0.6,
)

ax1.set_ylabel("Range occupied (%)", fontsize=12)
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
ax1.set_xticks(range(len(top25)))
ax1.set_xticklabels(top25["region_name"], rotation=45, ha="right")

# Line chart (mean abundance)
ax2 = ax1.twinx()
ymax = top25["abundance_mean"].max()
ax2.set_ylim(0, ymax*1.1)  # Slightly expand right axis by 10%
line = ax2.plot(
    top25["region_name"],
    top25["abundance_mean"],
    color="black",
    marker="o",
    markersize=6,
    linewidth=2.2,
    label="Mean abundance"
)
ax2.set_ylabel("Mean abundance", fontsize=12)

# === 6. Remove gridlines ===
ax1.grid(False)
ax2.grid(False)

# === 7. Adjust spines ===
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# === 8. Legend positioning ===
continent_handles = [Patch(color=color, label=continent) for continent, color in continent_palette.items()]
line_handle = line[0]
ax1.legend(
    handles=continent_handles + [line_handle],
    loc="upper center",
    frameon=False,
    ncol=7,
    bbox_to_anchor=(0.5, 1.52)  # Fine-tune position above chart
)

# === 9. Title and source text adjustments ===
fig.suptitle(
    "Rock Pigeon Abundance and Range Coverage — Top 25 Countries (2023)",
    fontsize=13, fontweight="bold",
    y=1.05  # Adjust title position
)
fig.text(
    0.99, -0.015,  # Adjust source position
    "Source: eBird Status & Trends 2023",
    ha="right",
    fontsize=8,
    color="gray"
)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.78)  # Leave space for title and legend

plt.show()
