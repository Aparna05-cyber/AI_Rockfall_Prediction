# generate_3d_terrain.py

import pandas as pd
from utils import generate_realistic_dem, compute_risk_map_with_dataset, plot_interactive_risk_3d, add_rocks_to_plot

# -----------------------------
# 1️⃣ Load Synthetic Dataset
# -----------------------------
dataset_path = "scripts/synthetic_dataset/synthetic_rockfall_dataset.csv"
rockfall_data = pd.read_csv(dataset_path)

# -----------------------------
# 2️⃣ Generate DEM
# -----------------------------
dem = generate_realistic_dem(rows=150, cols=150, scale=50, octaves=6)

# -----------------------------
# 3️⃣ Compute Risk Map using DEM + Dataset
# -----------------------------
risk_map = compute_risk_map_with_dataset(dem, rockfall_data, slope_weight=0.7, data_weight=0.3, use_elevation=True)

# -----------------------------
# 4️⃣ Plot Interactive Risk Map
# -----------------------------
# 4️⃣ Plot Interactive Risk Map
fig = plot_interactive_risk_3d(dem, risk_map, dataset=rockfall_data, output_path="outputs/terrain_risk_map.html")


# -----------------------------
# 5️⃣ Add Rocks to High-Risk Areas
# -----------------------------
fig = add_rocks_to_plot(fig, dem, risk_map, num_rocks=50)

# -----------------------------
# 6️⃣ Show Final Visualization
# -----------------------------
fig.show()
