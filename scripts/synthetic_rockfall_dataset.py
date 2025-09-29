import numpy as np
import pandas as pd
from utils import generate_realistic_dem, compute_risk_map
import os

# ----------------------------
# Settings
# ----------------------------
rows, cols = 150, 150

# ----------------------------
# 1️⃣ Generate DEM and risk
# ----------------------------
dem = generate_realistic_dem(rows=rows, cols=cols, scale=100, octaves=6)
risk_map = compute_risk_map(dem)

# ----------------------------
# 2️⃣ Generate additional synthetic attributes
# ----------------------------
np.random.seed(42)

# Geography: categorical
geography_types = ["hill", "plain", "valley", "cliff"]
geography = np.random.choice(geography_types, size=(rows, cols))

# Rainfall (mm/year)
rainfall = np.random.normal(loc=1200, scale=300, size=(rows, cols))
rainfall = np.clip(rainfall, 500, 2000)

# Rainfall pattern
rainfall_patterns = ["seasonal", "uniform", "irregular"]
rainfall_pattern = np.random.choice(rainfall_patterns, size=(rows, cols))

# Soil type
soil_types = ["clay", "sand", "loam", "rocky"]
soil_type = np.random.choice(soil_types, size=(rows, cols))

# Livelihood (settlement density)
livelihood_types = ["low", "medium", "high"]
livelihood = np.random.choice(livelihood_types, size=(rows, cols))

# Industrial activity
industrial_types = ["low", "medium", "high"]
industrial = np.random.choice(industrial_types, size=(rows, cols))

# Pitmine radius (m)
pitmine_radius = np.random.uniform(50, 500, size=(rows, cols))

# Pitmine depth (m)
pitmine_depth = np.random.uniform(10, 100, size=(rows, cols))

# Disaster type based on risk
disaster = np.where(risk_map > 0.6, "rockfall",
            np.where(risk_map > 0.3, "landslide", "none"))

# ----------------------------
# 3️⃣ Flatten all arrays for CSV
# ----------------------------
gx, gy = np.gradient(dem)
slope_angle = np.sqrt(gx**2 + gy**2).flatten()  # approximate slope

data = {
    "X": np.repeat(np.arange(cols), rows),
    "Y": np.tile(np.arange(rows), cols),
    "Elevation": dem.flatten(),
    "Slope Angle": slope_angle,
    "Geography": geography.flatten(),
    "Rainfall": rainfall.flatten(),
    "Rainfall Pattern": rainfall_pattern.flatten(),
    "Soil Type": soil_type.flatten(),
    "Livelihood": livelihood.flatten(),
    "Industrial Activity": industrial.flatten(),
    "Pitmine Radius": pitmine_radius.flatten(),
    "Pitmine Depth": pitmine_depth.flatten(),
    "Risk": risk_map.flatten(),
    "Disaster": disaster.flatten()
}

df = pd.DataFrame(data)

# ----------------------------
# 4️⃣ Save CSV
# ----------------------------
output_folder = os.path.join(os.path.dirname(__file__), "synthetic_dataset")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "synthetic_rockfall_dataset.csv")

# Save CSV with UTF-8 BOM (Excel-friendly)
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Synthetic rockfall dataset saved to:\n{output_file}")
