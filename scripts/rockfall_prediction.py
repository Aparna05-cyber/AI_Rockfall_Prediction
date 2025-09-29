import numpy as np
from utils import generate_mock_dem, plot_heatmap, plot_interactive_risk_3d
import os

# Ensure outputs folder exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
heatmap_path = os.path.join(output_dir, "heatmap.png")

# Generate simulated DEM
dem = generate_mock_dem(rows=200, cols=200)

# Calculate slope and risk
gradient_x, gradient_y = np.gradient(dem)
slope = np.sqrt(gradient_x**2 + gradient_y**2)
slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())
risk_map = slope_normalized  # 0=low, 1=high

# 2D heatmap
plot_heatmap(dem, risk_map, heatmap_path)

# Interactive 3D risk map
plot_interactive_risk_3d(dem, risk_map)
