import numpy as np
from perlin_noise import PerlinNoise
import plotly.graph_objects as go
import os

# -----------------------------
# 1️⃣ Generate Realistic DEM
# -----------------------------
def generate_realistic_dem(rows=150, cols=150, scale=50, octaves=6):
    """
    Generate a realistic DEM using Perlin noise.
    rows, cols: dimensions of DEM
    scale: maximum elevation
    octaves: noise complexity
    """
    noise = PerlinNoise(octaves=octaves)
    dem = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            dem[i, j] = noise([i / rows, j / cols])
    # Normalize to 0-1 and scale
    dem = (dem - dem.min()) / (dem.max() - dem.min()) * scale
    return dem


# -----------------------------
# 2️⃣ Compute Risk Map (Slope only)
# -----------------------------
def compute_risk_map(dem):
    """
    Compute slope-based rockfall risk map.
    Risk is higher where slope is steeper.
    Returns a normalized risk map (0-1).
    """
    gx, gy = np.gradient(dem)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    risk_map = (slope - slope.min()) / (slope.max() - slope.min())
    return risk_map


# -----------------------------
# 3️⃣ Compute Risk Map (DEM + Dataset)
# -----------------------------
def compute_risk_map_with_dataset(dem, dataset, slope_weight=0.7, data_weight=0.3, use_elevation=False):
    """
    Compute combined risk map using DEM slope + dataset risk values.
    - slope_weight: contribution from terrain slope
    - data_weight: contribution from dataset's "Risk"
    - use_elevation: if True, dataset Elevation overrides DEM at those points
    """
    # Optionally override DEM with dataset's Elevation values
    if use_elevation and 'Elevation' in dataset.columns:
        for _, row in dataset.iterrows():
            x, y, elev = int(row['X']), int(row['Y']), row['Elevation']
            if 0 <= y < dem.shape[0] and 0 <= x < dem.shape[1]:
                dem[y, x] = elev

    # Slope-based risk
    gx, gy = np.gradient(dem)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    slope_risk = (slope - slope.min()) / (slope.max() - slope.min())

    # Dataset-based risk
    data_risk = np.zeros_like(dem)
    if 'Risk' in dataset.columns:
        for _, row in dataset.iterrows():
            x, y, risk = int(row['X']), int(row['Y']), row['Risk']
            if 0 <= y < dem.shape[0] and 0 <= x < dem.shape[1]:
                data_risk[y, x] = risk

    # Weighted combination
    combined_risk = slope_weight * slope_risk + data_weight * data_risk
    combined_risk = np.clip(combined_risk, 0, 1)
    return combined_risk


# -----------------------------
# 4️⃣ Plot Interactive Risk Map
# -----------------------------
def plot_interactive_risk_3d(dem, risk_map, dataset=None, output_path=None):
    """
    Plot interactive 3D surface showing risk (color = risk, height = elevation).
    If dataset is provided, additional attributes are shown on hover.
    """
    rows, cols = dem.shape
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    X, Y = np.meshgrid(x, y)

    # Risk categories
    risk_percent = risk_map * 100
    categories = np.where(
        risk_percent < 33, "Low",
        np.where(risk_percent < 66, "Medium", "High")
    )

    # Build customdata array for hover
    customdata = np.stack([risk_map, risk_percent, categories], axis=-1)

    hovertemplate = (
        "X: %{x}<br>"
        "Y: %{y}<br>"
        "Elevation: %{z:.1f} m<br>"
        "Risk: %{customdata[1]:.1f}%<br>"
        "Category: %{customdata[2]}"
    )

    # 🔹 Add dataset attributes if available
    if dataset is not None:
        # Initialize attribute grids
        attr_grids = {}
        for col in dataset.columns:
            if col in ["X", "Y"]:  # Skip coordinates
                continue
            attr_grids[col] = np.full_like(dem, "", dtype=object)

        # Fill grids with dataset values
        for _, row in dataset.iterrows():
            x, y = int(row["X"]), int(row["Y"])
            if 0 <= y < dem.shape[0] and 0 <= x < dem.shape[1]:
                for col in attr_grids.keys():
                    attr_grids[col][y, x] = row[col]

        # Add them into customdata
        extra_attrs = [attr_grids[col] for col in attr_grids.keys()]
        customdata = np.stack([risk_map, risk_percent, categories] + extra_attrs, axis=-1)

        # Extend hover template
        for i, col in enumerate(attr_grids.keys(), start=3):
            hovertemplate += f"<br>{col}: %{{customdata[{i}]}}"

    hovertemplate += "<extra></extra>"

    # Custom colorscale: green → yellow → red
    danger_colorscale = [
        [0.0, "green"],   # low risk
        [0.5, "yellow"],  # medium risk
        [1.0, "red"]      # high risk
    ]

    fig = go.Figure(data=[go.Surface(
        z=dem,
        x=X,
        y=Y,
        surfacecolor=risk_map,
        colorscale=danger_colorscale,
        cmin=0, cmax=1,
        colorbar=dict(title="Rockfall Risk (%)"),
        customdata=customdata,
        hovertemplate=hovertemplate
    )])

    fig.update_layout(
        title="Interactive 3D Rockfall Risk Map with Attributes",
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Elevation (m)")
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Interactive risk map saved to {output_path}")

    return fig


# -----------------------------
# 5️⃣ Add Rocks on High-Risk Areas
# -----------------------------
def add_rocks_to_plot(fig, dem, risk_map, num_rocks=30):
    """
    Overlay rocks as scatter3d markers on high-risk areas.
    """
    rows, cols = dem.shape
    high_risk_indices = np.argwhere(risk_map > 0.6)
    if len(high_risk_indices) == 0:
        return fig

    chosen_indices = high_risk_indices[np.random.choice(
        len(high_risk_indices),
        min(num_rocks, len(high_risk_indices)),
        replace=False
    )]

    x = chosen_indices[:, 1]
    y = chosen_indices[:, 0]
    z = dem[y, x]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z + 1,  # rocks slightly above terrain
        mode="markers",
        marker=dict(size=4, color="gray", symbol="circle"),
        name="Rocks"
    ))

    return fig
