"""
Synthetic Geospatial Satellite Data Generator & Sub-Pixel Processor
Generates multi-spectral satellite tiles (10m medium resolution and 2.5m target resolution)
"""

import numpy as np

def generate_synthetic_satellite_tile(height: int = 128, width: int = 128, num_bands: int = 4):
    """
    Simulates a multi-spectral medium-resolution satellite tile.
    Bands: 0: Red, 1: Green, 2: Blue, 3: Near-Infrared (NIR)
    """
    np.random.seed(42)
    # Generate structured spatial patterns (simulating urban, vegetation, and water bodies)
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    
    pattern1 = np.sin(xx) * np.cos(yy) # Urban / Built-up structure
    pattern2 = np.cos(xx / 2) * np.sin(yy / 2) # Vegetation / Forest
    water_mask = (pattern1 + pattern2) < -0.5 # Water body
    
    # Construct 4 spectral channels
    red = np.clip(0.3 + 0.4 * pattern1 + 0.1 * np.random.randn(height, width), 0.0, 1.0)
    green = np.clip(0.4 + 0.3 * pattern2 + 0.1 * np.random.randn(height, width), 0.0, 1.0)
    blue = np.clip(0.2 + 0.2 * pattern1 + 0.1 * np.random.randn(height, width), 0.0, 1.0)
    nir = np.clip(0.6 * pattern2 + 0.3 * (1 - water_mask) + 0.1 * np.random.randn(height, width), 0.0, 1.0)
    
    # Apply water body spectral signature (Low NIR, High Blue/Green)
    blue[water_mask] = 0.8
    green[water_mask] = 0.4
    red[water_mask] = 0.1
    nir[water_mask] = 0.05
    
    tile = np.stack([red, green, blue, nir], axis=0) # [4, H, W]
    return tile

def calculate_ndvi(tile: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index (NDVI = (NIR - Red) / (NIR + Red))"""
    red = tile[0]
    nir = tile[3]
    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

if __name__ == "__main__":
    tile = generate_synthetic_satellite_tile(64, 64)
    ndvi = calculate_ndvi(tile)
    print(f"Generated Satellite Tile shape: {tile.shape}")
    print(f"NDVI Min: {ndvi.min():.3f}, Max: {ndvi.max():.3f}, Mean: {ndvi.mean():.3f}")
