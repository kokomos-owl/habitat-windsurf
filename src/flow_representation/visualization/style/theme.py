"""Theme configuration for climate risk visualizations."""

# Background colors
BACKGROUND_COLOR = '#22333b'  # Dark blue-gray background

# Metric colors - each metric has a light and dark variant
COLORS = {
    'coherence': ['#90e0ef', '#0077b6'],           # Light blue to dark blue
    'cross_pattern_flow': ['#fcbf49', '#d62828'],  # Yellow to red
    'emergence_rate': ['#80ed99', '#38a3a5'],      # Light green to teal
    'social_support': ['#c9ada7', '#4a4e69']       # Light pink to purple
}

# Text colors
TEXT_COLOR = '#ffffff'  # White text
LABEL_COLOR = '#cccccc'  # Light gray labels

# Map styling
MAP_EDGE_COLOR = 'black'
MAP_EDGE_WIDTH = 0.4

# Plot dimensions
FIGURE_SIZE = (8, 10)  # Width, height in inches
DPI = 100  # Dots per inch for raster output

# Font configuration
FONT_FAMILY = 'sans-serif'
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 8

# Layout configuration
GRID_SPEC = {
    'nrows': 3,
    'ncols': 2,
    'height_ratios': [1, 1, 1],
    'width_ratios': [1, 1]
}

# Map bounds for Martha's Vineyard
MAP_BOUNDS = {
    'x_min': -70.85,  # Western longitude
    'x_max': -70.45,  # Eastern longitude
    'y_min': 41.30,   # Southern latitude
    'y_max': 41.50    # Northern latitude
}
