"""One colour language for the whole course.

Every page uses the same roles so a student learns them once:
  grey dashed  -> the ORIGINAL (before) shape
  blue         -> the TRANSFORMED (after) shape
  orange arrow -> where basis vector e1 = (1, 0) lands  (1st column of the matrix)
  green arrow  -> where basis vector e2 = (0, 1) lands  (2nd column of the matrix)
"""

ORIGINAL = "#8f8d87"      # neutral grey - the reference / "before"
TRANSFORMED = "#2a78d6"   # blue - the result / "after"
E1 = "#eb6834"            # orange - image of e1
E2 = "#1baf7a"            # aqua/green - image of e2
HIGHLIGHT = "#eda100"     # amber - special points (midpoints, intersections, pivots)
MAGENTA = "#d55181"       # a 4th role when needed (e.g. points at infinity, horizon)
OUTLIER = "#e34948"       # red - outliers / errors (always paired with a label or symbol)

GRID_ORIGINAL = "rgba(143,141,135,0.28)"
GRID_TRANSFORMED = "rgba(42,120,214,0.42)"
AXIS = "rgba(128,128,128,0.65)"
GRID = "rgba(128,128,128,0.14)"


def rgba(hex_color: str, alpha: float) -> str:
    """'#2a78d6', 0.2 -> 'rgba(42,120,214,0.2)'"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
