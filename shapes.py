"""lidarv2 shapes — classify square / rectangle / cube / cuboid.

Focused on the primitives the user cares about: squares, rectangles and cubes.
A 3-D box is called a cube when all sides match, a cuboid when it is genuinely
thick (thickness at least half the smaller face dimension), and a rectangle
when the thickness is a thin panel.
"""


def _close(a, b, tol):
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


def classify_shape(w_mm, h_mm, d_mm=None, tol=0.08):
    """Return a shape description for an emulated width/height/thickness."""
    w = abs(float(w_mm or 0.0))
    h = abs(float(h_mm or 0.0))
    d = abs(float(d_mm)) if d_mm else None
    if w <= 0 or h <= 0:
        return {"shape": "unknown", "is_square": False, "is_rectangle": False,
                "is_cube": False, "is_cuboid": False, "is_lumber": False,
                "aspect": None}

    is_square = _close(w, h, tol)
    is_cube = bool(is_square and d is not None and _close(w, d, tol))
    aspect = max(w, h) / min(w, h)
    is_lumber = aspect >= 4.0
    is_thick = bool(d is not None and d >= 0.5 * min(w, h))

    if is_cube:
        shape = "cube"
    elif is_square:
        shape = "square"
    elif is_lumber:
        shape = "lumber"
    elif is_thick:
        shape = "cuboid"
    else:
        shape = "rectangle"

    return {
        "shape": shape,
        "is_square": is_square,
        "is_rectangle": shape in ("rectangle", "lumber"),
        "is_cube": is_cube,
        "is_cuboid": shape == "cuboid",
        "is_lumber": is_lumber,
        "aspect": round(aspect, 3),
        "faces": {"square": 1, "rectangle": 1, "lumber": 1,
                  "cube": 6, "cuboid": 6}.get(shape, 0),
    }
