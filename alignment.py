"""
Shared geometry for locating the invoice table's grid border in a scan,
and for mapping between pixel coordinates and PROPORTIONS of that
border ("34.2% down from the border's top"), so a single one-time
calibration (see calibrate_template.py) can be reused on every new
scan without needing a full geometric image warp -- just "find the
border, then look at a known percentage across/down it."

This module only knows about the *border* (its 4 corners) and the
proportion <-> pixel math. It has no idea what rows/columns mean --
that's calibrate_template.py's job (recording proportions) and
extract_invoice.py's job (consuming them), both built on top of this.
"""

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort four unordered 2D points into a consistent
    (top-left, top-right, bottom-right, bottom-left) order.

    Contour/corner detection returns points in an arbitrary order that
    can vary scan to scan, but every other function in this module
    assumes a fixed order -- without this, proportions computed during
    calibration wouldn't line up with proportions applied later.

    Args:
        pts: array of shape (4, 2), four (x, y) corner points in any order.

    Returns:
        Array of shape (4, 2), the same four points reordered as
        [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype=np.float64)

    # A point's (x + y) sum is smallest at the top-left corner (small
    # x, small y) and largest at the bottom-right (large x, large y).
    # Its (x - y) difference is smallest at the top-right (large x,
    # small y -> very negative) and largest at the bottom-left (small
    # x, large y -> very positive). This holds for any convex
    # quadrilateral that's roughly "upright" (not rotated close to 45
    # degrees), which a scanned invoice page always is.
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmax(diff)]
    bottom_left = pts[np.argmin(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left])


def detect_border_corners(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Find the four corners of the invoice table's outer grid border.

    The table isn't outlined by one bold rectangle -- it's a uniform
    mesh of thin ruled lines (row dividers, column dividers, and the
    outer edge, all the same weight). So instead of looking for "the
    biggest rectangle," this isolates long horizontal and long
    vertical line segments specifically (via morphological
    opening -- erode then dilate with a wide/tall kernel, which wipes
    out anything that isn't a long straight line, like handwriting or
    printed text), unions them into one mask, and takes the outer
    boundary of that mask. Because every ruled line in the table
    touches its neighbors, that mask forms one connected mesh whose
    external contour is exactly the table's own outer border --
    unaffected by text/handwriting elsewhere on the page, since none
    of that survives the line-isolating step.

    Args:
        image_bgr: a full invoice scan as a BGR image array (e.g. from
            cv2.imread), at typical 300 DPI scan resolution.

    Returns:
        A (4, 2) array of [top-left, top-right, bottom-right,
        bottom-left] pixel coordinates, or None if no plausible grid
        border was found -- callers must treat None as "flag this
        invoice for manual handling," not fall back to guessing.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding (rather than one fixed global threshold)
    # copes with real scans having uneven lighting/shadow across the
    # page. THRESH_BINARY_INV makes ink (dark on the original scan)
    # come out white/foreground in the binary image, which is what
    # the morphology and contour steps below expect.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    # Kernel lengths are relative to image size rather than a fixed
    # pixel count, so this works the same whether a scan came in at
    # 300 DPI full-page or some other resolution/crop.
    height, width = gray.shape
    horizontal_kernel_len = max(10, width // 30)
    vertical_kernel_len = max(10, height // 30)

    # A wide-but-1-tall kernel: erosion with it only survives where
    # ink is continuously present across that whole width, i.e. long
    # horizontal strokes -- printed text and handwriting are too short
    # and irregular to survive. Dilating back afterward (the second
    # half of "opening") restores the surviving lines to roughly their
    # original thickness. Same idea rotated 90 degrees for vertical
    # lines.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # RETR_EXTERNAL only returns each connected shape's outermost
    # boundary, ignoring the many internal holes the grid's individual
    # cells form -- exactly what we want, since we only care about the
    # mesh's overall outer perimeter.
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # A real table border should cover a large, deliberate fraction of
    # the page -- a stray line fragment surviving the morphology step
    # (e.g. from a ruled signature line) would produce a tiny contour
    # that's obviously not the table. Rejecting it here means the
    # caller reliably gets either "the table" or None, never noise.
    page_area = width * height
    if cv2.contourArea(largest) < 0.15 * page_area:
        return None

    # minAreaRect finds the smallest rotated rectangle enclosing the
    # contour -- appropriate here since a scanned page's table is
    # never perfectly axis-aligned (slight rotation from how the page
    # sat on the scanner bed), but it isn't a general skewed
    # quadrilateral either (paper doesn't warp), so a rotated
    # rectangle is a truer fit than a raw 4-point polygon approximation
    # would be, and doesn't need a shape-tolerance parameter to tune.
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return order_corners(box)


def corner_distances(corners: np.ndarray) -> tuple[float, float]:
    """
    Measure the border's average width and height from its corners.

    Averaging the top/bottom edges (for width) and left/right edges
    (for height) is more robust to the small rotation/skew real scans
    have than trusting either single edge alone.

    Args:
        corners: (4, 2) array as returned by detect_border_corners,
            ordered [top-left, top-right, bottom-right, bottom-left].

    Returns:
        (width, height) in pixels.
    """
    top_left, top_right, bottom_right, bottom_left = corners

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    width = (top_width + bottom_width) / 2
    height = (left_height + right_height) / 2
    return float(width), float(height)


def corners_aspect_ratio(corners: np.ndarray) -> float:
    """
    Compute a detected border's width/height aspect ratio.

    Used to sanity-check a newly detected border against the
    calibrated reference template's own ratio (see
    validate_aspect_ratio) -- a real table border should always have
    roughly the same proportions, scan to scan, since it's the same
    printed form every time.

    Args:
        corners: (4, 2) array as returned by detect_border_corners.

    Returns:
        width / height as a float.
    """
    width, height = corner_distances(corners)
    return width / height


def validate_aspect_ratio(
    corners: np.ndarray, reference_ratio: float, tolerance: float = 0.12
) -> bool:
    """
    Check whether a detected border's shape plausibly matches the
    calibrated template, rather than being a misdetection (e.g. a
    stray line structure, or a different document entirely).

    This is the second half of the reliability guard described in the
    project plan: detect_border_corners() returning None handles
    outright detection failure, and this handles the "found *a*
    rectangle, but it's the wrong shape" case -- both should result in
    the invoice being flagged for manual handling rather than
    extraction proceeding on bad coordinates.

    Args:
        corners: (4, 2) array as returned by detect_border_corners.
        reference_ratio: the width/height ratio recorded once during
            calibration against the blank/reference template scan.
        tolerance: allowed fractional deviation from reference_ratio
            before flagging (default 12% -- a spot check across 6 real
            scans of this project's template found aspect ratios
            ranging 1.06-1.13 from normal scan-to-scan crop/skew
            variation alone, so anything much looser than that risks
            missing a real misdetection, and anything tighter risks
            flagging good scans depending on which one becomes the
            calibration reference).

    Returns:
        True if the detected border's aspect ratio is within
        tolerance of the reference; False otherwise.
    """
    ratio = corners_aspect_ratio(corners)
    deviation = abs(ratio - reference_ratio) / reference_ratio
    return deviation <= tolerance


def proportion_to_pixel(fx: float, fy: float, corners: np.ndarray) -> tuple[float, float]:
    """
    Map a proportion within the border (e.g. "34.2% across, 60% down")
    to an actual pixel coordinate in a specific scan.

    This is the core trick that avoids needing a full geometric image
    warp: bilinear interpolation across the quadrilateral formed by
    the border's 4 corners. fx=0/fy=0 is the top-left corner, fx=1 is
    the top-right/bottom-right edge, fy=1 is the bottom edge -- same
    idea as the border being its own little coordinate system,
    regardless of the scan's real size, position, or slight rotation.

    Args:
        fx: horizontal proportion across the border, 0.0 (left edge)
            to 1.0 (right edge). Not clamped -- a caller can pass
            slightly outside [0, 1] and still get a sensible
            extrapolated point.
        fy: vertical proportion down the border, 0.0 (top edge) to 1.0
            (bottom edge).
        corners: (4, 2) array as returned by detect_border_corners,
            ordered [top-left, top-right, bottom-right, bottom-left].

    Returns:
        (x, y) pixel coordinates in the scan this particular
        `corners` came from.
    """
    top_left, top_right, bottom_right, bottom_left = corners

    # Interpolate along the top edge and along the bottom edge first,
    # each by fx, then interpolate between those two intermediate
    # points by fy -- the standard bilinear-quadrilateral formula.
    top_point = top_left + fx * (top_right - top_left)
    bottom_point = bottom_left + fx * (bottom_right - bottom_left)
    point = top_point + fy * (bottom_point - top_point)

    return float(point[0]), float(point[1])


def pixel_to_proportion(
    x: float, y: float, corners: np.ndarray, max_iterations: int = 20, tolerance: float = 1e-6
) -> tuple[float, float]:
    """
    Invert proportion_to_pixel(): given a pixel point a human clicked
    during calibration, find the (fx, fy) proportion of the border it
    corresponds to.

    Bilinear interpolation has no simple closed-form inverse that's
    safe to implement without risking a subtle wrong-root bug (the
    forward map is quadratic in (fx, fy) jointly), so this solves it
    numerically instead: start from a plain axis-aligned estimate ("what
    fraction across the average left/right edge x-coordinates is this
    point"), then refine with a few Newton-Raphson steps against
    proportion_to_pixel's exact formula. For the near-rectangular
    quadrilaterals a flatbed scan produces (only slight rotation, no
    real perspective warp), this converges in just a couple of
    iterations.

    Args:
        x: pixel x-coordinate (e.g. of a calibration click).
        y: pixel y-coordinate.
        corners: (4, 2) array as returned by detect_border_corners.
        max_iterations: Newton-Raphson iteration cap -- a safety bound,
            not expected to be reached for realistic scans.
        tolerance: stop once the current (fx, fy) estimate reproduces
            (x, y) to within this many pixels.

    Returns:
        (fx, fy), each typically in [0, 1] for a point inside the
        border (points just outside it during calibration are fine
        too, and simply come back slightly outside that range).
    """
    top_left, top_right, bottom_right, bottom_left = corners
    target = np.array([x, y], dtype=np.float64)

    # Initial guess: treat the border as if it were an axis-aligned
    # rectangle using its corners' bounding extent. Close enough for
    # Newton's method to converge fast on the real, slightly-rotated
    # quadrilateral.
    width, height = corner_distances(corners)
    left_x = (top_left[0] + bottom_left[0]) / 2
    right_x = (top_right[0] + bottom_right[0]) / 2
    top_y = (top_left[1] + top_right[1]) / 2
    bottom_y = (bottom_left[1] + bottom_right[1]) / 2
    fx = (x - left_x) / (right_x - left_x) if right_x != left_x else 0.5
    fy = (y - top_y) / (bottom_y - top_y) if bottom_y != top_y else 0.5

    for _ in range(max_iterations):
        point = np.array(proportion_to_pixel(fx, fy, corners))
        error = target - point
        if np.linalg.norm(error) < tolerance:
            break

        # Analytic Jacobian of proportion_to_pixel with respect to
        # (fx, fy): d(point)/d(fx) and d(point)/d(fy), derived
        # directly from its bilinear formula.
        d_top = top_right - top_left
        d_bottom = bottom_right - bottom_left
        top_point = top_left + fx * d_top
        bottom_point = bottom_left + fx * d_bottom
        d_fx = d_top + fy * (d_bottom - d_top)
        d_fy = bottom_point - top_point

        jacobian = np.array([[d_fx[0], d_fy[0]], [d_fx[1], d_fy[1]]])
        try:
            delta = np.linalg.solve(jacobian, error)
        except np.linalg.LinAlgError:
            # A singular Jacobian means a degenerate (zero-area) quad
            # -- not a real table border, so further refinement is
            # meaningless. Return the best estimate so far.
            break

        fx += delta[0]
        fy += delta[1]

    return float(fx), float(fy)
