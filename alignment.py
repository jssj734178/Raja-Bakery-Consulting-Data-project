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

from concurrent.futures import ThreadPoolExecutor

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


def _fit_line_in_band(mask: np.ndarray, axis: str, band_low: float, band_high: float):
    """
    Fit a straight line through every foreground pixel of `mask` that
    falls within a coordinate band -- e.g. "every horizontal-line pixel
    whose y is within 400px of the table's approximate top edge."

    Deliberately does NOT require those pixels to form one connected
    component. Testing against this project's real scans showed the
    outer border's OWN line, specifically, breaks into many small
    disconnected fragments in the isolated horizontal/vertical-line
    masks -- it's crossed by every internal row/column divider along
    its full length, and each crossing turned out to interrupt
    connectivity there even though internal single dividers (crossed
    far less) stayed intact. Gathering every pixel in the band
    regardless of which fragment it belongs to sidesteps that
    fragmentation entirely. A robust distance metric (Huber, not plain
    least-squares) keeps the fit from being thrown off by the rare
    stray pixel that lands in the band without actually being part of
    this border (e.g. a jagged mesh-notch corner).

    Args:
        mask: horizontal_lines or vertical_lines mask.
        axis: "horizontal" or "vertical" -- which coordinate (y or x)
            the band is measured along.
        band_low: lower bound of the band, in pixels.
        band_high: upper bound of the band, in pixels.

    Returns:
        (point, direction) -- a point on the fitted line and its unit
        direction vector, both as length-2 float arrays -- or None if
        no foreground pixels fall inside the band at all.
    """
    # Search only the band's own strip of the mask, not the whole page
    # -- scanning a full 85-megapixel page for every edge cost ~1s each.
    # Points come out in the same order either way, so the fit is
    # unchanged.
    low = max(0, int(np.ceil(band_low)))
    high = int(np.floor(band_high)) + 1
    if axis == "horizontal":
        ys, xs = np.nonzero(mask[low:high, :])
        ys = ys + low
    else:
        ys, xs = np.nonzero(mask[:, low:high])
        xs = xs + low
    if len(xs) == 0:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect_lines(point1, direction1, point2, direction2):
    """
    Find where two infinite lines (each given as a point + direction)
    cross, by solving point1 + t1*direction1 == point2 + t2*direction2
    for (t1, t2) as a 2x2 linear system.

    Used to turn the four fitted border lines (top/bottom/left/right)
    into the four actual corner points -- each corner is just the
    intersection of the two border lines that meet there.
    """
    a = np.array([[direction1[0], -direction2[0]], [direction1[1], -direction2[1]]])
    b = np.array(point2) - np.array(point1)
    t1, _ = np.linalg.solve(a, b)
    return np.array(point1) + t1 * np.array(direction1)


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
    printed text), uses their combined shape only to get a rough
    bounding box for the whole table, then -- within a narrow search
    band around each of that box's 4 edges -- gathers every surviving
    line pixel in the band and fits one robust line through it per
    edge, finally intersecting adjacent pairs (top/left, top/right,
    bottom/left, bottom/right) to get the actual 4 corners.

    (Three more direct approaches were tried first and all rejected
    after testing against this project's real scans. minAreaRect over
    the combined mask's own contour overshot one corner at a time,
    pulled by whichever single jagged mesh-notch pixel happened to
    stick out furthest. approxPolyDP-on-convex-hull was worse,
    occasionally locking onto a small ink-blob protrusion on an
    otherwise-straight edge as one of only 4 allowed vertices instead
    of the true corner -- both failure modes come from reducing one
    noisy contour down to a handful of points, where a single outlier
    pixel can dominate the result. Requiring each border line to
    survive as its own single CONNECTED component (rather than
    searching a band for any matching pixels) also failed: the outer
    border, being crossed by every internal row/column divider along
    its full length, turned out to fragment into many small
    disconnected pieces at those crossings even though internal
    dividers -- crossed far less -- stayed intact; requiring
    connectivity meant only picking up a stray internal divider's
    fragment instead of the true, fragmented border. Gathering every
    band pixel regardless of which fragment it belongs to, and fitting
    with a robust (Huber) distance metric rather than plain
    least-squares, sidesteps both problems at once.)

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
    # the morphology and line-fitting steps below expect.
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

    # Combine into one mask purely to get an approximate bounding box
    # for the whole table -- every ruled line touches its neighbors,
    # so this forms one connected mesh whose bounding box reliably
    # brackets the true border (verified against real scans), even
    # though extracting exact CORNERS from this mask directly proved
    # unreliable (see the two rejected approaches in this function's
    # docstring). It's only used here as a rough anchor for where to
    # search for each of the 4 border lines individually, below.
    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)

    page_area = width * height
    if cv2.contourArea(largest) < 0.15 * page_area:
        return None

    approx_left, approx_top, box_w, box_h = cv2.boundingRect(largest)
    approx_right = approx_left + box_w
    approx_bottom = approx_top + box_h

    # Search bands around each approximate edge -- generous enough to
    # comfortably contain the true border despite the page's own
    # small rotation (observed up to roughly 1 degree on real scans,
    # i.e. well under 200px of vertical/horizontal drift across the
    # whole table), while staying far short of the gap to the next
    # real content (the nearest unrelated printed line on this
    # project's template, a signature underline, sits several hundred
    # pixels further away still).
    band_y = max(60, int(0.03 * height))
    band_x = max(60, int(0.03 * width))

    # The four edges are fitted at the same time on separate cores --
    # each fit is independent and takes about half a second on its own.
    with ThreadPoolExecutor(max_workers=4) as pool:
        top_line, bottom_line, left_line, right_line = pool.map(
            lambda args: _fit_line_in_band(*args),
            [
                (horizontal_lines, "horizontal", approx_top - band_y, approx_top + band_y),
                (horizontal_lines, "horizontal", approx_bottom - band_y, approx_bottom + band_y),
                (vertical_lines, "vertical", approx_left - band_x, approx_left + band_x),
                (vertical_lines, "vertical", approx_right - band_x, approx_right + band_x),
            ],
        )

    if None in (top_line, bottom_line, left_line, right_line):
        return None

    try:
        top_left = _intersect_lines(*top_line, *left_line)
        top_right = _intersect_lines(*top_line, *right_line)
        bottom_left = _intersect_lines(*bottom_line, *left_line)
        bottom_right = _intersect_lines(*bottom_line, *right_line)
    except np.linalg.LinAlgError:
        # A singular system means two "border" lines came out parallel
        # to each other instead of perpendicular -- not a real table.
        return None

    return order_corners(np.array([top_left, top_right, bottom_right, bottom_left]))


def ruled_line_positions(binary: np.ndarray, axis: int, run_length: int) -> list:
    """
    Locate a form's printed ruled lines in an already-deskewed,
    thresholded image, by isolating strokes that run straight for at
    least `run_length` pixels -- long enough that handwriting or
    printed text never qualifies, but a printed ruled line always does.

    Shared by extract_invoice.py (finding a cell's own bounding lines,
    to snap a calibrated box onto them -- see snap_cell_box) and by
    calibrate_total_price.py (finding the Total Price column's own left
    divider once, during calibration) -- both need exactly the same
    "where are the long straight lines" primitive, just aimed at
    different crops.

    Args:
        binary: thresholded image, ink as foreground (255), already
            deskewed -- a rotated line drifts out of a single
            column/row over its own length and won't be found by this.
        axis: 0 to find vertical lines (returning x positions), 1 to
            find horizontal ones (returning y positions).
        run_length: how long a straight run has to be to count as a
            printed line rather than handwriting.

    Returns:
        The centre position of each detected line, in ascending order.
        Adjacent columns/rows of the same line are collapsed into one
        entry -- a printed line is several pixels thick, and what the
        caller wants is one number per line.
    """
    kernel_size = (1, run_length) if axis == 0 else (run_length, 1)
    lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    )

    # Count ink along each line's own direction, leaving one value per
    # candidate position, then keep the positions where a line
    # actually is. Half the run length is a deliberately forgiving
    # cut: it takes a real line to get near it, but a line broken up
    # where other rules cross it still clears it.
    profile = lines.sum(axis=axis) / 255
    present = profile > run_length / 2

    positions = []
    start = None
    for index, is_line in enumerate(present):
        if is_line and start is None:
            start = index
        elif not is_line and start is not None:
            positions.append((start + index - 1) / 2)
            start = None
    if start is not None:
        positions.append((start + len(present) - 1) / 2)
    return positions


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
