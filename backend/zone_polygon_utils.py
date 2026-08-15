"""
zone_polygon_utils.py
----------------------
Adds two things your synopsis promises but the current backend doesn't
actually implement yet:

  1. Mouse-drawn polygon zones (instead of a fixed 3x3 grid)
  2. A real density heatmap (Gaussian-accumulated, colorized) instead of
     colored grid rectangles

Designed to drop into dashboard_app.py with minimal changes. Falls back
to the old fixed-grid behavior automatically if no polygons have been
saved yet, so nothing breaks while you're wiring this in.
"""

import json
import os
import time
import numpy as np
import cv2

# ---------------------------------------------------------------------
# 1. Polygon zone storage
# ---------------------------------------------------------------------
# File format (zones_config.json):
# {
#   "frame_width": 1920, "frame_height": 1080,
#   "zones": [
#       {"id": "Z1", "name": "Entrance", "points": [[x1,y1],[x2,y2],...]},
#       ...
#   ]
# }
# Points are stored in ABSOLUTE pixel coords of the reference snapshot
# they were drawn on. We rescale them at runtime if the live frame size
# differs (e.g. camera resolution changed).

ZONES_CONFIG_FILE = "zones_config.json"


def load_zone_polygons(base_dir):
    path = os.path.join(base_dir, ZONES_CONFIG_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data.get("zones"):
            return None
        return data
    except Exception as e:
        print(f"⚠️ Failed to load {ZONES_CONFIG_FILE}: {e}")
        return None


def save_zone_polygons(base_dir, zones, frame_width, frame_height):
    """zones: list of {"id": str, "name": str, "points": [[x,y], ...]}"""
    path = os.path.join(base_dir, ZONES_CONFIG_FILE)
    data = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "zones": zones,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _scale_points(points, src_w, src_h, dst_w, dst_h):
    if src_w == dst_w and src_h == dst_h:
        return points
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    return [[p[0] * sx, p[1] * sy] for p in points]


def assign_point_to_zone(cx, cy, zone_polygons):
    """
    Returns the zone_id whose polygon contains (cx, cy), or None.
    zone_polygons: list of {"id", "points": [[x,y],...]} already scaled
    to the current frame size.
    """
    for zone in zone_polygons:
        pts = np.array(zone["points"], dtype=np.int32)
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            return zone["id"]
    return None


def get_runtime_zones(zone_config, frame_w, frame_h):
    """Rescale saved polygons to the current frame resolution."""
    if not zone_config:
        return None
    src_w = zone_config.get("frame_width", frame_w)
    src_h = zone_config.get("frame_height", frame_h)
    out = []
    for z in zone_config["zones"]:
        out.append({
            "id": z["id"],
            "name": z.get("name", z["id"]),
            "points": _scale_points(z["points"], src_w, src_h, frame_w, frame_h),
        })
    return out


def draw_zone_polygons(frame, runtime_zones, zone_levels):
    """
    zone_levels: {zone_id: (count, level)} — same shape you already
    build in process_frame's frame_data.
    """
    def color_for(level):
        return {
            "Low": (0, 255, 0),
            "Medium": (0, 255, 255),
            "High": (0, 165, 255),
            "Critical": (0, 0, 255),
        }.get(level, (0, 255, 0))

    overlay = frame.copy()
    for zone in runtime_zones:
        pts = np.array(zone["points"], dtype=np.int32)
        count, level = zone_levels.get(zone["id"], (0, "Low"))
        color = color_for(level)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        label = f"{zone.get('name', zone['id'])}: {level} ({count})"
        cv2.putText(frame, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    # semi-transparent zone fill so the video underneath still shows
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


# ---------------------------------------------------------------------
# 2. Real density heatmap
# ---------------------------------------------------------------------
class HeatmapAccumulator:
    """
    Maintains a decaying density surface. Call add_points() every frame
    with detected person centroids, then render() to get a colorized
    overlay. Old activity fades out (decay), new activity blooms in —
    this looks like a real crowd heatmap instead of a static snapshot.
    """

    def __init__(self, width, height, decay=0.90, blob_radius=40):
        self.width = width
        self.height = height
        self.decay = decay
        self.blob_radius = blob_radius
        self.accum = np.zeros((height, width), dtype=np.float32)

    def _resize_if_needed(self, w, h):
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self.accum = np.zeros((h, w), dtype=np.float32)

    def add_points(self, points, frame_w, frame_h):
        """points: list of (cx, cy) centroids in current frame's pixel space."""
        self._resize_if_needed(frame_w, frame_h)
        self.accum *= self.decay  # fade previous activity
        blob = np.zeros_like(self.accum)
        for (cx, cy) in points:
            cx, cy = int(cx), int(cy)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                blob[cy, cx] += 1.0
        if points:
            k = self.blob_radius | 1  # must be odd
            blob = cv2.GaussianBlur(blob, (k, k), 0)
        self.accum += blob

    def render(self, base_frame, alpha=0.55):
        """Return base_frame with a JET-colormap heatmap overlay blended in."""
        if self.accum.max() <= 0:
            return base_frame
        norm = cv2.normalize(self.accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=3)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # only blend where there's meaningful density, so empty areas stay clear
        mask = (norm > 15).astype(np.uint8)
        mask3 = cv2.merge([mask, mask, mask])
        blended = cv2.addWeighted(colored, alpha, base_frame, 1 - alpha, 0)
        return np.where(mask3 > 0, blended, base_frame)