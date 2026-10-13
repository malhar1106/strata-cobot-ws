"""
Reusable HSV color-box detection logic.

Same algorithm as the standalone OpenCV script:
  BGR -> HSV -> per-color mask -> morphological cleanup -> contours -> boxes.

Kept separate from the ROS2 node so it can be unit-tested or reused
outside of ROS entirely (e.g. in the plain script version).
"""

import cv2
import numpy as np

# Tune these to your lighting / sim. Sample real pixel HSV values from your
# camera feed and set ranges a bit wider than what you observe.
# Keys are the mineral names used for ore_ids (e.g. azurite_ore_1).
COLOR_RANGES = {
    "azurite":    [((100, 100, 50), (130, 255, 255))],  # blue
    "malachite":  [((40, 60, 40), (85, 255, 255))],     # green
    "vanadinite": [((0, 210, 170), (18, 255, 255))],    # orange
}

DRAW_COLOR = {
    "azurite":    (255, 0, 0),
    "malachite":  (0, 255, 0),
    "vanadinite": (0, 140, 255),
}

MIN_AREA = 150  # pixels; tune to your camera resolution / distance to boxes


def detect_color_boxes(image, color_ranges=COLOR_RANGES, min_area=MIN_AREA):
    """Returns list of dicts:
    {color, name, bbox(x,y,w,h), centroid(cx,cy), area}

    'name' is a per-color-numbered id like 'azurite_ore_1', 'azurite_ore_2',
    numbering restarts at 1 for each mineral/color and follows contour
    discovery order within that color (not guaranteed stable frame-to-frame
    if boxes move -- fine for a static sorting scene, but don't rely on it
    as a persistent object ID if things get shuffled).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    detections = []
    for color_name, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ore_index = 1  # per-color counter, e.g. azurite_ore_1, azurite_ore_2
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
            detections.append({
                "color": color_name,
                "name": f"{color_name}_ore_{ore_index}",
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "area": area,
            })
            ore_index += 1
    return detections


def draw_detections(image, detections):
    out = image.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = d["centroid"]
        color = DRAW_COLOR.get(d["color"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(out, d["name"], (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
