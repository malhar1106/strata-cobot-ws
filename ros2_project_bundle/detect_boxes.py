"""
Color-based ore/box detection with OpenCV.

Pipeline:
  1. Load image, convert BGR -> HSV
  2. Threshold each target color into a binary mask
  3. Clean masks with morphological ops
  4. Find contours -> filter by area -> compute bounding box + centroid
  5. Draw + print results

Usage:
  python3 detect_boxes.py <input_image> <output_image>
"""

import sys
import cv2
import numpy as np

# --- 1. Define HSV color ranges for each box color -------------------------
# HSV in OpenCV: H in [0,179], S in [0,255], V in [0,255]
# These are starting points -- tune them for your lighting (see notes below).
COLOR_RANGES = {
    "blue":   [((100, 100, 50), (130, 255, 255))],
    "green":  [((40, 60, 40), (85, 255, 255))],
    # orange/red often wraps around hue 0, so give it one band near 0-15
    # and (if needed) one near 170-180. Here the boxes are orange, so one band suffices.
    "orange": [((0, 200, 150), (20, 255, 255))],
}

# BGR colors just for drawing labels/boxes on screen
DRAW_COLOR = {
    "blue":   (255, 0, 0),
    "green":  (0, 255, 0),
    "orange": (0, 140, 255),
}

MIN_AREA = 150  # pixels; filters out tiny noise blobs -- tune to your image scale


def detect_color_boxes(image, color_ranges=COLOR_RANGES, min_area=MIN_AREA):
    """Returns list of dicts: {color, bbox(x,y,w,h), centroid(cx,cy), area}"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    detections = []
    for color_name, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # clean up the mask: remove speckle noise, then fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "area": area,
            })
    return detections


def draw_detections(image, detections):
    out = image.copy()
    for i, d in enumerate(detections):
        x, y, w, h = d["bbox"]
        cx, cy = d["centroid"]
        color = DRAW_COLOR.get(d["color"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)
        label = f'{d["color"]}_{i}'
        cv2.putText(out, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"

    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    dets = detect_color_boxes(img)
    result_img = draw_detections(img, dets)
    cv2.imwrite(out_path, result_img)

    print(f"Found {len(dets)} boxes:")
    for i, d in enumerate(dets):
        print(f'  [{i}] color={d["color"]:<7} centroid={d["centroid"]} '
              f'bbox={d["bbox"]} area={d["area"]:.0f}')
