import cv2
import numpy as np
from .env import WORKSPACE, COLOR_RGB, ColorName, Pose2D

class Vision:
    def __init__(self, image_width: int = 640, image_height: int = 480) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.hsv_ranges = {
            "red": ([0, 100, 80], [10, 255, 255]),
            "green": ([40, 50, 50], [90, 255, 255]),
            "blue": ([95, 50, 50], [140, 255, 255]),
        }

    def detect(self, image: np.ndarray) -> list:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detections = []
        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if color == "red":
                mask2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask, mask2)
                
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50: continue
                
                rect = cv2.minAreaRect(cnt)
                center, dimensions, angle_deg = rect
                u, v = int(center[0]), int(center[1])
                
                detections.append({"color": color, "pixel": (u, v), "angle": angle_deg, "rect": rect})
        return detections

    def annotate(self, image: np.ndarray, detections: list) -> np.ndarray:
        canvas = image.copy()
        if canvas.ndim == 2: canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        for det in detections:
            rect = det["rect"]
            color = det["color"]
            rgb = tuple(int(255 * c) for c in COLOR_RGB[color])
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(canvas, [box], 0, rgb, 2)
            u, v = det["pixel"]
            cv2.circle(canvas, (u, v), 3, (255, 255, 255), -1)
            cv2.putText(canvas, f"{color}", (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
        return canvas

    def track_target(self, image: np.ndarray, target_color: str) -> dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        shadow_ranges = {
            "red": ([0, 40, 20], [10, 255, 255]),
            "green": ([40, 30, 20], [90, 255, 255]),
            "blue": ([90, 30, 20], [140, 255, 255]),
        }
        lower, upper = shadow_ranges[target_color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if target_color == "red":
            mask2 = cv2.inRange(hsv, np.array([170, 40, 20]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
            
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_u, best_v, best_angle, best_area = None, None, 0.0, 0
        min_dist = float('inf')
        cx, cy = self.image_width // 2, self.image_height // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                rect = cv2.minAreaRect(cnt)
                center, dimensions, angle_deg = rect
                u, v = int(center[0]), int(center[1])
                dist = (u - cx)**2 + (v - cy)**2
                if dist < min_dist:
                    min_dist = dist
                    best_u, best_v = u, v
                    best_area = area
                    best_angle = angle_deg
                    
        if best_u is not None:
            return {"pixel": (best_u, best_v), "area": best_area, "angle": best_angle}
        return None