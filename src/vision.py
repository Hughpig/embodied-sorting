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

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50: 
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                detections.append({"color": color, "pixel": (u, v)})
        return detections

    def annotate(self, image: np.ndarray, detections: list) -> np.ndarray:
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        for det in detections:
            u, v = det["pixel"]
            color = det["color"]
            rgb = tuple(int(255 * c) for c in COLOR_RGB[color])
            cv2.circle(canvas, (int(u), int(v)), 8, rgb, 2)
            cv2.putText(canvas, color, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
        return canvas

    # === [为 Servo 模式追加的函数] ===
    def track_target(self, image: np.ndarray, target_color: str) -> dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 🎓 核心修复 1：降低亮度和饱和度下限，即使在机械臂的极度深沉阴影下，也能看清方块！
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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_u, best_v, best_area = None, None, 0
        min_dist = float('inf')
        cx, cy = self.image_width // 2, self.image_height // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 🎓 核心修复 2：只要看到 10 个像素就咬住不放，防止边缘裁切丢失！
            if area > 10: 
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"])
                    v = int(M["m01"] / M["m00"])
                    dist = (u - cx)**2 + (v - cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_u, best_v = u, v
                        best_area = area
                        
        if best_u is not None:
            return {"pixel": (best_u, best_v), "area": best_area}
        return None