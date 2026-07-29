import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE
from .vision import Vision
from .controller import Controller

class Planner:
    def __init__(self, env: SortingEnv, vision: Vision, controller: Controller) -> None:
        self.env = env
        self.vision = vision
        self.controller = controller
        self.state = "SCAN"
        self.detections = []
        self.current = None
        self.done = False

    def step(self) -> str:
        if self.state == "SCAN":
            image = self.env.get_camera_image()
            raw_dets = self.vision.detect(image)
            self.detections = []
            
            for d in raw_dets:
                u, v = d["pixel"]
                x_vision, y_vision = self.env.pixel_to_world(u, v)
                
                # ==========================================
                # 🎓 核心绝杀：相似三角形视差补偿 (Parallax Correction)
                # 相机在 X=0.50, Y=0.0, Z=1.40。
                # 视觉交点在桌面 Z=0.625，方块顶面在 Z=0.665。
                # ==========================================
                cam_x, cam_y, cam_z = 0.50, 0.0, 1.40
                table_z = WORKSPACE["z_table"]
                block_z = table_z + 0.04
                
                # 相似三角形比例 = (相机到方块顶部的高度) / (相机到桌面的高度)
                ratio = (cam_z - block_z) / (cam_z - table_z)
                
                # 修正后的真实物理坐标
                x_real = cam_x + (x_vision - cam_x) * ratio
                y_real = cam_y + (y_vision - cam_y) * ratio
                
                # ROI 空间过滤
                if 0.25 <= x_real <= 0.70 and -0.10 <= y_real <= 0.35:
                    d["world"] = (x_real, y_real)
                    self.detections.append(d)

            if not self.detections:
                self.state = "FINISH"
            else:
                self.state = "SELECT"

        elif self.state == "SELECT":
            self.current = self.detections[0] if self.detections else None
            if self.current:
                start = self.current["world"]
                print(f"[Planner] GRASPING {self.current['color']} block at ({start[0]:.2f}, {start[1]:.2f})")
                self.state = "PICK_AND_PLACE"
            else:
                self.state = "FINISH"

        elif self.state == "PICK_AND_PLACE":
            start = self.current["world"]
            goal = self.env.get_target_pose(self.current["color"])[:2]
            
            self.controller.pick_and_place(start, goal)
            self.state = "RETURN_HOME"

        elif self.state == "RETURN_HOME":
            self.env.go_home()
            for _ in range(240):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
            self.state = "SCAN"

        elif self.state == "FINISH":
            self.done = True

        return self.state