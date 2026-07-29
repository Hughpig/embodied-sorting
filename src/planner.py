import numpy as np
import pybullet as p
import time
from .env import SortingEnv
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
                x, y = self.env.pixel_to_world(u, v)
                
                # ROI 空间过滤
                if 0.25 <= x <= 0.70 and -0.10 <= y <= 0.35:
                    d["world"] = (x, y)
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
            # 核心绝杀：让物理引擎多跑一会儿，等机械臂彻底退出相机视野！
            for _ in range(240):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
            self.state = "SCAN"

        elif self.state == "FINISH":
            self.done = True

        return self.state