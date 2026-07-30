import numpy as np
import pybullet as p
import time
import random
from .env import SortingEnv, WORKSPACE
from .vision import Vision
from .controller import Controller

class Planner:
    def __init__(self, env: SortingEnv, vision: Vision, controller: Controller) -> None:
        self.env = env
        self.vision = vision
        self.controller = controller
        self.state = "WAIT_FOR_COMMAND"
        self.detections = []
        self.current = None
        self.done = False
        
        self.placed_counts = {"red": 0, "green": 0, "blue": 0}
        self.place_offsets = [
            (0.0, 0.0), (0.0, 0.05), (0.0, -0.05),
            (0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05),
        ]
        
        self.task_queue = []
        self.mode = "language"  # 支持: "language", "random", "nearest"

    def set_mode(self, mode: str, tasks: list = None):
        """外部调用以设置工作模式"""
        self.mode = mode
        if tasks:
            self.task_queue = tasks.copy()
            
        print(f"\n[Planner] 系统已切换至模式: {self.mode.upper()}")
        self.state = "SCAN"

    def step(self) -> str:
        if self.state == "WAIT_FOR_COMMAND":
            pass
            
        elif self.state == "SCAN":
            image = self.env.get_camera_image()
            raw_dets = self.vision.detect(image)
            self.detections = []
            
            for d in raw_dets:
                u, v = d["pixel"]
                x_vision, y_vision = self.env.pixel_to_world(u, v)
                
                cam_x, cam_y, cam_z = 0.50, 0.0, 1.40
                table_z = WORKSPACE["z_table"]
                block_z = table_z + 0.04
                ratio = (cam_z - block_z) / (cam_z - table_z)
                
                x_real = cam_x + (x_vision - cam_x) * ratio
                y_real = cam_y + (y_vision - cam_y) * ratio
                
                if 0.25 <= x_real <= 0.70 and -0.10 <= y_real <= 0.35:
                    d["world"] = (x_real, y_real)
                    self.detections.append(d)

            # 如果检测不到任何方块（或者语言队列空了），任务结束
            if not self.detections or (self.mode == "language" and not self.task_queue):
                self.state = "FINISH"
            else:
                self.state = "SELECT"

        elif self.state == "SELECT":
            # ==========================================
            # 🎓 核心拓展：多模态分拣策略 (Multi-modal Sorting Strategy)
            # ==========================================
            if self.mode == "language":
                target_color = self.task_queue[0]
                self.current = next((d for d in self.detections if d["color"] == target_color), None)
                
                if not self.current:
                    print(f"[Planner] ⚠️ 画面中无 {target_color}，跳过此任务。")
                    self.task_queue.pop(0)
                    self.state = "SCAN"
                    return self.state
                    
            elif self.mode == "random":
                # 随机策略：从当前视野中随机挑一个方块
                self.current = random.choice(self.detections)
                
            elif self.mode == "nearest":
                # 最近优先策略：计算每个方块到机械臂基座 (0,0) 的欧氏距离
                # 距离 = sqrt(x^2 + y^2)，找出距离最小的那个
                self.current = min(self.detections, key=lambda d: d["world"][0]**2 + d["world"][1]**2)

            # --- 选定目标，准备抓取 ---
            start = self.current["world"]
            print(f"[Planner] [{self.mode.upper()}] 选定目标: {self.current['color']} 方块 at ({start[0]:.2f}, {start[1]:.2f})")
            self.state = "PICK_AND_PLACE"

        elif self.state == "PICK_AND_PLACE":
            start = self.current["world"]
            color = self.current["color"]
            base_goal = self.env.get_target_pose(color)[:2]
            
            count = self.placed_counts[color]
            idx = count % len(self.place_offsets)
            dx, dy = self.place_offsets[idx]
            goal = (base_goal[0] + dx, base_goal[1] + dy)
            
            self.controller.pick_and_place(start, goal)
            
            self.placed_counts[color] += 1
            if self.mode == "language":
                self.task_queue.pop(0) 
                
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