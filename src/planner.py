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
        
        # ==========================================
        # 🎓 核心拓展：状态记忆与平铺码垛算法 (Palletizing)
        # 记录每种颜色已经成功放置了多少个方块
        # ==========================================
        self.placed_counts = {"red": 0, "green": 0, "blue": 0}
        # 预设的座位表（相对于目标中心点的坐标偏移量 dx, dy）
        self.place_offsets = [
            (0.0, 0.0),       # 第1个：正中心
            (0.0, 0.05),      # 第2个：靠上
            (0.0, -0.05),     # 第3个：靠下
            (0.05, 0.05),     # 第4个：右上
            (-0.05, -0.05),   # 第5个：左下
            (0.05, -0.05),    # 第6个：右下
            (-0.05, 0.05),    # 第7个：左上
        ]

    def step(self) -> str:
        if self.state == "SCAN":
            image = self.env.get_camera_image()
            raw_dets = self.vision.detect(image)
            self.detections = []
            
            for d in raw_dets:
                u, v = d["pixel"]
                x_vision, y_vision = self.env.pixel_to_world(u, v)
                
                # 视差补偿 (Parallax Correction)
                cam_x, cam_y, cam_z = 0.50, 0.0, 1.40
                table_z = WORKSPACE["z_table"]
                block_z = table_z + 0.04
                ratio = (cam_z - block_z) / (cam_z - table_z)
                
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
            color = self.current["color"]
            base_goal = self.env.get_target_pose(color)[:2]
            
            # ==========================================
            # 分配座位：根据已放置的数量，获取偏移量
            # ==========================================
            count = self.placed_counts[color]
            idx = count % len(self.place_offsets) # 防止越界
            dx, dy = self.place_offsets[idx]
            
            # 计算最终的平铺目标坐标
            goal = (base_goal[0] + dx, base_goal[1] + dy)
            
            # 执行抓放
            self.controller.pick_and_place(start, goal)
            
            # 更新计数器
            self.placed_counts[color] += 1
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