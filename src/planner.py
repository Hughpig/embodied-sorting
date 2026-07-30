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
        self.mode = "language"
        
        # === [为 Servo 模式追加的变量] ===
        self.servo_target_color = None
        self.servo_lost_frames = 0

    def set_mode(self, mode: str, tasks: list = None):
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

            if not self.detections or (self.mode == "language" and not self.task_queue):
                self.state = "FINISH"
            else:
                self.state = "SELECT"

        elif self.state == "SELECT":
            if self.mode == "language":
                target_color = self.task_queue[0]
                self.current = next((d for d in self.detections if d["color"] == target_color), None)
                if not self.current:
                    self.task_queue.pop(0)
                    self.state = "SCAN"
                    return self.state
            elif self.mode in ["random", "servo"]:
                self.current = random.choice(self.detections)
            elif self.mode == "nearest":
                self.current = min(self.detections, key=lambda d: d["world"][0]**2 + d["world"][1]**2)

            start = self.current["world"]
            
            # === [旁路分流：Servo 走独立的追踪循环] ===
            if self.mode == "servo":
                self.servo_target_color = self.current['color']
                print(f"[Servo] 启动动态追踪！目标: {self.servo_target_color}")
                self.controller.open_gripper()
                
                # 🎓 核心修复 1：起步飞高一点 (0.85)，获得超大视野，绝对不会一上来就脱靶！
                self.controller.move_to(start, 0.85, steps=150)
                for _ in range(60): 
                    p.stepSimulation()
                    time.sleep(1.0/240.0)
                self.servo_lost_frames = 0
                self.state = "VS_TRACKING"
            else:
                # 你的稳定版开环分支
                print(f"[Planner] [{self.mode.upper()}] 选定目标: {self.current['color']} 方块 at ({start[0]:.2f}, {start[1]:.2f})")
                self.state = "PICK_AND_PLACE"

        # === [Servo 独立跟踪逻辑] ===
        elif self.state == "VS_TRACKING":
            eye_img = self.env.get_eye_in_hand_image()
            target_info = self.vision.track_target(eye_img, self.servo_target_color)
            
            if target_info is None:
                self.servo_lost_frames += 1
                if self.servo_lost_frames > 15:
                    print("[Servo] 目标跟丢了！手臂复位，避免遮挡全局相机...")
                    self.state = "RETURN_HOME" 
                return self.state
                
            self.servo_lost_frames = 0
            u, v = target_info["pixel"]
            err_x = u - 320
            err_y = v - 240
            
            Kp = 0.00010
            dx = np.clip(err_x * Kp, -0.015, 0.015)
            dy = np.clip(-err_y * Kp, -0.015, 0.015)
            
            current_pose, _ = self.env.get_ee_pose()
            
            # 🎓 核心修复 2：贴脸打击前，强制校验十字准星！
            if current_pose[2] <= self.controller.z_pick + 0.03:
                # 只有偏差小于 15 像素（极其精准），才允许咬下去！
                if abs(err_x) < 15 and abs(err_y) < 15:
                    print(f"[Servo] 目标绝对锁定，执行致命打击！")
                    self.controller.move_to([current_pose[0], current_pose[1]], self.controller.z_pick, steps=40)
                    for _ in range(20): p.stepSimulation() 
                    self.controller.close_gripper()
                    self.state = "VS_DELIVER"
                else:
                    # 如果高度到了但没瞄准，就在方块头上悬停平移，直到对准为止！
                    self.controller.move_delta(dx, dy, 0.0, steps=15)
            else:
                # 在高空逼近时
                if abs(err_x) < 30 and abs(err_y) < 30:
                    dz = -0.015  
                    dx *= 0.4  # 下降时减速平移，防止螺旋抖动
                    dy *= 0.4
                else:
                    dz = 0.0     
                self.controller.move_delta(dx, dy, dz, steps=15)
                
        elif self.state == "VS_DELIVER":
            color = self.servo_target_color
            base_goal = self.env.get_target_pose(color)[:2]
            count = self.placed_counts[color]
            idx = count % len(self.place_offsets)
            dx, dy = self.place_offsets[idx]
            goal = (base_goal[0] + dx, base_goal[1] + dy)
            
            current_pose, _ = self.env.get_ee_pose()
            start = current_pose[:2]
            
            num_waypoints = 12
            for i in range(1, num_waypoints + 1):
                interp_z = self.controller.z_pick + (self.controller.z_safe - self.controller.z_pick) * (i / num_waypoints)
                self.controller.move_to([start[0], start[1]], interp_z, steps=15)
                
            self.controller.move_to([start[0], start[1]], self.controller.z_safe, steps=40)
            self.controller.move_to(goal, self.controller.z_safe, steps=200)
            self.controller.move_to(goal, self.controller.z_drop, steps=100)
            self.controller.open_gripper()
            for _ in range(40): p.stepSimulation(); time.sleep(1.0/240.0)
            self.controller.move_to(goal, self.controller.z_safe, steps=100)
            
            self.placed_counts[color] += 1
            self.state = "RETURN_HOME"

        # === [你的稳定版抓取逻辑] ===
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