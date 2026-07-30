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
            (0.0, 0.0), (0.0, 0.05), (0.0, -0.05), (0.05, 0.0), (-0.05, 0.0)
        ]
        self.task_queue = []
        self.mode = "nearest" 
        self.servo_target_color = None
        self.servo_lost_frames = 0
        self.debug_counter = 0

    def set_mode(self, mode: str, tasks: list = None):
        self.mode = mode
        if tasks: self.task_queue = tasks.copy()
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
            
            if self.mode == "servo":
                self.servo_target_color = self.current['color']
                print(f"[Servo] 启动动态追踪！目标: {self.servo_target_color}")
                self.controller.open_gripper(width=0.060)
                self.controller.move_to(start, 0.85, orn=self.controller.down_orn, steps=150)
                for _ in range(60): p.stepSimulation(); time.sleep(1.0/240.0)
                self.servo_lost_frames = 0
                self.state = "VS_TRACKING"
            else:
                print(f"[Planner] 选定目标: {self.current['color']} 方块 at ({start[0]:.2f}, {start[1]:.2f})")
                self.state = "PICK_AND_PLACE"

        elif self.state == "VS_TRACKING":
            eye_img = self.env.get_eye_in_hand_image()
            target_info = self.vision.track_target(eye_img, self.servo_target_color)
            
            if target_info is None:
                self.servo_lost_frames += 1
                if self.servo_lost_frames > 15:
                    print("[Servo] 目标跟丢了！手臂复位...")
                    self.state = "RETURN_HOME" 
                return self.state
                
            self.servo_lost_frames = 0
            u, v = target_info["pixel"]
            angle_deg = target_info["angle"]
            angle = angle_deg % 90
            if angle > 45: angle -= 90
            yaw = -np.deg2rad(angle)
            yaw_deg = np.rad2deg(yaw)
            
            err_x = u - 320
            err_y = v - 240
            
            self.debug_counter += 1
            if self.debug_counter % 10 == 0:
                current_pose, _ = self.env.get_ee_pose()
                print(f"[DEBUG Servo] eX:{err_x:4d} | eY:{err_y:4d} | eYaw:{yaw_deg:5.1f}° | Z:{current_pose[2]:.3f}")
            
            Kp = 0.00010
            dx = np.clip(err_x * Kp, -0.015, 0.015)
            dy = np.clip(-err_y * Kp, -0.015, 0.015)
            
            current_pose, _ = self.env.get_ee_pose()
            target_yaw_world = np.pi / 2 + yaw
            dynamic_orn = p.getQuaternionFromEuler([np.pi, 0.0, target_yaw_world])
            
            if current_pose[2] <= self.controller.z_pick + 0.08:
                if abs(err_x) < 15 and abs(err_y) < 15:
                    print(f"\n[Servo] 🎯 目标绝对锁定！")
                    print(f"[Servo] 高空悬停平滑扭腕 (目标角度: {yaw_deg:.1f}°)...")
                    
                    self.controller.smooth_twist([current_pose[0], current_pose[1]], current_pose[2], target_yaw_world, num_waypoints=20)
                    for _ in range(10): p.stepSimulation()
                    
                    print(f"[Servo] 扭腕完毕，直插打击！")
                    self.controller.vertical_move([current_pose[0], current_pose[1]], current_pose[2], self.controller.z_pick, orn=dynamic_orn, num_waypoints=10)
                    
                    for _ in range(20): p.stepSimulation() 
                    self.controller.close_gripper()
                    self.state = "VS_DELIVER"
                else:
                    self.controller.move_delta(dx, dy, 0.0, orn=self.controller.down_orn, steps=15)
            else:
                if abs(err_x) < 30 and abs(err_y) < 30:
                    dz = -0.015  
                    dx *= 0.4
                    dy *= 0.4
                else:
                    dz = 0.0     
                self.controller.move_delta(dx, dy, dz, orn=self.controller.down_orn, steps=15)
                
        elif self.state == "VS_DELIVER":
            color = self.servo_target_color
            base_goal = self.env.get_target_pose(color)[:2]
            count = self.placed_counts[color]
            idx = count % len(self.place_offsets)
            dx, dy = self.place_offsets[idx]
            goal = (base_goal[0] + dx, base_goal[1] + dy)
            
            current_pose, current_orn = self.env.get_ee_pose()
            start = current_pose[:2]
            
            self.controller.vertical_move([start[0], start[1]], self.controller.z_pick, self.controller.z_safe, orn=current_orn, num_waypoints=10)
            self.controller.smooth_twist([start[0], start[1]], self.controller.z_safe, np.pi / 2, num_waypoints=20)
            self.controller.horizontal_move(start, goal, self.controller.z_safe, orn=self.controller.down_orn, num_waypoints=15)
            
            self.controller.move_to(goal, self.controller.z_drop, orn=self.controller.down_orn, steps=100)
            self.controller.open_gripper(width=0.08)
            for _ in range(40): p.stepSimulation(); time.sleep(1.0/240.0)
            self.controller.move_to(goal, self.controller.z_safe, orn=self.controller.down_orn, steps=100)

            for _ in range(120): 
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

            self.placed_counts[color] += 1
            self.state = "RETURN_HOME"

        elif self.state == "PICK_AND_PLACE":
            start = self.current["world"]
            angle_deg = self.current["angle"]
            angle = angle_deg % 90
            if angle > 45: angle -= 90
            yaw = -np.deg2rad(angle)
            
            color = self.current["color"]
            base_goal = self.env.get_target_pose(color)[:2]
            count = self.placed_counts[color]
            idx = count % len(self.place_offsets)
            dx, dy = self.place_offsets[idx]
            goal = (base_goal[0] + dx, base_goal[1] + dy)
            
            self.controller.pick_and_place(start, goal, yaw=yaw)
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