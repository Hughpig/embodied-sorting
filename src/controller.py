import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        # 抓取高度保持在方块偏上部位
        self.z_pick = WORKSPACE["z_table"] + 0.025
        self.down_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2])

    def move_to(self, xy: list, z: float, steps: int = 150) -> None:
        self.env.move_ee_pose([xy[0], xy[1], z], self.down_orn, steps=steps)

    def open_gripper(self) -> None:
        self.env.set_gripper(0.08)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close_gripper(self) -> None:
        # ==========================================
        # 🎓 核心绝杀：柔性抓取 (Soft Grasping)
        # 绝不闭合到 0.0 产生暴力挤压！
        # 目标宽度设为 0.032 (略小于方块的 0.04)，实现温柔而坚定的夹持。
        # ==========================================
        self.env.set_gripper(0.032)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def pick_and_place(self, start_xy: list, target_xy: list) -> None:
        start = np.array(start_xy, dtype=float)
        goal = np.array(target_xy, dtype=float)

        self.open_gripper()
        self.move_to(start, self.z_safe, steps=150)
        
        # 笛卡尔垂直插值下降
        num_waypoints = 12
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_safe + (self.z_pick - self.z_safe) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        self.move_to(start, self.z_pick, steps=40)
        
        # 柔性抓取
        self.close_gripper()
        
        # 笛卡尔垂直插值上升，防止出坑甩飞
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_pick + (self.z_safe - self.z_pick) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        self.move_to(start, self.z_safe, steps=40)
        
        self.move_to(goal, self.z_safe, steps=200)
        
        z_drop = WORKSPACE["z_table"] + 0.12
        self.move_to(goal, z_drop, steps=100)
        
        self.open_gripper()
        
        self.move_to(goal, self.z_safe, steps=100)