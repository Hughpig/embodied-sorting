import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        # 核心修复：把抓取高度从 z_table + 0.025 降低到 z_table + 0.010
        # 让手指深深地插到底，彻底抱住方块的两侧，无视几毫米的视觉误差！
        self.z_pick = WORKSPACE["z_table"] + 0.010
        self.down_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2])

    def move_to(self, xy: list, z: float, steps: int = 150) -> None:
        self.env.move_ee_pose([xy[0], xy[1], z], self.down_orn, steps=steps)

    def open_gripper(self) -> None:
        self.env.set_gripper(0.08)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close_gripper(self) -> None:
        self.env.set_gripper(0.0)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def pick_and_place(self, start_xy: list, target_xy: list) -> None:
        start = np.array(start_xy, dtype=float)
        goal = np.array(target_xy, dtype=float)

        self.open_gripper()
        self.move_to(start, self.z_safe)
        
        self.move_to(start, self.z_pick, steps=100)
        
        self.close_gripper()
        
        self.move_to(start, self.z_safe, steps=100)
        
        self.move_to(goal, self.z_safe, steps=200)
        
        z_drop = WORKSPACE["z_table"] + 0.12
        self.move_to(goal, z_drop, steps=100)
        
        self.open_gripper()
        
        self.move_to(goal, self.z_safe, steps=100)