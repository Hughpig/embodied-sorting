import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        # 你的稳定版高度
        self.z_pick = WORKSPACE["z_table"] + 0.025
        # 加上 z_drop 供 Servo 的独立投放逻辑使用
        self.z_drop = WORKSPACE["z_table"] + 0.12
        self.down_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2])

    def move_to(self, xy: list, z: float, steps: int = 150) -> None:
        self.env.move_ee_pose([xy[0], xy[1], z], self.down_orn, steps=steps)

    def open_gripper(self) -> None:
        self.env.set_gripper(0.08)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close_gripper(self) -> None:
        # 你的稳定版柔性宽度
        self.env.set_gripper(0.032)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def pick_and_place(self, start_xy: list, target_xy: list) -> None:
        # 这里的逻辑 100% 没动你的，保证 Nearest 和 Random 绝对稳定！
        start = np.array(start_xy, dtype=float)
        goal = np.array(target_xy, dtype=float)

        self.open_gripper()
        self.move_to(start, self.z_safe, steps=150)
        
        num_waypoints = 12
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_safe + (self.z_pick - self.z_safe) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        self.move_to(start, self.z_pick, steps=40)
        self.close_gripper()
        
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_pick + (self.z_safe - self.z_pick) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        self.move_to(start, self.z_safe, steps=40)
        self.move_to(goal, self.z_safe, steps=200)
        
        z_drop = WORKSPACE["z_table"] + 0.12
        self.move_to(goal, z_drop, steps=100)
        
        self.open_gripper()
        self.move_to(goal, self.z_safe, steps=100)

    # === [为 Servo 模式追加的函数] ===
    def move_delta(self, dx: float, dy: float, dz: float, steps: int = 15) -> None:
        current_pose, _ = self.env.get_ee_pose()
        target_x = current_pose[0] + dx
        target_y = current_pose[1] + dy
        target_z = current_pose[2] + dz
        self.env.move_ee_pose([target_x, target_y, target_z], self.down_orn, steps=steps)