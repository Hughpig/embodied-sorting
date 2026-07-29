import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        # 指尖刚好捏在方块上半部
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
        self.env.set_gripper(0.0)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def pick_and_place(self, start_xy: list, target_xy: list) -> None:
        start = np.array(start_xy, dtype=float)
        goal = np.array(target_xy, dtype=float)

        self.open_gripper()
        # 1. 飞到方块正上方，停稳
        self.move_to(start, self.z_safe, steps=150)
        
        # ==================================================
        # 🎓 核心绝杀：笛卡尔空间垂直电梯式插值 (Cartesian Z-Waypoints)
        # 强制锁死 X 和 Y 不变，将 Z 轴下降拆分成 10 份。
        # 彻底消灭关节空间插值带来的“画弧线踢飞方块”效应！
        # ==================================================
        num_waypoints = 10
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_safe + (self.z_pick - self.z_safe) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        # 确保下降到位后彻底停稳
        self.move_to(start, self.z_pick, steps=40)
        
        # 3. 稳稳夹紧
        self.close_gripper()
        
        # 4. 同样像电梯一样垂直抬起，防止出坑时把方块甩出去
        for i in range(1, num_waypoints + 1):
            interp_z = self.z_pick + (self.z_safe - self.z_pick) * (i / num_waypoints)
            self.move_to(start, interp_z, steps=15)
        
        self.move_to(start, self.z_safe, steps=40)
        
        # 5. 飞往目标区域
        self.move_to(goal, self.z_safe, steps=200)
        
        # 6. 下降到安全高度投放
        z_drop = WORKSPACE["z_table"] + 0.12
        self.move_to(goal, z_drop, steps=100)
        
        self.open_gripper()
        
        # 7. 回归安全高度
        self.move_to(goal, self.z_safe, steps=100)