import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        self.z_pick = WORKSPACE["z_table"] + 0.015
        self.z_drop = WORKSPACE["z_table"] + 0.050
        self.down_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2])

    def move_to(self, xy: list, z: float, orn: list = None, steps: int = 150) -> None:
        if orn is None: orn = self.down_orn
        self.env.move_ee_pose([xy[0], xy[1], z], orn, steps=steps)

    def move_delta(self, dx: float, dy: float, dz: float, orn: list = None, steps: int = 15) -> None:
        if orn is None: orn = self.down_orn
        current_pose, _ = self.env.get_ee_pose()
        target_x = current_pose[0] + dx
        target_y = current_pose[1] + dy
        target_z = current_pose[2] + dz
        self.env.move_ee_pose([target_x, target_y, target_z], orn, steps=steps)

    def vertical_move(self, xy: list, start_z: float, end_z: float, orn: list = None, num_waypoints: int = 10) -> None:
        if orn is None: orn = self.down_orn
        for i in range(1, num_waypoints + 1):
            interp_z = start_z + (end_z - start_z) * (i / num_waypoints)
            self.move_to(xy, interp_z, orn=orn, steps=15)
        self.move_to(xy, end_z, orn=orn, steps=30)

    def horizontal_move(self, start_xy: list, goal_xy: list, z: float, orn: list = None, num_waypoints: int = 15) -> None:
        if orn is None: orn = self.down_orn
        start = np.array(start_xy, dtype=float)
        goal = np.array(goal_xy, dtype=float)
        for i in range(1, num_waypoints + 1):
            interp_xy = start + (goal - start) * (i / num_waypoints)
            self.move_to([interp_xy[0], interp_xy[1]], z, orn=orn, steps=15)
        self.move_to([goal[0], goal[1]], z, orn=orn, steps=30)

    # ==========================================
    # 🎓 核心绝杀：显式角度传参（彻底免疫万向节死锁！）
    # 绝对不去读取 getEulerFromQuaternion，直接从 start_yaw 线性插值到 target_yaw！
    # ==========================================
    def smooth_twist(self, xy: list, z: float, start_yaw: float, target_yaw: float, num_waypoints: int = 20) -> None:
        diff = (target_yaw - start_yaw + np.pi) % (2 * np.pi) - np.pi
        for i in range(1, num_waypoints + 1):
            interp_yaw = start_yaw + diff * (i / num_waypoints)
            interp_orn = p.getQuaternionFromEuler([np.pi, 0.0, interp_yaw])
            self.move_to([xy[0], xy[1]], z, orn=interp_orn, steps=5)

    def open_gripper(self, width: float = 0.060) -> None:
        self.env.set_gripper(width)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def close_gripper(self) -> None:
        self.env.set_gripper(0.032)
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def pick_and_place(self, start_xy: list, target_xy: list, yaw: float = 0.0) -> None:
        start = np.array(start_xy, dtype=float)
        goal = np.array(target_xy, dtype=float)

        target_yaw_world = np.pi / 2 + yaw
        dynamic_orn = p.getQuaternionFromEuler([np.pi, 0.0, target_yaw_world])

        self.open_gripper(width=0.060)
        
        # 1. 飞往方块上方
        self.move_to(start, self.z_safe, orn=self.down_orn, steps=150)
        
        # 2. 从 90度 (np.pi/2) 平滑转到目标角度，显式传参！
        self.smooth_twist(start, self.z_safe, start_yaw=np.pi/2, target_yaw=target_yaw_world, num_waypoints=20)
        
        # 3. 垂直下落
        self.vertical_move(start, self.z_safe, self.z_pick, orn=dynamic_orn, num_waypoints=12)
        
        for _ in range(30): p.stepSimulation(); time.sleep(1.0 / 240.0)
        self.close_gripper()
        
        # 4. 垂直拔起
        self.vertical_move(start, self.z_pick, self.z_safe, orn=dynamic_orn, num_waypoints=12)
        
        # 5. 从目标角度平滑拧回 90度 (np.pi/2)
        self.smooth_twist(start, self.z_safe, start_yaw=target_yaw_world, target_yaw=np.pi/2, num_waypoints=20)
        
        # 6. 端正平移
        self.horizontal_move(start, goal, self.z_safe, orn=self.down_orn, num_waypoints=15)
        
        self.move_to(goal, self.z_drop, orn=self.down_orn, steps=100)
        self.open_gripper(width=0.080)
        for _ in range(40): p.stepSimulation(); time.sleep(1.0 / 240.0)
        self.move_to(goal, self.z_safe, orn=self.down_orn, steps=100)