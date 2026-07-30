import numpy as np
import pybullet as p
import time
from .env import SortingEnv, WORKSPACE

class Controller:
    def __init__(self, env: SortingEnv) -> None:
        self.env = env
        self.z_safe = 0.85
        # 恢复老版本稳定高度：0.025
        self.z_pick = WORKSPACE["z_table"] + 0.025
        # 恢复老版本投放高度：0.12
        self.z_drop = WORKSPACE["z_table"] + 0.12
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

    def smooth_twist(self, xy: list, z: float, target_yaw: float, num_waypoints: int = 20) -> None:
        current_pose, current_orn = self.env.get_ee_pose()
        start_euler = p.getEulerFromQuaternion(current_orn)
        start_yaw = start_euler[2]
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
        # 恢复老版本黄金柔性宽度：0.032
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
        
        self.move_to(start, self.z_safe, orn=self.down_orn, steps=150)
        self.smooth_twist(start, self.z_safe, target_yaw_world, num_waypoints=20)
        
        self.vertical_move(start, self.z_safe, self.z_pick, orn=dynamic_orn, num_waypoints=12)
        
        for _ in range(30): p.stepSimulation(); time.sleep(1.0 / 240.0)
        self.close_gripper()
        
        self.vertical_move(start, self.z_pick, self.z_safe, orn=dynamic_orn, num_waypoints=12)
        self.smooth_twist(start, self.z_safe, np.pi / 2, num_waypoints=20)
        
        self.horizontal_move(start, goal, self.z_safe, orn=self.down_orn, num_waypoints=15)
        
        self.move_to(goal, self.z_drop, orn=self.down_orn, steps=100)
        self.open_gripper(width=0.080)
        for _ in range(40): p.stepSimulation(); time.sleep(1.0 / 240.0)
        self.move_to(goal, self.z_safe, orn=self.down_orn, steps=100)

        for _ in range(120): 
            p.stepSimulation()
            time.sleep(1.0 / 240.0)