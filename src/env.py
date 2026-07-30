"""Visual-guided robotic arm desktop sorting environment."""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

ColorName = str
RGB = Tuple[float, float, float]
Pose2D = Tuple[float, float]

COLOR_RGB: Dict[ColorName, RGB] = {
    "red": (0.95, 0.12, 0.10),
    "green": (0.10, 0.78, 0.18),
    "blue": (0.12, 0.35, 0.95),
}

WORKSPACE = {
    "x_min": 0.30,
    "x_max": 0.70,
    "y_min": -0.20,
    "y_max": 0.30,
    "z_table": 0.625,
}

TARGET_POSES: Dict[ColorName, Pose2D] = {
    "red": (0.35, -0.35),
    "green": (0.50, -0.35),
    "blue": (0.65, -0.35),
}

@dataclass
class BlockInfo:
    body_id: int
    color: ColorName
    size: float = 0.04

class SortingEnv:
    def __init__(self, gui: bool = True, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(0)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.35,
            cameraYaw=55,
            cameraPitch=-40,
            cameraTargetPosition=[0.45, 0.0, 0.55],
        )

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, WORKSPACE["z_table"]], useFixedBase=True)

        self.arm_joint_indices = list(range(7))
        self.finger_joint_indices = [9, 10]
        self.ee_link_index = 11
        self.home_joint_positions = [0.0, -0.55, 0.0, -2.0, 0.0, 1.5, 0.8]
        self.down_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

        self.blocks: List[BlockInfo] = []
        self.target_ids: Dict[ColorName, int] = {}
        self._create_target_zones()
        self._configure_camera()
        self.reset_arm()

    def _create_target_zones(self) -> None:
        for color, (x, y) in TARGET_POSES.items():
            pad_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.09, 0.09, 0.002])
            pad_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.09, 0.09, 0.002], rgbaColor=[0.75, 0.75, 0.75, 0.85])
            pad = p.createMultiBody(0, pad_col, pad_vis, [x, y, WORKSPACE["z_table"] + 0.001])
            chip_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.01, 0.001], rgbaColor=[*COLOR_RGB[color], 1.0])
            chip = p.createMultiBody(0, -1, chip_vis, [x, y - 0.07, WORKSPACE["z_table"] + 0.004])
            self.target_ids[color] = pad

    def _configure_camera(self) -> None:
        self.cam_eye = [0.50, 0.0, 1.40]
        self.cam_target = [0.50, 0.0, WORKSPACE["z_table"]]
        self.cam_up = [1.0, 0.0, 0.0]
        self.cam_fov = 55.0
        self.cam_near = 0.05
        self.cam_far = 3.0
        aspect = self.width / float(self.height)
        self.view_matrix = p.computeViewMatrix(self.cam_eye, self.cam_target, self.cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(self.cam_fov, aspect, self.cam_near, self.cam_far)

    def _spawn_block(self, color: ColorName, xy: Pose2D, size: float = 0.04) -> BlockInfo:
        half = size / 2.0
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[*COLOR_RGB[color], 1.0])
        z = WORKSPACE["z_table"] + half + 0.001
        body = p.createMultiBody(0.04, col, vis, [xy[0], xy[1], z])
        p.changeDynamics(body, -1, lateralFriction=1.5, spinningFriction=0.2)
        return BlockInfo(body_id=body, color=color, size=size)

    def clear_blocks(self) -> None:
        for block in self.blocks:
            try:
                p.removeBody(block.body_id)
            except Exception:
                pass
        self.blocks = []

    def _sample_positions(self, n: int) -> List[Pose2D]:
        positions = []
        for _ in range(1000):
            if len(positions) >= n:
                break
            x = float(self.rng.uniform(0.35, 0.60))
            y = float(self.rng.uniform(0.00, 0.28))
            if all(math.hypot(x - px, y - py) >= 0.06 for px, py in positions):
                positions.append((x, y))
        while len(positions) < n:
            positions.append((0.40, 0.12))
        return positions

    def reset(self, n_blocks: int = 4) -> List[BlockInfo]:
        self.clear_blocks()
        self.reset_arm()
        available_colors = ["red", "green", "blue"]
        colors = [str(self.rng.choice(available_colors)) for _ in range(n_blocks)]
        positions = self._sample_positions(n_blocks)
        self.blocks = [self._spawn_block(c, xy) for c, xy in zip(colors, positions)]
        for _ in range(60):
            p.stepSimulation()
        return list(self.blocks)

    def reset_arm(self) -> None:
        for i, q in zip(self.arm_joint_indices, self.home_joint_positions):
            p.resetJointState(self.robot_id, i, q)
        for j in self.finger_joint_indices:
            p.resetJointState(self.robot_id, j, 0.04)
        self.set_gripper(0.08)
        for _ in range(10):
            p.stepSimulation()

    def set_gripper(self, open_width: float = 0.08) -> None:
        half = float(np.clip(open_width / 2.0, 0.0, 0.04))
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=half, force=40, maxVelocity=0.2)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        st = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(st[0], dtype=float), np.array(st[1], dtype=float)

    def move_ee_pose(self, position: Sequence[float], orn: Optional[Sequence[float]] = None, steps: int = 140) -> float:
        if orn is None:
            orn = self.down_orn
        joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_link_index, position, orn, lowerLimits=[-2.9]*7, upperLimits=[2.9]*7, jointRanges=[5.8]*7, restPoses=self.home_joint_positions, maxNumIterations=200, residualThreshold=1e-5)
        for i, q in zip(self.arm_joint_indices, joint_poses[:7]):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=q, force=200, maxVelocity=1.0)
        for _ in range(steps):
            p.stepSimulation()
        ee, _ = self.get_ee_pose()
        return float(np.linalg.norm(ee - np.asarray(position, dtype=float)))

    def go_home(self, steps: int = 120) -> None:
        for i, q in zip(self.arm_joint_indices, self.home_joint_positions):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=q, force=200, maxVelocity=1.0)
        self.set_gripper(0.08)
        for _ in range(steps):
            p.stepSimulation()

    def get_camera_image(self) -> np.ndarray:
        _, _, rgba, depth, seg = p.getCameraImage(width=self.width, height=self.height, viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(np.array(rgba, dtype=np.uint8), (self.height, self.width, 4))[:, :, :3]
        return rgb

    def pixel_to_world(self, u: float, v: float) -> Pose2D:
        fov_rad = math.radians(self.cam_fov)
        aspect = self.width / float(self.height)
        ndc_x = (2.0 * u / self.width) - 1.0
        ndc_y = 1.0 - (2.0 * v / self.height)
        tan_half = math.tan(fov_rad / 2.0)
        forward = np.array(self.cam_target, dtype=float) - np.array(self.cam_eye, dtype=float)
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        up = np.array(self.cam_up, dtype=float)
        up = up / (np.linalg.norm(up) + 1e-9)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)
        dir_cam = forward + ndc_x * tan_half * aspect * right + ndc_y * tan_half * up
        dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-9)
        origin = np.array(self.cam_eye, dtype=float)
        z_table = WORKSPACE["z_table"]
        if abs(dir_cam[2]) < 1e-9:
            return (float(self.cam_target[0]), float(self.cam_target[1]))
        t = (z_table - origin[2]) / dir_cam[2]
        point = origin + t * dir_cam
        return (float(point[0]), float(point[1]))

    def get_block_pose(self, block: BlockInfo) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(block.body_id)
        return np.array(pos, dtype=float), np.array(orn, dtype=float)

    def get_target_pose(self, color: ColorName) -> np.ndarray:
        x, y = TARGET_POSES[color]
        return np.array([x, y, WORKSPACE["z_table"] + 0.02], dtype=float)

    def is_block_in_target(self, block: BlockInfo, tol: float = 0.10) -> bool:
        pos, _ = self.get_block_pose(block)
        target = self.get_target_pose(block.color)
        return float(np.linalg.norm(pos[:2] - target[:2])) <= tol

    def count_success(self) -> Tuple[int, int]:
        ok = sum(1 for b in self.blocks if self.is_block_in_target(b))
        return ok, len(self.blocks)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # === [为 Servo 模式追加的函数] ===
    def get_eye_in_hand_image(self) -> np.ndarray:
        ee_pos, _ = self.get_ee_pose()
        cam_eye = [ee_pos[0], ee_pos[1], ee_pos[2] + 0.08]
        cam_target = [ee_pos[0], ee_pos[1], 0.0]
        cam_up = [0.0, 1.0, 0.0] 
        view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(60.0, self.width/self.height, 0.01, 2.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return np.reshape(np.array(rgba, dtype=np.uint8), (self.height, self.width, 4))[:, :, :3]