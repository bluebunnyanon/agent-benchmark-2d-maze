import math
import os

import gymnasium
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class PointyEnv(MujocoEnv, utils.EzPickle):
    """PointMass environment with a directional pyramid marker.

    Action semantics are (delta_forward, delta_heading), where heading is in radians.
    This keeps the same point-mass physics as PointEnv, but renders a pointy
    visual marker that indicates heading.
    """

    xml_file = os.path.join(os.path.dirname(__file__), 'assets', 'pointy.xml')
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 10,
    }
    if gymnasium.__version__ >= '1.1.0':
        metadata['render_modes'] += ['rgbd_tuple']

    def __init__(
        self,
        xml_file=None,
        render_mode='rgb_array',
        width=200,
        height=200,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = self.xml_file
        utils.EzPickle.__init__(
            self,
            xml_file,
            **kwargs,
        )

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )
        self._marker_height = 0.05
        self._marker_heading = np.array([1.0, 0.0], dtype=np.float64)
        self._forward_scale = 0.2
        self._heading_scale = 0.35

    def _update_marker_pose(self, heading_xy=None):
        if heading_xy is not None:
            heading_xy = np.asarray(heading_xy, dtype=np.float64)
            heading_norm = np.linalg.norm(heading_xy)
            if heading_norm > 1e-8:
                self._marker_heading = heading_xy / heading_norm

        marker_angle = math.atan2(self._marker_heading[1], self._marker_heading[0])
        self.data.mocap_pos[0] = np.array([self.data.qpos[0], self.data.qpos[1], self._marker_height])
        self.data.mocap_quat[0] = np.array([
            math.cos(marker_angle / 2.0),
            0.0,
            0.0,
            math.sin(marker_angle / 2.0),
        ])
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size < 2:
            padded = np.zeros(2, dtype=np.float64)
            padded[: action.size] = action
            action = padded

        delta_forward = self._forward_scale * float(action[0])
        delta_heading = self._heading_scale * float(action[1])

        # Rotate current heading by delta_heading.
        c = math.cos(delta_heading)
        s = math.sin(delta_heading)
        hx, hy = self._marker_heading
        new_heading = np.array([c * hx - s * hy, s * hx + c * hy], dtype=np.float64)
        heading_norm = np.linalg.norm(new_heading)
        if heading_norm > 1e-8:
            self._marker_heading = new_heading / heading_norm

        delta_xy = delta_forward * self._marker_heading

        # Kinematic update: apply intended translation directly. Maze collision logic
        # in the wrapper decides whether to keep or reject the move.
        self.data.qpos[:] = self.data.qpos + delta_xy
        self.data.qvel[:] = np.array([0.0, 0.0])
        self._update_marker_pose()

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        observation = self.get_ob()

        if self.render_mode == 'human':
            self.render()

        return (
            observation,
            0.0,
            False,
            False,
            {
                'xy': self.get_xy(),
                'prev_qpos': prev_qpos,
                'prev_qvel': prev_qvel,
                'qpos': qpos,
                'qvel': qvel,
            },
        )

    def get_ob(self):
        return self.data.qpos.flat.copy()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        self.set_state(qpos, qvel)
        self._marker_heading = np.array([1.0, 0.0], dtype=np.float64)
        self._update_marker_pose()

        return self.get_ob()

    def get_xy(self):
        return self.data.qpos.copy()

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:] = xy
        self.set_state(qpos, qvel)
        self._update_marker_pose()