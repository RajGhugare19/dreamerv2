import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

import os

from mujoco_py.generated import const


def get_state_block(state):
    x = state[2].item()
    y = state[3].item()

    if -1 < x < 1:
        x_block = 'low'
    elif 1 < x < 3:
        x_block = 'mid'
    elif 3 < x < 5:
        x_block = 'high'
    else:
        raise Exception

    if -1 < y < 1:
        y_block = 'left'
    elif 1 < y < 3:
        y_block = 'center'
    elif 3 < y < 5:
        y_block = 'right'
    else:
        raise Exception

    if x_block == 'low' and y_block == 'left':
        return 0
    elif x_block == 'low' and y_block == 'center':
        return 1
    elif x_block == 'low' and y_block == 'right':
        return 2
    elif x_block == 'mid' and y_block == 'right':
        return 3
    elif x_block == 'high' and y_block == 'right':
        return 4
    elif x_block == 'high' and y_block == 'center':
        return 5
    elif x_block == 'high' and y_block == 'left':
        return 6


def rate_buffer(buffer):
    visited_blocks = [get_state_block(state) for state in buffer.states]
    n_unique = len(set(visited_blocks))
    return n_unique


class MagellanAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Observation Space:
        - x torso COM velocity
        - y torso COM velocity
        - 15 joint positions
        - 14 joint velocities
        - (optionally, commented for now) 84 contact forces
    """
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant_maze.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def contact_forces(self):
        return np.clip(self.sim.data.cfrc_ext, -1, 1)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        return obs, None, False, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt

        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent
        self.viewer.cam.lookat[0] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -85
        self.viewer.cam.azimuth = 235

    @property
    def tasks(self):
        t = dict()
        return t
