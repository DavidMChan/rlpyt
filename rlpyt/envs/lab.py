from collections import namedtuple

import numpy as np
import cv2

from rlpyt.envs.base import Env, EnvSpaces, EnvStep
from rlpyt.utils.collections import is_namedtuple_class, is_namedtuple
from rlpyt.spaces.int_box import IntBox

import deepmind_lab

EnvInfo = namedtuple("EnvInfo", ['total_reward', 'traj_done'])

LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
'nav_maze_random_goal_01','nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_static_01', \
'nav_maze_static_02', 'seekavoid_arena_01', 'stairway_to_melon', 'tests/empty_room_test']


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DeepmindLabEnv(Env):

    def __init__(self, level, width=84, height=84, horizon=27000):

        if level not in LEVELS:
            raise ValueError('Level {} not found in deepmind lab'.format(level))

        self._level = level
        self._width = width
        self._height = height
        self._horizon = int(horizon)
        self._lab = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], {
            'fps': '60',
            'width': str(self._width),
            'height': str(self._height),
        })

        # Setup the available actions
        self._action_set = [
            _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
            _action(20, 0, 0, 0, 0, 0, 0),  # look_right
            #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
            #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
            #_action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
            #_action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
            _action(0, 0, 0, 1, 0, 0, 0),  # forward
            _action(0, 0, 0, -1, 0, 0, 0),  # backward
            #_action(  0,   0,  0,  0, 1, 0, 0), # fire
            #_action(  0,   0,  0,  0, 0, 1, 0), # jump
            #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
        ]

        self._action_space = IntBox(low=0, high=len(self._action_set))
        obs_shape = (3, self._height, self._width)  # Only works for RGB_INTERLEAVED
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape, dtype="uint8")
        self._obs = np.zeros(obs_shape, dtype=np.uint8)
        self._total_reward = 0
        self.reset()

    def _update_obs(self):
        obs = self._lab.observations()['RGB_INTERLEAVED']
        self._obs = np.transpose(obs, axes=(2, 0, 1))

    def step(self, action):
        reward = self._lab.step(self._action_set[action])
        finished = not self._lab.is_running()
        if not finished:
            self._update_obs()
        self._total_reward += reward
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, finished, EnvInfo(
            total_reward=self._total_reward,
            traj_done=finished,
        ))

    def reset(self):
        self._lab.reset()
        self._step_counter = 0
        self._total_reward = 0
        self._update_obs()
        return self.get_obs()

    def render(self, wait=10):
        img = self.get_obs()
        cv2.imshow(self._level, np.transpose(img, axes=(1, 2, 0)))
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        return self._horizon
