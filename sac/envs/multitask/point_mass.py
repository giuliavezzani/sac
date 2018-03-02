from rllab.envs.base import Step

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import math
from rllab.mujoco_py import glfw

from rllab.envs.mujoco.mujoco_env import MujocoEnv

from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np
import pickle


class PointEnv(MujocoEnv, Serializable):

    """
    Use Left, Right, Up, Down, A (steer left), D (steer right)
    """

    #global FILE = 'point.xml'
    global FILE

    #def __init__(self, *args, **kwargs):
    def __init__(self,file_goals, file_env, goal= None, *args, **kwargs):
        FILE=file_env
        #import IPython
        #IPython.embed()
        self._goal_idx=goal
        print(goal)
        self.all_goals = 3 * pickle.load(open(file_goals, "rb"))
        super(PointEnv, self).__init__(*args, **kwargs, file_path=FILE)
        Serializable.quick_init(self, locals())

    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.geom_xpos[-6:-1, :2].flat,
            self.model.data.qvel.flat,
            ]).reshape(-1)

    def get_xy(self):
        qpos = self.model.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def set_xy(self, xy):
        qpos = np.copy(self.model.data.qpos)
        qpos[0, 0] = xy[0]
        qpos[1, 0] = xy[1]
        self.model.data.qpos = qpos
        self.model.forward()

    def step(self, action):
        qpos = np.copy(self.model.data.qpos)
        qpos[2, 0] += action[1]
        ori = qpos[2, 0]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0, 0] = np.clip(qpos[0, 0] + 0.5 * dx, -7, 7)
        qpos[1, 0] = np.clip(qpos[1, 0] + 0.5 * dy, -7, 7)
        self.model.data.qpos = qpos
        self.model.forward()
        next_obs = self.get_current_obs()
        self.current_com = self.model.data.com_subtree[0]

        reward = -np.linalg.norm(qpos[0:2, 0] - self.all_goals[self._goal_idx])
        #infos = {'goal': self._goal_idx}
        return Step(next_obs, reward, False)

    @overrides
    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0]*0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0]*0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1]*0.3, 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1]*0.3, 0])
        else:
            return np.array([0, 0])
