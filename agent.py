from viewer import Viewer
import gym
import numpy as np
from gym.spaces import Discrete, Box


class myEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, target_coords=[250, 303], mode="hard"):
        self.state_dim = 7
        self.dt = 0.1  # refresh rate

        self.arm1_long = 100
        self.arm2_long = 100
        self.arm1_ang = 0
        self.arm2_ang = 0
        self.arm1_coords = np.array([0, 0])
        self.arm2_coords = np.array([0, 0])

        self.viewer = None
        self.viewer_xy = (400, 400)
        self.got_target = False
        self.mouse_in = np.array([False])
        self.target_width = 15

        self.action_space = Discrete(3 * 3)  #
        # observation:
        self.observation_space = Box(
            low=-1e10, high=1e10, shape=(self.state_dim,), dtype=np.float32
        )
        self.observation_space.n = self.state_dim
        self.mode = mode

        self.target_coords = np.array(target_coords)
        self.target_coords_init = self.target_coords.copy()
        self.center_coords = np.array(self.viewer_xy) / 2

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(
                *self.viewer_xy,
                self.arm1_coords,
                self.arm2_coords,
                self.arm1_ang,
                self.arm2_ang,
                self.target_coords,
                self.target_width,
                self.mouse_in
            )
        self.viewer.render(
            self.arm1_coords, self.arm2_coords, self.arm1_ang,
            self.arm2_ang, self.target_coords
        )

    def reset(self):

        if self.mode == "hard":
            pxy = np.random.randint(low=-self.viewer_xy[0] + self.center_coords[0],
                                    high=self.viewer_xy[0] - self.center_coords[0],
                                    size=2)
            self.target_coords[:] = np.clip(pxy, 100, 300)

        elif self.mode == "easy":
            self.arm1_ang = 0
            self.arm2_ang = 0
            self.arm1_coords = np.array([0, 0])
            self.arm2_coords = np.array([0, 0])
            self.target_coords[:] = self.target_coords_init
        elif self.mode == "supereasy":
            # Do not restart anything
            pass
        elif self.mode == "follow":
            self.target_coords += np.random.randint(low = -20, high = 20, size = 2)
            self.target_coords = np.clip(self.target_coords, 100, 300)

        s = self._get_state()
        self.got_target = self._got_target()
        return s  # observation

    def _got_target(self):
        dist = np.linalg.norm(self.arm2_coords - self.target_coords)
        return dist < self.target_width

    def _get_state(self):
        """
        Returns arm coordinates
        """
        a = (self.target_coords - self.center_coords)/200
        b = (self.target_coords - self.arm2_coords)/200
        return np.array([*a, *b])

    def step(self, act):
        action1 = act // 3 - 1
        action2 = act % 3 - 1

        self.arm1_ang += action1 * self.dt
        self.arm1_ang %= np.pi * 2
        self.arm2_ang += action2 * self.dt
        self.arm2_ang %= np.pi * 2

        self.arm1_coords = self.center_coords.copy()
        self.arm1_coords[0] += self.arm1_long * np.cos(self.arm1_ang)
        self.arm1_coords[1] += self.arm1_long * np.sin(self.arm1_ang)

        self.arm2_coords = self.arm1_coords.copy()
        self.arm2_coords[0] += self.arm2_long * np.cos(self.arm1_ang + self.arm2_ang)
        self.arm2_coords[1] += self.arm2_long * np.sin(self.arm1_ang + self.arm2_ang)

        s = self._get_state()

        # Euclidean distance between
        dist = np.linalg.norm(self.arm2_coords - self.target_coords)
        r = -dist/200

        if dist < self.target_width:
            r += 1
            if self.got_target:
                r += 1
            else:
                self.got_target = True
        else:
            self.got_target = False

        return s, r, self.got_target, {}

    def close(self):
        if self.viewer:
            self.viewer.close()
