import gym
import numpy as np
from gym.spaces import Discrete, Box


class myEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 50}

    def __init__(self, threshold=30, target_coords=[250, 300], mode="hard"):
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
        self.steps_in_target = 0
        self.threshold = threshold
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
        if self.mode == "supereasy":  # Do not reset anything
            pass
        elif self.mode == "easy":  # Reset arm to (0, 0)
            self.arm1_ang = 0
            self.arm2_ang = 0
            self.arm1_coords = np.array([0, 0])
            self.arm2_coords = np.array([0, 0])
        elif self.mode == 'medium':  # Reset arm to random position
            self.arm1_ang = np.random.uniform(0, 2*np.pi)
            self.arm2_ang = np.random.uniform(0, 2*np.pi)
            self.arm1_coords = np.random.randint(low=100, high=300, size=2)
            self.arm2_coords = np.random.randint(low=100, high=300, size=2)
        elif self.mode == "hard":  # Move target to random position
            pxy = np.random.randint(low=100, high=300, size=2)
            self.target_coords[:] = pxy
        elif self.mode == "follow":  # Move target to random nearby position
            self.target_coords += np.random.randint(low=-20, high=20, size=2)
            self.target_coords = np.clip(self.target_coords, 100, 300)

        s = self._get_state()

        return s

    def _get_state(self):
        """
        Returns internal state
        """
        target_pos = (self.target_coords - self.center_coords)/200
        arm1_pos = (self.arm1_coords - self.center_coords)/200
        arm2_dist = (self.target_coords - self.arm2_coords)/200
        has_target = 1 if self.steps_in_target > 0 else 0
        return np.array([*target_pos, *arm1_pos, *arm2_dist, has_target])

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
        self.arm2_coords[0] += self.arm2_long * \
            np.cos(self.arm1_ang + self.arm2_ang)
        self.arm2_coords[1] += self.arm2_long * \
            np.sin(self.arm1_ang + self.arm2_ang)

        s = self._get_state()

        # Euclidean distance between target and arm2
        dist = np.linalg.norm(self.arm2_coords - self.target_coords)
        r = -dist/200.

        if not self.got_target and dist <= self.target_width:  # if touching the target
            r += 1
            self.steps_in_target += 1
            if self.steps_in_target > self.threshold:
                r += 10
                self.got_target = True
        elif dist > self.target_width:
            self.steps_in_target = 0
            self.got_target = False

        return s, r, self.got_target, {}

    def close(self):
        if self.viewer:
            self.viewer.close()
