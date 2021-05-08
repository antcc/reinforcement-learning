import pyglet

import numpy as np
import gym
from gym.spaces import Discrete, Box


class Viewer(pyglet.window.Window):
    color = {"background": [1] * 3 + [1]}
    fps_display = pyglet.clock.Clock()
    bar_thc = 5

    def __init__(
        self,
        width,
        height,
        arm1_coords,
        arm2_coords,
        arm1_ang,
        arm2_ang,
        target_coords,
        target_width,
        mouse_in,
    ):
        super(Viewer, self).__init__(
            width, height, resizable=False, caption="Arm", vsync=False
        )  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color["background"])

        self.arm1_coords = arm1_coords
        self.arm2_coords = arm2_coords
        self.arm1_ang = arm1_ang
        self.arm2_ang = arm2_ang

        self.target_coords = target_coords
        self.mouse_in = mouse_in
        self.target_width = target_width

        self.center_coords = np.array((min(width, height) / 2,) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, target_box = [0] * 8, [0] * 8, [0] * 8
        c1, c2, c3 = (249, 86, 86) * 4, (86, 109, 249) * 4, (249, 39, 65) * 4
        self.target = self.batch.add(
            4, pyglet.gl.GL_QUADS, None, ("v2f", target_box), ("c3B", c2)
        )
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None, ("v2f", arm1_box), ("c3B", c1)
        )
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None, ("v2f", arm2_box), ("c3B", c1)
        )

    def render(self, arm1_coords, arm2_coords, arm1_ang, arm2_ang):
        self.arm1_coords = arm1_coords
        self.arm2_coords = arm2_coords
        self.arm1_ang = arm1_ang
        self.arm2_ang = arm2_ang

        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        target_width = self.target_width
        target_coords = self.target_coords
        target_box = (
            target_coords[0] - target_width,
            target_coords[1] - target_width,
            target_coords[0] + target_width,
            target_coords[1] - target_width,
            target_coords[0] + target_width,
            target_coords[1] + target_width,
            target_coords[0] - target_width,
            target_coords[1] + target_width,
        )
        self.target.vertices = target_box

        arm1_aux = np.hstack((self.center_coords, self.arm1_coords))  # (x0, y0, x1, y1)
        arm2_aux = np.hstack((self.arm1_coords, self.arm2_coords))  # (x1, y1, x2, y2)

        arm1_thick_rad = np.pi / 2 - self.arm1_ang
        x01 = arm1_aux[0] - np.cos(arm1_thick_rad) * self.bar_thc
        y01 = arm1_aux[1] + np.sin(arm1_thick_rad) * self.bar_thc

        x02 = arm1_aux[0] + np.cos(arm1_thick_rad) * self.bar_thc
        y02 = arm1_aux[1] - np.sin(arm1_thick_rad) * self.bar_thc

        x11 = arm1_aux[2] + np.cos(arm1_thick_rad) * self.bar_thc
        y11 = arm1_aux[3] - np.sin(arm1_thick_rad) * self.bar_thc

        x12 = arm1_aux[2] - np.cos(arm1_thick_rad) * self.bar_thc
        y12 = arm1_aux[3] + np.sin(arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)

        arm2_thick_rad = np.pi / 2 - (self.arm1_ang + self.arm2_ang)
        x11_ = arm2_aux[0] + np.cos(arm2_thick_rad) * self.bar_thc
        y11_ = arm2_aux[1] - np.sin(arm2_thick_rad) * self.bar_thc

        x12_ = arm2_aux[0] - np.cos(arm2_thick_rad) * self.bar_thc
        y12_ = arm2_aux[1] + np.sin(arm2_thick_rad) * self.bar_thc

        x21 = arm2_aux[2] - np.cos(arm2_thick_rad) * self.bar_thc
        y21 = arm2_aux[3] + np.sin(arm2_thick_rad) * self.bar_thc

        x22 = arm2_aux[2] + np.cos(arm2_thick_rad) * self.bar_thc
        y22 = arm2_aux[3] - np.sin(arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)

        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm1_ang += 0.1 * 10
            print(
                self.arm1_coords - self.target_coords,
                self.arm1_ang,
                self.arm2_coords - self.target_coords,
                self.arm2_ang,
            )
        elif symbol == pyglet.window.key.DOWN:
            self.arm1_ang -= 0.1 * 10
            print(
                self.arm1_coords - self.target_coords,
                self.arm1_ang,
                self.arm2_coords - self.target_coords,
                self.arm2_ang,
            )
        elif symbol == pyglet.window.key.LEFT:
            self.arm2_ang += 0.1
            print(
                self.arm1_coords - self.target_coords,
                self.arm1_ang,
                self.arm2_coords - self.target_coords,
                self.arm2_ang,
            )
        elif symbol == pyglet.window.key.RIGHT:
            self.arm2_ang -= 0.1
            print(
                self.arm1_coords - self.target_coords,
                self.arm1_ang,
                self.arm2_coords - self.target_coords,
                self.arm2_ang,
            )
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.target_coords[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False


class myEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, target_coords=[250, 303]):
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
        self.grab_counter = 0

        self.action_space = Discrete(3 * 3)  #
        # observation:
        self.observation_space = Box(
            low=-1e10, high=1e10, shape=(self.state_dim,), dtype=np.float32
        )
        self.observation_space.n = self.state_dim

        self.target_coords = np.array(target_coords)
        self.target_coords_init = self.target_coords.copy()
        self.center_coords = np.array(self.viewer_xy) / 2

    def render(self, mode="human"):
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
            self.arm1_coords, self.arm2_coords, self.arm1_ang, self.arm2_ang
        )

    def reset(self):
        self.got_target = False
        self.grab_counter = 0
        self.target_coords = np.random.randint(low = -self.viewer_xy[0] + self.center_coords[0],
                                               high = self.viewer_xy[0] - self.center_coords[0],
                                               size = 2)
        pxy = np.clip(self.target_coords, 100, 300)
        self.target_coords[:] = pxy
        s = self._get_state()
        return s  # observation

    def _get_state(self):
        """
        Returns arm coordinates
        """
        return np.array([*self.arm1_coords, *self.arm2_coords, *self.target_coords])

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
        r = np.linalg.norm(self.arm2_coords - self.target_coords)
        
        if r < 10:
            self.got_target = True
        else:
            self.got_target = False

        return s, -r/self.viewer_xy[0], self.got_target, {}

    def close(self):
        if self.viewer:
            self.viewer.close()
