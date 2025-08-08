import math
from typing import Tuple

import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base.base_env import GameworldEnv


class Cartpole(GameworldEnv):
    """CartPole-v1 style dynamics with pixel observations.

    The observation returned by this environment is an RGB image of shape
    (210, 160, 3) where each pixel contains its color value.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Rendering params (match other gameworld bases)
        self.width: int = 160
        self.height: int = 210

        # Colors (R, G, B)
        self.bg_color: Tuple[int, int, int] = (50, 50, 100)
        self.track_color: Tuple[int, int, int] = (150, 150, 255)
        self.cart_color: Tuple[int, int, int] = (255, 255, 0)
        self.pole_color: Tuple[int, int, int] = (255, 0, 0)
        self.axle_color: Tuple[int, int, int] = (0, 255, 0)

        # Layout
        self.track_y: int = 150  # vertical pixel location of track center
        self.track_height: int = 6
        self.cart_width: int = 30
        self.cart_height: int = 15
        self.pole_length_px: int = 60
        self.pole_thickness_px: int = 6
        self.axle_radius_px: int = 3

        # Classic CartPole physics parameters (from Gymnasium)
        self.gravity: float = 9.8
        self.masscart: float = 1.0
        self.masspole: float = 0.1
        self.total_mass: float = self.masspole + self.masscart
        # length is actually half the pole's length in the classic implementation
        self.length: float = 0.5
        self.polemass_length: float = self.masspole * self.length
        self.force_mag: float = 10.0
        self.tau: float = 0.02  # seconds between state updates

        # Termination thresholds (from Gymnasium)
        self.theta_threshold_radians: float = 12 * 2 * math.pi / 360
        self.x_threshold: float = 2.4

        # State: [x, x_dot, theta, theta_dot]
        self.state: np.ndarray | None = None

        # Spaces
        self.action_space = spaces.Discrete(2)  # 0: push left, 1: push right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # Initialize
        self.reset(seed=0)

    def reset(self):
        # Same init noise as Gymnasium: small uniform noise around zero
        low = -0.05
        high = 0.05
        self.state = np.random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None, "Environment must be reset before stepping"

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Equations for the cart-pole system (from Gymnasium classic_control)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / (
            self.total_mass
        )
        theta_acc = (
            self.gravity * sintheta
            - costheta * temp
        ) / (
            self.length
            * (
                4.0 / 3.0
                - (self.masspole * costheta * costheta) / self.total_mass
            )
        )
        x_acc = temp - (self.polemass_length * theta_acc * costheta) / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0  # per-step reward until termination, as in classic CartPole

        return self._get_obs(), reward, terminated, False, {}

    def _world_to_pixel_x(self, x_world: float) -> int:
        # Map world x in [-x_threshold, x_threshold] to [0, width]
        # Keep a small margin so the cart stays fully visible
        margin = self.cart_width // 2 + 2
        usable_w = self.width - 2 * margin
        # Normalize to [0, 1]
        norm = (x_world + self.x_threshold) / (2 * self.x_threshold)
        pix = int(margin + norm * usable_w)
        return int(np.clip(pix, margin, self.width - margin))

    def _draw_scene(self) -> np.ndarray:
        # Start from a solid background
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Track
        ty0 = self.track_y - self.track_height // 2
        ty1 = ty0 + self.track_height
        draw.rectangle([0, ty0, self.width - 1, ty1], fill=self.track_color)

        # Cart position (x from state)
        x, _, theta, _ = self.state if self.state is not None else (0.0, 0.0, 0.0, 0.0)
        cart_cx = self._world_to_pixel_x(float(x))
        cart_top_y = self.track_y - self.cart_height // 2 - 1
        cart_left = cart_cx - self.cart_width // 2
        cart_right = cart_left + self.cart_width
        cart_bottom = cart_top_y + self.cart_height
        draw.rectangle(
            [cart_left, cart_top_y, cart_right, cart_bottom], fill=self.cart_color
        )

        # Pole from axle at top center of cart
        axle_x = cart_cx
        axle_y = cart_top_y
        pole_x = axle_x + self.pole_length_px * math.sin(theta)
        pole_y = axle_y - self.pole_length_px * math.cos(theta)
        draw.line(
            [axle_x, axle_y, pole_x, pole_y],
            fill=self.pole_color,
            width=self.pole_thickness_px,
            joint="curve",
        )

        # Axle
        r = self.axle_radius_px
        draw.ellipse([axle_x - r, axle_y - r, axle_x + r, axle_y + r], fill=self.axle_color)

        return np.array(img, dtype=np.uint8)

    def _get_obs(self):
        return self._draw_scene()


