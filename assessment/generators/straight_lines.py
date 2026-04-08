"""Straight line pattern generators (horizontal + angled)."""
import numpy as np
import cv2
from .base import (
    BaseGenerator, SyntheticVariation, register_generator,
    add_salt_pepper_noise, add_gaussian_noise, make_gradient_bg,
)


@register_generator
class StraightLinesGenerator(BaseGenerator):
    category = "straight_lines"

    def get_variations(self):
        return [
            SyntheticVariation("horiz_uniform", self.category, "primary", {
                "num_lines": 8, "thickness": 2, "angle": 0,
                "bg_type": "uniform", "bg_gray": 140, "noise": 0.01,
            }),
            SyntheticVariation("horiz_gradient_bg", self.category, "primary", {
                "num_lines": 8, "thickness": 2, "angle": 0,
                "bg_type": "horizontal_gradient", "noise": 0.01,
            }),
            SyntheticVariation("angled_45", self.category, "primary", {
                "num_lines": 10, "thickness": 2, "angle": 45,
                "bg_type": "uniform", "bg_gray": 140, "noise": 0.01,
            }),
            SyntheticVariation("angled_30", self.category, "secondary", {
                "num_lines": 10, "thickness": 2, "angle": 30,
                "bg_type": "uniform", "bg_gray": 140, "noise": 0.01,
            }),
            SyntheticVariation("angled_60", self.category, "secondary", {
                "num_lines": 10, "thickness": 2, "angle": 60,
                "bg_type": "uniform", "bg_gray": 140, "noise": 0.01,
            }),
            SyntheticVariation("angled_mixed", self.category, "secondary", {
                "angles": [30, -20, 70], "num_lines_per": 4, "thickness": 2,
                "bg_type": "uniform", "bg_gray": 150, "noise": 0.02,
            }),
            SyntheticVariation("horiz_varying_spacing", self.category, "secondary", {
                "spacing_type": "varying", "thickness": 2, "angle": 0,
                "bg_type": "uniform", "bg_gray": 140, "noise": 0.01,
            }),
            SyntheticVariation("horiz_noisy_thick", self.category, "secondary", {
                "num_lines": 6, "thickness": 6, "angle": 0,
                "bg_type": "uniform", "bg_gray": 130,
                "noise_type": "sp", "noise": 0.08,
            }),
            SyntheticVariation("horiz_very_close", self.category, "primary", {
                "num_lines": 25, "thickness": 1, "angle": 0,
                "bg_type": "uniform", "bg_gray": 145,
                "noise_type": "sp", "noise": 0.01,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("horiz_very_thin", self.category, "secondary", {
                "num_lines": 10, "thickness": 1, "angle": 0,
                "bg_type": "uniform", "bg_gray": 150,
                "noise_type": "sp", "noise": 0.0,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("horiz_gaussian_noise", self.category, "secondary", {
                "num_lines": 8, "thickness": 2, "angle": 0,
                "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "gaussian", "noise_sigma": 30,
            }),
        ]

    def generate(self, variation, size, rng):
        h, w = size
        p = variation.params

        # Background
        if p.get("bg_type") == "horizontal_gradient":
            image = make_gradient_bg(h, w, "horizontal", 100, 180)
        elif p.get("bg_type") == "vertical_gradient":
            image = make_gradient_bg(h, w, "vertical", 100, 180)
        else:
            image = np.full((h, w), p.get("bg_gray", 140), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)

        if "angles" in p:
            # Mixed angles mode
            for angle in p["angles"]:
                self._draw_angled_lines(
                    image, mask, h, w, p["num_lines_per"],
                    p["thickness"], angle, rng
                )
        elif p.get("spacing_type") == "varying":
            # Non-uniform spacing
            positions = []
            y = h * 0.1
            while y < h * 0.9:
                positions.append(int(y))
                y += rng.uniform(h * 0.05, h * 0.15)
            for yy in positions:
                cv2.line(image, (0, yy), (w - 1, yy), 0, p["thickness"])
                cv2.line(mask, (0, yy), (w - 1, yy), 255, p["thickness"])
        else:
            self._draw_angled_lines(
                image, mask, h, w, p["num_lines"],
                p["thickness"], p.get("angle", 0), rng
            )

        # Apply noise
        noise_type = p.get("noise_type", "sp")
        if noise_type == "gaussian" and p.get("noise_sigma", 0) > 0:
            image = add_gaussian_noise(image, p["noise_sigma"], rng)
        elif noise_type == "sp" and p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)

        return image, mask

    def _draw_angled_lines(self, image, mask, h, w, num_lines, thickness, angle, rng):
        """Draw parallel lines at a given angle."""
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Spacing perpendicular to the line direction
        diag = np.sqrt(h ** 2 + w ** 2)
        spacing = diag / (num_lines + 1)

        # Line direction vector and perpendicular
        dx, dy = cos_a, sin_a  # along line
        px, py = -sin_a, cos_a  # perpendicular

        cx, cy = w / 2, h / 2
        line_half_len = diag / 2

        for i in range(num_lines):
            offset = spacing * (i + 1) - diag / 2
            # Center of this line
            lx = cx + px * offset
            ly = cy + py * offset
            # Endpoints
            x1 = int(lx - dx * line_half_len)
            y1 = int(ly - dy * line_half_len)
            x2 = int(lx + dx * line_half_len)
            y2 = int(ly + dy * line_half_len)

            cv2.line(image, (x1, y1), (x2, y2), 0, thickness)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
