"""Wavy line pattern generators."""
import numpy as np
from .base import (
    BaseGenerator, SyntheticVariation, register_generator,
    add_salt_pepper_noise, add_gaussian_noise, make_gradient_bg, draw_line_on_images,
)


@register_generator
class WavyLinesGenerator(BaseGenerator):
    category = "wavy_lines"

    def get_variations(self):
        return [
            # --- Primary ---
            SyntheticVariation("wavy_thin_gray_bg", self.category, "primary", {
                "num_lines": 8, "thickness": 2, "amplitude": 5,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
            }),
            SyntheticVariation("wavy_thick_gradient_bg", self.category, "primary", {
                "num_lines": 6, "thickness": 5, "amplitude": 8,
                "frequency": 0.03, "bg_type": "vertical_gradient",
                "noise_type": "sp", "noise": 0.02, "drift": 0,
            }),
            SyntheticVariation("wavy_drift_noisy", self.category, "primary", {
                "num_lines": 7, "thickness": 3, "amplitude": 6,
                "frequency": 0.05, "bg_type": "uniform", "bg_gray": 150,
                "noise_type": "sp", "noise": 0.05, "drift": 0.3,
            }),
            SyntheticVariation("wavy_very_close", self.category, "primary", {
                "num_lines": 20, "thickness": 2, "amplitude": 3,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 150,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("wavy_gaussian_noise", self.category, "primary", {
                "num_lines": 7, "thickness": 3, "amplitude": 5,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "gaussian", "noise_sigma": 25, "drift": 0,
            }),
            # --- Secondary ---
            SyntheticVariation("wavy_dense_thin", self.category, "secondary", {
                "num_lines": 15, "thickness": 1, "amplitude": 3,
                "frequency": 0.06, "bg_type": "uniform", "bg_gray": 130,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("wavy_sparse_thick", self.category, "secondary", {
                "num_lines": 3, "thickness": 8, "amplitude": 10,
                "frequency": 0.02, "bg_type": "uniform", "bg_gray": 160,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
            }),
            SyntheticVariation("wavy_varying_thickness", self.category, "secondary", {
                "num_lines": 6, "thickness": "varying", "amplitude": 7,
                "frequency": 0.04, "bg_type": "horizontal_gradient",
                "noise_type": "sp", "noise": 0.02, "drift": 0.1,
            }),
            SyntheticVariation("wavy_crossed", self.category, "secondary", {
                "num_lines": 5, "thickness": 2, "amplitude": 5,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "sp", "noise": 0.02, "drift": 0, "crossed": True,
            }),
            SyntheticVariation("wavy_low_freq", self.category, "secondary", {
                "num_lines": 6, "thickness": 3, "amplitude": 15,
                "frequency": 0.008, "bg_type": "uniform", "bg_gray": 145,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
            }),
            SyntheticVariation("wavy_low_amplitude", self.category, "secondary", {
                "num_lines": 8, "thickness": 2, "amplitude": 1,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
            }),
            SyntheticVariation("wavy_high_freq", self.category, "secondary", {
                "num_lines": 6, "thickness": 2, "amplitude": 4,
                "frequency": 0.12, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "sp", "noise": 0.01, "drift": 0,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("wavy_ultra_thin", self.category, "secondary", {
                "num_lines": 10, "thickness": 1, "amplitude": 4,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 150,
                "noise_type": "sp", "noise": 0.0, "drift": 0,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("wavy_heavy_gaussian_noise", self.category, "secondary", {
                "num_lines": 7, "thickness": 3, "amplitude": 5,
                "frequency": 0.04, "bg_type": "uniform", "bg_gray": 140,
                "noise_type": "gaussian", "noise_sigma": 40, "drift": 0,
            }),
        ]

    def generate(self, variation, size, rng):
        h, w = size
        p = variation.params

        # Background
        if p.get("bg_type") == "vertical_gradient":
            image = make_gradient_bg(h, w, "vertical", 100, 180)
        elif p.get("bg_type") == "horizontal_gradient":
            image = make_gradient_bg(h, w, "horizontal", 100, 180)
        else:
            image = np.full((h, w), p.get("bg_gray", 140), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        x_arr = np.arange(w)

        spacing = h / (p["num_lines"] + 1)
        phase_offset = rng.uniform(0, 2 * np.pi)

        for i in range(p["num_lines"]):
            base_y = spacing * (i + 1)
            drift_offset = p.get("drift", 0) * i * spacing
            y_coords = base_y + drift_offset + p["amplitude"] * np.sin(
                2 * np.pi * p["frequency"] * x_arr + phase_offset + i * 0.5
            )

            if p.get("thickness") == "varying":
                for x in range(w):
                    t = max(1, int(2 + 4 * abs(np.sin(0.02 * x + i))))
                    cy = int(round(y_coords[x]))
                    for dy in range(-t // 2, t // 2 + 1):
                        y = cy + dy
                        if 0 <= y < h:
                            image[y, x] = 0
                            mask[y, x] = 255
            else:
                draw_line_on_images(image, mask, y_coords, 0, w, p["thickness"], 0)

        # Crossed: add a second set at slight angle
        if p.get("crossed"):
            for i in range(p["num_lines"]):
                base_y = spacing * (i + 1)
                slope = 0.15
                y_coords = base_y + slope * (x_arr - w / 2) + p["amplitude"] * np.sin(
                    2 * np.pi * p["frequency"] * 0.7 * x_arr + phase_offset + i * 0.8 + 1.0
                )
                draw_line_on_images(image, mask, y_coords, 0, w, p["thickness"], 0)

        # Apply noise
        noise_type = p.get("noise_type", "sp")
        if noise_type == "gaussian" and p.get("noise_sigma", 0) > 0:
            image = add_gaussian_noise(image, p["noise_sigma"], rng)
        elif noise_type == "sp" and p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)

        return image, mask
