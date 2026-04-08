"""Checkerboard pattern generators."""
import numpy as np
import cv2
from .base import (
    BaseGenerator, SyntheticVariation, register_generator,
    add_salt_pepper_noise, make_gradient_bg,
)


@register_generator
class CheckerboardGenerator(BaseGenerator):
    category = "checkerboard"

    def get_variations(self):
        return [
            SyntheticVariation("checker_uniform", self.category, "primary", {
                "cell_size": 16, "color_dark": 40, "color_light": 200,
                "noise": 0.01,
            }),
            SyntheticVariation("checker_gradient", self.category, "primary", {
                "cell_size": 20, "overlay": "vertical_gradient",
                "noise": 0.01,
            }),
            SyntheticVariation("checker_blurred", self.category, "secondary", {
                "cell_size": 16, "color_dark": 40, "color_light": 200,
                "blur_ksize": 5, "noise": 0.0,
            }),
            SyntheticVariation("checker_noisy", self.category, "secondary", {
                "cell_size": 16, "color_dark": 40, "color_light": 200,
                "noise": 0.08,
            }),
            SyntheticVariation("checker_rotated", self.category, "secondary", {
                "cell_size": 16, "color_dark": 40, "color_light": 200,
                "rotation": 15, "noise": 0.01,
            }),
        ]

    def generate(self, variation, size, rng):
        h, w = size
        p = variation.params
        cs = p["cell_size"]
        dark = p.get("color_dark", 40)
        light = p.get("color_light", 200)

        # Generate base checkerboard
        rows = (h + cs - 1) // cs
        cols = (w + cs - 1) // cs
        board = np.zeros((rows * cs, cols * cs), dtype=np.uint8)
        gt = np.zeros_like(board)

        for r in range(rows):
            for c in range(cols):
                val = light if (r + c) % 2 == 0 else dark
                board[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = val
                if (r + c) % 2 == 1:  # dark cells are "foreground"
                    gt[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = 255

        image = board[:h, :w]
        mask = gt[:h, :w]

        # Gradient overlay
        if p.get("overlay") == "vertical_gradient":
            grad = make_gradient_bg(h, w, "vertical", -40, 40)
            image = np.clip(image.astype(np.int16) + grad.astype(np.int16) - 128, 0, 255).astype(np.uint8)

        # Rotation
        if p.get("rotation", 0) != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, p["rotation"], 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=int((dark + light) / 2))
            mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)
            mask = (mask > 127).astype(np.uint8) * 255

        # Blur
        if p.get("blur_ksize", 0) > 0:
            ks = p["blur_ksize"]
            image = cv2.GaussianBlur(image, (ks, ks), 0)

        if p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)

        return image, mask
