"""Circle and ellipse pattern generators."""
import numpy as np
import cv2
from .base import (
    BaseGenerator, SyntheticVariation, register_generator,
    add_salt_pepper_noise,
)


@register_generator
class CirclesEllipsesGenerator(BaseGenerator):
    category = "circles_ellipses"

    def get_variations(self):
        return [
            SyntheticVariation("circles_concentric", self.category, "primary", {
                "type": "concentric", "num_rings": 6, "ring_thickness": 3,
                "bg_gray": 160, "noise": 0.01,
            }),
            SyntheticVariation("ellipses_scattered", self.category, "primary", {
                "type": "scattered", "num_shapes": 12, "bg_gray": 150,
                "noise": 0.02,
            }),
            SyntheticVariation("circles_grid", self.category, "secondary", {
                "type": "grid", "grid_size": 4, "fill": True,
                "bg_gray": 160, "noise": 0.01,
            }),
            SyntheticVariation("ellipses_oriented", self.category, "secondary", {
                "type": "oriented_ellipses", "num_shapes": 10,
                "orientation": 30, "bg_gray": 150, "noise": 0.01,
            }),
            SyntheticVariation("circles_noisy", self.category, "secondary", {
                "type": "concentric", "num_rings": 5, "ring_thickness": 4,
                "bg_gray": 140, "noise": 0.1,
            }),
            SyntheticVariation("circles_overlapping", self.category, "secondary", {
                "type": "overlapping", "num_shapes": 8,
                "bg_gray": 160, "noise": 0.02,
            }),
        ]

    def generate(self, variation, size, rng):
        h, w = size
        p = variation.params
        image = np.full((h, w), p.get("bg_gray", 160), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        if p["type"] == "concentric":
            cx, cy = w // 2, h // 2
            max_r = min(h, w) // 2 - 5
            spacing = max_r / (p["num_rings"] + 1)
            for i in range(1, p["num_rings"] + 1):
                r = int(spacing * i)
                cv2.circle(image, (cx, cy), r, 0, p["ring_thickness"])
                cv2.circle(mask, (cx, cy), r, 255, p["ring_thickness"])

        elif p["type"] == "scattered":
            for _ in range(p["num_shapes"]):
                cx = rng.randint(w // 6, 5 * w // 6)
                cy = rng.randint(h // 6, 5 * h // 6)
                a = rng.randint(min(h, w) // 15, min(h, w) // 6)
                b = rng.randint(min(h, w) // 15, min(h, w) // 6)
                angle = rng.randint(0, 180)
                cv2.ellipse(image, (cx, cy), (a, b), angle, 0, 360, 0, 2)
                cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 255, 2)

        elif p["type"] == "grid":
            gs = p["grid_size"]
            cell_w = w // gs
            cell_h = h // gs
            r = min(cell_w, cell_h) // 3
            for row in range(gs):
                for col in range(gs):
                    cx = col * cell_w + cell_w // 2
                    cy = row * cell_h + cell_h // 2
                    if p.get("fill"):
                        cv2.circle(image, (cx, cy), r, 0, -1)
                        cv2.circle(mask, (cx, cy), r, 255, -1)
                    else:
                        cv2.circle(image, (cx, cy), r, 0, 2)
                        cv2.circle(mask, (cx, cy), r, 255, 2)

        elif p["type"] == "oriented_ellipses":
            angle = p.get("orientation", 0)
            for _ in range(p["num_shapes"]):
                cx = rng.randint(w // 6, 5 * w // 6)
                cy = rng.randint(h // 6, 5 * h // 6)
                a = rng.randint(min(h, w) // 8, min(h, w) // 4)
                b = rng.randint(min(h, w) // 20, min(h, w) // 10)
                a_var = angle + rng.randint(-10, 10)
                cv2.ellipse(image, (cx, cy), (a, b), a_var, 0, 360, 0, 2)
                cv2.ellipse(mask, (cx, cy), (a, b), a_var, 0, 360, 255, 2)

        elif p["type"] == "overlapping":
            for _ in range(p["num_shapes"]):
                cx = rng.randint(w // 4, 3 * w // 4)
                cy = rng.randint(h // 4, 3 * h // 4)
                r = rng.randint(min(h, w) // 10, min(h, w) // 4)
                cv2.circle(image, (cx, cy), r, 0, 2)
                cv2.circle(mask, (cx, cy), r, 255, 2)

        if p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)

        return image, mask
