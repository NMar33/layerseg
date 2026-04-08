"""Composite and creative pattern generators."""
import numpy as np
import cv2
from .base import (
    BaseGenerator, SyntheticVariation, register_generator,
    add_salt_pepper_noise, add_gaussian_noise, make_gradient_bg,
)


@register_generator
class CompositeGenerator(BaseGenerator):
    category = "composite"

    def get_variations(self):
        return [
            SyntheticVariation("bands_layered", self.category, "primary", {
                "type": "bands", "num_bands": 8, "noise": 0.02,
            }),
            SyntheticVariation("texture_sine_waves", self.category, "primary", {
                "type": "sine_texture", "num_waves": 5, "noise": 0.01,
            }),
            SyntheticVariation("stripes_with_defects", self.category, "primary", {
                "type": "stripes_defects", "num_stripes": 10,
                "num_defects": 5, "noise": 0.02,
            }),
            SyntheticVariation("gradient_radial_rings", self.category, "secondary", {
                "type": "radial_rings", "num_rings": 6, "noise": 0.01,
            }),
            SyntheticVariation("mixed_geometry", self.category, "secondary", {
                "type": "mixed", "noise_type": "sp", "noise": 0.02,
            }),
            SyntheticVariation("geological_layers", self.category, "primary", {
                "type": "geological", "num_layers": 6,
                "noise_type": "gaussian", "noise_sigma": 15,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
            SyntheticVariation("geological_noisy", self.category, "secondary", {
                "type": "geological", "num_layers": 6,
                "noise_type": "gaussian", "noise_sigma": 35,
                "sizes_hint": [[256, 256], [512, 512]],
            }),
        ]

    def generate(self, variation, size, rng):
        h, w = size
        p = variation.params

        if p["type"] == "bands":
            return self._gen_bands(h, w, p, rng)
        elif p["type"] == "sine_texture":
            return self._gen_sine_texture(h, w, p, rng)
        elif p["type"] == "stripes_defects":
            return self._gen_stripes_defects(h, w, p, rng)
        elif p["type"] == "radial_rings":
            return self._gen_radial_rings(h, w, p, rng)
        elif p["type"] == "mixed":
            return self._gen_mixed(h, w, p, rng)
        elif p["type"] == "geological":
            return self._gen_geological(h, w, p, rng)

    def _gen_bands(self, h, w, p, rng):
        """Horizontal bands with varying intensity (geological layering)."""
        image = np.full((h, w), 160, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        band_h = h // p["num_bands"]

        for i in range(p["num_bands"]):
            y1 = i * band_h
            y2 = min((i + 1) * band_h, h)
            if i % 2 == 0:
                intensity = rng.randint(100, 180)
                image[y1:y2, :] = intensity
            else:
                intensity = rng.randint(20, 60)
                image[y1:y2, :] = intensity
                mask[y1:y2, :] = 255

        if p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)
        return image, mask

    def _gen_sine_texture(self, h, w, p, rng):
        """Smooth texture from summed sine waves (pseudo-Perlin)."""
        y_grid, x_grid = np.mgrid[:h, :w].astype(np.float64)
        texture = np.zeros((h, w), dtype=np.float64)

        for _ in range(p["num_waves"]):
            freq_x = rng.uniform(0.01, 0.08)
            freq_y = rng.uniform(0.01, 0.08)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(30, 60)
            texture += amp * np.sin(2 * np.pi * (freq_x * x_grid + freq_y * y_grid) + phase)

        # Normalize to 0-255
        texture = texture - texture.min()
        texture = texture / (texture.max() + 1e-8) * 255
        image = texture.astype(np.uint8)

        # Threshold at median for GT mask
        median = np.median(image)
        mask = np.where(image < median, 255, 0).astype(np.uint8)

        if p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)
        return image, mask

    def _gen_stripes_defects(self, h, w, p, rng):
        """Regular vertical stripes with random rectangular defects."""
        stripe_w = w // p["num_stripes"]
        image = np.full((h, w), 160, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(p["num_stripes"]):
            if i % 2 == 1:
                x1 = i * stripe_w
                x2 = min((i + 1) * stripe_w, w)
                image[:, x1:x2] = 30
                mask[:, x1:x2] = 255

        # Add defects (random bright patches on dark stripes)
        for _ in range(p["num_defects"]):
            dx = rng.randint(5, w // 6)
            dy = rng.randint(5, h // 6)
            x0 = rng.randint(0, w - dx)
            y0 = rng.randint(0, h - dy)
            image[y0:y0 + dy, x0:x0 + dx] = rng.randint(140, 200)

        if p.get("noise", 0) > 0:
            image = add_salt_pepper_noise(image, p["noise"], rng)
        return image, mask

    def _gen_radial_rings(self, h, w, p, rng):
        """Radial gradient with concentric intensity rings."""
        bg = make_gradient_bg(h, w, "radial", 80, 200)
        image = bg.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        max_r = min(h, w) // 2 - 5
        spacing = max_r / (p["num_rings"] + 1)

        for i in range(1, p["num_rings"] + 1):
            r = int(spacing * i)
            cv2.circle(image, (cx, cy), r, 0, 3)
            cv2.circle(mask, (cx, cy), r, 255, 3)

        image = self._apply_noise(image, p, rng)
        return image, mask

    def _gen_mixed(self, h, w, p, rng):
        """Mix of horizontal lines + circles + gradient."""
        image = make_gradient_bg(h, w, "vertical", 120, 180)
        mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(5):
            y = int(h * (i + 1) / 6)
            cv2.line(image, (0, y), (w - 1, y), 0, 2)
            cv2.line(mask, (0, y), (w - 1, y), 255, 2)

        for _ in range(3):
            cx = rng.randint(w // 4, 3 * w // 4)
            cy = rng.randint(h // 4, 3 * h // 4)
            r = rng.randint(min(h, w) // 10, min(h, w) // 5)
            cv2.circle(image, (cx, cy), r, 0, 2)
            cv2.circle(mask, (cx, cy), r, 255, 2)

        image = self._apply_noise(image, p, rng)
        return image, mask

    def _gen_geological(self, h, w, p, rng):
        """Simulates geological thin section: wavy bands + grain texture."""
        image = np.full((h, w), 160, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        x_arr = np.arange(w)
        n_layers = p.get("num_layers", 6)
        layer_h = h / (n_layers + 1)

        for i in range(n_layers):
            # Wavy boundary between layers
            base_y = layer_h * (i + 1)
            freq = rng.uniform(0.01, 0.04)
            amp = rng.uniform(3, layer_h * 0.3)
            phase = rng.uniform(0, 2 * np.pi)
            y_boundary = base_y + amp * np.sin(2 * np.pi * freq * x_arr + phase)

            # Draw band (fill between boundaries)
            thickness = max(2, int(rng.uniform(2, layer_h * 0.4)))
            intensity = rng.randint(10, 70)  # dark bands
            for x in range(w):
                cy = int(round(y_boundary[x]))
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    y = cy + dy
                    if 0 <= y < h:
                        image[y, x] = intensity
                        mask[y, x] = 255

        # Add grain texture (small random spots)
        n_grains = int(h * w * 0.005)
        for _ in range(n_grains):
            gx = rng.randint(0, w)
            gy = rng.randint(0, h)
            gr = rng.randint(1, 3)
            gv = rng.randint(100, 200)
            cv2.circle(image, (gx, gy), gr, int(gv), -1)

        image = self._apply_noise(image, p, rng)
        return image, mask

    @staticmethod
    def _apply_noise(image, p, rng):
        noise_type = p.get("noise_type", "sp")
        if noise_type == "gaussian" and p.get("noise_sigma", 0) > 0:
            return add_gaussian_noise(image, p["noise_sigma"], rng)
        elif noise_type == "sp" and p.get("noise", 0) > 0:
            return add_salt_pepper_noise(image, p["noise"], rng)
        return image
