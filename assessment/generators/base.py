"""Base class and registry for synthetic image generators."""
from abc import ABC, abstractmethod
import numpy as np


class SyntheticVariation:
    """Describes one image variation to generate."""

    def __init__(self, name, category, priority, params):
        self.name = name            # e.g. "wavy_thin_gray_bg"
        self.category = category    # e.g. "wavy_lines"
        self.priority = priority    # "primary" or "secondary"
        self.params = params        # dict of generation params

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "params": self.params,
        }


class BaseGenerator(ABC):
    """Abstract base for synthetic image generators."""

    category = ""  # override in subclass

    @abstractmethod
    def get_variations(self):
        """Return list of SyntheticVariation descriptors."""
        pass

    @abstractmethod
    def generate(self, variation, size, rng):
        """Generate (image, ground_truth_mask) for a variation and size.

        Args:
            variation: SyntheticVariation descriptor
            size: (height, width) tuple
            rng: numpy RandomState for reproducibility

        Returns:
            (image, mask): both uint8 HxW numpy arrays.
                image: grayscale 0-255
                mask: binary 0 or 255
        """
        pass


# Registry
_GENERATORS = {}


def register_generator(cls):
    _GENERATORS[cls.category] = cls
    return cls


def get_all_generators():
    return dict(_GENERATORS)


# --- Utility functions for generators ---

def add_salt_pepper_noise(image, density, rng):
    """Add salt-and-pepper noise to a grayscale image."""
    out = image.copy()
    n_pixels = image.size
    n_salt = int(n_pixels * density / 2)
    n_pepper = int(n_pixels * density / 2)

    # Salt
    coords = [rng.randint(0, i, n_salt) for i in image.shape]
    out[coords[0], coords[1]] = 255

    # Pepper
    coords = [rng.randint(0, i, n_pepper) for i in image.shape]
    out[coords[0], coords[1]] = 0

    return out


def add_gaussian_noise(image, sigma, rng):
    """Add Gaussian noise to a grayscale image."""
    noise = rng.normal(0, sigma, image.shape)
    out = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return out


def make_gradient_bg(h, w, direction="vertical", low=80, high=180):
    """Create a gradient background."""
    if direction == "vertical":
        grad = np.linspace(low, high, h, dtype=np.float64)
        bg = np.tile(grad[:, None], (1, w))
    elif direction == "horizontal":
        grad = np.linspace(low, high, w, dtype=np.float64)
        bg = np.tile(grad[None, :], (h, 1))
    elif direction == "radial":
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)
        bg = low + (high - low) * dist / max_dist
    else:
        bg = np.full((h, w), (low + high) / 2, dtype=np.float64)
    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_line_on_images(image, mask, y_coords, x_start, x_end, thickness, intensity):
    """Draw a line defined by y_coords array onto image and mask.

    y_coords: array of y positions for each x in [x_start, x_end)
    """
    h, w = image.shape
    for x in range(max(0, x_start), min(w, x_end)):
        if x - x_start >= len(y_coords):
            break
        cy = int(round(y_coords[x - x_start]))
        for dy in range(-thickness // 2, thickness // 2 + 1):
            y = cy + dy
            if 0 <= y < h:
                image[y, x] = intensity
                mask[y, x] = 255
