# label_noise.py
import numpy as np
from typing import Dict, Optional, Sequence


class LabelNoiseInjector:
    def __init__(
        self,
        noise_rate: float,
        mode: str = "symmetric",   # "symmetric" | "asymmetric"
        mapping: Optional[Dict[int, int]] = None,
        n_classes: Optional[int] = None,
        seed: int = 0,
    ):
        self.noise_rate = noise_rate
        self.mode = mode
        self.mapping = mapping
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def transform(self, y: Sequence[int]) -> np.ndarray:
        y = np.asarray(y).astype(int).copy()

        if self.mode == "symmetric":
            return self._symmetric_noise(y)
        elif self.mode == "asymmetric":
            return self._asymmetric_noise(y)
        else:
            raise ValueError("mode must be 'symmetric' or 'asymmetric'")

    # -------------------------
    # SYMMETRIC
    # -------------------------
    def _symmetric_noise(self, y: np.ndarray) -> np.ndarray:
        if self.n_classes is None:
            n_classes = int(np.max(y)) + 1
        else:
            n_classes = self.n_classes

        mask = self.rng.random(len(y)) < self.noise_rate
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return y

        new_labels = self.rng.integers(0, n_classes - 1, size=len(idx))
        orig = y[idx]
        new_labels = new_labels + (new_labels >= orig)
        y[idx] = new_labels
        return y

    # -------------------------
    # ASYMMETRIC
    # -------------------------
    def _asymmetric_noise(self, y: np.ndarray) -> np.ndarray:
        if self.mapping is None:
            raise ValueError("Asymmetric mode requires a mapping dict")

        mask = self.rng.random(len(y)) < self.noise_rate
        for i in np.where(mask)[0]:
            yi = int(y[i])
            if yi in self.mapping:
                y[i] = int(self.mapping[yi])
        return y

    # -------------------------
    # UTIL
    # -------------------------
    @staticmethod
    def make_default_mapping(classes: Sequence[int]) -> Dict[int, int]:
        classes = list(map(int, classes))
        return {
            classes[i]: classes[(i + 1) % len(classes)]
            for i in range(len(classes))
        }
