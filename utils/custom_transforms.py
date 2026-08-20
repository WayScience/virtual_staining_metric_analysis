"""
Custom Albumentations transforms for image degradation.
"""

from typing import Any

import albumentations as A
import cv2
import numpy as np


class Dilate(A.ImageOnlyTransform):
    """Dilate bright regions with a square OpenCV kernel."""

    def __init__(self, kernel_size: int = 3, iterations: int = 1, p: float = 1.0):
        super().__init__(p=p)
        self.kernel_size = kernel_size | 1
        self.iterations = iterations

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        return cv2.dilate(image, kernel, iterations=self.iterations)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("kernel_size", "iterations")


class Erode(A.ImageOnlyTransform):
    """Erode bright regions with a square OpenCV kernel."""

    def __init__(self, kernel_size: int = 3, iterations: int = 1, p: float = 1.0):
        super().__init__(p=p)
        self.kernel_size = kernel_size | 1
        self.iterations = iterations

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        return cv2.erode(image, kernel, iterations=self.iterations)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("kernel_size", "iterations")
