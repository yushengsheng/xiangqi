"""
静态/模拟截屏器 (Mock Capture)
用于离线测试、断点调试与跨平台算法回归测试。
支持单张图片或多张走子截图轮播。
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from capture.base import BaseCapture

class MockCapture(BaseCapture):
    def __init__(self, image_paths: List[str], loop: bool = True):
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        if not self.image_paths:
            raise FileNotFoundError(f"None of the specified mock images exist: {image_paths}")
        self.loop = loop
        self.current_idx = 0
        self.cached_images: List[np.ndarray] = []
        for p in self.image_paths:
            img = cv2.imread(p)
            if img is not None:
                self.cached_images.append(img)
            else:
                raise ValueError(f"Failed to load mock image: {p}")

    def capture(self) -> Optional[np.ndarray]:
        if not self.cached_images:
            return None
        img = self.cached_images[self.current_idx].copy()
        if self.loop or self.current_idx + 1 < len(self.cached_images):
            self.current_idx = (self.current_idx + 1) % len(self.cached_images)
        return img

    def is_available(self) -> bool:
        return len(self.cached_images) > 0

    def get_info(self) -> Dict[str, Any]:
        cur_img = self.cached_images[self.current_idx]
        h, w = cur_img.shape[:2]
        return {
            "mode": "mock",
            "images_count": len(self.cached_images),
            "current_index": self.current_idx,
            "current_file": self.image_paths[self.current_idx],
            "resolution": f"{w}x{h}"
        }
