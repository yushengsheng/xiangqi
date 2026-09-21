"""
截屏抽象基类 (Base Capture)
定义跨平台窗口与画面捕获的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np

class BaseCapture(ABC):
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像，返回 BGR 格式的 numpy 数组 (H, W, 3)；若捕获失败返回 None
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查当前捕获器是否就绪（例如窗口是否存在、权限是否具备）
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        返回当前捕获器状态信息（模式、分辨率、窗口标题、坐标等）
        """
        pass
