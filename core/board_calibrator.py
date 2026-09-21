"""
棋盘标定与网格计算模块 (Board Calibrator)
计算象棋 9 列 x 10 行 (共 90 个交叉点) 的精确像素坐标，支持任意分辨率缩放。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2

@dataclass
class BoardConfig:
    """
    棋盘相对坐标配置 (归一化比例，适配任意窗口缩放比例)
    以 918x1704 标准参考分辨率为基准
    """
    base_width: int = 918
    base_height: int = 1704
    
    # 90 个交叉点左上角与右下角网格的像素基准
    x_min: float = 60.0
    x_max: float = 858.0
    y_min: float = 449.0
    y_max: float = 1315.0
    
    # 棋子采样半径 (基准分辨率下)
    base_piece_radius: int = 38

    # 可选人工标定的四个最外层交叉点，顺序：左上、右上、右下、左下。
    # 存储为相对图像宽高的比例，适用于同一窗口的缩放。
    normalized_corners: Optional[List[Tuple[float, float]]] = None
    
    @property
    def x_min_ratio(self) -> float:
        return self.x_min / self.base_width
        
    @property
    def x_max_ratio(self) -> float:
        return self.x_max / self.base_width
        
    @property
    def y_min_ratio(self) -> float:
        return self.y_min / self.base_height
        
    @property
    def y_max_ratio(self) -> float:
        return self.y_max / self.base_height


class BoardGrid:
    def __init__(self, config: Optional[BoardConfig] = None):
        self.config = config or BoardConfig()

    def compute_grid(self, image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        根据当前画面的宽高，计算 9 列和 10 行的绝对像素坐标以及棋子切片半径
        返回: (col_coords (9,), row_coords (10,), piece_radius)
        """
        x0 = self.config.x_min_ratio * image_width
        x1 = self.config.x_max_ratio * image_width
        y0 = self.config.y_min_ratio * image_height
        y1 = self.config.y_max_ratio * image_height
        
        cols = np.linspace(x0, x1, 9)
        rows = np.linspace(y0, y1, 10)
        
        scale = image_width / self.config.base_width
        piece_radius = max(15, int(round(self.config.base_piece_radius * scale)))
        
        return cols, rows, piece_radius

    def compute_points(self, image_width: int, image_height: int) -> Tuple[np.ndarray, int]:
        """返回形如 (10, 9, 2) 的交叉点坐标和采样半径。

        未标定时沿用旧版固定比例；标定后使用四边形双线性插值，能容忍窗口
        内容区轻微偏移和透视变形。
        """
        if not self.config.normalized_corners:
            cols, rows, radius = self.compute_grid(image_width, image_height)
            return np.array([[[x, y] for x in cols] for y in rows], dtype=np.float32), radius

        corners = self.config.normalized_corners
        if len(corners) != 4:
            raise ValueError("normalized_corners 必须有左上、右上、右下、左下四点")
        tl, tr, br, bl = np.array(
            [(x * image_width, y * image_height) for x, y in corners], dtype=np.float32
        )
        points = np.zeros((10, 9, 2), dtype=np.float32)
        for row in range(10):
            vertical = row / 9
            left = tl * (1 - vertical) + bl * vertical
            right = tr * (1 - vertical) + br * vertical
            for col in range(9):
                horizontal = col / 8
                points[row, col] = left * (1 - horizontal) + right * horizontal

        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)
        reference_step = (self.config.x_max - self.config.x_min) / 8
        scale = ((top_width + bottom_width) / 2 / 8) / reference_step
        radius = max(15, int(round(self.config.base_piece_radius * scale)))
        return points, radius

    def get_points(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
        """获取 90 个交叉点列表: (col_idx, row_idx, cx, cy)。"""
        grid, _ = self.compute_points(image_width, image_height)
        return [
            (col, row, int(round(grid[row, col, 0])), int(round(grid[row, col, 1])))
            for row in range(10) for col in range(9)
        ]

    def draw_debug_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        在图像上绘制标定网格和 90 个交叉点蓝圈红点，方便肉眼校准
        """
        h, w = image.shape[:2]
        points, radius = self.compute_points(w, h)
        annotated = image.copy()

        for row in range(10):
            cv2.polylines(annotated, [np.round(points[row]).astype(np.int32)], False, (0, 255, 0), 1)
        for col in range(9):
            cv2.polylines(annotated, [np.round(points[:, col]).astype(np.int32)], False, (0, 255, 0), 1)
        for row in range(10):
            for col in range(9):
                cx, cy = np.round(points[row, col]).astype(int)
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
                cv2.circle(annotated, (cx, cy), radius, (255, 0, 0), 1)
        return annotated
