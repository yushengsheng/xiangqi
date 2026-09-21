"""棋盘人工标定的持久化与交互工具。"""

import json
import os
from typing import List, Tuple

import cv2
import numpy as np


CornerList = List[Tuple[float, float]]


def load_normalized_corners(path: str) -> CornerList:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    corners = data.get("normalized_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError("标定文件必须包含 4 个 normalized_corners 点")
    result = []
    for point in corners:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("每个标定点必须是 [x_ratio, y_ratio]")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("标定点必须位于图像范围内")
        result.append((x, y))
    return result


def save_normalized_corners(path: str, corners: CornerList, width: int, height: int) -> None:
    if len(corners) != 4:
        raise ValueError("必须恰好提供 4 个角点")
    data = {
        "version": 1,
        "description": "棋盘 9x10 交叉点外框；顺序为左上、右上、右下、左下",
        "normalized_corners": [
            [round(x / width, 8), round(y / height, 8)] for x, y in corners
        ],
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def interactive_calibrate(frame: np.ndarray) -> CornerList:
    """让用户按左上、右上、右下、左下依次点击四个棋盘交叉点。"""
    points: CornerList = []
    window_name = "棋盘标定：左上 → 右上 → 右下 → 左下，R 重置，Enter 保存，Esc 取消"

    def redraw() -> np.ndarray:
        canvas = frame.copy()
        labels = ["左上", "右上", "右下", "左下"]
        for index, (x, y) in enumerate(points):
            cv2.circle(canvas, (round(x), round(y)), 10, (0, 0, 255), 2)
            cv2.putText(canvas, labels[index], (round(x) + 12, round(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) == 4:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            cv2.imshow(window_name, redraw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(points) == 4:
                return points
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27:
                raise RuntimeError("用户取消了棋盘标定")
    finally:
        cv2.destroyWindow(window_name)
