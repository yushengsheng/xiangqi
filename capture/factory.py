"""
捕获器工厂 (Capture Factory)
根据平台环境与指定参数（指定窗口标题、Window ID、Mock 截图等）统一创建捕获器。
"""

import sys
from typing import Optional, List, Dict, Any
from capture.base import BaseCapture
from capture.mock_capture import MockCapture
from capture.win_capture import WinCapture
from capture.mac_capture import MacCapture

def list_available_windows() -> List[Dict[str, Any]]:
    """
    跨平台获取当前所有可见窗口列表
    返回统一结构: [{'id': ..., 'title': ..., 'bounds': ...}, ...]
    """
    if sys.platform == "win32":
        raw = WinCapture.list_windows()
        return [{"id": w["id"], "title": w["title"], "app": w["title"], "bounds": w["rect"]} for w in raw]
    elif sys.platform == "darwin":
        return MacCapture.list_windows()
    return []

def create_capture(mode: str = "auto", 
                   window_title: Optional[str] = None,
                   window_id: Optional[int] = None,
                   mock_images: Optional[List[str]] = None,
                   target_keywords: Optional[List[str]] = None) -> BaseCapture:
    """
    创建指定窗口捕获器
    mode:
      - 'auto': 自动查找 JJ 象棋/应用宝窗口；找不到时返回不可用的窗口捕获器
      - 'window': 强制使用指定窗口；找不到时由调用方显示明确状态
      - 'mock': 仅在显式指定时使用静态图片模拟
    """
    default_mock = ["test_images/board_test_01.png"]
    mock_paths = mock_images or default_mock

    if mode == "mock":
        return MockCapture(mock_paths)

    # 生产模式绝不静默回退到 mock：否则会把静态测试盘面伪装成实时数据。
    if sys.platform == "win32":
        return WinCapture(target_title=window_title, target_hwnd=window_id, fallback_keywords=target_keywords)

    if sys.platform == "darwin":
        return MacCapture(target_title=window_title, target_wid=window_id, fallback_keywords=target_keywords)

    raise RuntimeError("当前平台不支持窗口捕获；请显式使用 --mode mock 进行离线测试")
