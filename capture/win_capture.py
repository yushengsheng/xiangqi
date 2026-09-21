"""
Windows 指定窗口捕获器 (Windows Window-Specific Capture)
支持指定窗口标题、窗口类名或 HWND 句柄进行精确独立截屏（而非全屏幕截屏）。
优先使用 PrintWindow/BitBlt 截取后台/独立窗口图像，并备选精确客户区坐标截屏。
"""

import sys
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from capture.base import BaseCapture

class WinCapture(BaseCapture):
    def __init__(self, 
                 target_title: Optional[str] = None, 
                 target_hwnd: Optional[int] = None,
                 fallback_keywords: Optional[List[str]] = None):
        self.target_title = target_title
        self.target_hwnd = target_hwnd
        self.fallback_keywords = fallback_keywords or [
            "天天象棋", "应用宝", "腾讯手游助手", "JJ象棋", "JJ 象棋", "QQ游戏", "雷电", "MuMu"
        ]
        self.hwnd = target_hwnd
        self.window_info: Dict[str, Any] = {}
        self.game_id = "unknown"
        self.game_name = "未识别"
        self.sct = None
        
        if sys.platform == "win32":
            self._init_win32()

    def _init_win32(self):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            import mss
            self.sct = mss.mss()
        except Exception as e:
            print(f"[WinCapture] Init warning: {e}")

    @staticmethod
    def list_windows() -> List[Dict[str, Any]]:
        """列出所有当前屏幕可见的应用窗口"""
        if sys.platform != "win32":
            return []

        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32

        windows = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

        def enum_callback(hwnd, extra):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()
            
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 100 and h > 100:
                windows.append({
                    "id": hwnd,
                    "title": title,
                    "rect": (rect.left, rect.top, w, h)
                })
            return True

        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
        return windows

    def _update_game(self, title: str) -> None:
        lowered = title.lower()
        if "天天象棋" in lowered:
            self.game_id, self.game_name = "tiantian", "天天象棋"
        elif "jj象棋" in lowered or "jj 象棋" in lowered:
            self.game_id, self.game_name = "jj", "JJ象棋"

    def find_target_window(self) -> Optional[int]:
        """根据指定的标题、HWND 或关键词查找目标窗口"""
        if sys.platform != "win32":
            return None

        import ctypes
        user32 = ctypes.windll.user32

        # 1. 若指定了 HWND；客户端重启后旧句柄失效，再按标题自动重找。
        if self.target_hwnd:
            if user32.IsWindow(self.target_hwnd) and user32.IsWindowVisible(self.target_hwnd):
                self.hwnd = self.target_hwnd
                length = user32.GetWindowTextLengthW(self.hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(self.hwnd, buff, length + 1)
                self.window_info["title"] = buff.value.strip()
                self._update_game(self.window_info["title"])
                return self.hwnd
            self.target_hwnd = None
            self.hwnd = None

        # 2. 若指定了窗口标题
        windows = self.list_windows()
        if self.target_title:
            for w in windows:
                if self.target_title.lower() in w["title"].lower():
                    self.hwnd = self.target_hwnd = w["id"]
                    self.window_info = w
                    self._update_game(w["title"])
                    return self.hwnd

        # 3. 从预设关键词中匹配
        for kw in self.fallback_keywords:
            for w in windows:
                if kw.lower() in w["title"].lower():
                    self.hwnd = self.target_hwnd = w["id"]
                    self.window_info = w
                    self._update_game(w["title"])
                    return self.hwnd

        return None

    def capture_via_printwindow(self, hwnd: int) -> Optional[np.ndarray]:
        """
        使用 Windows PrintWindow API 直接抓取目标窗口的画面缓存
        优点：即使窗口被其他软件遮挡或处于后台，也能正确抓取该窗口内部画面
        """
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 50 or h <= 50:
                return None

            hwnd_dc = user32.GetDC(hwnd)
            mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
            bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
            gdi32.SelectObject(mem_dc, bitmap)

            # PW_RENDERFULLCONTENT = 2
            result = user32.PrintWindow(hwnd, mem_dc, 2)
            if not result:
                # 尝试 PW_DEFAULT = 0
                result = user32.PrintWindow(hwnd, mem_dc, 0)

            if result:
                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ('biSize', wintypes.DWORD),
                        ('biWidth', wintypes.LONG),
                        ('biHeight', wintypes.LONG),
                        ('biPlanes', wintypes.WORD),
                        ('biBitCount', wintypes.WORD),
                        ('biCompression', wintypes.DWORD),
                        ('biSizeImage', wintypes.DWORD),
                        ('biXPelsPerMeter', wintypes.LONG),
                        ('biYPelsPerMeter', wintypes.LONG),
                        ('biClrUsed', wintypes.DWORD),
                        ('biClrImportant', wintypes.DWORD),
                    ]

                class BITMAPINFO(ctypes.Structure):
                    _fields_ = [
                        ('bmiHeader', BITMAPINFOHEADER),
                        ('bmiColors', wintypes.DWORD * 3),
                    ]

                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = w
                bmi.bmiHeader.biHeight = -h  # 负值表示从上到下扫描
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 32
                bmi.bmiHeader.biCompression = 0  # BI_RGB

                buffer = ctypes.create_string_buffer(w * h * 4)
                gdi32.GetDIBits(mem_dc, bitmap, 0, h, buffer, ctypes.byref(bmi), 0)

                img = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, 4))
                bgr = img[:, :, :3].copy()
                
                # 清理 GDI 资源
                gdi32.DeleteObject(bitmap)
                gdi32.DeleteDC(mem_dc)
                user32.ReleaseDC(hwnd, hwnd_dc)
                return bgr

            # 清理 GDI 资源
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)
        except Exception as e:
            print(f"[WinCapture] PrintWindow failed: {e}")
        return None

    def capture_via_rect(self, hwnd: int) -> Optional[np.ndarray]:
        """
        根据指定窗口客户区坐标截取该窗口范围（精确只截该窗口矩形）
        """
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            import mss

            client_rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(client_rect))
            point = wintypes.POINT(client_rect.left, client_rect.top)
            user32.ClientToScreen(hwnd, ctypes.byref(point))

            w = client_rect.right - client_rect.left
            h = client_rect.bottom - client_rect.top
            x = point.x
            y = point.y

            if w <= 50 or h <= 50:
                return None

            if self.sct is None:
                self.sct = mss.mss()

            monitor = {"left": x, "top": y, "width": w, "height": h}
            sct_img = self.sct.grab(monitor)
            return np.array(sct_img)[:, :, :3]
        except Exception as e:
            print(f"[WinCapture] Rect grab failed: {e}")
            return None

    def capture(self) -> Optional[np.ndarray]:
        """执行指定窗口的截屏"""
        if sys.platform != "win32":
            return None

        self.find_target_window()

        if self.hwnd is None:
            return None

        # 优先使用 PrintWindow 独立截取指定窗口
        img = self.capture_via_printwindow(self.hwnd)
        if img is not None:
            return img

        # 回退到精确窗口区域截图
        return self.capture_via_rect(self.hwnd)

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False
        return self.find_target_window() is not None

    def get_info(self) -> Dict[str, Any]:
        return {
            "mode": "windows_specific_window",
            "hwnd": self.hwnd,
            "title": self.window_info.get("title", ""),
            "rect": self.window_info.get("rect"),
            "game_id": self.game_id,
            "game_name": self.game_name,
            "source": "printwindow",
        }
