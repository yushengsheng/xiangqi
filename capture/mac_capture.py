"""macOS 指定窗口捕获器。

使用 CoreGraphics 直接按 CGWindowID 抓取窗口内容，而不是按桌面坐标调用 mss。
后者在未获录屏权限或 Retina 缩放环境中可能返回桌面壁纸，无法证明拿到的
是目标窗口。本模块从不主动请求录屏权限，只做预检并将状态暴露给上层。
"""

import glob
import os
import re
import struct
import subprocess
import sys
import time
from typing import Optional, Dict, Any, List

import numpy as np

from capture.base import BaseCapture


class MacCapture(BaseCapture):
    def __init__(
        self,
        target_title: Optional[str] = None,
        target_wid: Optional[int] = None,
        fallback_keywords: Optional[List[str]] = None,
    ):
        self.target_title = target_title
        self.target_wid = target_wid
        self.fallback_keywords = fallback_keywords or [
            "JJ象棋", "JJ 象棋", "腾讯应用宝", "应用宝", "天天象棋",
            "微信", "WeChat", "Weixin",
        ]
        self.target_window_info: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.frame_count = 0
        self.last_raw_resolution: Optional[str] = None
        self.last_content_crop: Optional[Dict[str, int]] = None
        self.permission_state = "unknown"
        self.game_id = "unknown"
        self.game_name = "未识别"
        self._wechat_window = bool(target_title and self._is_wechat_name(target_title))
        self.capture_source = "coregraphics"
        self._adb_path: Optional[str] = None
        self._adb_device: Optional[str] = None
        self._adb_display_id: Optional[str] = None
        self._display_checked_at = 0.0
        self._game_checked_at = 0.0
        self._tiantian_source = "auto"
        self._source_bad_frames = 0
        self._last_window_probe = 0.0
        self._last_adb_probe = 0.0
        self._probing_window = False
        self._quartz = None
        self._load_quartz()
        self._discover_yyb_adb()

    def _load_quartz(self) -> None:
        if sys.platform != "darwin":
            self.last_error = "macOS window capture is only available on macOS"
            return
        try:
            import Quartz
            self._quartz = Quartz
            self.permission_state = (
                "granted" if Quartz.CGPreflightScreenCaptureAccess() else "not_granted"
            )
        except Exception as exc:
            self.last_error = f"无法加载 macOS Quartz 捕获组件: {exc}"
            self.permission_state = "unavailable"

    def _discover_yyb_adb(self) -> None:
        """发现应用宝自带 ADB；只使用现有组件，不额外安装或常驻新服务。"""
        patterns = [
            os.path.expanduser(
                "~/Library/Application Support/com.tencent.yybmac/reshub/*/app/online/*/vms/tools/adb"
            ),
            os.path.expanduser(
                "~/Library/Application Support/com.tencent.yybmac/**/tools/adb"
            ),
        ]
        matches: List[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern, recursive="**" in pattern))
        self._adb_path = next((path for path in matches if os.access(path, os.X_OK)), None)
        if not self._adb_path:
            return
        try:
            result = subprocess.run(
                [self._adb_path, "devices"], capture_output=True, text=True, timeout=1.0,
            )
            devices = [
                line.split()[0] for line in result.stdout.splitlines()[1:]
                if line.strip().endswith("device")
            ]
            self._adb_device = next(
                (device for device in devices if device.startswith("emulator-")),
                devices[0] if devices else "emulator-5554",
            )
        except Exception:
            # 应用宝默认控制端口；后续命令会再次验证，避免一次瞬时超时导致漏识别。
            self._adb_device = "emulator-5554"

    def _adb(self, *args: str, timeout: float = 1.5, text: bool = False):
        if not self._adb_path or not self._adb_device:
            return None
        try:
            return subprocess.run(
                [self._adb_path, "-s", self._adb_device, *args],
                capture_output=True, text=text, timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _detect_yyb_game(self, force: bool = False) -> str:
        """从应用宝 Android 前台 Activity 区分 JJ 象棋与天天象棋。"""
        now = time.monotonic()
        # 已经识别游戏后无需每秒启动一个 dumpsys 子进程。
        # 未识别阶段仍保持 1 秒重试，切换已知游戏最多延迟 3 秒。
        refresh_after = 3.0 if self.game_id in ("jj", "tiantian") else 1.0
        if not force and now - self._game_checked_at < refresh_after:
            return self.game_id
        self._game_checked_at = now
        result = self._adb("shell", "dumpsys", "window", timeout=1.0, text=True)
        if result and result.returncode == 0:
            packages = re.findall(
                r"(?:mCurrentFocus|mFocusedApp)=[^\n]*?\s(com\.[\w.]+)/", result.stdout
            )
            known = [package for package in packages if package in {
                "com.tencent.qqgame.xq", "com.tencent.tmgp.cn.jj.chess2"
            }]
            if known:
                previous_game = self.game_id
                package = known[-1]
                if package == "com.tencent.qqgame.xq":
                    self.game_id, self.game_name = "tiantian", "天天象棋"
                    self.permission_state = (
                        "granted" if self._quartz is not None
                        and self._quartz.CGPreflightScreenCaptureAccess()
                        else "not_required"
                    )
                else:
                    self.game_id, self.game_name = "jj", "JJ象棋"
                    if self._quartz is not None:
                        self.permission_state = (
                            "granted" if self._quartz.CGPreflightScreenCaptureAccess()
                            else "not_granted"
                        )
                if self.game_id != previous_game:
                    self._adb_display_id = None
                    self._tiantian_source = (
                        "adb" if self.game_id == "tiantian"
                        and self._adb_path and self._adb_device else "auto"
                    )
                    self._source_bad_frames = 0
                    self._probing_window = False
                return self.game_id
        # 前台暂时处于启动页时保留已知平台，避免短暂切换造成整盘重置。
        return self.game_id

    def _get_adb_display_id(self) -> Optional[str]:
        now = time.monotonic()
        if self._adb_display_id and now - self._display_checked_at < 2.0:
            return self._adb_display_id
        result = self._adb("shell", "dumpsys", "display", timeout=1.0, text=True)
        self._display_checked_at = now
        self._adb_display_id = None
        if not result or result.returncode != 0:
            self.last_error = "无法读取应用宝显示器列表，等待重试"
            return None
        for viewport in re.findall(r"DisplayViewport\{type=EXTERNAL[^}]*\}", result.stdout):
            if "valid=false" in viewport or "isActive=false" in viewport:
                continue
            match = re.search(r"uniqueId='local:(\d+)'", viewport)
            if match:
                self._adb_display_id = match.group(1)
                break
        if self._adb_display_id is None:
            self.last_error = "应用宝当前没有有效的外部游戏显示器"
        return self._adb_display_id

    def _capture_tiantian_adb(self) -> Optional[np.ndarray]:
        """抓取应用宝外部游戏显示；天天象棋因此不依赖 macOS 录屏授权。"""
        display_id = self._get_adb_display_id()
        if not display_id:
            return None
        result = self._adb(
            "exec-out", "screencap", "-d", display_id, timeout=1.5, text=False,
        )
        if not result or result.returncode != 0 or len(result.stdout) < 16:
            self._adb_display_id = None
            self.last_error = "应用宝截帧失败，将重新查找游戏显示器"
            return None
        try:
            width, height, pixel_format, _ = struct.unpack_from("<4I", result.stdout, 0)
            if pixel_format != 1 or width <= 0 or height <= 0:
                raise ValueError("无效的 RGBA 帧头")
            expected = width * height * 4
            if len(result.stdout) < 16 + expected:
                raise ValueError("RGBA 帧数据不完整")
            pixels = np.frombuffer(result.stdout, dtype=np.uint8, offset=16, count=expected)
            rgba = pixels.reshape(height, width, 4)
            frame = rgba[:, :, :3][:, :, ::-1].copy()
            self.last_raw_resolution = f"{width}x{height}"
            self.last_content_crop = None
            self.frame_count += 1
            self.capture_source = "yyb_adb"
            self.last_error = None
            return frame
        except (ValueError, struct.error) as exc:
            self._adb_display_id = None
            self.last_error = f"应用宝截帧解码失败：{exc}"
            return None

    @staticmethod
    def list_windows() -> List[Dict[str, Any]]:
        """列出当前屏幕上可见的主窗口，不触发录屏授权请求。"""
        if sys.platform != "darwin":
            return []
        try:
            import Quartz
            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
            )
            results = []
            for window in windows:
                owner = str(window.get("kCGWindowOwnerName", "") or "")
                name = str(window.get("kCGWindowName", "") or "")
                bounds = window.get("kCGWindowBounds", {})
                width = int(bounds.get("Width", 0))
                height = int(bounds.get("Height", 0))
                if window.get("kCGWindowLayer", 0) != 0 or width <= 120 or height <= 120:
                    continue
                results.append({
                    "id": int(window.get("kCGWindowNumber")),
                    "app": owner,
                    "title": name,
                    "display_name": f"{owner} - {name}" if name else owner,
                    "bounds": (
                        int(bounds.get("X", 0)), int(bounds.get("Y", 0)), width, height
                    ),
                })
            return results
        except Exception:
            return []

    @staticmethod
    def _is_wechat_name(value: str) -> bool:
        lowered = value.lower()
        return any(name in lowered for name in ("微信", "wechat", "weixin"))

    @classmethod
    def _is_wechat_window(cls, window: Dict[str, Any]) -> bool:
        return cls._is_wechat_name(str(window.get("app", ""))) or cls._is_wechat_name(
            str(window.get("title", ""))
        )

    @staticmethod
    def _wechat_window_priority(window: Dict[str, Any]):
        title = str(window.get("title", "")).lower()
        bounds = window.get("bounds") or (0, 0, 0, 0)
        width, height = int(bounds[2]), int(bounds[3])
        return (
            int("象棋" in title or "chess" in title),
            int(width > height),
            width * height,
        )

    def _activate_window(self, window: Dict[str, Any]) -> Dict[str, Any]:
        is_wechat = self._is_wechat_window(window)
        changed = self.target_wid != window["id"] or self._wechat_window != is_wechat
        if changed:
            self._adb_display_id = None
            self._tiantian_source = (
                "window" if is_wechat else (
                    "adb" if self.game_id == "tiantian"
                    and self._adb_path and self._adb_device else "auto"
                )
            )
            self._source_bad_frames = 0
            self._probing_window = False
        if is_wechat:
            # Desktop WeChat is not the Android emulator.  Never use its ADB
            # channel even when another Tiantian game happens to be open.
            self.game_id, self.game_name = "tiantian", "天天象棋（微信）"
        elif self._wechat_window:
            self.game_id, self.game_name = "unknown", "未识别"
        self._wechat_window = is_wechat
        self.target_window_info = window
        self.target_wid = window["id"]
        self.last_error = None
        return window

    def find_target_window(self) -> Optional[Dict[str, Any]]:
        windows = self.list_windows()
        if self.target_wid is not None:
            for window in windows:
                if window["id"] == self.target_wid:
                    return self._activate_window(window)
            # 客户端重启后 Window ID 会变化；旧 ID 失效时继续按标题/应用名找新窗口。
            self.last_error = f"Window ID={self.target_wid} 已失效，正在自动重新查找象棋窗口"

        candidates = [self.target_title] if self.target_title else self.fallback_keywords
        for candidate in candidates:
            if not candidate:
                continue
            needle = candidate.lower()
            matching = []
            for window in windows:
                if self._wechat_window and not self._is_wechat_window(window):
                    continue
                searchable = " ".join(
                    [window["display_name"], window["app"], window["title"]]
                ).lower()
                if needle in searchable:
                    matching.append(window)
            if matching:
                if self._is_wechat_name(candidate):
                    matching.sort(key=self._wechat_window_priority, reverse=True)
                return self._activate_window(matching[0])

        self.target_window_info = None
        self.last_error = "未找到象棋/应用宝/微信窗口；可运行 --list-windows 后用 --window-id 指定"
        return None

    def _cgimage_to_bgr(self, image: Any) -> Optional[np.ndarray]:
        quartz = self._quartz
        if image is None or quartz is None:
            return None
        width = quartz.CGImageGetWidth(image)
        height = quartz.CGImageGetHeight(image)
        bytes_per_row = quartz.CGImageGetBytesPerRow(image)
        provider = quartz.CGImageGetDataProvider(image)
        raw = quartz.CGDataProviderCopyData(provider)
        pixels = np.frombuffer(raw, dtype=np.uint8)
        if width <= 0 or height <= 0 or pixels.size < height * bytes_per_row:
            return None
        # CoreGraphics 的 32-bit little-endian 窗口图像在 Apple Silicon 上为 BGRA。
        bgra = pixels[:height * bytes_per_row].reshape(height, bytes_per_row)
        bgra = bgra[:, :width * 4].reshape(height, width, 4)
        return bgra[:, :, :3].copy()

    def request_screen_permission(self) -> bool:
        """显式请求录屏；天天象棋使用应用宝画面通道，无需系统授权。"""
        self.find_target_window()
        if not self._wechat_window:
            self._detect_yyb_game(force=True)
        if (
            not self._wechat_window and self.game_id == "tiantian"
            and self._adb_path and self._adb_device
        ):
            # ADB 画面已包含完整实盘和落点光圈，不为天天象棋
            # 额外申请可捕获整个桌面的 macOS 录屏权限。
            self.permission_state = "not_required"
            self.last_error = None
            return True
        if self._quartz is None:
            return False
        try:
            granted = bool(self._quartz.CGRequestScreenCaptureAccess())
            self.permission_state = "granted" if granted else "not_granted"
            if not granted:
                self.last_error = "系统尚未授予屏幕录制权限"
            return granted
        except Exception as exc:
            self.last_error = f"请求屏幕录制权限失败: {exc}"
            return False

    @staticmethod
    def _trim_empty_border(frame: np.ndarray) -> np.ndarray:
        """移除 CGWindowListCreateImage 在部分 Retina 窗口外附带的纯黑画布。

        只有四周存在纯黑边、且有效内容仍占原图大部分时才裁切，避免把游戏内
        的黑色 UI 当成边框。坐标仍对应窗口内容，供固定比例或人工标定使用。
        """
        active = np.any(frame > 10, axis=2)
        ys = np.flatnonzero(np.any(active, axis=1))
        xs = np.flatnonzero(np.any(active, axis=0))
        if not len(xs) or not len(ys):
            return frame
        x0, x1 = int(xs[0]), int(xs[-1] + 1)
        y0, y1 = int(ys[0]), int(ys[-1] + 1)
        height, width = frame.shape[:2]
        content_width, content_height = x1 - x0, y1 - y0
        # 未出现边框或裁切会丢失过多内容时保留原始帧。
        if (x0 == 0 and y0 == 0 and x1 == width and y1 == height) or (
            content_width < width * 0.7 or content_height < height * 0.7
        ):
            return frame
        return frame[y0:y1, x0:x1].copy()

    def capture(self) -> Optional[np.ndarray]:
        """优先捕获已授权宿主窗口，天天象棋可回退至应用宝显示通道。"""
        if self.target_window_info is None and (self.target_title or self.target_wid):
            self.find_target_window()
        if not self._wechat_window:
            self._detect_yyb_game()
        if self.game_id == "tiantian" and not self._wechat_window:
            window_allowed = (
                self._quartz is not None and self._quartz.CGPreflightScreenCaptureAccess()
            )
            # 当前应用宝版本的 ADB 外部显示画面比宿主窗口更稳定，
            # 且包含真人对局和最后一步光圈。只要 ADB 持续识别到双将就
            # 保持该来源；ADB 失败或盘面无效时仍会自动回退窗口捕获。
            now = time.monotonic()
            probe_adb = (
                self._tiantian_source == "window"
                and now - self._last_adb_probe >= 5.0
            )
            use_adb = self._tiantian_source == "adb" or not window_allowed or probe_adb
            if use_adb:
                if probe_adb:
                    self._last_adb_probe = now
                frame = self._capture_tiantian_adb()
                if frame is not None:
                    # ADB 画面不依赖宿主窗口几何信息；低频刷新即可
                    # 发现应用宝窗口重开，避免每帧枚举全部 macOS 窗口。
                    now = time.monotonic()
                    if now - self._last_window_probe >= 2.0:
                        self._last_window_probe = now
                        self.find_target_window()
                    return frame
                self.last_error = "天天象棋 ADB 画面暂不可用，正在尝试窗口捕获"

        if self._quartz is None:
            return None
        if not self._quartz.CGPreflightScreenCaptureAccess():
            self.permission_state = "not_granted"
            self.last_error = "未授予 Python/启动器“屏幕录制”权限；已停止实时截取，不会重复弹窗"
            return None
        self.permission_state = "granted"

        window = self.find_target_window()
        if window is None:
            return None

        try:
            image = self._quartz.CGWindowListCreateImage(
                self._quartz.CGRectNull,
                self._quartz.kCGWindowListOptionIncludingWindow,
                window["id"],
                self._quartz.kCGWindowImageDefault,
            )
            frame = self._cgimage_to_bgr(image)
            if frame is None or frame.size == 0:
                if self.game_id == "tiantian" and not self._wechat_window:
                    self._tiantian_source = "adb"
                self.last_error = "CoreGraphics 未返回有效窗口图像"
                return None
            self.last_raw_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
            raw_height, raw_width = frame.shape[:2]
            active = np.any(frame > 10, axis=2)
            active_ys = np.flatnonzero(np.any(active, axis=1))
            active_xs = np.flatnonzero(np.any(active, axis=0))
            frame = self._trim_empty_border(frame)
            if frame.shape[:2] != (raw_height, raw_width):
                self.last_content_crop = {
                    "x": int(active_xs[0]),
                    "y": int(active_ys[0]),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            else:
                self.last_content_crop = None
            # 极低方差的全黑/透明画面通常是被系统或应用阻止的窗口抓取结果。
            if float(frame.std()) < 2.0:
                if self.game_id == "tiantian" and not self._wechat_window:
                    self._tiantian_source = "adb"
                self.last_error = "目标窗口图像近乎全黑，可能被应用或系统阻止捕获"
                return None
            self.frame_count += 1
            self.capture_source = "wechat_coregraphics" if self._wechat_window else "coregraphics"
            self.last_error = None
            return frame
        except Exception as exc:
            if self.game_id == "tiantian" and not self._wechat_window:
                self._tiantian_source = "adb"
            self.last_error = f"CoreGraphics 窗口截取失败: {exc}"
            return None

    def report_recognition(self, piece_count: int, has_both_kings: bool) -> None:
        """自动选择天天象棋的人机 ADB 画面或真人宿主窗口画面。"""
        if self.game_id != "tiantian" or self._wechat_window:
            return
        if has_both_kings:
            self._source_bad_frames = 0
            self._probing_window = False
            self._tiantian_source = "adb" if self.capture_source == "yyb_adb" else "window"
            return
        if self.capture_source == "coregraphics" and self._probing_window:
            self._probing_window = False
            self._source_bad_frames = 0
            self._tiantian_source = "adb"
            return
        self._source_bad_frames += 1
        if self.capture_source == "yyb_adb":
            # 截帧期间的过场动画偶尔会造成单帧缺子，不能因此
            # 立即切换到几何不同的宿主窗口。只有连续失败才回退；
            # 在 window 模式下的低频 ADB 探测失败则继续保持 window。
            if self._tiantian_source == "window":
                self._source_bad_frames = 0
                return
            if (
                self._source_bad_frames >= 4
                and self._quartz is not None
                and self._quartz.CGPreflightScreenCaptureAccess()
            ):
                self._tiantian_source = "window"
                self._source_bad_frames = 0
        elif self.capture_source == "coregraphics" and self._source_bad_frames >= 8:
            self._tiantian_source = "adb"
            self._source_bad_frames = 0

    def is_available(self) -> bool:
        window_available = self.find_target_window() is not None
        if self._wechat_window:
            return self._quartz is not None and window_available
        self._detect_yyb_game(force=True)
        return (self._quartz is not None and window_available) or (
            self.game_id == "tiantian" and self._adb_path is not None and self._adb_device is not None
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "mode": "macos_window",
            "platform": sys.platform,
            "window": self.target_window_info,
            "target_title": self.target_title,
            "target_window_id": self.target_wid,
            "permission": self.permission_state,
            "game_id": self.game_id,
            "game_name": self.game_name,
            "window_kind": "wechat" if self._wechat_window else "native",
            "source": self.capture_source,
            "frame_count": self.frame_count,
            "raw_resolution": self.last_raw_resolution,
            "content_crop": self.last_content_crop,
            "last_error": self.last_error,
        }
