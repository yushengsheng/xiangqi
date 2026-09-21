"""
中国象棋盘面秒级同步主程序 (Main Entry)
支持跨平台 (Windows 生产运行 / macOS 本地开发与运行)
功能:
- 列出系统可见窗口: python main.py --list-windows
- 指定目标窗口标题: python main.py --window "天天象棋" 或 --window "腾讯应用宝"
- 指定 Window ID / HWND: python main.py --window-id <当前ID>
- 离线测试模拟模式: python main.py --mode mock
"""

import sys
import time
import signal
import argparse
import json
import os

import cv2
from typing import Optional

from capture.factory import create_capture, list_available_windows
from core.adaptive_recognizer import AdaptiveBoardRecognizer
from core.board_calibrator import BoardConfig
from core.calibration_store import interactive_calibrate, load_normalized_corners, save_normalized_corners
from core.debouncer import BoardDebouncer
from server.sync_server import SyncServer
from server.ports import select_port_pair, write_runtime_info


def print_windows_table():
    """列出当前操作系统上所有可见窗口"""
    windows = list_available_windows()
    print("=" * 75)
    print(f"{'ID / HWND':<12} | {'所属应用 (App)':<25} | {'窗口标题 / 几何信息'}")
    print("-" * 75)
    if not windows:
        print("未检测到有效顶层窗口 (或未赋予屏幕/窗口枚举权限)")
    else:
        for w in windows:
            wid = str(w.get("id", ""))
            app = str(w.get("app", w.get("title", "")))[:24]
            title = str(w.get("title", ""))[:30]
            bounds = str(w.get("bounds", ""))
            print(f"{wid:<12} | {app:<25} | {title} {bounds}")
    print("=" * 75)
    print("提示: 可使用 --window \"关键词\" 或 --window-id <ID> 来指定目标窗口")


def main():
    parser = argparse.ArgumentParser(description="中国象棋盘面秒级同步工具 (支持 JJ象棋 / 天天象棋 模拟器)")
    parser.add_argument("--list-windows", action="store_true", help="列出当前系统所有可见窗口并退出")
    parser.add_argument("--window", type=str, default=None, help="指定目标窗口标题关键字 (如 '天天象棋' 或 '腾讯应用宝')")
    parser.add_argument("--window-id", type=int, default=None, help="指定目标窗口 ID (macOS WindowNumber 或 Windows HWND)")
    parser.add_argument("--mode", type=str, choices=["auto", "window", "mock"], default="auto", 
                        help="运行模式: auto(自动检测), window(仅窗口模式), mock(静态测试图模拟)")
    parser.add_argument("--mock-image", type=str, default="test_images/board_test_01.png", help="Mock 模式下的测试图片路径")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket 广播端口 (默认: 8765)")
    parser.add_argument("--http-port", type=int, default=8766, help="HTTP API 与 Web 看板端口 (默认: 8766；冲突时自动避让)")
    parser.add_argument("--runtime-file", default="logs/runtime.json", help="写入实际 HTTP/WebSocket 入口的运行时文件")
    parser.add_argument("--interval", type=float, default=0.12, help="同步轮询间隔秒数 (默认: 0.12s，约 8 FPS)")
    parser.add_argument("--idle-timeout", type=float, default=12.0, help="看板无 WebSocket 连接后自动退出秒数；0 表示不退出")
    parser.add_argument("--test-run", action="store_true", help="自检运行 1 秒后自动退出 (用于自动化回归测试)")
    parser.add_argument("--capture-once", metavar="PATH", help="抓取一帧并保存到 PATH 后退出，用于验证实时窗口采集")
    parser.add_argument("--calibrate", action="store_true", help="抓取当前窗口并点击四个棋盘角点后保存标定")
    parser.add_argument("--calibration-file", default="config/board_calibration.json", help="棋盘标定文件路径")
    parser.add_argument("--request-screen-permission", action="store_true", help="仅一次：请求 macOS 屏幕录制授权后退出")
    parser.add_argument("--no-ai", action="store_true", help="关闭 AI（默认使用本地 Pikafish，始终执画面下方）")
    parser.add_argument("--ai-settings-file", default="config/ai_settings.json", help="保存网页 AI 强度档位的配置文件")
    parser.add_argument("--ai-engine", choices=["pikafish", "builtin"], default=None, help="覆盖已保存的 AI 引擎")
    parser.add_argument("--ai-time", type=float, default=None, help="覆盖已保存的每步思考时间（秒）")
    parser.add_argument("--ai-depth", type=int, default=None, help="覆盖已保存的最大搜索深度")
    parser.add_argument("--ai-threads", type=int, default=None, help="覆盖已保存的 Pikafish CPU 线程数")
    parser.add_argument("--ai-hash", type=int, default=None, help="覆盖已保存的 Pikafish Hash 内存 MB")

    args = parser.parse_args()

    saved_ai = {}
    if args.ai_settings_file and os.path.exists(args.ai_settings_file):
        try:
            with open(args.ai_settings_file, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                saved_ai = loaded
        except Exception as exc:
            print(f"[!] 忽略无效 AI 设置文件 {args.ai_settings_file}: {exc}")
    def saved_number(key, default):
        try:
            return float(saved_ai.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    ai_engine = args.ai_engine or saved_ai.get("engine_kind", "pikafish")
    ai_time = args.ai_time if args.ai_time is not None else saved_number("time_ms", 1000) / 1000.0
    ai_depth = args.ai_depth if args.ai_depth is not None else saved_number("max_depth", 60)
    ai_threads = args.ai_threads if args.ai_threads is not None else saved_number("engine_threads", 2)
    ai_hash = args.ai_hash if args.ai_hash is not None else saved_number("engine_hash_mb", 64)

    # 1. 仅列出窗口
    if args.list_windows:
        print_windows_table()
        sys.exit(0)

    try:
        ws_port, http_port, used_fallback_ports = select_port_pair("127.0.0.1", args.ws_port, args.http_port)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        sys.exit(2)
    args.ws_port, args.http_port = ws_port, http_port

    print("=" * 60)
    print("      中国象棋实时盘面秒级同步引擎      ")
    print(f"  平台: {sys.platform} | 轮询周期: {args.interval}s")
    if used_fallback_ports:
        print(f"  [端口避让] 默认端口被占用，改用 WS={args.ws_port} / HTTP={args.http_port}")
    print("=" * 60)

    # 2. 创建指定窗口捕获器
    capture = create_capture(
        mode=args.mode,
        window_title=args.window,
        window_id=args.window_id,
        mock_images=[args.mock_image]
    )

    # 仅枚举目标窗口，不会触发 macOS 的录屏授权弹窗。
    capture.is_available()
    cap_info = capture.get_info()
    print(f"[*] 捕获器初始化完成: 模式={cap_info.get('mode')}")
    if "window" in cap_info and cap_info["window"]:
        win = cap_info["window"]
        print(f"[*] 已锁定目标窗口: App={win.get('app')}, ID={win.get('id')}, 区域={win.get('bounds')}")
    elif "hwnd" in cap_info and cap_info["hwnd"]:
        print(f"[*] 已锁定目标窗口: HWND={cap_info.get('hwnd')}, 区域={cap_info.get('rect')}")
    elif "current_file" in cap_info:
        print(f"[*] 使用 Mock 测试源: {cap_info.get('current_file')} (分辨率 {cap_info.get('resolution')})")
    else:
        print(f"[!] 尚未锁定实时窗口: {cap_info.get('last_error', '未知原因')}")

    if args.request_screen_permission:
        request_permission = getattr(capture, "request_screen_permission", None)
        if request_permission is None:
            print("[!] 当前捕获器不支持请求屏幕录制权限")
            sys.exit(2)
        if request_permission():
            permission = capture.get_info().get("permission")
            if permission == "not_required":
                print("[*] 已识别天天象棋，将使用应用宝画面通道，无需屏幕录制授权。")
            else:
                print("[*] 已获得屏幕录制权限。")
            sys.exit(0)
        print("[!] 尚未获得屏幕录制权限；请在系统设置中允许 Python 后重试。")
        sys.exit(2)

    if args.calibrate or args.capture_once:
        frame = capture.capture()
        if frame is None:
            print(f"[!] 抓帧失败: {capture.get_info().get('last_error', '未知原因')}")
            sys.exit(2)
        if args.calibrate:
            try:
                corners = interactive_calibrate(frame)
                save_normalized_corners(args.calibration_file, corners, frame.shape[1], frame.shape[0])
                print(f"[*] 标定已保存: {os.path.abspath(args.calibration_file)}")
            except Exception as exc:
                print(f"[!] 标定未保存: {exc}")
                sys.exit(4)
        if args.capture_once:
            output_path = os.path.abspath(args.capture_once)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not cv2.imwrite(output_path, frame):
                print(f"[!] 无法写入抓帧文件: {output_path}")
                sys.exit(3)
            print(f"[*] 实时窗口帧已保存: {output_path} ({frame.shape[1]}x{frame.shape[0]})")
        sys.exit(0)

    # 3. 初始化识别器与状态机，并在有标定文件时加载它。
    grid_config = BoardConfig()
    if os.path.exists(args.calibration_file):
        try:
            grid_config.normalized_corners = load_normalized_corners(args.calibration_file)
            print(f"[*] 已加载棋盘标定: {args.calibration_file}")
        except Exception as exc:
            print(f"[!] 忽略无效标定文件 {args.calibration_file}: {exc}")
    recognizer = AdaptiveBoardRecognizer(jj_grid_config=grid_config)
    debouncer = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.08)

    # 4. 写入实际入口并初始化服务。启动器/下游应读取此文件而非假设固定端口。
    write_runtime_info(args.runtime_file, "127.0.0.1", args.ws_port, args.http_port, used_fallback_ports)
    server = SyncServer(
        capture=capture,
        recognizer=recognizer,
        debouncer=debouncer,
        host="127.0.0.1",
        ws_port=args.ws_port,
        http_port=args.http_port,
        interval=args.interval,
        idle_timeout=args.idle_timeout,
        ai_enabled=not args.no_ai,
        ai_time_ms=int(max(0.08, min(30.0, float(ai_time))) * 1000),
        ai_max_depth=max(1, min(128, int(ai_depth))),
        ai_engine=ai_engine if ai_engine in ("pikafish", "builtin") else "pikafish",
        ai_threads=max(1, min(32, int(ai_threads))),
        ai_hash_mb=max(16, min(2048, int(ai_hash))),
        ai_settings_path=args.ai_settings_file,
    )

    def handle_signal(sig, frame):
        print("\n[!] 接收到退出信号，正在停止服务...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    server.start()

    print("\n服务就绪！你可以:")
    print(f"  1. 浏览器打开 http://127.0.0.1:{args.http_port}/ 查看实时盘面看板")
    print(f"  2. 下游引擎/程序连接 WebSocket: ws://127.0.0.1:{args.ws_port}")
    print(f"  3. HTTP 获取最新盘面: curl http://127.0.0.1:{args.http_port}/fen")
    print("\n按 Ctrl+C 退出程序...\n")

    if args.test_run:
        time.sleep(1.0)
        server.stop()
        print("[*] 自检运行完成，正常退出。")
        sys.exit(0)

    # 服务因网页关闭而空闲退出时，主进程也随之结束，避免残留后台资源。
    try:
        while server.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        handle_signal(None, None)


if __name__ == "__main__":
    main()
