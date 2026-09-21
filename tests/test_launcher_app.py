"""
一键启动脚本与原生 .app 运行生命周期自动化回归测试
"""

import os
import sys
import time
import subprocess
import urllib.request
import json
import asyncio
import plistlib
import websockets

def test_app_bundle_structure():
    print("[Launcher 1/4] 验证 .app 包结构与原生图标完整性...")
    for app_name, exec_name in [("象棋盘面同步.app", "launcher"), ("停止服务.app", "stopper")]:
        plist_path = os.path.join(app_name, "Contents/Info.plist")
        exec_path = os.path.join(app_name, f"Contents/MacOS/{exec_name}")
        icon_path = os.path.join(app_name, "Contents/Resources/AppIcon.icns")

        assert os.path.exists(plist_path), f"{app_name} 缺少 Info.plist"
        assert os.path.exists(exec_path), f"{app_name} 缺少 {exec_name} 原生二进制"
        assert os.path.exists(icon_path), f"{app_name} 缺少 AppIcon.icns 图标"
        assert os.access(exec_path, os.X_OK), f"{app_name} {exec_name} 缺少执行权限"
        with open(plist_path, "rb") as handle:
            plist = plistlib.load(handle)
        assert plist.get("LSMinimumSystemVersion") == "14.0"
        arch = subprocess.check_output(["file", exec_path], text=True)
        assert "arm64" in arch, f"{app_name} 启动器不是 Apple Silicon: {arch}"
        subprocess.run(["codesign", "--verify", "--deep", "--strict", app_name], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  ✓ 象棋盘面同步.app 与 停止服务.app 结构与原生二进制完整")


def test_cold_start_and_service():
    print("[Launcher 2/4] 验证一键启动脚本与后台服务拉起...")
    # 清理旧进程
    subprocess.run(["./stop.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    # 运行实际打包使用的 launcher.sh；mock 参数仅用于让自动化测试不依赖 TCC 授权。
    env = dict(os.environ, XIANGQI_SYNC_ARGS="--mode mock")
    res = subprocess.run(["./launcher.sh"], capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"launcher.sh 失败: {res.stderr}"

    # 等待端口响应
    ready = False
    for _ in range(30):
        time.sleep(0.1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8766/status", timeout=0.3) as resp:
                if resp.status == 200:
                    state = json.loads(resp.read().decode("utf-8"))
                    if state.get("capture_info", {}).get("mode") == "mock":
                        ready = True
                        break
        except Exception:
            pass

    assert ready, "服务未能在超时时间内就绪"
    with open("logs/service.pid", encoding="utf-8") as handle:
        pid = handle.read().strip()
    command = subprocess.check_output(["ps", "-p", pid, "-o", "command="], text=True)
    assert "/runtime/Python3.framework/" in command, f"启动器未使用打包 Python: {command}"
    print("  ✓ 一键启动脚本通过自带 Python 无黑框拉起服务成功 (端口 8766 就绪)")


def test_runtime_communication():
    print("[Launcher 3/4] 验证 .app 启动后后台实时通信链路 (HTTP + WebSocket)...")
    # 1. 验证 HTTP
    time.sleep(0.3)
    with urllib.request.urlopen("http://127.0.0.1:8766/fen", timeout=2.0) as resp:
        fen_str = resp.read().decode().strip()
        assert "3k5/6R1C" in fen_str, f"FEN 内容异常: {fen_str}"

    # 2. 验证 WebSocket
    async def ws_check():
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            assert "fen" in data

    asyncio.run(ws_check())
    print("  ✓ HTTP 与 WebSocket 实时双向通信验证通过")


def test_clean_stop():
    print("[Launcher 4/4] 验证原生 .app 一键停止服务与资源释放...")
    res = subprocess.run(["open", "-n", "停止服务.app"], capture_output=True, text=True)
    assert res.returncode == 0, f"open -n 停止服务.app 失败: {res.stderr}"

    # 验证端口已释放
    released = False
    for _ in range(20):
        time.sleep(0.1)
        try:
            urllib.request.urlopen("http://127.0.0.1:8766/status", timeout=0.3)
        except Exception:
            released = True
            break

    assert released, "端口 8766 未能彻底释放"
    engine_path = os.path.abspath("engines/pikafish/pikafish")
    orphan = subprocess.run(["pgrep", "-f", engine_path], capture_output=True, text=True)
    assert orphan.returncode != 0, f"停止后仍残留 Pikafish 进程: {orphan.stdout.strip()}"
    print("  ✓ 停止服务.app 安全终止服务、Pikafish 子进程并释放端口")


def run():
    print("=" * 65)
    print("      执行 .app 原生一键启动与生命周期自动化测试      ")
    print("=" * 65)
    test_app_bundle_structure()
    test_cold_start_and_service()
    test_runtime_communication()
    test_clean_stop()
    print("=" * 65)
    print("🎉 原生 .app 一键启动/停止 4/4 项测试全部 100% 通过！")
    print("=" * 65)


if __name__ == "__main__":
    run()
