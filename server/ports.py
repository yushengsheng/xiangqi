"""本地服务端口选择与运行时入口文件。"""

import json
import os
import socket
import tempfile
from typing import Dict, Tuple


def port_is_available(host: str, port: int) -> bool:
    """通过 bind 检查 TCP 端口是否可用，不连接代理或其他监听器。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def select_port_pair(host: str, preferred_ws: int, preferred_http: int) -> Tuple[int, int, bool]:
    """优先使用默认端口；被占用时选择 18000~18999 范围内的相邻空闲端口。"""
    if port_is_available(host, preferred_ws) and port_is_available(host, preferred_http):
        return preferred_ws, preferred_http, False

    # 与 Shadowrocket 等代理常用的低位端口隔离，且方便用户识别。
    for ws_port in range(18000, 19000, 2):
        http_port = ws_port + 1
        if port_is_available(host, ws_port) and port_is_available(host, http_port):
            return ws_port, http_port, True
    raise RuntimeError("未找到可用的本地 HTTP/WebSocket 端口对（18000-18999）")


def write_runtime_info(path: str, host: str, ws_port: int, http_port: int, fallback: bool) -> None:
    """原子写入启动器与下游程序可读取的实际入口。"""
    info: Dict[str, object] = {
        "host": host,
        "ws_port": ws_port,
        "http_port": http_port,
        "ws_url": f"ws://{host}:{ws_port}",
        "http_url": f"http://{host}:{http_port}",
        "fallback_ports": fallback,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".runtime-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
