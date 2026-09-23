"""
秒级盘面同步服务 (Sync Server)
包含:
1. WebSocket 广播服务 (默认端口 8765): 秒级推送最新 FEN 与走子事件
2. HTTP REST 接口 (默认端口 8766): 提供 /fen, /status, /move 及内置可视化 Web 看板
"""

import json
import logging
import os
import tempfile
import time
import asyncio
import threading
import http.server
from typing import Optional, Dict, Any, Set
import numpy as np

import websockets
from capture.base import BaseCapture
from core.recognizer import BoardRecognizer
from core.debouncer import BoardDebouncer
from core.fen import board_to_text
from core.ai_advisor import AIAdvisor


WEB_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>中国象棋实时盘面同步看板</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #1e1e2e; color: #cdd6f4; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #45475a; padding-bottom: 15px; margin-bottom: 20px; }
        .status-badge { padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: bold; background: #a6e3a1; color: #11111b; }
        .disconnected { background: #f38ba8; color: #11111b; }
        .card { background: #313244; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .fen-box { background: #181825; padding: 12px 16px; border-radius: 6px; font-family: monospace; font-size: 15px; word-break: break-all; color: #89b4fa; display: flex; justify-content: space-between; align-items: center; }
        .btn-copy { background: #89b4fa; color: #11111b; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn-copy:hover { background: #b4befe; }
        pre.board-text { background: #181825; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 16px; line-height: 1.4; color: #fab387; overflow-x: auto; }
        .history-list { max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 13px; }
        .history-item { padding: 6px 10px; border-bottom: 1px solid #45475a; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>中国象棋实时盘面同步看板</h2>
            <div id="status" class="status-badge disconnected">正在连接...</div>
        </div>

        <div class="card">
            <h3>最新 FEN 字符串</h3>
            <div class="fen-box">
                <span id="fen-text">等待数据...</span>
                <button class="btn-copy" onclick="copyFen()">复制</button>
            </div>
            <p id="move-desc" style="color: #a6adc8; margin-top: 10px; font-size: 14px;">走法描述: 暂无变动</p>
        </div>

        <div class="card">
            <h3>字符盘面预览</h3>
            <pre class="board-text" id="board-text">等待加载...</pre>
        </div>

        <div class="card">
            <h3>走子历史记录</h3>
            <div class="history-list" id="history-list"></div>
        </div>
    </div>

    <script>
        const wsPort = window.location.port ? 8765 : 8765;
        const wsUrl = `ws://${window.location.hostname}:${wsPort}`;
        let ws;

        function connect() {
            ws = new WebSocket(wsUrl);
            const statusEl = document.getElementById('status');

            ws.onopen = () => {
                statusEl.textContent = '已连接实时同步';
                statusEl.className = 'status-badge';
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.fen) {
                        document.getElementById('fen-text').textContent = data.fen;
                    }
                    if (data.text_board) {
                        document.getElementById('board-text').textContent = data.text_board;
                    }
                    if (data.move && data.move.description) {
                        document.getElementById('move-desc').textContent = '走法描述: ' + data.move.description;
                        addHistory(data.move.description, data.timestamp);
                    } else if (data.description) {
                        document.getElementById('move-desc').textContent = '状态: ' + data.description;
                    }
                } catch (e) {
                    console.error('JSON parse error:', e);
                }
            };

            ws.onclose = () => {
                statusEl.textContent = '连接断开 (正在重连...)';
                statusEl.className = 'status-badge disconnected';
                setTimeout(connect, 1000);
            };
        }

        function addHistory(desc, ts) {
            const list = document.getElementById('history-list');
            const item = document.createElement('div');
            item.className = 'history-item';
            const timeStr = new Date(ts * 1000).toLocaleTimeString();
            item.innerHTML = `<span>${desc}</span><span style="color:#6c7086">${timeStr}</span>`;
            list.prepend(item);
        }

        function copyFen() {
            const fen = document.getElementById('fen-text').textContent;
            navigator.clipboard.writeText(fen).then(() => {
                alert('FEN 已复制到剪贴板！');
            });
        }

        connect();
    </script>
</body>
</html>
"""


# 覆盖旧版信息面板，只保留浅色背景的 JJ 风格棋盘。
from server.dashboard import WEB_DASHBOARD_HTML


class SyncHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, sync_server):
        super().__init__(server_address, RequestHandlerClass)
        self.sync_server = sync_server


class SyncHTTPHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_bytes(self, payload: bytes, content_type: str,
                    status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self):
        sync = self.server.sync_server
        state = sync.get_latest_state()

        if self.path == "/" or self.path == "/index.html":
            dashboard = WEB_DASHBOARD_HTML.replace("__WS_PORT__", str(sync.ws_port))
            self._send_bytes(dashboard.encode("utf-8"), "text/html; charset=utf-8")
            return

        if self.path == "/fen":
            self._send_bytes(
                state.get("fen", "").encode("utf-8"), "text/plain; charset=utf-8",
            )
            return

        if self.path == "/status" or self.path == "/api/status":
            self._send_json(state)
            return

        if self.path == "/move" or self.path == "/api/move":
            self._send_json(state.get("last_move") or {})
            return

        if self.path == "/api/ai":
            self._send_json(state.get("ai") or {})
            return

        self._send_bytes(b"", "text/plain; charset=utf-8", 404)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        sync = self.server.sync_server
        return origin in {
            f"http://{sync.host}:{sync.http_port}",
            f"http://localhost:{sync.http_port}",
            f"http://127.0.0.1:{sync.http_port}",
        }

    def do_OPTIONS(self):
        if not self._origin_allowed():
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if not self._origin_allowed():
            self._send_json({"error": "禁止来自外部网页的本地控制请求"}, 403)
            return
        sync = self.server.sync_server
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        advisor = sync.advisor
        if path == "/api/ai/config":
            requested_engine = body.get("engine_kind")
            if requested_engine not in (None, "pikafish"):
                self._send_json({"error": "当前版本仅支持 Pikafish，不会启用内置 AI"}, 400)
                return
            try:
                advisor.configure(
                    enabled=body.get("enabled"),
                    mode=body.get("mode"),
                    time_ms=body.get("time_ms"),
                    max_depth=body.get("max_depth"),
                    engine_kind=body.get("engine_kind"),
                    engine_threads=body.get("engine_threads"),
                    engine_hash_mb=body.get("engine_hash_mb"),
                )
                sync.persist_ai_settings()
            except (TypeError, ValueError) as exc:
                self._send_json({"error": f"AI 参数无效: {exc}"}, 400)
                return
        elif path == "/api/ai/start":
            advisor.start_local_game()
        elif path == "/api/ai/follow":
            advisor.follow_live()
        elif path == "/api/live/reset":
            self._send_json(sync.request_live_reset())
            return
        elif path == "/api/ai/turn":
            side = body.get("side")
            if side in ("r", "b"):
                advisor.set_turn(side)
        elif path == "/api/ai/undo":
            advisor.undo()
        elif path == "/api/ai/play":
            src = body.get("from") or {}
            dst = body.get("to") or {}
            try:
                advisor.play_move(int(src["row"]), int(src["col"]), int(dst["row"]), int(dst["col"]))
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
        else:
            self._send_bytes(b"", "text/plain; charset=utf-8", 404)
            return

        self._send_json(sync.publish_ai())

    def _send_json(self, payload, status=200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        origin = self.headers.get("Origin")
        if origin and self._origin_allowed():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        # 静默常规请求日志，避免刷屏
        pass


class SyncServer:
    def __init__(self,
                 capture: BaseCapture,
                 recognizer: Optional[BoardRecognizer] = None,
                 debouncer: Optional[BoardDebouncer] = None,
                 host: str = "127.0.0.1",
                 ws_port: int = 8765,
                 http_port: int = 8766,
                 interval: float = 0.12,
                 idle_timeout: float = 12.0,
                 ai_enabled: bool = True,
                 ai_time_ms: int = 700,
                 ai_max_depth: int = 60,
                 ai_engine: str = "pikafish",
                 ai_threads: int = 1,
                 ai_hash_mb: int = 32,
                 ai_allow_builtin_fallback: bool = False,
                 ai_settings_path: Optional[str] = None):
        self.capture = capture
        self.recognizer = recognizer or BoardRecognizer()
        self.debouncer = debouncer or BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.08)
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.interval = interval
        self.idle_timeout = max(0.0, idle_timeout)
        self.ai_settings_path = ai_settings_path
        self._settings_lock = threading.Lock()

        self.running = False
        self._stop_event = threading.Event()
        self._ws_clients: Set[Any] = set()
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_server = None
        self._http_server: Optional[SyncHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._idle_thread: Optional[threading.Thread] = None
        self._last_client_seen = time.monotonic()
        self._active_game_id = "unknown"
        self._reset_requested = threading.Event()
        self._manual_reset_active = False
        self._manual_reset_deadline = 0.0
        self._manual_reset_timeout = 2.5
        self._pending_since = 0.0
        self._last_tracking_refresh = 0.0
        self._consecutive_pipeline_errors = 0

        # 线程安全共享状态
        self._lock = threading.Lock()
        self._latest_state: Dict[str, Any] = {
            "fen": "",
            "board": [],
            "text_board": "",
            "piece_count": 0,
            "last_move": None,
            "timestamp": 0.0,
            "description": "尚未捕获画面",
            "fps": 0.0,
            "capture_status": "waiting",
            "session_revision": 0,
            "last_frame_at": 0.0,
            "capture_info": self.capture.get_info(),
            "manual_reset_pending": False,
            "ai": None,
        }
        self.advisor = AIAdvisor(
            time_ms=ai_time_ms,
            max_depth=ai_max_depth,
            enabled=ai_enabled,
            engine_kind=ai_engine,
            engine_threads=ai_threads,
            engine_hash_mb=ai_hash_mb,
            allow_builtin_fallback=ai_allow_builtin_fallback,
            on_update=self._on_ai_update,
        )
        self._latest_state["ai"] = self.advisor.snapshot()

    def _effective_poll_interval(
        self,
        capture_info: Optional[Dict[str, Any]] = None,
        *,
        has_stable_board: bool = False,
        recognition_pending: bool = False,
    ) -> float:
        """根据画面来源自适应降频。

        棋盘稳定时无需持续满速截图；一旦发现候选变化，下一帧
        立即恢复快速轮询以完成双帧确认。
        """
        source = (capture_info or {}).get("source")
        if recognition_pending:
            return self.interval
        if not has_stable_board and source in ("yyb_adb", "coregraphics"):
            return max(self.interval, 0.5)
        if source == "yyb_adb":
            return max(self.interval, 0.75)
        if source == "coregraphics" and has_stable_board:
            return max(self.interval, 0.30)
        if source == "windows_graphics_capture" and has_stable_board:
            return max(self.interval, 0.25)
        return self.interval

    @staticmethod
    def _looks_like_single_move(stable_board, observed_board) -> bool:
        """仅对“一个来源格 + 一个目标格”的候选快速补帧。"""
        if stable_board is None or observed_board is None:
            return False
        differences = [
            (row, col, stable_board[row][col], observed_board[row][col])
            for row in range(10) for col in range(9)
            if stable_board[row][col] != observed_board[row][col]
        ]
        if len(differences) != 2:
            return False
        departures = [item for item in differences if item[2] is not None and item[3] is None]
        destinations = [item for item in differences if item[3] is not None and item[3] != item[2]]
        return (
            len(departures) == 1
            and len(destinations) == 1
            and departures[0][2] == destinations[0][3]
        )

    def get_latest_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_state)

    def persist_ai_settings(self) -> None:
        """原子保存网页选择的 AI 档位，供下次启动恢复。"""
        if not self.ai_settings_path:
            return
        snap = self.advisor.snapshot()
        payload = {
            "engine_kind": snap["engine_kind"],
            "time_ms": snap["time_ms"],
            "max_depth": snap["max_depth"],
            "engine_threads": snap["engine_threads"],
            "engine_hash_mb": snap["engine_hash_mb"],
        }
        path = os.path.abspath(self.ai_settings_path)
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with self._settings_lock:
            fd, temporary = tempfile.mkstemp(prefix=".ai-settings-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                os.replace(temporary, path)
            except Exception as exc:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                print(f"[SyncServer] 无法保存 AI 设置 {path}: {exc}")

    def _update_state(self, updates: Dict[str, Any]):
        with self._lock:
            self._latest_state.update(updates)

    def _ai_overlay(self, state_update: Dict[str, Any]) -> None:
        snap = self.advisor.snapshot()
        public_ai = {key: value for key, value in snap.items() if key != "text_board"}
        state_update["ai"] = public_ai
        state_update["side_to_move"] = self.debouncer.active_side
        # 主状态永远保留真实识别盘面。网页模拟盘只存在于 ai.board，避免
        # /status、WebSocket 与下游把 AI 预演的一步误当成实盘已经走出。

    def publish_ai(self) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        self._ai_overlay(updates)
        self._update_state(updates)
        payload = self.get_latest_state()
        payload["event_type"] = "ai_update"
        self._broadcast_event(payload)
        return payload

    def request_live_reset(self) -> Dict[str, Any]:
        """由捕获线程安全地清空识别状态，并立即向界面反馈。"""
        self._manual_reset_active = True
        self._manual_reset_deadline = time.monotonic() + self._manual_reset_timeout
        self._reset_requested.set()
        self.advisor.follow_live()
        now = time.time()
        self._update_state({
            "raw_piece_count": 0,
            "last_move": None,
            "timestamp": now,
            "recognition_pending": True,
            "recognition_rejection": "正在后台重新读取当前对局",
            "capture_status": "resetting",
            "manual_reset_pending": True,
            "description": "正在后台重新读取当前对局，旧盘面会保留到新盘面就绪",
        })
        payload = self.get_latest_state()
        payload["event_type"] = "reset_requested"
        self._broadcast_event(payload)
        return payload

    def _expire_manual_reset(self) -> None:
        """有界结束重读；无论捕获是否有帧都不能无限等待。"""
        if not self._manual_reset_active:
            return
        self._reset_requested.clear()
        cancel_reanchor = getattr(self.debouncer, "cancel_reanchor", None)
        if cancel_reanchor:
            cancel_reanchor()
        self._manual_reset_active = False
        self._manual_reset_deadline = 0.0
        current = self.get_latest_state()
        has_board = bool(current.get("board"))
        self._update_state({
            "manual_reset_pending": False,
            "capture_status": "ok" if has_board else "waiting",
            "description": (
                "未获得新的完整帧，已继续使用当前稳定盘面"
                if has_board else "等待棋盘画面恢复"
            ),
            "timestamp": time.time(),
        })

    def _on_ai_update(self) -> None:
        if not self.running:
            return
        self.publish_ai()

    def _broadcast_event(self, event_data: Dict[str, Any]):
        """向所有已连接的 WebSocket 客户端线程安全广播事件"""
        if not self._ws_clients or not self._ws_loop or not self._ws_loop.is_running():
            return

        msg = json.dumps(event_data, ensure_ascii=False)

        async def send_all():
            dead_clients = set()
            for ws in list(self._ws_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead_clients.add(ws)
            self._ws_clients.difference_update(dead_clients)

        asyncio.run_coroutine_threadsafe(send_all(), self._ws_loop)

    async def _ws_handler(self, websocket):
        self._ws_clients.add(websocket)
        self._last_client_seen = time.monotonic()
        try:
            # 刚连上时立即向客户端发送当前最新状态
            state = self.get_latest_state()
            await websocket.send(json.dumps(state, ensure_ascii=False))
            await websocket.wait_closed()
        finally:
            self._ws_clients.discard(websocket)
            if not self._ws_clients:
                self._last_client_seen = time.monotonic()

    def _run_idle_watchdog(self):
        """没有看板 WebSocket 时，延时停止整个本地服务。"""
        if self.idle_timeout <= 0:
            return
        while not self._stop_event.wait(0.25):
            if not self._ws_clients and time.monotonic() - self._last_client_seen >= self.idle_timeout:
                print(f"[SyncServer] 看板已断开超过 {self.idle_timeout:g}s，自动停止服务。")
                self.stop()
                return

    def _run_ws_server(self):
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)

        async def main():
            # websockets 默认会把被来源策略拒绝的恶意握手打印完整堆栈；
            # 这是预期的 403，不应污染用户发布版日志。
            ws_logger = logging.getLogger(f"xiangqi.websocket.{id(self)}")
            ws_logger.addHandler(logging.NullHandler())
            ws_logger.propagate = False
            ws_logger.setLevel(logging.CRITICAL)
            allowed_origins = [
                None,
                f"http://{self.host}:{self.http_port}",
                f"http://127.0.0.1:{self.http_port}",
                f"http://localhost:{self.http_port}",
            ]
            async with websockets.serve(
                self._ws_handler, self.host, self.ws_port,
                origins=allowed_origins, logger=ws_logger,
            ) as server:
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.05)
                server.close()
                await server.wait_closed()

        try:
            self._ws_loop.run_until_complete(main())
        except Exception as e:
            if self.running:
                print(f"[SyncServer] WS server exception: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self._ws_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._ws_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._ws_loop.run_until_complete(self._ws_loop.shutdown_asyncgens())
            except Exception:
                pass
            self._ws_loop.close()

    def _run_http_server(self):
        try:
            self._http_server = SyncHTTPServer((self.host, self.http_port), SyncHTTPHandler, self)
            self._http_server.serve_forever()
        except Exception as e:
            if self.running:
                print(f"[SyncServer] HTTP server exception: {e}")

    def _run_capture_loop(self):
        """核心捕获与识别轮询主循环"""
        frame_times = []
        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            poll_interval = self.interval
            if (
                self._manual_reset_active
                and self._manual_reset_deadline > 0
                and time.monotonic() >= self._manual_reset_deadline
            ):
                self._expire_manual_reset()
            try:
                frame = self.capture.capture()
            except Exception as exc:
                # 取窗驱动、WGC 或模拟器偶发重建表面时，单帧异常不能杀死
                # 整个后台线程。保留最后稳定盘面，下轮重新尝试捕获。
                self._consecutive_pipeline_errors += 1
                try:
                    capture_info = self.capture.get_info()
                except Exception:
                    capture_info = {"mode": "unknown"}
                poll_interval = self._effective_poll_interval(capture_info)
                self._update_state({
                    "capture_status": "error",
                    "recognition_pending": True,
                    "recognition_rejection": "画面通道短暂异常，正在自动重新连接",
                    "manual_reset_pending": self._manual_reset_active,
                    "capture_info": capture_info,
                    "poll_interval": poll_interval,
                    "description": "画面通道短暂异常，已保留当前盘面并自动重试",
                    "timestamp": time.time(),
                })
                print(f"[SyncServer] Capture error: {exc}")
                elapsed = time.perf_counter() - t_start
                self._stop_event.wait(max(0.01, poll_interval - elapsed))
                continue
            if frame is not None:
                try:
                    if self._reset_requested.is_set():
                        self._reset_requested.clear()
                        request_reanchor = getattr(self.debouncer, "request_reanchor", None)
                        if request_reanchor:
                            request_reanchor()
                        else:
                            self.debouncer.reset()
                        reset_tracking = getattr(self.recognizer, "reset_tracking", None)
                        if reset_tracking:
                            reset_tracking()
                    capture_info = self.capture.get_info()
                    game_id = capture_info.get("game_id", "unknown")
                    set_capture_source = getattr(self.recognizer, "set_capture_source", None)
                    if set_capture_source:
                        set_capture_source(capture_info.get("source"))
                    # 两平台均双帧确认；天天仅缩短时间门槛。
                    self.debouncer.configure_timing(game_id == "tiantian")
                    set_game = getattr(self.recognizer, "set_game", None)
                    if set_game and set_game(game_id):
                        self.debouncer.reset()
                    if game_id in ("jj", "tiantian"):
                        self._active_game_id = game_id
                    rec_result = self.recognizer.recognize(frame)
                    raw_fen = rec_result["fen"]
                    raw_piece_count = rec_result["piece_count"]
                    pieces = [piece for row in rec_result["board"] for piece in row if piece]
                    report_recognition = getattr(self.capture, "report_recognition", None)
                    if report_recognition:
                        report_recognition(
                            raw_piece_count,
                            pieces.count("r_k") == 1 and pieces.count("b_k") == 1,
                        )

                    # 送入防抖状态机。即使当前整盘字形校验暂时失败，只要已有
                    # 稳定锚点和 90 格占位信息，仍允许状态机从“来源空、目标被
                    # 同色棋子占据”推断唯一合法走子。将/帅落点光圈正是此类帧。
                    occupancy = rec_result.get("occupancy")
                    can_reconcile_degraded = (
                        self.debouncer.last_stable_board is not None
                        and isinstance(occupancy, list)
                        and len(occupancy) == 10
                        and all(isinstance(row, list) and len(row) == 9 for row in occupancy)
                    )
                    recognition_valid = rec_result.get("recognition_valid", True)
                    occupied_count = rec_result.get("occupied_count")
                    manual_candidate_complete = (
                        recognition_valid
                        and (
                            occupied_count is None
                            or occupied_count == raw_piece_count
                        )
                    )
                    # 清缓存重读期间只允许完整字形帧接管锚点。正常走子仍可
                    # 用占位图恢复被光圈遮住的目标棋子，但这个宽松通道不能
                    # 用于整盘重建，否则一个动画帧就会永久丢子或留下旧子。
                    if self._manual_reset_active:
                        accept_observation = manual_candidate_complete
                    else:
                        accept_observation = recognition_valid or can_reconcile_degraded
                    if accept_observation:
                        event = self.debouncer.update(
                            raw_fen,
                            rec_result["board"],
                            rec_result.get("last_move_side"),
                            occupancy,
                            rec_result.get("occupied_sides"),
                            rec_result.get("last_visual_move"),
                        )
                    else:
                        event = None
                        note_unstable = getattr(self.debouncer, "note_unstable_source", None)
                        if note_unstable:
                            note_unstable(raw_piece_count)
                        if self._manual_reset_active:
                            self.debouncer.last_rejection_reason = (
                                "正在等待连续完整盘面，当前动画帧不会覆盖旧盘面"
                            )
                        elif self.debouncer.last_stable_board is None:
                            self.debouncer.last_rejection_reason = (
                                "正在等待聊天/弹窗关闭或棋盘画面稳定，旧盘面会暂时保留"
                            )
                        else:
                            self.debouncer.last_rejection_reason = (
                                "棋盘暂时被动画、聊天或弹窗遮挡，已保留上一盘面并自动重试"
                            )
                    stable_fen, stable_board = self.debouncer.get_stable_state()
                    if self._manual_reset_active:
                        if event and event.get("event_type") in {
                            "manual_reanchor", "initial", "session_reset",
                        }:
                            self._manual_reset_active = False
                            self._manual_reset_deadline = 0.0
                        elif time.monotonic() >= self._manual_reset_deadline:
                            self._expire_manual_reset()
                    if stable_fen is None or stable_board is None:
                        poll_interval = self._effective_poll_interval(
                            capture_info, has_stable_board=False, recognition_pending=False,
                        )
                        self._update_state({
                            "raw_piece_count": raw_piece_count,
                            "recognition_pending": True,
                            "capture_status": "waiting",
                            "manual_reset_pending": self._manual_reset_active,
                            "description": self.debouncer.last_rejection_reason or "等待进入有效棋局",
                            "recognition_rejection": (
                                self.debouncer.last_rejection_reason or "等待进入有效棋局"
                            ),
                            "capture_info": capture_info,
                            "game_id": rec_result.get("game_id", game_id),
                            "game_name": rec_result.get("game_name", capture_info.get("game_name", "象棋")),
                            "grid_source": rec_result.get("grid_source"),
                            "frame_resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                            "poll_interval": poll_interval,
                        })
                        elapsed = time.perf_counter() - t_start
                        self._stop_event.wait(max(0.01, poll_interval - elapsed))
                        continue
                    piece_count = sum(1 for row in stable_board for piece in row if piece is not None)

                    now = time.time()
                    now_perf = time.perf_counter()
                    frame_times.append(now_perf)
                    # 保留最近 10 次耗时以估算实际 FPS
                    frame_times = [t for t in frame_times if now_perf - t <= 2.0]
                    fps = len(frame_times) / 2.0 if len(frame_times) > 1 else 1.0 / self.interval

                    recognition_pending = rec_result["board"] != stable_board
                    now_monotonic = time.monotonic()
                    if recognition_pending:
                        if self._pending_since <= 0:
                            self._pending_since = now_monotonic
                        elif (
                            now_monotonic - self._pending_since >= 1.5
                            and now_monotonic - self._last_tracking_refresh >= 2.0
                            and not self._manual_reset_active
                        ):
                            refresh_tracking = getattr(self.recognizer, "reset_tracking", None)
                            if refresh_tracking:
                                refresh_tracking()
                            self._last_tracking_refresh = now_monotonic
                    else:
                        self._pending_since = 0.0
                    poll_interval = self._effective_poll_interval(
                        capture_info,
                        has_stable_board=True,
                        recognition_pending=recognition_pending,
                    )

                    state_update = {
                        "fen": stable_fen,
                        "board": stable_board,
                        "text_board": board_to_text(stable_board),
                        "piece_count": piece_count,
                        "raw_piece_count": raw_piece_count,
                        "recognition_pending": recognition_pending,
                        "recognition_rejection": (
                            self.debouncer.last_rejection_reason if recognition_pending else None
                        ),
                        "session_revision": self.debouncer.session_revision,
                        "timestamp": now,
                        "fps": round(fps, 1),
                        "capture_status": "resetting" if self._manual_reset_active else "ok",
                        "manual_reset_pending": self._manual_reset_active,
                        "last_frame_at": now,
                        "frame_resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "capture_info": capture_info,
                        "poll_interval": poll_interval,
                        "game_id": rec_result.get("game_id", self._active_game_id),
                        "game_name": rec_result.get("game_name", capture_info.get("game_name", "象棋")),
                        "grid_source": rec_result.get("grid_source"),
                        "side_to_move": self.debouncer.active_side,
                    }
                    self.advisor.on_live_position(
                        stable_board, self.debouncer.active_side,
                        session_revision=self.debouncer.session_revision,
                    )
                    self._ai_overlay(state_update)

                    if event:
                        state_update["last_move"] = event.get("move")
                        state_update["description"] = event.get("description")
                        self._update_state(state_update)

                        # 广播新事件
                        broadcast_payload = dict(state_update)
                        broadcast_payload["event_type"] = event.get("event_type")
                        broadcast_payload["move"] = event.get("move")
                        if "moves" in event:
                            broadcast_payload["moves"] = event["moves"]
                        self._broadcast_event(broadcast_payload)
                    else:
                        self._update_state(state_update)

                    self._consecutive_pipeline_errors = 0

                except Exception as e:
                    # 识别器内部任何一次异常都只影响当前帧。清理视觉跟踪缓存，
                    # 下一帧从当前画面重建；已确认盘面始终保留。
                    self._consecutive_pipeline_errors += 1
                    reset_tracking = getattr(self.recognizer, "reset_tracking", None)
                    if reset_tracking:
                        try:
                            reset_tracking()
                        except Exception:
                            pass
                    self._pending_since = 0.0
                    try:
                        capture_info = self.capture.get_info()
                    except Exception:
                        capture_info = {"mode": "unknown"}
                    poll_interval = self._effective_poll_interval(capture_info)
                    self._update_state({
                        "capture_status": "error",
                        "recognition_pending": True,
                        "recognition_rejection": "识别遇到临时异常，正在自动重建",
                        "manual_reset_pending": self._manual_reset_active,
                        "capture_info": capture_info,
                        "poll_interval": poll_interval,
                        "description": "识别遇到临时异常，已保留当前盘面并自动恢复",
                        "timestamp": time.time(),
                    })
                    print(f"[SyncServer] Recognition error: {e}")
            else:
                try:
                    capture_info = self.capture.get_info()
                except Exception as exc:
                    capture_info = {"mode": "unknown", "last_error": str(exc)}
                poll_interval = self._effective_poll_interval(capture_info)
                capture_error = capture_info.get("last_error")
                self._update_state({
                    "capture_status": "error",
                    "manual_reset_pending": self._manual_reset_active,
                    "capture_info": capture_info,
                    "poll_interval": poll_interval,
                    "description": capture_error or "未获取到画面帧（窗口可能已关闭、最小化或权限受限）"
                })

            t_elapsed = time.perf_counter() - t_start
            sleep_time = max(0.01, poll_interval - t_elapsed)
            self._stop_event.wait(sleep_time)

    def start(self):
        """启动同步服务 (非阻塞)"""
        self.running = True
        self._stop_event.clear()

        # 1. 启动 HTTP 线程
        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()

        # 2. 启动 WebSocket 线程
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        # 3. 启动截图识别轮询线程
        self._capture_thread = threading.Thread(target=self._run_capture_loop, daemon=True)
        self._capture_thread.start()

        # 4. 页面关闭后自动释放服务；留出首次打开浏览器的宽限期。
        self._last_client_seen = time.monotonic()
        self._idle_thread = threading.Thread(target=self._run_idle_watchdog, daemon=True)
        self._idle_thread.start()

        print(f"[SyncServer] 同步服务已启动:")
        print(f"  - WebSocket 实时流: ws://{self.host}:{self.ws_port}")
        print(f"  - HTTP 接口/Web看板: http://{self.host}:{self.http_port}/")
        print(f"  - 纯文本 FEN 获取:   http://{self.host}:{self.http_port}/fen")
        print(
            f"  - AI 执画面下方（{'开' if self.advisor.enabled else '关'}，"
            f"{self.advisor.engine_kind}，{self.advisor.time_ms}ms / 深度 {self.advisor.max_depth}，"
            f"{self.advisor.engine_threads}线程 / {self.advisor.engine_hash_mb}MB Hash）"
        )
        if self.idle_timeout > 0:
            print(f"  - 看板关闭 {self.idle_timeout:g}s 后自动停止（--idle-timeout 0 可关闭）")

    def stop(self):
        """停止同步服务"""
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        self.advisor.stop()
        close_capture = getattr(self.capture, "close", None)
        if close_capture:
            try:
                close_capture()
            except Exception:
                pass

        if self._http_server:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=1.0)
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=1.0)

        print("[SyncServer] 同步服务已安全退出。")
