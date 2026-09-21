"""Pikafish UCI subprocess backend.

The recognizer stores pieces in screen coordinates. Pikafish expects canonical Xiangqi
coordinates (black at the top, red at the bottom), so black-at-bottom positions are
rotated 180 degrees before search and all returned moves are rotated back.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from core.engine import MATE, SearchResult
from core.fen import matrix_to_fen
from core.xiangqi import (
    Board,
    Move,
    apply_move,
    copy_board,
    infer_bottom_side,
    is_legal_move,
    parse_uci,
)


class PikafishError(RuntimeError):
    pass


def rotate_board_180(board: Board) -> Board:
    return [list(reversed(row)) for row in reversed(board)]


def rotate_move_180(move: Move, target_board: Board) -> Move:
    sr, sc = 9 - move.sr, 8 - move.sc
    dr, dc = 9 - move.dr, 8 - move.dc
    piece = target_board[sr][sc]
    if piece is None:
        raise PikafishError(f"旋转后的来源格为空: {move.uci}")
    return Move(sr, sc, dr, dc, piece, target_board[dr][dc])


class PikafishEngine:
    """A persistent, thread-safe-enough UCI engine for one active search at a time."""

    def __init__(self, binary_path: Optional[str] = None,
                 nnue_path: Optional[str] = None,
                 threads: int = 2, hash_mb: int = 64) -> None:
        root = Path(__file__).resolve().parent.parent
        engine_dir = root / "engines" / "pikafish"
        self.binary_path = Path(binary_path) if binary_path else engine_dir / "pikafish"
        self.nnue_path = Path(nnue_path) if nnue_path else engine_dir / "pikafish.nnue"
        self.threads = max(1, min(32, int(threads)))
        self.hash_mb = max(16, min(2048, int(hash_mb)))

        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._write_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._searching = threading.Event()
        self._cancel_requested = threading.Event()
        self._options_dirty = True
        self._new_game_pending = True
        self._last_error: Optional[str] = None
        self._engine_name = "Pikafish"

    @property
    def name(self) -> str:
        return self._engine_name

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def is_available(self) -> bool:
        return (
            self.binary_path.is_file()
            and os.access(self.binary_path, os.X_OK)
            and self.nnue_path.is_file()
        )

    def configure(self, threads: Optional[int] = None,
                  hash_mb: Optional[int] = None) -> None:
        changed = False
        if threads is not None:
            value = max(1, min(32, int(threads)))
            if value != self.threads:
                self.threads = value
                self._options_dirty = True
                changed = True
        if hash_mb is not None:
            value = max(16, min(2048, int(hash_mb)))
            if value != self.hash_mb:
                self.hash_mb = value
                self._options_dirty = True
                changed = True
        # macOS malloc may keep the old Hash allocation resident after setoption.
        # Restarting on a strength-tier change both applies settings atomically and
        # releases the previous engine's memory immediately.
        if changed and self._process is not None:
            self.request_stop()
            self.close()

    def new_game(self) -> None:
        self._new_game_pending = True
        self.request_stop()

    @staticmethod
    def _reader_loop(process: subprocess.Popen,
                     output: "queue.Queue[Optional[str]]") -> None:
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                output.put(raw.rstrip("\r\n"))
        finally:
            output.put(None)

    def _send(self, command: str) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise PikafishError("Pikafish 进程未运行")
        with self._write_lock:
            try:
                process.stdin.write(command + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise PikafishError(f"Pikafish 通信失败: {exc}") from exc

    def _wait_for(self, predicate, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        recent: List[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PikafishError("等待 Pikafish 响应超时" + (f": {recent[-1]}" if recent else ""))
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise PikafishError("等待 Pikafish 响应超时") from exc
            if line is None:
                code = self._process.poll() if self._process else None
                raise PikafishError(f"Pikafish 意外退出（code={code}）")
            if line:
                recent.append(line)
                recent = recent[-12:]
                if line.startswith("id name "):
                    self._engine_name = line[8:].strip()
                if "CRITICAL ERROR" in line:
                    raise PikafishError(line)
            if predicate(line):
                return line

    def _start(self) -> None:
        with self._start_lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.is_available():
                raise PikafishError(
                    f"缺少 Pikafish 资源: {self.binary_path} / {self.nnue_path}"
                )
            self.close()
            self._lines = queue.Queue()
            try:
                self._process = subprocess.Popen(
                    [str(self.binary_path)],
                    cwd=str(self.binary_path.parent),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._process = None
                raise PikafishError(f"无法启动 Pikafish: {exc}") from exc
            output = self._lines
            self._reader = threading.Thread(
                target=self._reader_loop, args=(self._process, output), daemon=True,
                name="pikafish-stdout",
            )
            self._reader.start()
            self._send("uci")
            self._wait_for(lambda line: line == "uciok", 10.0)
            self._options_dirty = True
            self._new_game_pending = True
            self._apply_options()
            self._last_error = None

    def _apply_options(self) -> None:
        if not self._options_dirty:
            return
        self._send(f"setoption name EvalFile value {self.nnue_path.resolve()}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("setoption name MultiPV value 1")
        self._send("setoption name Ponder value false")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("isready")
        self._wait_for(lambda line: line == "readyok", 20.0)
        self._options_dirty = False

    @staticmethod
    def _parse_info(line: str) -> Optional[dict]:
        if not line.startswith("info ") or " pv " not in line:
            return None
        tokens = line.split()
        info = {"depth": 0, "nodes": 0, "time": 0, "score": 0, "mate": False, "pv": []}
        try:
            if "multipv" in tokens and int(tokens[tokens.index("multipv") + 1]) != 1:
                return None
            if "depth" in tokens:
                info["depth"] = int(tokens[tokens.index("depth") + 1])
            if "nodes" in tokens:
                info["nodes"] = int(tokens[tokens.index("nodes") + 1])
            if "time" in tokens:
                info["time"] = int(tokens[tokens.index("time") + 1])
            if "score" in tokens:
                idx = tokens.index("score")
                kind, raw = tokens[idx + 1], int(tokens[idx + 2])
                if kind == "mate":
                    info["mate"] = True
                    info["score"] = (MATE - min(999, abs(raw))) * (1 if raw >= 0 else -1)
                elif kind == "cp":
                    info["score"] = raw
            idx = tokens.index("pv")
            info["pv"] = tokens[idx + 1:]
            return info
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _convert_uci_move(canonical_board: Board, screen_board: Board,
                          uci: str, flipped: bool) -> Optional[Move]:
        move = parse_uci(canonical_board, uci)
        if move is None:
            return None
        return rotate_move_180(move, screen_board) if flipped else move

    @classmethod
    def _convert_pv(cls, canonical_board: Board, screen_board: Board,
                    pv_tokens: List[str], flipped: bool) -> List[Move]:
        result: List[Move] = []
        canonical = copy_board(canonical_board)
        screen = copy_board(screen_board)
        for token in pv_tokens:
            move = parse_uci(canonical, token)
            if move is None:
                break
            converted = rotate_move_180(move, screen) if flipped else move
            result.append(converted)
            canonical = apply_move(canonical, move)
            screen = apply_move(screen, converted)
        return result

    def search(self, board: Board, side: str, time_ms: int = 1000,
               max_depth: int = 60) -> SearchResult:
        with self._search_lock:
            self._cancel_requested.clear()
            try:
                self._start()
                self._apply_options()
                if self._cancel_requested.is_set():
                    raise PikafishError("Pikafish 搜索已取消")
                if self._new_game_pending:
                    self._send("ucinewgame")
                    self._send("isready")
                    self._wait_for(lambda line: line == "readyok", 20.0)
                    self._new_game_pending = False
                if self._cancel_requested.is_set():
                    raise PikafishError("Pikafish 搜索已取消")

                flipped = infer_bottom_side(board) == "b"
                canonical = rotate_board_180(board) if flipped else copy_board(board)
                fen = matrix_to_fen(canonical, "w" if side == "r" else "b")
                self._send(f"position fen {fen}")
                # A high depth cap protects against accidental unbounded searches; wall time
                # remains the normal strength control and usually stops first.
                depth_cap = max(1, min(128, int(max_depth)))
                think_ms = max(50, min(30000, int(time_ms)))
                # go 与 stop 必须有确定顺序；否则 stop 可能先于 go 被引擎忽略。
                with self._control_lock:
                    if self._cancel_requested.is_set():
                        raise PikafishError("Pikafish 搜索已取消")
                    self._searching.set()
                    self._send(f"go movetime {think_ms} depth {depth_cap}")

                latest = None
                bestmove = None
                deadline = time.monotonic() + think_ms / 1000.0 + 8.0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.request_stop()
                        raise PikafishError("Pikafish 搜索超时")
                    try:
                        line = self._lines.get(timeout=remaining)
                    except queue.Empty as exc:
                        self.request_stop()
                        raise PikafishError("Pikafish 搜索超时") from exc
                    if line is None:
                        code = self._process.poll() if self._process else None
                        raise PikafishError(f"Pikafish 搜索期间退出（code={code}）")
                    if "CRITICAL ERROR" in line:
                        raise PikafishError(line)
                    parsed = self._parse_info(line)
                    if parsed is not None:
                        latest = parsed
                    if line.startswith("bestmove "):
                        parts = line.split()
                        bestmove = parts[1] if len(parts) > 1 else "(none)"
                        break

                if bestmove in (None, "(none)"):
                    return SearchResult(
                        move=None,
                        score=latest["score"] if latest else -MATE,
                        depth=latest["depth"] if latest else 0,
                        nodes=latest["nodes"] if latest else 0,
                        time_ms=latest["time"] if latest else think_ms,
                        pv=[], mate=bool(latest and latest["mate"]),
                        reason="Pikafish：无合法着法", engine="pikafish",
                    )
                move = self._convert_uci_move(canonical, board, bestmove, flipped)
                if move is None or not is_legal_move(board, move, side):
                    raise PikafishError(f"Pikafish 返回非法着法: {bestmove}")
                pv_tokens = latest["pv"] if latest else [bestmove]
                pv = self._convert_pv(canonical, board, pv_tokens, flipped)
                if not pv or pv[0].uci != move.uci:
                    pv = [move]
                return SearchResult(
                    move=move,
                    score=latest["score"] if latest else 0,
                    depth=latest["depth"] if latest else 0,
                    nodes=latest["nodes"] if latest else 0,
                    time_ms=latest["time"] if latest else think_ms,
                    pv=pv,
                    mate=bool(latest and latest["mate"]),
                    reason="Pikafish NNUE", engine="pikafish",
                )
            except Exception as exc:
                self._last_error = str(exc)
                if self._process is not None and self._process.poll() is not None:
                    self.close()
                raise
            finally:
                self._searching.clear()

    def request_stop(self) -> None:
        self._cancel_requested.set()
        with self._control_lock:
            try:
                if self._process is not None and self._process.poll() is None:
                    self._send("stop")
            except PikafishError as exc:
                logging.getLogger(__name__).warning("停止 Pikafish 时进程已不可用：%s", exc)

    def close(self) -> None:
        process = self._process
        reader = self._reader
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                with self._write_lock:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.wait(timeout=1.5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logging.getLogger(__name__).debug("Pikafish 未正常退出，尝试终止：%s", exc)
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired) as terminate_exc:
                logging.getLogger(__name__).warning("Pikafish 终止失败，尝试强制退出：%s", terminate_exc)
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired) as kill_exc:
                    logging.getLogger(__name__).error("无法回收 Pikafish 进程：%s", kill_exc)
        finally:
            if process.poll() is not None:
                if reader is not None and reader is not threading.current_thread():
                    reader.join(timeout=1.0)
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError as exc:
                            logging.getLogger(__name__).warning("关闭 Pikafish 管道失败：%s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
