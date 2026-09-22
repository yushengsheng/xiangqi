"""AI 顾问：始终执画面下方，与上方对战。

- coach：跟随实盘，只给出下方着法建议，绝不点击外部游戏窗口。
- play：本地对弈，人点上方棋子，AI 自动走下方。
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from core.engine import SearchResult, XiangqiEngine
from core.pikafish_engine import PikafishEngine
from core.fen import board_to_text, matrix_to_fen
from core.xiangqi import (
    Board,
    Move,
    apply_move,
    copy_board,
    generate_legal_moves,
    infer_bottom_side,
    is_in_check,
    is_legal_move,
    legal_move_payloads,
    move_payload,
    other_side,
)

OnUpdate = Callable[[], None]


class AIAdvisor:
    def __init__(self, time_ms: int = 700, max_depth: int = 60,
                 enabled: bool = True, on_update: Optional[OnUpdate] = None,
                 engine_kind: str = "pikafish", engine_threads: int = 1,
                 engine_hash_mb: int = 32):
        self.time_ms = time_ms
        self.max_depth = max_depth
        self.enabled = enabled
        self.engine_kind = engine_kind if engine_kind in ("pikafish", "builtin") else "pikafish"
        self.engine_threads = max(1, min(32, int(engine_threads)))
        self.engine_hash_mb = max(16, min(2048, int(engine_hash_mb)))
        self.active_engine = self.engine_kind
        self.engine_error: Optional[str] = None
        self.mode = "coach"  # coach | play
        self.on_update = on_update

        self.engine = XiangqiEngine()
        self.pikafish = PikafishEngine(
            threads=self.engine_threads, hash_mb=self.engine_hash_mb,
        )
        self._lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._job_id = 0
        self._worker: Optional[threading.Thread] = None

        self.live_board: Optional[Board] = None
        self.live_to_move = "r"
        self.live_session_revision: Optional[int] = None
        self.play_board: Optional[Board] = None
        self.play_session_revision: Optional[int] = None
        self.play_to_move = "r"
        self.thinking = False
        self.last_result: Optional[SearchResult] = None
        self.last_result_board: Optional[Board] = None
        self.status = "等待盘面"
        self.error: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        self.play_states: List[Dict[str, Any]] = []
        self.play_cursor: int = -1
        self.variations: List[Dict[str, Any]] = []
        self.game_over = False
        self._running_engine: Optional[XiangqiEngine] = None
        self._think_signature = None

    def stop(self) -> None:
        if self._running_engine is not None:
            self._running_engine.request_stop()
        self.engine.request_stop()
        self.pikafish.request_stop()
        with self._lock:
            self._job_id += 1
            self.thinking = False
        self.pikafish.close()

    def configure(self, enabled: Optional[bool] = None, mode: Optional[str] = None,
                  time_ms: Optional[int] = None,
                  max_depth: Optional[int] = None,
                  engine_kind: Optional[str] = None,
                  engine_threads: Optional[int] = None,
                  engine_hash_mb: Optional[int] = None) -> Dict[str, Any]:
        restart_search = False
        reconfigure_pikafish = False
        close_pikafish = False
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if mode in ("coach", "play"):
                self._invalidate_search_unlocked()
                if mode == "play" and self.play_board is None:
                    source = self.live_board
                    if source is not None:
                        self.play_board = copy_board(source)
                        self.play_to_move = self.live_to_move
                        self._reset_play_timeline_unlocked(self.play_board, self.play_to_move)
                        self.play_session_revision = self.live_session_revision
                        self.last_result = None
                        self.game_over = False
                        self.status = "本地对弈：请走上方"
                if mode == "coach":
                    self.play_board = None
                    self.history = []
                    self.play_states = []
                    self.play_cursor = -1
                    self.play_session_revision = None
                    self.variations = []
                    self.game_over = False
                self.mode = mode
            if time_ms is not None:
                new_time = max(80, min(30000, int(time_ms)))
                restart_search = restart_search or new_time != self.time_ms
                self.time_ms = new_time
            if max_depth is not None:
                new_depth = max(1, min(128, int(max_depth)))
                restart_search = restart_search or new_depth != self.max_depth
                self.max_depth = new_depth
            if engine_kind is not None:
                if engine_kind not in ("pikafish", "builtin"):
                    raise ValueError("engine_kind 必须是 pikafish 或 builtin")
                changed_engine = engine_kind != self.engine_kind
                restart_search = restart_search or changed_engine
                close_pikafish = changed_engine and engine_kind == "builtin"
                self.engine_kind = engine_kind
            if engine_threads is not None:
                new_threads = max(1, min(32, int(engine_threads)))
                restart_search = restart_search or new_threads != self.engine_threads
                reconfigure_pikafish = reconfigure_pikafish or new_threads != self.engine_threads
                self.engine_threads = new_threads
            if engine_hash_mb is not None:
                new_hash = max(16, min(2048, int(engine_hash_mb)))
                restart_search = restart_search or new_hash != self.engine_hash_mb
                reconfigure_pikafish = reconfigure_pikafish or new_hash != self.engine_hash_mb
                self.engine_hash_mb = new_hash
            if restart_search:
                self._job_id += 1
                self.thinking = False
                self._think_signature = None
                self.last_result = None
        if restart_search and self._running_engine is not None:
            self._running_engine.request_stop()
        if reconfigure_pikafish:
            self.pikafish.configure(
                threads=self.engine_threads, hash_mb=self.engine_hash_mb,
            )
        if close_pikafish:
            self.pikafish.close()
        if self.enabled:
            self._maybe_think()
        else:
            self.stop()
            with self._lock:
                self.status = "AI 已关闭"
                self.last_result = None
        return self.snapshot()

    def start_local_game(self, board: Optional[Board] = None, to_move: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            source = board if board is not None else self.live_board
            if source is None:
                self.error = "还没有可对弈的盘面"
                return self._snapshot_unlocked()
            self._invalidate_search_unlocked()
            self.mode = "play"
            self.play_board = copy_board(source)
            self.play_to_move = to_move or self.live_to_move
            self._reset_play_timeline_unlocked(self.play_board, self.play_to_move)
            self.play_session_revision = self.live_session_revision
            self.last_result = None
            self.error = None
            self.game_over = False
            self.status = "本地对弈已开始"
        self.pikafish.new_game()
        self._maybe_think()
        return self.snapshot()

    def follow_live(self) -> Dict[str, Any]:
        with self._lock:
            self._invalidate_search_unlocked()
            self.error = None
            self.mode = "coach"
            self.play_board = None
            self.history = []
            self.play_states = []
            self.play_cursor = -1
            self.play_session_revision = None
            self.variations = []
            self.last_result = None
            self.game_over = False
            self.status = "已回到跟随实盘"
        self._maybe_think()
        return self.snapshot()

    def set_turn(self, side: str) -> Dict[str, Any]:
        if side not in ("r", "b"):
            return self.snapshot()
        with self._lock:
            self._invalidate_search_unlocked()
            self.error = None
            if self.mode == "play" and self.play_board is not None:
                self.play_to_move = side
            else:
                self.live_to_move = side
            self.last_result = None
        self._maybe_think()
        return self.snapshot()

    @staticmethod
    def _matching_live_move(before: Board, after: Board, side: str) -> Optional[Move]:
        """找出能把 AI 时间线盘面精确变成实盘的新走子。"""
        for move in generate_legal_moves(before, side, infer_bottom_side(before)):
            if apply_move(before, move) == after:
                return move
        return None

    def on_live_position(self, board: Board, to_move: str,
                         session_revision: Optional[int] = None) -> None:
        """接收实盘。

        play 模式下 AI 盘面可能领先实盘一手（等待用户在 JJ 执行 AI 着法）。
        旧盘面只视为延迟；当实盘在当前 AI 盘面上出现一个合法上方走子时，
        自动写入时间线并立即触发 AI 下一手。
        """
        should_think = False
        cancel_search = False
        reset_engine_game = False
        with self._lock:
            signature = self._signature(board, to_move)
            changed = self.live_board is None or self._signature(self.live_board, self.live_to_move) != signature
            revision_changed = (
                session_revision is not None
                and self.live_session_revision is not None
                and session_revision != self.live_session_revision
            )
            self.live_board = copy_board(board)
            self.live_to_move = to_move
            if session_revision is not None:
                self.live_session_revision = session_revision

            if self.mode == "play" and self.play_board is not None:
                if revision_changed and self.play_session_revision != session_revision:
                    self._job_id += 1
                    self.thinking = False
                    cancel_search = True
                    self.play_board = copy_board(board)
                    self.play_to_move = to_move
                    self._reset_play_timeline_unlocked(self.play_board, self.play_to_move)
                    self.play_session_revision = session_revision
                    self.last_result = None
                    self.game_over = False
                    self.error = None
                    self.status = "检测到新局，已重置 AI 对局"
                    should_think = True
                    reset_engine_game = True
                elif board == self.play_board:
                    # 中盘接入后落点提示可能只更新回合，不改变布局。
                    if changed and self.play_to_move != to_move:
                        self._invalidate_search_unlocked()
                        self.play_to_move = to_move
                        self.play_states[self.play_cursor]["to_move"] = to_move
                        should_think = True
                elif any(
                    state.get("board") == board
                    for state in self.play_states[:self.play_cursor + 1]
                ):
                    # AI 刚给出并应用着法时，实盘会短暂停留在上一状态。
                    self.status = "等待实盘执行 AI 着法"
                else:
                    bottom = infer_bottom_side(self.play_board)
                    top = other_side(bottom)
                    move = None
                    if self.play_to_move == top:
                        move = self._matching_live_move(self.play_board, board, top)
                    if move is not None:
                        self._apply_unlocked(move, "opponent_live")
                        self.error = None
                        self.status = "已同步对手走子，下方思考中…"
                        should_think = True
                    elif changed:
                        # 输入已经过稳定器确认。变招、补回遮挡棋子或漏采多步不能
                        # 永久停在旧模拟盘上；以稳定实盘重新锚定时间线和回合。
                        self._invalidate_search_unlocked()
                        self.play_board = copy_board(board)
                        self.play_to_move = to_move
                        self._reset_play_timeline_unlocked(board, to_move)
                        self.play_session_revision = session_revision
                        self.game_over = False
                        self.error = None
                        self.status = "已重新对齐稳定实盘"
                        should_think = True
            elif self.mode == "coach" and changed:
                should_think = True

        if cancel_search and self._running_engine is not None:
            self._running_engine.request_stop()
        if reset_engine_game:
            self.pikafish.new_game()
        if should_think:
            self._maybe_think()

    def play_move(self, src_row: int, src_col: int, dst_row: int, dst_col: int) -> Dict[str, Any]:
        with self._lock:
            if self.mode != "play" or self.play_board is None:
                self.error = "尚未开始本地对弈"
                return self._snapshot_unlocked()
            board = self.play_board
            bottom = infer_bottom_side(board)
            human = other_side(bottom)
            if self.play_to_move != human:
                self.error = "现在轮到下方 AI"
                return self._snapshot_unlocked()
            piece = board[src_row][src_col]
            if piece is None or piece[0] != human:
                self.error = "请走上方的棋子"
                return self._snapshot_unlocked()
            move = Move(src_row, src_col, dst_row, dst_col, piece, board[dst_row][dst_col])
            if not is_legal_move(board, move, human):
                self.error = "非法走法"
                return self._snapshot_unlocked()
            self._apply_unlocked(move, "human")
            self.error = None
        self._maybe_think()
        return self.snapshot()

    def _reset_play_timeline_unlocked(self, board: Board, to_move: str) -> None:
        self.history = []
        self.variations = []
        self.play_states = [{
            "board": copy_board(board), "to_move": to_move,
            "move": None, "actor": None,
        }]
        self.play_cursor = 0

    def _archive_future_unlocked(self) -> None:
        if self.play_cursor < 0 or self.play_cursor >= len(self.play_states) - 1:
            return
        moves = [state["move"] for state in self.play_states[self.play_cursor + 1:] if state.get("move")]
        if moves:
            self.variations.append({"branch_at": self.play_cursor, "moves": moves})
        self.play_states = self.play_states[:self.play_cursor + 1]

    def _rebuild_history_unlocked(self) -> None:
        self.history = [
            state["move"] for state in self.play_states[1:self.play_cursor + 1]
            if state.get("move")
        ]

    def undo(self) -> Dict[str, Any]:
        """本地对弈回退一回合；随后用户可走不同着形成变招分支。"""
        if self._running_engine is not None:
            self._running_engine.request_stop()
        with self._lock:
            self._job_id += 1
            self.thinking = False
            self.last_result = None
            if self.mode != "play" or self.play_board is None or self.play_cursor <= 0:
                self.error = "当前没有可回退的着法"
                return self._snapshot_unlocked()
            target = self.play_cursor - 1
            # AI 已应手时同时撤销 AI 与上一手人类着，直接回到可变招节点。
            if (
                self.play_states[self.play_cursor].get("actor") == "ai"
                and target > 0
                and self.play_states[target].get("actor") == "human"
            ):
                target -= 1
            self.play_cursor = target
            state = self.play_states[target]
            self.play_board = copy_board(state["board"])
            self.play_to_move = state["to_move"]
            self._rebuild_history_unlocked()
            self.game_over = False
            self.error = None
            self.status = "已回退，请为上方选择变招"
            return self._snapshot_unlocked()

    def _apply_unlocked(self, move: Move, actor: str) -> None:
        assert self.play_board is not None
        self._archive_future_unlocked()
        payload = move_payload(self.play_board, move)
        payload["actor"] = actor
        self.play_board = apply_move(self.play_board, move)
        self.play_to_move = other_side(move.piece[0])
        self.play_states.append({
            "board": copy_board(self.play_board),
            "to_move": self.play_to_move,
            "move": payload,
            "actor": actor,
        })
        self.play_cursor += 1
        self._rebuild_history_unlocked()
        bottom = infer_bottom_side(self.play_board)
        if not generate_legal_moves(self.play_board, self.play_to_move, bottom):
            loser = "下方" if self.play_to_move == bottom else "上方"
            checked = is_in_check(self.play_board, self.play_to_move, bottom)
            self.game_over = True
            self.status = f"{loser}被将死" if checked else f"{loser}困毙"
            self.last_result = None
        elif actor in ("human", "opponent_live"):
            self.status = "下方思考中…"
        else:
            self.status = "请走上方"

    @staticmethod
    def _signature(board: Board, to_move: str):
        return (tuple(piece for row in board for piece in row), to_move)

    def _active_position_unlocked(self) -> Optional[tuple]:
        if self.mode == "play" and self.play_board is not None:
            return self.play_board, self.play_to_move
        if self.live_board is not None:
            return self.live_board, self.live_to_move
        return None

    def _invalidate_search_unlocked(self) -> None:
        self._job_id += 1
        self.thinking = False
        self._think_signature = None
        self.last_result = None
        self.last_result_board = None
        if self._running_engine is not None:
            self._running_engine.request_stop()

    def _maybe_think(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            position = self._active_position_unlocked()
            if position is None:
                return
            if self.game_over:
                self.thinking = False
                return
            board, to_move = position
            bottom = infer_bottom_side(board)
            signature = self._signature(board, to_move)
            if to_move not in ("r", "b"):
                if self.thinking or self._running_engine is not None:
                    self._job_id += 1
                self.thinking = False
                self._think_signature = signature
                self.last_result = None
                if self._running_engine is not None:
                    self._running_engine.request_stop()
                self.status = "等待首个真实走子确认回合"
                return
            if to_move != bottom:
                if self.thinking or self._running_engine is not None:
                    self._job_id += 1
                self.thinking = False
                self._think_signature = signature
                self.last_result = None
                if self._running_engine is not None:
                    self._running_engine.request_stop()
                if self.mode == "play":
                    self.status = "请走上方"
                else:
                    self.status = "等待上方走子"
                return
            if not generate_legal_moves(board, bottom, bottom):
                self.thinking = False
                self.status = "下方无棋可走"
                return
            if self.thinking and signature == self._think_signature:
                return
            if self._running_engine is not None:
                self._running_engine.request_stop()
            self._job_id += 1
            job_id = self._job_id
            snapshot = copy_board(board)
            requested_engine = self.engine_kind
            self.thinking = True
            self._think_signature = signature
            self.status = "下方思考中…"
        worker = threading.Thread(
            target=self._search_job,
            args=(job_id, snapshot, bottom, requested_engine),
            daemon=True,
        )
        self._worker = worker
        worker.start()

    def _search_job(self, job_id: int, board: Board, side: str,
                    requested_engine: str) -> None:
        # 搜索串行执行，取消后排队的旧任务不能重新启动引擎或覆盖当前任务句柄。
        with self._search_lock:
            with self._lock:
                if job_id != self._job_id:
                    return
            try:
                self._run_search_job(job_id, board, side, requested_engine)
            except Exception as exc:
                with self._lock:
                    if job_id != self._job_id:
                        return
                    self.thinking = False
                    self.last_result = None
                    self.error = f"AI 搜索失败：{exc}"
                    self.status = self.error
                if self.on_update:
                    self.on_update()
            finally:
                with self._lock:
                    self._running_engine = None

    def _run_search_job(self, job_id: int, board: Board, side: str,
                        requested_engine: str) -> None:
        fallback_error = None
        if requested_engine == "pikafish":
            self._running_engine = self.pikafish
            try:
                result = self.pikafish.search(
                    board, side, time_ms=self.time_ms, max_depth=self.max_depth,
                )
            except Exception as exc:
                first_error = str(exc)
                with self._lock:
                    if job_id != self._job_id:
                        return
                # 当前搜索期间若引擎进程意外退出，先透明重启并重试一次；
                # 只有资源持续不可用时才降级，避免一次偶发崩溃降低整局棋力。
                try:
                    self._running_engine = self.pikafish
                    result = self.pikafish.search(
                        board, side, time_ms=self.time_ms, max_depth=self.max_depth,
                    )
                except Exception as retry_exc:
                    fallback_error = f"{first_error}；重试失败：{retry_exc}"
                    with self._lock:
                        if job_id != self._job_id:
                            return
                    engine = XiangqiEngine()
                    self._running_engine = engine
                    result = engine.search(
                        board, side, time_ms=self.time_ms, max_depth=self.max_depth,
                    )
                    result.engine = "builtin-fallback"
                    result.reason = f"Pikafish 不可用，已回退内置引擎：{fallback_error}"
        else:
            engine = XiangqiEngine()
            self._running_engine = engine
            result = engine.search(
                board, side, time_ms=self.time_ms, max_depth=self.max_depth,
            )
            result.engine = "builtin"
        with self._lock:
            if job_id != self._job_id:
                return
            position = self._active_position_unlocked()
            if position is None or self._signature(*position) != self._signature(board, side):
                self.thinking = False
                return
            self._running_engine = None
            self.thinking = False
            self.error = None
            self.last_result = result
            self.active_engine = result.engine
            self.engine_error = fallback_error
            self.last_result_board = copy_board(board)
            if result.move is None:
                self.status = result.reason or "下方无棋可走"
            elif self.mode == "play" and self.play_board is not None:
                current_bottom = infer_bottom_side(self.play_board)
                if self.play_to_move == current_bottom:
                    self._apply_unlocked(result.move, "ai")
                else:
                    self.status = "请走上方"
            else:
                zh = result.to_dict(board).get("zh") or result.move.uci
                self.status = f"下方建议 {zh}"
        if self.on_update:
            self.on_update()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Dict[str, Any]:
        position = self._active_position_unlocked()
        board = position[0] if position else None
        to_move = position[1] if position else "r"
        bottom = infer_bottom_side(board) if board else "r"
        top = other_side(bottom)
        suggestion = None
        if self.last_result is not None:
            source = self.last_result_board or board
            suggestion = self.last_result.to_dict(source)
        legal = []
        if board is not None and self.mode == "play" and to_move == top:
            legal = legal_move_payloads(board, top)
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "time_ms": self.time_ms,
            "max_depth": self.max_depth,
            "engine_kind": self.engine_kind,
            "active_engine": self.active_engine,
            "engine_threads": self.engine_threads,
            "engine_hash_mb": self.engine_hash_mb,
            "pikafish_available": self.pikafish.is_available(),
            "engine_error": self.engine_error,
            "thinking": self.thinking,
            "status": self.status,
            "error": self.error,
            "bottom_side": bottom,
            "top_side": top,
            "bottom_label": "红" if bottom == "r" else "黑",
            "top_label": "红" if top == "r" else "黑",
            "to_move": to_move,
            "to_move_label": (
                "待确认" if to_move not in ("r", "b")
                else ("下方" if to_move == bottom else "上方")
            ),
            "suggestion": suggestion,
            "legal_moves": legal,
            "history": list(self.history[-40:]),
            "can_undo": self.mode == "play" and self.play_cursor > 0,
            "ply": max(0, self.play_cursor),
            "variation_count": len(self.variations),
            "variations": list(self.variations[-10:]),
            "live_session_revision": self.live_session_revision,
            "play_session_revision": self.play_session_revision,
            "board": copy_board(board) if board else None,
            "fen": matrix_to_fen(board, "b" if to_move == "b" else "w") if board else "",
            "text_board": board_to_text(board) if board else "",
            "play_mode": self.mode == "play",
            "game_over": self.game_over,
        }
