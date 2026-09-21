"""逐格盘面稳定器。

90 个交叉点各自追踪观测值。单独出现的“棋子消失”永远不会直接改写稳定盘面；
只有来源格持续为空、目标格持续出现同一枚棋子，并且两格组成可行走子时，才原子
提交这两个格。这样选中发光、弹起、缩放等特效不会让看板丢子。
"""

import time
from typing import Optional, Dict, Any, Callable, List, Tuple

from core.fen import fen_to_matrix, matrix_to_fen, PIECE_NAMES_ZH
from core.xiangqi import (
    FLIPPED_START_FEN, START_FEN, apply_move, generate_legal_moves,
    infer_bottom_side, is_pseudo_legal_move, other_side,
)

Board = List[List[Optional[str]]]


class BoardDebouncer:
    def __init__(self,
                 required_stable_frames: int = 2,
                 min_stable_seconds: float = 0.2,
                 on_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.required_stable_frames = max(1, required_stable_frames)
        self.min_stable_seconds = max(0.0, min_stable_seconds)
        self.recovery_frames = max(4, self.required_stable_frames * 2)
        self.resync_frames = max(4, self.required_stable_frames + 2)
        self.resync_seconds = max(0.8, self.min_stable_seconds * 4)
        self.on_change_callback = on_change_callback

        self.last_stable_fen: Optional[str] = None
        self.last_stable_board: Optional[Board] = None
        self.move_history: List[Dict[str, Any]] = []
        self.last_rejection_reason: Optional[str] = None
        self.session_revision: int = 0
        self.active_side: str = "r"

        self._candidate: List[List[Optional[str]]] = [[None] * 9 for _ in range(10)]
        self._candidate_count: List[List[int]] = [[0] * 9 for _ in range(10)]
        self._candidate_since: List[List[float]] = [[0.0] * 9 for _ in range(10)]
        self._resync_signature: Optional[Tuple[Optional[str], ...]] = None
        self._resync_count: int = 0
        self._resync_since: float = 0.0
        self._hint_signature = None
        self._hint_count = 0
        self._hint_since = 0.0

    def configure_timing(self, fast_exact_source: bool) -> None:
        """所有实时源至少双帧确认；天天窗口仅缩短时间门槛，不牺牲回合稳定性。"""
        self.required_stable_frames = 2
        self.min_stable_seconds = 0.05 if fast_exact_source else 0.08
        self.recovery_frames = 4
        self.resync_frames = 3 if fast_exact_source else 4
        self.resync_seconds = 0.5 if fast_exact_source else 0.8

    def reset(self) -> None:
        """切换游戏平台或捕获源时清空盘面，但保留递增的会话编号。"""
        self.last_stable_fen = None
        self.last_stable_board = None
        self.move_history = []
        self.last_rejection_reason = None
        self.active_side = "r"
        self._hint_signature = None
        self._hint_count = 0
        self._candidate = [[None] * 9 for _ in range(10)]
        self._candidate_count = [[0] * 9 for _ in range(10)]
        self._candidate_since = [[0.0] * 9 for _ in range(10)]
        self._reset_resync_candidate()

    @staticmethod
    def _initial_side(board: Board) -> str:
        placement = matrix_to_fen(board).split()[0]
        starts = {START_FEN.split()[0], FLIPPED_START_FEN.split()[0]}
        # 标准开局始终红先；从中盘接入时无法可靠猜测，必须等待首个真实走子锚定回合。
        return "r" if placement in starts else "unknown"

    @staticmethod
    def _copy_board(board: Board) -> Board:
        return [row[:] for row in board]

    @staticmethod
    def _board_from_input(current_fen: str, current_board: Optional[Board]) -> Board:
        if current_board is not None:
            return [row[:] for row in current_board]
        board, _, _, _ = fen_to_matrix(current_fen)
        return board

    def _reset_cell(self, row: int, col: int) -> None:
        stable = self.last_stable_board[row][col] if self.last_stable_board else None
        self._candidate[row][col] = stable
        self._candidate_count[row][col] = 0
        self._candidate_since[row][col] = 0.0

    def _reset_all_candidates(self) -> None:
        for row in range(10):
            for col in range(9):
                self._reset_cell(row, col)

    def _reset_resync_candidate(self) -> None:
        self._resync_signature = None
        self._resync_count = 0
        self._resync_since = 0.0

    @staticmethod
    def _signature(board: Board) -> Tuple[Optional[str], ...]:
        return tuple(piece for row in board for piece in row)

    @staticmethod
    def _is_plausible_board(board: Board) -> bool:
        """过滤菜单、转场和严重误识别，仅允许合法棋子数量范围内的真实盘面。"""
        counts: Dict[str, int] = {}
        for row in board:
            for piece in row:
                if piece is not None:
                    counts[piece] = counts.get(piece, 0) + 1
        total = sum(counts.values())
        if total < 2 or total > 32:
            return False
        if counts.get("r_k", 0) != 1 or counts.get("b_k", 0) != 1:
            return False
        limits = {"k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5}
        return all(count <= limits.get(piece[2], 0) for piece, count in counts.items())

    def _track_resync_candidate(self, observed: Board, now: float) -> None:
        if self.last_stable_board == observed or not self._is_plausible_board(observed):
            self._reset_resync_candidate()
            return
        signature = self._signature(observed)
        if signature == self._resync_signature:
            self._resync_count += 1
        else:
            self._resync_signature = signature
            self._resync_count = 1
            self._resync_since = now

    def _should_resync(self, observed: Board, now: float) -> bool:
        if (
            self._resync_signature != self._signature(observed)
            or self._resync_count < self.resync_frames
            or now - self._resync_since < self.resync_seconds
            or not self._is_plausible_board(observed)
        ):
            return False
        assert self.last_stable_board is not None
        differences = sum(
            observed[row][col] != self.last_stable_board[row][col]
            for row in range(10) for col in range(9)
        )
        stable_plausible = self._is_plausible_board(self.last_stable_board)
        # 完整32子不代表新开局；否则同方连走或非法跳子会绕过回合校验。
        return differences >= 6 or self._initial_side(observed) == "r" or not stable_plausible

    def _track_observations(self, observed: Board, now: float) -> None:
        assert self.last_stable_board is not None
        for row in range(10):
            for col in range(9):
                value = observed[row][col]
                stable = self.last_stable_board[row][col]
                if value == stable:
                    self._reset_cell(row, col)
                elif value == self._candidate[row][col]:
                    self._candidate_count[row][col] += 1
                else:
                    self._candidate[row][col] = value
                    self._candidate_count[row][col] = 1
                    self._candidate_since[row][col] = now

    def _cell_ready(self, row: int, col: int, now: float, frames: Optional[int] = None) -> bool:
        needed = frames or self.required_stable_frames
        return (
            self._candidate_count[row][col] >= needed
            and now - self._candidate_since[row][col] >= self.min_stable_seconds
        )

    def _find_catchup_path(self, observed: Board, max_plies: int = 2,
                           last_move_side: Optional[str] = None):
        """用上一稳定盘面推演漏采的 1～2 步，处理对手在两帧之间快速走完的情况。"""
        if self.last_stable_board is None:
            return None
        starts = [self.active_side] if self.active_side in ("r", "b") else ["r", "b"]
        bottom = infer_bottom_side(self.last_stable_board)
        matches = []
        for start_side in starts:
            for first in generate_legal_moves(self.last_stable_board, start_side, bottom):
                after_first = apply_move(self.last_stable_board, first)
                if after_first == observed:
                    matches.append(([first], other_side(start_side)))
                    continue
                if max_plies < 2:
                    continue
                second_side = other_side(start_side)
                for second in generate_legal_moves(after_first, second_side, bottom):
                    if apply_move(after_first, second) == observed:
                        matches.append(([first, second], start_side))
        if last_move_side in ("r", "b") and self.active_side == "unknown":
            matches = [(path, side) for path, side in matches
                       if path[-1].piece[0] == last_move_side]
        if not matches:
            return None
        # 不同走法顺序可能到达同一盘面；只要最终轮次一致即可安全恢复。
        resulting_sides = {side for _, side in matches}
        if len(resulting_sides) != 1:
            return None
        return matches[0][0], matches[0][1]

    def _commit_catchup(self, observed: Board, path, next_side: str) -> Dict[str, Any]:
        old_fen = self.last_stable_fen
        board = self._copy_board(self.last_stable_board)
        payloads = []
        for move in path:
            payload = move.to_dict()
            payload["description"] = (
                f"{PIECE_NAMES_ZH.get(move.piece, '?')} 从 "
                f"({move.sc}, {move.sr}) 走到 ({move.dc}, {move.dr})"
            )
            payloads.append(payload)
            board = apply_move(board, move)
        self.last_stable_board = self._copy_board(observed)
        self.active_side = next_side
        self.last_stable_fen = matrix_to_fen(
            self.last_stable_board, "w" if next_side == "r" else "b"
        )
        self.move_history.extend(payloads)
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self.last_rejection_reason = None
        event = {
            "event_type": "move" if len(path) == 1 else "catchup",
            "fen": self.last_stable_fen,
            "old_fen": old_fen,
            "timestamp": time.time(),
            "move": payloads[-1],
            "moves": payloads,
            "description": (
                payloads[-1]["description"] if len(path) == 1
                else f"检测到两帧间已连续完成 {len(path)} 步，已追上实盘"
            ),
        }
        if self.on_change_callback:
            self.on_change_callback(event)
        return event

    def update(self, current_fen: str, current_board: Optional[Board] = None,
               last_move_side: Optional[str] = None) -> Optional[Dict[str, Any]]:
        observed = self._board_from_input(current_fen, current_board)
        now = time.perf_counter()
        # 落点提示必须与同一布局一起稳定出现，且只辅助未知回合，不覆盖历史差分。
        hint = (self._signature(observed), last_move_side) if last_move_side in ("r", "b") else None
        if hint is not None and hint == self._hint_signature:
            self._hint_count += 1
        else:
            self._hint_signature = hint
            self._hint_count = 1 if hint else 0
            self._hint_since = now
        confirmed_side = last_move_side if (
            self._hint_count >= self.required_stable_frames
            and now - self._hint_since >= self.min_stable_seconds
        ) else None

        if self.last_stable_board is None:
            if not self._is_plausible_board(observed):
                self.last_rejection_reason = "等待包含双方将帅的有效棋局盘面"
                return None
            self.last_stable_board = self._copy_board(observed)
            self.active_side = self._initial_side(self.last_stable_board)
            self.last_stable_fen = matrix_to_fen(
                self.last_stable_board, "b" if self.active_side == "b" else "w"
            )
            self.session_revision += 1
            self._reset_all_candidates()
            self._reset_resync_candidate()
            event = {
                "event_type": "initial",
                "fen": self.last_stable_fen,
                "timestamp": time.time(),
                "move": None,
                "description": "初始盘面加载成功",
            }
            if self.on_change_callback:
                self.on_change_callback(event)
            return event

        if observed == self.last_stable_board and self.active_side == "unknown" and confirmed_side:
            self.active_side = other_side(confirmed_side)
            self.last_stable_fen = matrix_to_fen(observed, "b" if self.active_side == "b" else "w")
            event = {
                "event_type": "turn_confirmed", "fen": self.last_stable_fen,
                "timestamp": time.time(), "move": None,
                "description": "已根据稳定落点标记确认行棋方",
            }
            if self.on_change_callback:
                self.on_change_callback(event)
            return event

        self._track_observations(observed, now)
        self._track_resync_candidate(observed, now)
        differences = sum(
            observed[row][col] != self.last_stable_board[row][col]
            for row in range(10) for col in range(9)
        )

        # 1. 寻找“来源格消失 + 目标格出现同一棋子”的稳定配对。
        departures = []
        destinations = []
        for row in range(10):
            for col in range(9):
                if not self._cell_ready(row, col, now):
                    continue
                stable = self.last_stable_board[row][col]
                candidate = self._candidate[row][col]
                if stable is not None and candidate is None:
                    departures.append((col, row, stable))
                elif candidate is not None and candidate != stable:
                    destinations.append((col, row, candidate))

        legal_moves = []
        for from_col, from_row, piece in departures:
            for to_col, to_row, observed_piece in destinations:
                if observed_piece != piece:
                    continue
                if self.active_side in ("r", "b") and piece[0] != self.active_side:
                    continue
                if self._is_legal_piece_move(
                    self.last_stable_board, from_col, from_row, to_col, to_row, piece
                ):
                    score = (
                        self._candidate_count[from_row][from_col]
                        + self._candidate_count[to_row][to_col]
                    )
                    legal_moves.append((score, from_col, from_row, to_col, to_row, piece))

        if legal_moves and differences == 2 and len(departures) + len(destinations) == 2:
            # 单步最多涉及来源与目标两个格。多个候选时只接受证据最强且唯一
            # 的配对，避免特效误分类造成跳子。
            legal_moves.sort(reverse=True)
            if len(legal_moves) == 1 or legal_moves[0][0] > legal_moves[1][0]:
                _, from_col, from_row, to_col, to_row, piece = legal_moves[0]
                return self._commit_move(from_col, from_row, to_col, to_row, piece)
            self.last_rejection_reason = "存在多个同强度走子候选，等待整盘稳定或消除歧义"

        # 对手走得很快时，采集可能直接从上一步跳到两步后的盘面。用上一稳定
        # 布局按合法走法推演，而不是依赖必须捕捉到中间动画帧。
        if (
            self._resync_signature == self._signature(observed)
            and self._resync_count >= self.required_stable_frames
            and now - self._resync_since >= self.min_stable_seconds
        ):
            differences = sum(
                observed[row][col] != self.last_stable_board[row][col]
                for row in range(10) for col in range(9)
            )
            if 2 <= differences <= 4 and self._is_plausible_board(observed):
                catchup = self._find_catchup_path(observed, max_plies=2,
                                                  last_move_side=confirmed_side)
                if catchup is not None:
                    path, next_side = catchup
                    return self._commit_catchup(observed, path, next_side)

        # 2. 新开一局、退出后重开通常会同时改变多个格子。只有完整盘面在
        # 连续帧和时间上都稳定后才整盘切换，菜单/空画面不会清空当前盘面。
        if self._should_resync(observed, now):
            return self._commit_resync(observed, confirmed_side)

        # 3. 单独缺子/原位变色一律不提交。只允许空格上持续出现棋子，用于
        # 服务恰好在选中特效期间启动后，取消选中时逐格补回缺失棋子。
        recovered = []
        has_pending_departure = any(
            self.last_stable_board[row][col] is not None
            and self._candidate[row][col] is None
            and self._candidate_count[row][col] > 0
            for row in range(10) for col in range(9)
        )
        if not has_pending_departure:
            for row in range(10):
                for col in range(9):
                    if (
                        self.last_stable_board[row][col] is None
                        and self._candidate[row][col] is not None
                        and self._cell_ready(row, col, now, self.recovery_frames)
                    ):
                        self.last_stable_board[row][col] = self._candidate[row][col]
                        recovered.append((col, row, self._candidate[row][col]))

        if recovered:
            old_fen = self.last_stable_fen
            self.last_stable_fen = matrix_to_fen(
                self.last_stable_board, "b" if self.active_side == "b" else "w"
            )
            self._reset_all_candidates()
            self._reset_resync_candidate()
            self.last_rejection_reason = None
            event = {
                "event_type": "recovered",
                "fen": self.last_stable_fen,
                "old_fen": old_fen,
                "timestamp": time.time(),
                "move": None,
                "description": f"逐格恢复 {len(recovered)} 枚被特效遮挡的棋子",
            }
            if self.on_change_callback:
                self.on_change_callback(event)
            return event

        if departures or destinations:
            self.last_rejection_reason = "格子变化尚未组成唯一有效走子"
        return None

    def _commit_resync(self, observed: Board, last_move_side: Optional[str] = None) -> Dict[str, Any]:
        old_fen = self.last_stable_fen
        self.last_stable_board = self._copy_board(observed)
        self.active_side = self._initial_side(self.last_stable_board)
        if self.active_side == "unknown" and last_move_side in ("r", "b"):
            self.active_side = other_side(last_move_side)
        self.last_stable_fen = matrix_to_fen(
            self.last_stable_board, "b" if self.active_side == "b" else "w"
        )
        self.session_revision += 1
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self.last_rejection_reason = None
        event = {
            "event_type": "session_reset",
            "fen": self.last_stable_fen,
            "old_fen": old_fen,
            "timestamp": time.time(),
            "move": None,
            "session_revision": self.session_revision,
            "description": f"检测到新对局/重开，已同步整盘（会话 {self.session_revision}）",
        }
        if self.on_change_callback:
            self.on_change_callback(event)
        return event

    def _commit_move(self, from_col: int, from_row: int,
                     to_col: int, to_row: int, piece: str) -> Dict[str, Any]:
        assert self.last_stable_board is not None
        old_fen = self.last_stable_fen
        captured = self.last_stable_board[to_row][to_col]
        self.last_stable_board[from_row][from_col] = None
        self.last_stable_board[to_row][to_col] = piece
        self.active_side = other_side(piece[0])
        self.last_stable_fen = matrix_to_fen(
            self.last_stable_board, "w" if self.active_side == "r" else "b"
        )
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self.last_rejection_reason = None

        piece_name = PIECE_NAMES_ZH.get(piece, "?")
        captured_name = PIECE_NAMES_ZH.get(captured, "?") if captured else None
        uci = f"{chr(97 + from_col)}{9 - from_row}{chr(97 + to_col)}{9 - to_row}"
        description = f"{piece_name} 从 ({from_col}, {from_row}) "
        description += f"吃到 ({to_col}, {to_row}) {captured_name}" if captured else f"走到 ({to_col}, {to_row})"
        move = {
            "piece": piece,
            "piece_name": piece_name,
            "from": {"col": from_col, "row": from_row},
            "to": {"col": to_col, "row": to_row},
            "captured": captured,
            "uci": uci,
            "description": description,
        }
        if captured:
            move["captured_name"] = captured_name
        event = {
            "event_type": "move",
            "fen": self.last_stable_fen,
            "old_fen": old_fen,
            "timestamp": time.time(),
            "move": move,
            "description": description,
        }
        self.move_history.append(event)
        if self.on_change_callback:
            self.on_change_callback(event)
        return event

    @staticmethod
    def _is_legal_piece_move(board: Board, sc: int, sr: int,
                             dc: int, dr: int, piece: str) -> bool:
        return is_pseudo_legal_move(board, sc, sr, dc, dr, piece)

    def get_stable_state(self) -> Tuple[Optional[str], Optional[Board]]:
        return self.last_stable_fen, self._copy_board(self.last_stable_board) if self.last_stable_board else None
