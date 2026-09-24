"""规则驱动的实时棋局跟踪器。

视觉识别只提供观测，内部盘面只能由标准开局、新会话或经过象棋规则证明的走子
改变。单独丢子、多子、原位变色和无法解释的整盘跳变，无论持续多少帧都不会
覆盖稳定盘面。这样短暂高亮、动画、漏帧和字形误判只能造成延迟，不能破坏棋局。
"""

import time
from collections import Counter
from typing import Optional, Dict, Any, Callable, List, Tuple

from core.fen import fen_to_matrix, matrix_to_fen, PIECE_NAMES_ZH
from core.position_validation import is_structurally_valid_board
from core.xiangqi import (
    FLIPPED_START_FEN, START_FEN, Move, apply_move, generate_legal_moves,
    infer_bottom_side, is_legal_move, other_side,
)

Board = List[List[Optional[str]]]


class BoardDebouncer:
    def __init__(self,
                 required_stable_frames: int = 2,
                 min_stable_seconds: float = 0.2,
                 on_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.required_stable_frames = max(1, required_stable_frames)
        self.min_stable_seconds = max(0.0, min_stable_seconds)
        self.resync_frames = max(4, self.required_stable_frames + 2)
        self.resync_seconds = max(0.8, self.min_stable_seconds * 4)
        self.initial_frames = 1
        self.initial_seconds = 0.0
        self.on_change_callback = on_change_callback

        self.last_stable_fen: Optional[str] = None
        self.last_stable_board: Optional[Board] = None
        self.move_history: List[Dict[str, Any]] = []
        # Exact positions preceding verified moves.  They are the only states
        # an in-game undo may automatically return to.
        self._undo_positions: List[Tuple[Board, str]] = []
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
        self._source_discontinuity_count = 0
        self._source_discontinuity = False
        self._force_reanchor = False

    def configure_timing(self, fast_exact_source: bool) -> None:
        """所有实时源至少双帧确认；天天窗口仅缩短时间门槛，不牺牲回合稳定性。"""
        self.required_stable_frames = 2
        self.min_stable_seconds = 0.05 if fast_exact_source else 0.08
        # 单步仍使用快速双帧确认；整盘重锚必须更保守。小程序的抬子、落子
        # 和吃子动画可能连续维持两三帧，过早重锚会同时产生丢子和幽灵棋子。
        self.resync_frames = 8 if fast_exact_source else 5
        self.resync_seconds = 1.0 if fast_exact_source else 1.2
        # 首帧可能恰好处于抬子/落子动画，不能把一批暂时消失的棋子
        # 当成初始稳定盘面展示。正常运行中的走子仍使用快速双帧确认。
        self.initial_frames = 8 if fast_exact_source else 3
        self.initial_seconds = 1.0 if fast_exact_source else 0.35

    def reset(self) -> None:
        """切换游戏平台或捕获源时清空盘面，但保留递增的会话编号。"""
        self.last_stable_fen = None
        self.last_stable_board = None
        self.move_history = []
        self._undo_positions = []
        self.last_rejection_reason = None
        self.active_side = "r"
        self._hint_signature = None
        self._hint_count = 0
        self._candidate = [[None] * 9 for _ in range(10)]
        self._candidate_count = [[0] * 9 for _ in range(10)]
        self._candidate_since = [[0.0] * 9 for _ in range(10)]
        self._reset_resync_candidate()
        self._source_discontinuity_count = 0
        self._source_discontinuity = False
        self._force_reanchor = False

    def request_reanchor(self) -> None:
        """保留当前稳定盘面，只把下一张完整盘面作为新的原子锚点。"""
        if self.last_stable_board is None:
            self.reset()
            return
        self._force_reanchor = True
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self.last_rejection_reason = "正在后台校验新的完整盘面"

    def cancel_reanchor(self) -> None:
        """重读超时时恢复原锚点，避免界面无限停在等待状态。"""
        self._force_reanchor = False
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self.last_rejection_reason = None

    def note_unstable_source(self, piece_count: int = 0) -> None:
        """Remember a real menu/transition gap without publishing an empty board."""
        if piece_count <= 2:
            self._source_discontinuity_count += 1
            if self._source_discontinuity_count >= 2:
                self._source_discontinuity = True

    def _clear_source_discontinuity(self) -> None:
        self._source_discontinuity_count = 0
        self._source_discontinuity = False

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

    def _reconcile_occupancy(
        self,
        observed: Board,
        occupancy: Optional[List[List[bool]]],
        occupied_sides: Optional[List[List[Optional[str]]]],
    ) -> Board:
        """Infer a highlighted destination from occupancy and a legal source.

        Exact glyph recognition is still used for initial positions.  During a
        live move, however, the piece identity is already known at its source;
        requiring the halo-covered destination glyph to be perfect only creates
        lag.  A unique legal source/destination pair is safe to reconcile.
        """
        if self.last_stable_board is None or occupancy is None:
            return observed
        if len(occupancy) != 10 or any(len(row) != 9 for row in occupancy):
            return observed

        reconciled = self._copy_board(observed)
        sides = occupied_sides or [[None] * 9 for _ in range(10)]

        # An occupied cell whose glyph is temporarily unreadable has not lost
        # its stable piece.  A same-side piece also cannot legally change type
        # in place, so a transient 象→炮 / 马→车 classification must keep the
        # stable identity.  Opposite-side occupancy is retained as a possible
        # capture destination and is handled by legal move pairing below.
        for row in range(10):
            for col in range(9):
                stable = self.last_stable_board[row][col]
                side = sides[row][col] if row < len(sides) and col < len(sides[row]) else None
                if (
                    occupancy[row][col]
                    and stable is not None
                    and (
                        reconciled[row][col] is None
                        or side == stable[0]
                    )
                ):
                    reconciled[row][col] = stable

        departures = [
            (col, row, self.last_stable_board[row][col])
            for row in range(10) for col in range(9)
            if self.last_stable_board[row][col] is not None and not occupancy[row][col]
        ]
        arrivals = []
        for row in range(10):
            for col in range(9):
                if not occupancy[row][col]:
                    continue
                stable = self.last_stable_board[row][col]
                side = sides[row][col] if row < len(sides) and col < len(sides[row]) else None
                if (
                    stable is None
                    or (side in ("r", "b") and stable[0] != side)
                    or (reconciled[row][col] is not None and reconciled[row][col] != stable)
                ):
                    arrivals.append((col, row, side))

        legal_pairs = []
        for from_col, from_row, piece in departures:
            if piece is None:
                continue
            for to_col, to_row, side in arrivals:
                if side in ("r", "b") and piece[0] != side:
                    continue
                if self._is_legal_piece_move(
                    self.last_stable_board,
                    from_col, from_row, to_col, to_row, piece,
                ):
                    legal_pairs.append((from_col, from_row, to_col, to_row, piece))

        if len(legal_pairs) == 1:
            from_col, from_row, to_col, to_row, piece = legal_pairs[0]
            reconciled[from_row][from_col] = None
            reconciled[to_row][to_col] = piece
        return reconciled

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
        return is_structurally_valid_board(board)

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

    @classmethod
    def _standard_start_from_observation(
        cls,
        observed: Board,
        occupancy: Optional[List[List[bool]]],
        occupied_sides: Optional[List[List[Optional[str]]]],
    ) -> Optional[Board]:
        """Return an exact standard start when the frame is an unambiguous subset.

        New games have a known state, so a selected/highlighted source square must
        not make cold start permanently omit that piece.  Every recognized piece
        and every positive occupancy observation must agree with one orientation;
        at least 28 occupied/recognized cells are required to avoid mistaking a
        later sparse position for a fresh game.
        """
        standards = (
            fen_to_matrix(START_FEN)[0],
            fen_to_matrix(FLIPPED_START_FEN)[0],
        )
        occupancy_valid = (
            isinstance(occupancy, list)
            and len(occupancy) == 10
            and all(isinstance(row, list) and len(row) == 9 for row in occupancy)
        )
        sides = occupied_sides if isinstance(occupied_sides, list) else None
        for standard in standards:
            evidence = 0
            valid = True
            for row in range(10):
                for col in range(9):
                    expected = standard[row][col]
                    seen = observed[row][col]
                    if seen is not None:
                        evidence += 1
                        if seen != expected:
                            valid = False
                            break
                    if occupancy_valid and occupancy[row][col]:
                        evidence += 1 if seen is None else 0
                        if expected is None:
                            valid = False
                            break
                        side = None
                        if (
                            sides
                            and row < len(sides)
                            and isinstance(sides[row], list)
                            and col < len(sides[row])
                        ):
                            side = sides[row][col]
                        if side in ("r", "b") and expected[0] != side:
                            valid = False
                            break
                if not valid:
                    break
            if valid and evidence >= 28:
                return cls._copy_board(standard)
        return None

    def _should_resync(self, observed: Board, now: float) -> bool:
        if (
            self._resync_signature != self._signature(observed)
            or self._resync_count < self.resync_frames
            or now - self._resync_since < self.resync_seconds
            or not self._is_plausible_board(observed)
        ):
            return False
        assert self.last_stable_board is not None
        # A menu/empty transition is not proof that the next arbitrary board is
        # a different game: capture interruptions also produce empty frames.
        # Only the known 32-piece opening can be adopted automatically.  A
        # midgame restart needs an explicit manual re-read.
        return self._initial_side(observed) == "r"

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

    @staticmethod
    def _mismatch_count(left: Board, right: Board) -> int:
        return sum(
            left[row][col] != right[row][col]
            for row in range(10) for col in range(9)
        )

    @staticmethod
    def _inventory(board: Board) -> Counter:
        return Counter(piece for row in board for piece in row if piece)

    def _remember_undo_position(self, board: Board, mover: str) -> None:
        self._undo_positions.append((self._copy_board(board), mover))
        if len(self._undo_positions) > 24:
            self._undo_positions.pop(0)

    def _find_undo_transition(
        self, observed: Board, confirmed_side: Optional[str],
        visual: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, str, Optional[Move]]]:
        """Accept only an exact recorded rollback or a proven alternative move."""
        for index in range(len(self._undo_positions) - 1, -1, -1):
            historical, mover = self._undo_positions[index]
            if historical == observed:
                return index, mover, None

        # If the other player replays before a capture frame shows the reverted
        # position, the last visual marker must identify the alternative move.
        if not isinstance(visual, dict):
            return None
        source, target = visual.get("from") or {}, visual.get("to") or {}
        try:
            sr, sc = int(source["row"]), int(source["col"])
            dr, dc = int(target["row"]), int(target["col"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= dr < 10 and 0 <= dc < 9):
            return None
        matches = []
        for index, (historical, mover) in enumerate(self._undo_positions):
            piece = historical[sr][sc]
            if (
                piece is None or piece[0] != mover
                or visual.get("side") != mover
                or visual.get("piece") != piece
                or confirmed_side != mover
            ):
                continue
            move = Move(sr, sc, dr, dc, piece, historical[dr][dc])
            if not is_legal_move(historical, move, mover):
                continue
            if apply_move(historical, move) == observed:
                matches.append((index, other_side(mover), move))
        return matches[0] if len(matches) == 1 else None

    def _commit_undo_transition(
        self, observed: Board, transition: Tuple[int, str, Optional[Move]],
    ) -> Dict[str, Any]:
        index, next_side, move = transition
        old_fen = self.last_stable_fen
        historical = self._copy_board(self._undo_positions[index][0])
        self._undo_positions = self._undo_positions[:index]
        self.move_history = []
        if move is not None:
            self._remember_undo_position(historical, move.piece[0])
        self.last_stable_board = self._copy_board(observed)
        self.active_side = next_side
        self.last_stable_fen = matrix_to_fen(
            observed, "b" if next_side == "b" else "w",
        )
        self.session_revision += 1
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self._clear_source_discontinuity()
        self.last_rejection_reason = None
        payload = move.to_dict() if move is not None else None
        description = "检测到悔棋后改走，已按历史盘面及合法走子同步" if move else "检测到悔棋，已回到记录过的盘面"
        event = {
            "event_type": "undo_branch" if move else "undo",
            "fen": self.last_stable_fen,
            "old_fen": old_fen,
            "timestamp": time.time(),
            "session_revision": self.session_revision,
            "move": payload,
            "description": description,
        }
        if self.on_change_callback:
            self.on_change_callback(event)
        return event

    def _find_catchup_path(self, observed: Board, max_plies: int = 4,
                           last_move_side: Optional[str] = None):
        """Find a proven 1..N-ply path from the stable board to the observation.

        Search is beam-bounded and ordered by distance to the observed board.
        Importantly, a board is committed only on an *exact* legal match; beam
        scoring can affect latency but can never authorize an invalid state.
        """
        if self.last_stable_board is None:
            return None
        before_inventory = self._inventory(self.last_stable_board)
        target_inventory = self._inventory(observed)
        if any(
            count > before_inventory.get(piece, 0)
            for piece, count in target_inventory.items()
        ):
            return None
        capture_delta = sum(before_inventory.values()) - sum(target_inventory.values())
        if capture_delta < 0 or capture_delta > max_plies:
            return None
        changed_cells = {
            (row, col)
            for row in range(10) for col in range(9)
            if self.last_stable_board[row][col] != observed[row][col]
        }
        # A visual last-mover hint identifies a side, not the number of plies.
        # A two-cell OCR error must never be rationalized as a four-ply path
        # padded with reversible moves that leave no trace in the final frame.
        max_plies = min(max_plies, (len(changed_cells) + 1) // 2)
        if max_plies < 1:
            return None

        if self.active_side in ("r", "b"):
            starts = [self.active_side]
            # 只有画面中的稳定落点明确表明回合锁已漂移时，才尝试另一方；
            # 否则不能让“同一方连续走两次”的异常盘面绕过回合保护。
            if last_move_side in ("r", "b") and last_move_side != self.active_side:
                starts.append(other_side(self.active_side))
        else:
            starts = ["r", "b"]
        bottom = infer_bottom_side(self.last_stable_board)
        target_signature = self._signature(observed)
        beam_width = 192

        def search_from(start_side: str):
            # One node is (board, side_to_move, path, captures_so_far).
            initial = self._copy_board(self.last_stable_board)
            frontier = [(initial, start_side, [], 0)]
            seen = {(self._signature(initial), start_side, 0)}
            for depth in range(1, max_plies + 1):
                remaining = max_plies - depth
                next_nodes = []
                matches = []
                for board, side, path, captures in frontier:
                    for move in generate_legal_moves(board, side, bottom):
                        if (
                            (move.sr, move.sc) not in changed_cells
                            and (move.dr, move.dc) not in changed_cells
                        ):
                            continue
                        after = apply_move(board, move)
                        next_captures = captures + (1 if move.captured else 0)
                        if next_captures > capture_delta:
                            continue
                        inventory = self._inventory(after)
                        if any(
                            inventory.get(piece, 0) < count
                            for piece, count in target_inventory.items()
                        ):
                            continue
                        captures_left = (
                            sum(inventory.values()) - sum(target_inventory.values())
                        )
                        if captures_left < 0 or captures_left > remaining:
                            continue
                        next_side = other_side(side)
                        next_path = path + [move]
                        signature = self._signature(after)
                        if signature == target_signature:
                            if (
                                last_move_side not in ("r", "b")
                                or move.piece[0] == last_move_side
                            ):
                                matches.append((next_path, next_side))
                            continue

                        mismatches = self._mismatch_count(after, observed)
                        # Every inferred ply must leave evidence in a cell that
                        # differs in the final observed position.
                        if (mismatches + 1) // 2 > remaining:
                            continue
                        key = (signature, next_side, depth)
                        if key in seen:
                            continue
                        seen.add(key)
                        next_nodes.append((
                            mismatches,
                            captures_left,
                            after,
                            next_side,
                            next_path,
                            next_captures,
                        ))

                if matches:
                    resulting_sides = {next_side for _, next_side in matches}
                    if len(resulting_sides) == 1:
                        return matches[0]
                    return None
                next_nodes.sort(key=lambda item: (item[0], item[1], len(item[4])))
                frontier = [
                    (board, side, path, captures)
                    for _, _, board, side, path, captures in next_nodes[:beam_width]
                ]
                if not frontier:
                    break
            return None

        matches = []
        for start_side in starts:
            match = search_from(start_side)
            if match is not None:
                if self.active_side in ("r", "b"):
                    return match
                matches.append(match)
        if matches:
            resulting_sides = {next_side for _, next_side in matches}
            if len(resulting_sides) == 1:
                return matches[0]
        return None

    def _commit_catchup(self, observed: Board, path, next_side: str) -> Dict[str, Any]:
        old_fen = self.last_stable_fen
        board = self._copy_board(self.last_stable_board)
        payloads = []
        for move in path:
            self._remember_undo_position(board, move.piece[0])
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
        self._clear_source_discontinuity()
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
               last_move_side: Optional[str] = None,
               occupancy: Optional[List[List[bool]]] = None,
               occupied_sides: Optional[List[List[Optional[str]]]] = None,
               last_visual_move: Optional[Dict[str, Any]] = None,
               complete_observation: bool = True,
               ) -> Optional[Dict[str, Any]]:
        observed = self._board_from_input(current_fen, current_board)
        if self.last_stable_board is None or self._force_reanchor:
            standard = self._standard_start_from_observation(
                observed, occupancy, occupied_sides,
            )
            if standard is not None:
                observed = standard
        if not self._force_reanchor:
            observed = self._reconcile_occupancy(observed, occupancy, occupied_sides)
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
                self._reset_resync_candidate()
                self.last_rejection_reason = "等待包含双方将帅的有效棋局盘面"
                return None
            self._track_resync_candidate(observed, now)
            if (
                self._resync_count < self.initial_frames
                or now - self._resync_since < self.initial_seconds
            ):
                self.last_rejection_reason = "等待完整盘面连续稳定"
                return None
            self.last_stable_board = self._copy_board(observed)
            self.active_side = self._initial_side(self.last_stable_board)
            self.last_stable_fen = matrix_to_fen(
                self.last_stable_board, "b" if self.active_side == "b" else "w"
            )
            self.session_revision += 1
            self._undo_positions = []
            self._reset_all_candidates()
            self._reset_resync_candidate()
            self._clear_source_discontinuity()
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

        # 菜单、转场或动画导致的不完整盘面只用于标记捕获源中断，绝不参与
        # 逐格候选和整盘重锚。这样 direct/unit 调用与服务端过滤路径行为一致。
        if not self._is_plausible_board(observed):
            self.note_unstable_source(sum(
                piece is not None for row in observed for piece in row
            ))
            self._reset_resync_candidate()
            self.last_rejection_reason = "等待包含双方将帅的有效棋局盘面"
            return None

        # 手动重读使用独立的双帧候选，连“新盘面与旧盘面完全相同”的情况
        # 也能正常完成。普通重同步会忽略相同盘面，不能复用它的跟踪入口。
        if self._force_reanchor:
            if observed != self.last_stable_board and self._initial_side(observed) != "r":
                departures = any(
                    self.last_stable_board[row][col] is not None
                    and observed[row][col] is None
                    for row in range(10) for col in range(9)
                )
                arrivals = any(
                    observed[row][col] is not None
                    and observed[row][col] != self.last_stable_board[row][col]
                    for row in range(10) for col in range(9)
                )
                if departures and not arrivals:
                    self._reset_resync_candidate()
                    self.last_rejection_reason = (
                        "重读画面只有孤立少子，已保留上一正确盘面"
                    )
                    return None
            signature = self._signature(observed)
            if signature == self._resync_signature:
                self._resync_count += 1
            else:
                self._resync_signature = signature
                self._resync_count = 1
                self._resync_since = now
            reanchor_frames = max(self.required_stable_frames, self.initial_frames)
            reanchor_seconds = max(self.min_stable_seconds, self.initial_seconds)
            if (
                self._resync_count >= reanchor_frames
                and now - self._resync_since >= reanchor_seconds
            ):
                return self._commit_manual_reanchor(observed, confirmed_side)
            self.last_rejection_reason = "正在后台校验新的完整盘面"
            return None

        # A reverted rook/horse/general position can itself look like a legal
        # reverse move.  If it exactly matches a recorded pre-move board and
        # the current turn belongs to the other side, wait for the stronger
        # undo confirmation instead of committing that inverse move early.
        undo_candidate = complete_observation and any(
            historical == observed and mover != self.active_side
            for historical, mover in self._undo_positions
        )

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

        # 天天象棋会同时给出来源白圈和目标灰环。若这两个坐标、棋子身份、
        # 完整盘面差分及象棋走法四者完全一致，就无需再等第二个轮询周期；
        # 这能让带高亮的马/将/帅立即跟上，同时不会把单帧 OCR 猜测写入盘面。
        visual = last_visual_move if isinstance(last_visual_move, dict) else None
        if visual and not undo_candidate:
            source = visual.get("from") or {}
            target = visual.get("to") or {}
            piece = visual.get("piece")
            try:
                from_row, from_col = int(source["row"]), int(source["col"])
                to_row, to_col = int(target["row"]), int(target["col"])
            except (KeyError, TypeError, ValueError):
                from_row = from_col = to_row = to_col = -1
            if (
                isinstance(piece, str)
                and len(piece) >= 3
                and 0 <= from_row < 10 and 0 <= from_col < 9
                and 0 <= to_row < 10 and 0 <= to_col < 9
                and self.last_stable_board[from_row][from_col] == piece
                and observed[from_row][from_col] is None
                and observed[to_row][to_col] == piece
                and visual.get("side") == piece[0]
                and self._is_legal_piece_move(
                    self.last_stable_board,
                    from_col, from_row, to_col, to_row, piece,
                )
            ):
                expected = self._copy_board(self.last_stable_board)
                expected[from_row][from_col] = None
                expected[to_row][to_col] = piece
                if expected == observed:
                    return self._commit_move(
                        from_col, from_row, to_col, to_row, piece,
                    )

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
                if self._is_legal_piece_move(
                    self.last_stable_board, from_col, from_row, to_col, to_row, piece
                ):
                    score = (
                        self._candidate_count[from_row][from_col]
                        + self._candidate_count[to_row][to_col]
                    )
                    legal_moves.append((score, from_col, from_row, to_col, to_row, piece))

        if (
            legal_moves and not undo_candidate and differences == 2
            and len(departures) + len(destinations) == 2
        ):
            # 单步最多涉及来源与目标两个格。多个候选时只接受证据最强且唯一
            # 的配对，避免特效误分类造成跳子。
            if self.active_side in ("r", "b"):
                on_turn = [move for move in legal_moves if move[5][0] == self.active_side]
                hinted = [move for move in legal_moves if move[5][0] == confirmed_side]
                legal_moves = on_turn or hinted
            legal_moves.sort(reverse=True)
            if legal_moves and (len(legal_moves) == 1 or legal_moves[0][0] > legal_moves[1][0]):
                _, from_col, from_row, to_col, to_row, piece = legal_moves[0]
                return self._commit_move(from_col, from_row, to_col, to_row, piece)
            self.last_rejection_reason = "存在多个同强度走子候选，等待整盘稳定或消除歧义"

        # 捕获线程偶尔可能直接从上一步跳到数步后的盘面。使用上一稳定布局
        # 枚举合法路径，只有精确到达当前观测时才一次性追上；不再使用未经
        # 走法证明的整盘覆盖。
        if (
            not undo_candidate
            and self._resync_signature == self._signature(observed)
            and self._resync_count >= self.required_stable_frames
            and now - self._resync_since >= self.min_stable_seconds
        ):
            differences = sum(
                observed[row][col] != self.last_stable_board[row][col]
                for row in range(10) for col in range(9)
            )
            if 2 <= differences <= 8 and self._is_plausible_board(observed):
                max_catchup_plies = 4 if confirmed_side in ("r", "b") else 2
                catchup = self._find_catchup_path(
                    observed,
                    max_plies=max_catchup_plies,
                    last_move_side=confirmed_side,
                )
                if catchup is not None:
                    path, next_side = catchup
                    return self._commit_catchup(observed, path, next_side)

        if (
            complete_observation
            and self._resync_signature == self._signature(observed)
            and self._resync_count >= self.resync_frames
            and now - self._resync_since >= self.resync_seconds
        ):
            transition = self._find_undo_transition(
                observed, confirmed_side, visual,
            )
            if transition is not None:
                return self._commit_undo_transition(observed, transition)

        # 2. 新开一局、退出后重开通常会同时改变多个格子。只有完整盘面在
        # 连续帧和时间上都稳定后才整盘切换，菜单/空画面不会清空当前盘面。
        if self._should_resync(observed, now):
            return self._commit_resync(observed, confirmed_side)

        if departures or destinations:
            self.last_rejection_reason = "画面变化尚未组成唯一合法走子，已保留上一正确盘面"
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
        self._undo_positions = []
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self._clear_source_discontinuity()
        self._force_reanchor = False
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

    def _commit_manual_reanchor(
        self, observed: Board, last_move_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        old_fen = self.last_stable_fen
        same_board = observed == self.last_stable_board
        previous_side = self.active_side
        historical_side = next((
            mover for board, mover in reversed(self._undo_positions)
            if board == observed
        ), None)
        self.last_stable_board = self._copy_board(observed)
        inferred_side = self._initial_side(self.last_stable_board)
        if historical_side in ("r", "b"):
            inferred_side = historical_side
        elif inferred_side == "unknown" and last_move_side in ("r", "b"):
            inferred_side = other_side(last_move_side)
        elif inferred_side == "unknown" and same_board:
            inferred_side = previous_side
        self.active_side = inferred_side
        self.last_stable_fen = matrix_to_fen(
            self.last_stable_board, "b" if self.active_side == "b" else "w"
        )
        self.session_revision += 1
        self.move_history = []
        self._undo_positions = []
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self._clear_source_discontinuity()
        self._force_reanchor = False
        self.last_rejection_reason = None
        event = {
            "event_type": "manual_reanchor",
            "fen": self.last_stable_fen,
            "old_fen": old_fen,
            "timestamp": time.time(),
            "move": None,
            "session_revision": self.session_revision,
            "description": "已清理识别缓存并重新读取当前对局",
        }
        if self.on_change_callback:
            self.on_change_callback(event)
        return event

    def _commit_move(self, from_col: int, from_row: int,
                     to_col: int, to_row: int, piece: str) -> Dict[str, Any]:
        assert self.last_stable_board is not None
        old_fen = self.last_stable_fen
        captured = self.last_stable_board[to_row][to_col]
        self._remember_undo_position(self.last_stable_board, piece[0])
        self.last_stable_board[from_row][from_col] = None
        self.last_stable_board[to_row][to_col] = piece
        self.active_side = other_side(piece[0])
        self.last_stable_fen = matrix_to_fen(
            self.last_stable_board, "w" if self.active_side == "r" else "b"
        )
        self._reset_all_candidates()
        self._reset_resync_candidate()
        self._clear_source_discontinuity()
        self._force_reanchor = False
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
        if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= dr < 10 and 0 <= dc < 9):
            return False
        move = Move(sr, sc, dr, dc, piece, board[dr][dc])
        return is_legal_move(board, move, piece[0])

    def get_stable_state(self) -> Tuple[Optional[str], Optional[Board]]:
        return self.last_stable_fen, self._copy_board(self.last_stable_board) if self.last_stable_board else None
