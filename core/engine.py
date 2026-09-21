"""内置中国象棋搜索引擎（α-β + 迭代加深）。

作为 Pikafish 缺失或异常时的轻量回退，不依赖外部二进制。
评估与走法都以「画面下方」为己方底线，红黑均可执下。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.fen import matrix_to_fen
from core.xiangqi import (
    Board,
    Move,
    apply_move,
    generate_legal_moves,
    generate_pseudo_moves,
    infer_bottom_side,
    is_in_check,
    move_payload,
    other_side,
    parse_uci,
)

INF = 100000
MATE = 90000
MATERIAL = {"k": 10000, "r": 980, "c": 440, "n": 400, "b": 200, "a": 200, "p": 100}

# 以「在下方的一方」为视角的位置分。上方棋子用垂直镜像。
PAWN_PST = [
    [  9,  9,  9, 11, 13, 11,  9,  9,  9],
    [ 39, 59, 69, 79, 79, 79, 69, 59, 39],
    [ 39, 59, 69, 79, 79, 79, 69, 59, 39],
    [ 29, 48, 58, 68, 68, 68, 58, 48, 29],
    [ 11, 29, 39, 49, 49, 49, 39, 29, 11],
    [  0,  0,  8, 12, 15, 12,  8,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
]
HORSE_PST = [
    [  4,  8, 16, 12,  4, 12, 16,  8,  4],
    [  4, 10, 28, 16,  8, 16, 28, 10,  4],
    [ 12, 14, 16, 20, 18, 20, 16, 14, 12],
    [  8, 16, 22, 18, 16, 18, 22, 16,  8],
    [  4, 10, 14, 16, 21, 16, 14, 10,  4],
    [  4,  8, 12, 14, 16, 14, 12,  8,  4],
    [  0,  6,  8, 10, 12, 10,  8,  6,  0],
    [  0,  2,  4,  6,  8,  6,  4,  2,  0],
    [ -4,  0,  2,  0,  2,  0,  2,  0, -4],
    [ -8, -4,  0,  0,  0,  0,  0, -4, -8],
]
ROOK_PST = [
    [ 14, 14, 12, 18, 16, 18, 12, 14, 14],
    [ 16, 20, 18, 24, 26, 24, 18, 20, 16],
    [ 12, 12, 12, 18, 18, 18, 12, 12, 12],
    [ 12, 18, 16, 22, 22, 22, 16, 18, 12],
    [ 12, 14, 14, 18, 18, 18, 14, 14, 12],
    [  8, 10, 10, 14, 14, 14, 10, 10,  8],
    [  6,  8,  8, 12, 12, 12,  8,  8,  6],
    [  4,  6,  6, 10, 10, 10,  6,  6,  4],
    [  8,  8,  6, 10,  8, 10,  6,  8,  8],
    [  6,  8,  4, 14, 12, 14,  4,  8,  6],
]
CANNON_PST = [
    [  6,  6,  4,  6,  8,  6,  4,  6,  6],
    [  2,  2,  0,  4,  4,  4,  0,  2,  2],
    [  4,  4,  6,  8, 10,  8,  6,  4,  4],
    [  4,  6,  6,  8, 10,  8,  6,  6,  4],
    [  6,  8,  8, 10, 12, 10,  8,  8,  6],
    [  8,  8,  6,  8, 10,  8,  6,  8,  8],
    [ 10,  8,  8,  6,  6,  6,  8,  8, 10],
    [ 12, 10,  4,  0,  0,  0,  4, 10, 12],
    [  6,  4,  2,  0,  0,  0,  2,  4,  6],
    [  6,  4,  8,  0,  0,  0,  8,  4,  6],
]
KING_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  1,  2,  1,  0,  0,  0],
    [  0,  0,  0,  3,  3,  3,  0,  0,  0],
    [  0,  0,  0,  4,  8,  4,  0,  0,  0],
]
ADVISOR_PST = [
    [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9,
    [0] * 9, [0] * 9, [0] * 9,
    [0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 4, 0, 0, 0],
]
ELEPHANT_PST = [
    [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9,
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0] * 9,
    [0, 0, 4, 0, 0, 0, 4, 0, 0],
    [0] * 9,
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
]
PST = {
    "p": PAWN_PST, "n": HORSE_PST, "r": ROOK_PST, "c": CANNON_PST,
    "k": KING_PST, "a": ADVISOR_PST, "b": ELEPHANT_PST,
}

# 开局着法用视觉坐标，红在下与黑在下各写一套（都是「己方炮二平五」等）。
_BOOK: Dict[str, Sequence[str]] = {
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR": (
        "h2e2", "b2e2", "h0g2", "b0c2", "g0e2", "b0a2",
    ),
    "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr": (
        "h2e2", "b2e2", "h0g2", "b0c2", "g0e2", "b0a2",
    ),
}


@dataclass
class SearchResult:
    move: Optional[Move]
    score: int
    depth: int
    nodes: int
    time_ms: int
    pv: List[Move] = field(default_factory=list)
    book: bool = False
    mate: bool = False
    reason: str = ""
    engine: str = "builtin"

    def to_dict(self, board: Optional[Board] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "score": self.score,
            "depth": self.depth,
            "nodes": self.nodes,
            "time_ms": self.time_ms,
            "book": self.book,
            "mate": self.mate,
            "reason": self.reason,
            "engine": self.engine,
            "uci": self.move.uci if self.move else None,
            "pv": [item.uci for item in self.pv],
        }
        if self.move is not None and board is not None:
            payload.update(move_payload(board, self.move))
        elif self.move is not None:
            payload.update(self.move.to_dict())
        return payload


class XiangqiEngine:
    def __init__(self) -> None:
        self.nodes = 0
        self.stop = False
        self._deadline = 0.0
        self._killers: List[List[Optional[Move]]] = [[None, None] for _ in range(64)]
        self._rng = random.Random(7)

    def request_stop(self) -> None:
        self.stop = True

    def _timed_out(self) -> bool:
        return self.stop or time.perf_counter() >= self._deadline

    def search(self, board: Board, side: str, time_ms: int = 700,
               max_depth: int = 8) -> SearchResult:
        self.nodes = 0
        self.stop = False
        self._deadline = time.perf_counter() + max(0.05, time_ms / 1000.0)
        self._killers = [[None, None] for _ in range(64)]
        bottom = infer_bottom_side(board)
        started = time.perf_counter()

        book_move = self._probe_book(board, side)
        if book_move is not None:
            return SearchResult(
                move=book_move, score=25, depth=0, nodes=1,
                time_ms=int((time.perf_counter() - started) * 1000),
                pv=[book_move], book=True, reason="开局库",
            )

        moves = generate_legal_moves(board, side, bottom)
        if not moves:
            in_check = is_in_check(board, side, bottom)
            return SearchResult(
                move=None, score=-MATE, depth=0, nodes=1,
                time_ms=int((time.perf_counter() - started) * 1000),
                mate=True, reason="绝杀" if in_check else "困毙",
            )

        best_move = moves[0]
        best_score = -INF
        best_pv: List[Move] = [best_move]
        completed_depth = 0
        for depth in range(1, max_depth + 1):
            if self._timed_out():
                break
            score, move, pv = self._search_root(board, side, bottom, depth, moves)
            if self._timed_out() and completed_depth:
                break
            if move is None:
                break
            best_move, best_score, best_pv = move, score, pv
            completed_depth = depth
            if abs(score) >= MATE - 100:
                break
            # 下一层把当前最佳着放到最前。
            moves = [move] + [item for item in moves if item.uci != move.uci]

        elapsed = int((time.perf_counter() - started) * 1000)
        return SearchResult(
            move=best_move, score=best_score, depth=completed_depth,
            nodes=self.nodes, time_ms=elapsed, pv=best_pv,
            mate=abs(best_score) >= MATE - 100,
            reason="搜索完成",
        )

    def _probe_book(self, board: Board, side: str) -> Optional[Move]:
        fen = matrix_to_fen(board, "w" if side == "r" else "b")
        key = fen.split()[0]
        choices = _BOOK.get(key)
        if not choices:
            return None
        uci = choices[0] if len(choices) == 1 else self._rng.choice(list(choices[:3]))
        move = parse_uci(board, uci)
        if move is None:
            return None
        legal = generate_legal_moves(board, side)
        return move if any(item.uci == move.uci for item in legal) else None

    def _search_root(self, board: Board, side: str, bottom: str, depth: int,
                     moves: Sequence[Move]) -> Tuple[int, Optional[Move], List[Move]]:
        alpha = -INF
        beta = INF
        best_move: Optional[Move] = None
        best_pv: List[Move] = []
        ordered = self._order_moves(board, list(moves), None, 0)
        for move in ordered:
            if self._timed_out():
                break
            child = apply_move(board, move)
            score, pv = self._alphabeta(child, other_side(side), bottom, depth - 1,
                                        -beta, -alpha, 1)
            score = -score
            if score > alpha:
                alpha = score
                best_move = move
                best_pv = [move] + pv
        return alpha, best_move, best_pv

    def _alphabeta(self, board: Board, side: str, bottom: str, depth: int,
                   alpha: int, beta: int, ply: int) -> Tuple[int, List[Move]]:
        self.nodes += 1
        if self._timed_out():
            return evaluate(board, side, bottom), []
        if depth <= 0:
            return self._quiesce(board, side, bottom, alpha, beta, 4), []

        moves = generate_legal_moves(board, side, bottom)
        if not moves:
            return (-MATE + ply, []) if is_in_check(board, side, bottom) else (-MATE + ply, [])

        best_pv: List[Move] = []
        killer = self._killers[ply] if ply < len(self._killers) else [None, None]
        for move in self._order_moves(board, moves, killer, ply):
            if self._timed_out():
                break
            child = apply_move(board, move)
            score, pv = self._alphabeta(child, other_side(side), bottom, depth - 1,
                                        -beta, -alpha, ply + 1)
            score = -score
            if score >= beta:
                if move.captured is None and ply < len(self._killers):
                    self._killers[ply][1] = self._killers[ply][0]
                    self._killers[ply][0] = move
                return score, [move] + pv
            if score > alpha:
                alpha = score
                best_pv = [move] + pv
        return alpha, best_pv

    def _quiesce(self, board: Board, side: str, bottom: str,
                 alpha: int, beta: int, qply: int) -> int:
        self.nodes += 1
        stand = evaluate(board, side, bottom)
        if qply <= 0 or self._timed_out():
            return stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        captures = [move for move in generate_pseudo_moves(board, side, bottom) if move.captured]
        captures.sort(key=lambda move: MATERIAL.get(move.captured[2], 0) * 10
                      - MATERIAL.get(move.piece[2], 0), reverse=True)
        for move in captures:
            child = apply_move(board, move)
            if is_in_check(child, side, bottom):
                continue
            score = -self._quiesce(child, other_side(side), bottom, -beta, -alpha, qply - 1)
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    @staticmethod
    def _order_moves(board: Board, moves: List[Move],
                     killers: Optional[Sequence[Optional[Move]]], ply: int) -> List[Move]:
        def key(move: Move) -> Tuple[int, int, int]:
            capture = MATERIAL.get(move.captured[2], 0) if move.captured else 0
            victim = capture * 16 - MATERIAL.get(move.piece[2], 0)
            is_killer = 0
            if killers:
                if killers[0] is not None and move.uci == killers[0].uci:
                    is_killer = 2
                elif killers[1] is not None and move.uci == killers[1].uci:
                    is_killer = 1
            center = -abs(move.dc - 4) - abs(move.dr - 4)
            return (1 if capture else 0, is_killer, victim if capture else center)
        return sorted(moves, key=key, reverse=True)


def evaluate(board: Board, side: str, bottom: Optional[str] = None) -> int:
    if bottom is None:
        bottom = infer_bottom_side(board)
    total = 0
    for row_idx, row in enumerate(board):
        for col_idx, piece in enumerate(row):
            if piece is None:
                continue
            value = MATERIAL[piece[2]] + _pst(piece, row_idx, col_idx, bottom)
            total += value if piece[0] == side else -value
    if is_in_check(board, other_side(side), bottom):
        total += 18
    if is_in_check(board, side, bottom):
        total -= 18
    return total


def _pst(piece: str, row: int, col: int, bottom: str) -> int:
    table = PST.get(piece[2])
    if not table:
        return 0
    if piece[0] == bottom:
        return table[row][col]
    return table[9 - row][col]
