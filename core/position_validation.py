"""Cheap structural validation for recognized Xiangqi positions.

This is deliberately stricter than a generic FEN parser.  Vision may produce a
syntactically valid board while the whole lattice is shifted by one file/rank;
kings, advisors, elephants and unadvanced pawns make that error detectable
without running an engine.
"""

from typing import Dict, List, Optional, Tuple


Board = List[List[Optional[str]]]

_TOP_ADVISOR_POINTS = {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}
_BOTTOM_ADVISOR_POINTS = {(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)}
_TOP_ELEPHANT_POINTS = {
    (0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6),
}
_BOTTOM_ELEPHANT_POINTS = {
    (5, 2), (5, 6), (7, 0), (7, 4), (7, 8), (9, 2), (9, 6),
}
_PIECE_LIMITS = {"k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5}


def is_structurally_valid_board(board: Board) -> bool:
    """Return whether recognized pieces can occupy these absolute board cells.

    Missing non-king pieces are allowed because highlights can temporarily hide
    ink.  Impossible coordinates and excess pieces are rejected, which prevents
    an offset lattice from becoming the displayed stable position.
    """
    if len(board) != 10 or any(len(row) != 9 for row in board):
        return False

    locations: Dict[str, List[Tuple[int, int]]] = {}
    for row, cells in enumerate(board):
        for col, piece in enumerate(cells):
            if piece is None:
                continue
            if (
                len(piece) != 3 or piece[1] != "_"
                or piece[0] not in ("r", "b") or piece[2] not in _PIECE_LIMITS
            ):
                return False
            locations.setdefault(piece, []).append((row, col))

    for piece, cells in locations.items():
        if len(cells) > _PIECE_LIMITS[piece[2]]:
            return False
    if len(locations.get("r_k", [])) != 1 or len(locations.get("b_k", [])) != 1:
        return False

    red_king = locations["r_k"][0]
    black_king = locations["b_k"][0]
    if red_king[0] <= 2 and black_king[0] >= 7:
        top_side, bottom_side = "r", "b"
    elif black_king[0] <= 2 and red_king[0] >= 7:
        top_side, bottom_side = "b", "r"
    else:
        return False
    if not (3 <= red_king[1] <= 5 and 3 <= black_king[1] <= 5):
        return False

    for side, advisor_points, elephant_points, pawn_range in (
        (top_side, _TOP_ADVISOR_POINTS, _TOP_ELEPHANT_POINTS, range(3, 10)),
        (bottom_side, _BOTTOM_ADVISOR_POINTS, _BOTTOM_ELEPHANT_POINTS, range(0, 7)),
    ):
        if any(cell not in advisor_points for cell in locations.get(f"{side}_a", [])):
            return False
        if any(cell not in elephant_points for cell in locations.get(f"{side}_b", [])):
            return False
        if any(row not in pawn_range for row, _ in locations.get(f"{side}_p", [])):
            return False

    return True
