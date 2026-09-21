"""中国象棋规则：走法生成、将军、中文记谱。

几何以「谁在画面下方」为准，而不是写死红在底、黑在顶。
这样 JJ 执黑翻面后，下方仍是合法宫区/兵方向，AI 也能执下方对战。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

Board = List[List[Optional[str]]]
Coord = Tuple[int, int]

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
FLIPPED_START_FEN = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr w - - 0 1"

NOTATION_NAMES: Dict[str, str] = {
    "r_k": "帅", "r_a": "仕", "r_b": "相", "r_n": "马", "r_r": "车", "r_c": "炮", "r_p": "兵",
    "b_k": "将", "b_a": "士", "b_b": "象", "b_n": "马", "b_r": "车", "b_c": "炮", "b_p": "卒",
}
CN_NUM = "一二三四五六七八九"


@dataclass(frozen=True)
class Move:
    sr: int
    sc: int
    dr: int
    dc: int
    piece: str
    captured: Optional[str] = None

    @property
    def uci(self) -> str:
        return f"{chr(97 + self.sc)}{9 - self.sr}{chr(97 + self.dc)}{9 - self.dr}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "piece": self.piece,
            "piece_name": NOTATION_NAMES.get(self.piece, "?"),
            "from": {"row": self.sr, "col": self.sc},
            "to": {"row": self.dr, "col": self.dc},
            "captured": self.captured,
            "uci": self.uci,
        }


def other_side(side: str) -> str:
    return "b" if side == "r" else "r"


def copy_board(board: Board) -> Board:
    return [row[:] for row in board]


def infer_bottom_side(board: Board) -> str:
    """画面下方那一侧的颜色：看哪边将/帅的行号更大。"""
    red_row = black_row = None
    for row_idx, row in enumerate(board):
        for piece in row:
            if piece == "r_k":
                red_row = row_idx
            elif piece == "b_k":
                black_row = row_idx
    if red_row is not None and black_row is not None and red_row != black_row:
        return "r" if red_row > black_row else "b"
    if red_row is not None:
        return "r" if red_row >= 5 else "b"
    if black_row is not None:
        return "b" if black_row >= 5 else "r"
    return "r"


def side_is_bottom(side: str, bottom: str) -> bool:
    return side == bottom


def pawn_forward(side: str, bottom: str) -> int:
    return -1 if side_is_bottom(side, bottom) else 1


def palace_rows(side: str, bottom: str) -> range:
    return range(7, 10) if side_is_bottom(side, bottom) else range(0, 3)


def in_palace(col: int, row: int, side: str, bottom: str) -> bool:
    return 3 <= col <= 5 and row in palace_rows(side, bottom)


def elephant_stays_home(row: int, side: str, bottom: str) -> bool:
    return row >= 5 if side_is_bottom(side, bottom) else row <= 4


def pawn_crossed_river(row: int, side: str, bottom: str) -> bool:
    return row <= 4 if side_is_bottom(side, bottom) else row >= 5


def find_king(board: Board, side: str) -> Optional[Coord]:
    target = f"{side}_k"
    for row_idx, row in enumerate(board):
        for col_idx, piece in enumerate(row):
            if piece == target:
                return row_idx, col_idx
    return None


def count_between(board: Board, sr: int, sc: int, dr: int, dc: int) -> int:
    if sr != dr and sc != dc:
        return -1
    if sr == dr:
        step = 1 if dc > sc else -1
        return sum(1 for col in range(sc + step, dc, step) if board[sr][col] is not None)
    step = 1 if dr > sr else -1
    return sum(1 for row in range(sr + step, dr, step) if board[row][sc] is not None)


def is_pseudo_legal_move(board: Board, sc: int, sr: int, dc: int, dr: int,
                         piece: Optional[str] = None, bottom: Optional[str] = None) -> bool:
    """单棋种几何是否可走（不含自我将军）。供识别去抖与着法生成共用。"""
    if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= dr < 10 and 0 <= dc < 9):
        return False
    if (sc, sr) == (dc, dr):
        return False
    moving = piece or board[sr][sc]
    if moving is None:
        return False
    destination = board[dr][dc]
    if destination is not None and destination[0] == moving[0]:
        return False

    kind, side = moving[2], moving[0]
    dx, dy = abs(dc - sc), abs(dr - sr)
    if bottom is None:
        bottom = infer_bottom_side(board)
    forward = pawn_forward(side, bottom)

    if kind == "k":
        return dx + dy == 1 and in_palace(dc, dr, side, bottom)
    if kind == "a":
        return dx == 1 and dy == 1 and in_palace(dc, dr, side, bottom)
    if kind == "b":
        if dx != 2 or dy != 2 or not elephant_stays_home(dr, side, bottom):
            return False
        return board[(sr + dr) // 2][(sc + dc) // 2] is None
    if kind == "n":
        if (dx, dy) not in ((1, 2), (2, 1)):
            return False
        leg_row, leg_col = ((sr + dr) // 2, sc) if dy == 2 else (sr, (sc + dc) // 2)
        return board[leg_row][leg_col] is None
    if kind == "r":
        return (sc == dc or sr == dr) and count_between(board, sr, sc, dr, dc) == 0
    if kind == "c":
        screens = count_between(board, sr, sc, dr, dc)
        return (sc == dc or sr == dr) and screens == (1 if destination is not None else 0)
    if kind == "p":
        if dc == sc and dr - sr == forward:
            return True
        return pawn_crossed_river(sr, side, bottom) and dr == sr and dx == 1
    return False


def _ray_destinations(board: Board, row: int, col: int, side: str, cannon: bool) -> Iterable[Coord]:
    for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        screen = False
        nr, nc = row + d_row, col + d_col
        while 0 <= nr < 10 and 0 <= nc < 9:
            occupant = board[nr][nc]
            if cannon:
                if not screen:
                    if occupant is None:
                        yield nr, nc
                    else:
                        screen = True
                else:
                    if occupant is not None:
                        if occupant[0] != side:
                            yield nr, nc
                        break
            else:
                if occupant is None:
                    yield nr, nc
                else:
                    if occupant[0] != side:
                        yield nr, nc
                    break
            nr += d_row
            nc += d_col


def iter_destinations(board: Board, row: int, col: int, piece: str,
                      bottom: Optional[str] = None) -> Iterable[Coord]:
    kind, side = piece[2], piece[0]
    if bottom is None:
        bottom = infer_bottom_side(board)
    if kind == "k":
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if is_pseudo_legal_move(board, col, row, nc, nr, piece, bottom):
                yield nr, nc
        return
    if kind == "a":
        for nr, nc in ((row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)):
            if is_pseudo_legal_move(board, col, row, nc, nr, piece, bottom):
                yield nr, nc
        return
    if kind == "b":
        for nr, nc in ((row - 2, col - 2), (row - 2, col + 2), (row + 2, col - 2), (row + 2, col + 2)):
            if is_pseudo_legal_move(board, col, row, nc, nr, piece, bottom):
                yield nr, nc
        return
    if kind == "n":
        for nr, nc in (
            (row - 2, col - 1), (row - 2, col + 1), (row + 2, col - 1), (row + 2, col + 1),
            (row - 1, col - 2), (row - 1, col + 2), (row + 1, col - 2), (row + 1, col + 2),
        ):
            if is_pseudo_legal_move(board, col, row, nc, nr, piece, bottom):
                yield nr, nc
        return
    if kind == "r":
        yield from _ray_destinations(board, row, col, side, cannon=False)
        return
    if kind == "c":
        yield from _ray_destinations(board, row, col, side, cannon=True)
        return
    if kind == "p":
        forward = pawn_forward(side, bottom)
        for nr, nc in ((row + forward, col), (row, col - 1), (row, col + 1)):
            if is_pseudo_legal_move(board, col, row, nc, nr, piece, bottom):
                yield nr, nc


def kings_facing(board: Board) -> bool:
    red = find_king(board, "r")
    black = find_king(board, "b")
    if red is None or black is None or red[1] != black[1]:
        return False
    return count_between(board, red[0], red[1], black[0], black[1]) == 0


def is_in_check(board: Board, side: str, bottom: Optional[str] = None) -> bool:
    king = find_king(board, side)
    if king is None:
        return True
    if kings_facing(board):
        return True
    if bottom is None:
        bottom = infer_bottom_side(board)
    kr, kc = king
    attacker = other_side(side)
    for row_idx, row in enumerate(board):
        for col_idx, piece in enumerate(row):
            if piece is None or piece[0] != attacker:
                continue
            if is_pseudo_legal_move(board, col_idx, row_idx, kc, kr, piece, bottom):
                return True
    return False


def apply_move(board: Board, move: Move) -> Board:
    next_board = copy_board(board)
    next_board[move.sr][move.sc] = None
    next_board[move.dr][move.dc] = move.piece
    return next_board


def generate_pseudo_moves(board: Board, side: str, bottom: Optional[str] = None) -> List[Move]:
    if bottom is None:
        bottom = infer_bottom_side(board)
    moves: List[Move] = []
    for row_idx, row in enumerate(board):
        for col_idx, piece in enumerate(row):
            if piece is None or piece[0] != side:
                continue
            for dr, dc in iter_destinations(board, row_idx, col_idx, piece, bottom):
                moves.append(Move(row_idx, col_idx, dr, dc, piece, board[dr][dc]))
    return moves


def generate_legal_moves(board: Board, side: str, bottom: Optional[str] = None) -> List[Move]:
    if bottom is None:
        bottom = infer_bottom_side(board)
    legal: List[Move] = []
    for move in generate_pseudo_moves(board, side, bottom):
        next_board = apply_move(board, move)
        if not is_in_check(next_board, side, bottom):
            legal.append(move)
    return legal


def generate_captures(board: Board, side: str, bottom: Optional[str] = None) -> List[Move]:
    return [move for move in generate_legal_moves(board, side, bottom) if move.captured]


def parse_uci(board: Board, uci: str) -> Optional[Move]:
    if len(uci) < 4:
        return None
    sc = ord(uci[0]) - 97
    sr = 9 - int(uci[1])
    dc = ord(uci[2]) - 97
    dr = 9 - int(uci[3])
    if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= dr < 10 and 0 <= dc < 9):
        return None
    piece = board[sr][sc]
    if piece is None:
        return None
    return Move(sr, sc, dr, dc, piece, board[dr][dc])


def _file_digit(side: str, col: int, bottom: str) -> str:
    number = (9 - col) if side_is_bottom(side, bottom) else (col + 1)
    return CN_NUM[number - 1] if side == "r" else str(number)


def _step_digit(side: str, steps: int) -> str:
    return CN_NUM[steps - 1] if side == "r" else str(steps)


def move_to_chinese(board: Board, move: Move) -> str:
    piece = move.piece
    side = piece[0]
    name = NOTATION_NAMES.get(piece, "?")
    bottom = infer_bottom_side(board)
    same_file = [row for row in range(10) if row != move.sr and board[row][move.sc] == piece]
    if same_file:
        if side_is_bottom(side, bottom):
            is_front = move.sr < min(same_file)
        else:
            is_front = move.sr > max(same_file)
        origin = ("前" if is_front else "后") + name
    else:
        origin = name + _file_digit(side, move.sc, bottom)

    if move.sr == move.dr:
        verb, dest = "平", _file_digit(side, move.dc, bottom)
    else:
        going_forward = (move.dr < move.sr) if side_is_bottom(side, bottom) else (move.dr > move.sr)
        verb = "进" if going_forward else "退"
        if piece[2] in ("n", "a", "b"):
            dest = _file_digit(side, move.dc, bottom)
        else:
            dest = _step_digit(side, abs(move.dr - move.sr))
    return f"{origin}{verb}{dest}"


def move_payload(board: Board, move: Move) -> Dict[str, object]:
    data = move.to_dict()
    data["zh"] = move_to_chinese(board, move)
    data["description"] = data["zh"]
    if move.captured:
        data["captured_name"] = NOTATION_NAMES.get(move.captured, "?")
    return data


def legal_move_payloads(board: Board, side: str) -> List[Dict[str, object]]:
    return [move_payload(board, move) for move in generate_legal_moves(board, side)]


def is_legal_move(board: Board, move: Move, side: Optional[str] = None) -> bool:
    if side is None:
        side = move.piece[0]
    if move.piece[0] != side:
        return False
    return any(
        candidate.sr == move.sr and candidate.sc == move.sc
        and candidate.dr == move.dr and candidate.dc == move.dc
        for candidate in generate_legal_moves(board, side)
    )
