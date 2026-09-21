"""
中国象棋 FEN (Forsyth-Edwards Notation) 转换器
负责将 10x9 盘面状态矩阵与标准 FEN 字符串进行互相转换。
"""

from typing import List, Optional, Tuple, Dict

# 棋子编码映射表
PIECE_TO_FEN: Dict[str, str] = {
    "r_k": "K", "r_a": "A", "r_b": "B", "r_n": "N", "r_r": "R", "r_c": "C", "r_p": "P",
    "b_k": "k", "b_a": "a", "b_b": "b", "b_n": "n", "b_r": "r", "b_c": "c", "b_p": "p",
}

FEN_TO_PIECE: Dict[str, str] = {v: k for k, v in PIECE_TO_FEN.items()}

PIECE_NAMES_ZH: Dict[str, str] = {
    "r_k": "帥", "r_a": "仕", "r_b": "相", "r_n": "傌", "r_r": "俥", "r_c": "炮", "r_p": "兵",
    "b_k": "将", "b_a": "士", "b_b": "象", "b_n": "馬", "b_r": "車", "b_c": "砲", "b_p": "卒",
}


def matrix_to_fen(board: List[List[Optional[str]]], 
                  active_color: str = "w", 
                  halfmove: int = 0, 
                  fullmove: int = 1) -> str:
    """
    将 10x9 盘面矩阵转换为标准中国象棋 FEN 字符串
    board: 10 行 x 9 列，board[row][col] 为 'r_k', 'b_p' 或 None
    active_color: 'w' (红方走) 或 'b' (黑方走)
    """
    if len(board) != 10 or any(len(row) != 9 for row in board):
        raise ValueError(f"Invalid board dimensions: {len(board)}x{len(board[0]) if board else 0}, expected 10x9")

    rank_strs = []
    for row in board:
        empty_count = 0
        rank_parts = []
        for piece in row:
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_parts.append(str(empty_count))
                    empty_count = 0
                rank_parts.append(PIECE_TO_FEN[piece])
        if empty_count > 0:
            rank_parts.append(str(empty_count))
        rank_strs.append("".join(rank_parts))

    board_fen = "/".join(rank_strs)
    return f"{board_fen} {active_color} - - {halfmove} {fullmove}"


def fen_to_matrix(fen: str) -> Tuple[List[List[Optional[str]]], str, int, int]:
    """
    将标准中国象棋 FEN 字符串解析为 10x9 盘面矩阵及回合信息
    """
    parts = fen.strip().split()
    board_part = parts[0]
    active_color = parts[1] if len(parts) > 1 else "w"
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    rows = board_part.split("/")
    if len(rows) != 10:
        raise ValueError(f"Invalid FEN rank count: {len(rows)}, expected 10")

    matrix: List[List[Optional[str]]] = []
    for row_str in rows:
        matrix_row: List[Optional[str]] = []
        for ch in row_str:
            if ch.isdigit():
                matrix_row.extend([None] * int(ch))
            elif ch in FEN_TO_PIECE:
                matrix_row.append(FEN_TO_PIECE[ch])
            else:
                raise ValueError(f"Unknown piece symbol in FEN: {ch}")
        if len(matrix_row) != 9:
            raise ValueError(f"Invalid file count in FEN row '{row_str}': {len(matrix_row)}, expected 9")
        matrix.append(matrix_row)

    return matrix, active_color, halfmove, fullmove


def board_to_text(board: List[List[Optional[str]]]) -> str:
    """
    生成终端字符画形式的简易棋盘，方便查看与调试
    """
    lines = []
    lines.append("   0  1  2  3  4  5  6  7  8")
    lines.append("  ---------------------------")
    for r_idx, row in enumerate(board):
        chars = []
        for piece in row:
            if piece is None:
                chars.append(" ·")
            else:
                chars.append(f" {PIECE_NAMES_ZH.get(piece, '?')}")
        lines.append(f"{r_idx}|{''.join(chars)}")
        if r_idx == 4:
            lines.append("  |--- 楚 河 ----- 汉 界 ---|")
    return "\n".join(lines)
