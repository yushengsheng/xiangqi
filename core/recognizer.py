"""
核心棋盘识别器 (Board Recognizer)
负责从游戏画面中精确提取 90 个交叉点的棋子状态，并输出 10x9 矩阵与 FEN 串。
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from core.board_calibrator import BoardGrid, BoardConfig
from core.fen import matrix_to_fen, board_to_text, PIECE_NAMES_ZH


class BoardRecognizer:
    def __init__(self, 
                 templates_dir: str = "templates/pieces", 
                 grid_config: Optional[BoardConfig] = None):
        self.grid = BoardGrid(grid_config)
        self.templates_dir = templates_dir
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self):
        """加载所有 14 种标准棋子模板"""
        expected = [
            "r_k", "r_a", "r_b", "r_n", "r_r", "r_c", "r_p",
            "b_k", "b_a", "b_b", "b_n", "b_r", "b_c", "b_p"
        ]
        for name in expected:
            path = os.path.join(self.templates_dir, f"{name}.png")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required piece template: {path}")
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to read image template: {path}")
            self.templates[name] = img

    def is_square_occupied(self, patch: np.ndarray) -> Tuple[bool, float]:
        """
        根据木质棋子色调判断交叉点是否有棋子。
        真实棋子的木纹覆盖率通常在 0.35~0.55，而石板空位通常 <= 0.30。
        """
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # 木质棋子黄色/木纹色调
        wood_mask = cv2.inRange(hsv, np.array([12, 35, 90]), np.array([38, 210, 245]))
        wood_ratio = float(np.count_nonzero(wood_mask) / wood_mask.size)
        return (wood_ratio >= 0.32), wood_ratio

    def detect_piece_side(self, patch: np.ndarray) -> Tuple[str, int]:
        """
        判断棋子属于红方还是黑方
        返回: ('r' 或 'b', 红像素数量)
        """
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 45]), np.array([12, 255, 235]))
        mask2 = cv2.inRange(hsv, np.array([166, 70, 45]), np.array([180, 255, 235]))
        red_mask = mask1 | mask2
        red_px = int(np.count_nonzero(red_mask))
        side = "r" if red_px > 100 else "b"
        return side, red_px

    def classify_piece_ranked(self, patch: np.ndarray, side: str,
                              core_sizes: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        """返回按分数排序的棋种候选。

        JJ 的选中特效会把棋子放大并上浮，因此同时尝试多个字形核心尺寸；模板
        仍只匹配中心汉字，避免光圈、阴影和外缘动画影响。
        """
        candidate_pieces = ["k", "a", "b", "n", "r", "c", "p"]
        ph, pw = patch.shape[:2]
        sizes = core_sizes or [min(46, int(ph * 0.6))]
        scores = []
        for piece in candidate_pieces:
            name = f"{side}_{piece}"
            tpl = self.templates.get(name)
            if tpl is None:
                continue
            th, tw = tpl.shape[:2]
            best = -1.0
            for requested_size in sizes:
                core_size = min(requested_size, th, tw, ph, pw)
                if core_size < 20:
                    continue
                top = th // 2 - core_size // 2
                left = tw // 2 - core_size // 2
                t_core = tpl[top:top + core_size, left:left + core_size]
                result = cv2.matchTemplate(patch, t_core, cv2.TM_CCOEFF_NORMED)
                score = float(cv2.minMaxLoc(result)[1])
                best = max(best, score)
            scores.append((name, best))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def classify_piece(self, patch: np.ndarray, side: str) -> Tuple[str, float]:
        ranked = self.classify_piece_ranked(patch, side)
        return ranked[0] if ranked else (f"{side}_p", -1.0)

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        全盘识别入口函数
        返回包含:
        - 'board': 10x9 矩阵 (每个元素为 'r_k' 或 None)
        - 'fen': 中国象棋标准 FEN 串
        - 'active_color': 当前走子方 ('w' 或 'b')
        - 'piece_count': 棋子总数
        - 'pieces_detail': 识别详情列表
        """
        h, w = image.shape[:2]
        points, radius = self.grid.compute_points(w, h)

        board: List[List[Optional[str]]] = [[None for _ in range(9)] for _ in range(10)]
        pieces_detail = []
        piece_count = 0

        row_step = float(np.median(np.linalg.norm(points[1:] - points[:-1], axis=2)))
        # CoreGraphics 会随窗口缩放改变棋子像素尺寸。所有 patch 先归一化回模板
        # 的 76x76 基准，再匹配；额外的大 patch 用于 JJ 选中后的放大/上浮动画。
        base_patch_size = self.grid.config.base_piece_radius * 2
        search_specs = [(0.0, 1.0)] + [
            (offset, radius_scale)
            for offset in (-0.20, -0.30, -0.40, -0.50)
            for radius_scale in (1.0, 1.25)
        ]

        for r_idx in range(10):
            for c_idx in range(9):
                cx, cy = np.round(points[r_idx, c_idx]).astype(int)
                accepted = None
                for spec_index, (offset_ratio, radius_scale) in enumerate(search_specs):
                    offset_y = int(round(row_step * offset_ratio))
                    sample_y = cy + offset_y
                    sample_radius = max(radius, int(round(radius * radius_scale)))
                    x0, x1 = cx - sample_radius, cx + sample_radius
                    y0, y1 = sample_y - sample_radius, sample_y + sample_radius
                    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                        continue
                    raw_patch = image[y0:y1, x0:x1]
                    patch = cv2.resize(
                        raw_patch, (base_patch_size, base_patch_size),
                        interpolation=cv2.INTER_AREA if raw_patch.shape[0] > base_patch_size else cv2.INTER_CUBIC
                    )
                    occupied, wood_ratio = self.is_square_occupied(patch)
                    if not occupied:
                        # 已采集的正常、缩放及真实选中特效帧中，上浮棋子的
                        # 圆盘仍与原交点相交；原格完全无木色即可安全走快速空格路径。
                        if spec_index == 0:
                            break
                        continue
                    side, red_px = self.detect_piece_side(patch)
                    # 绝大多数普通棋子只需一次 46px 字形匹配；仅普通位置未能
                    # 确认时，才为选中特效尝试多种字形尺寸。
                    core_sizes = [46] if spec_index == 0 else [36, 40, 44, 46]
                    ranked = self.classify_piece_ranked(patch, side, core_sizes=core_sizes)
                    if not ranked:
                        continue
                    piece_name, score = ranked[0]
                    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
                    valid = score >= 0.65 and score - second_score >= 0.03
                    if valid and (accepted is None or score > accepted[1]):
                        accepted = (
                            piece_name, score, side, red_px, wood_ratio,
                            sample_y, offset_y, sample_radius
                        )
                        if spec_index == 0:
                            break

                if accepted is None:
                    continue
                piece_name, score, side, red_px, wood_ratio, sample_y, offset_y, sample_radius = accepted
                board[r_idx][c_idx] = piece_name
                piece_count += 1
                pieces_detail.append({
                    "col": c_idx,
                    "row": r_idx,
                    "cx": cx,
                    "cy": cy,
                    "sample_cy": sample_y,
                    "vertical_offset": offset_y,
                    "sample_radius": sample_radius,
                    "piece": piece_name,
                    "name_zh": PIECE_NAMES_ZH.get(piece_name, "?"),
                    "side": side,
                    "confidence": round(score, 3),
                    "wood_ratio": round(wood_ratio, 3)
                })

        # 判定红先还是黑先（默认 'w'，若红帅不在底线也可适应翻转视角）
        fen = matrix_to_fen(board, active_color="w")

        return {
            "board": board,
            "fen": fen,
            "piece_count": piece_count,
            "pieces_detail": pieces_detail,
            "text_board": board_to_text(board)
        }

    def render_debug_image(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        生成可视化识别结果图，标出所有识别出的棋子标签与可信度
        """
        out = self.grid.draw_debug_overlay(image)
        for p in result["pieces_detail"]:
            cx, cy = p["cx"], p["cy"]
            name = p["name_zh"]
            color = (0, 0, 255) if p["side"] == "r" else (50, 50, 50)
            # 绘制棋子中心标记
            cv2.circle(out, (cx, cy), 18, (255, 255, 255), -1)
            cv2.circle(out, (cx, cy), 18, color, 2)
            # 标注英文代号与置信度
            cv2.putText(out, p["piece"], (cx - 15, cy - 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(out, f"{p['confidence']:.2f}", (cx - 15, cy + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1)
        return out
