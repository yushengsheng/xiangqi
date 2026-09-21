"""JJ 象棋与天天象棋的自动识别配置。"""

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.board_calibrator import BoardConfig, BoardGrid
from core.fen import PIECE_NAMES_ZH, board_to_text, matrix_to_fen
from core.recognizer import BoardRecognizer
from core.xiangqi import is_pseudo_legal_move


class TiantianRecognizer:
    """天天象棋经典木纹盘面识别器。

    使用整枚棋子的形状匹配，空交叉点最高分与真实棋子之间有明显间隔；相比
    颜色阈值更不受木纹棋盘、窗口缩放和红黑字色影响。
    """

    def __init__(self, templates_dir: str = "templates/tiantian"):
        config = BoardConfig(
            base_width=1440,
            base_height=2560,
            x_min=109.0,
            x_max=1331.0,
            y_min=531.0,
            y_max=1909.0,
            base_piece_radius=76,
        )
        self.adb_grid = BoardGrid(config)
        self.host_grid = BoardGrid(BoardConfig(
            base_width=880,
            base_height=1636,
            x_min=66.0,
            x_max=812.0,
            y_min=433.0,
            y_max=1267.0,
            base_piece_radius=42,
        ))
        self.host_machine_grid = BoardGrid(BoardConfig(
            base_width=880,
            base_height=1636,
            x_min=66.0,
            x_max=812.0,
            y_min=403.0,
            y_max=1238.0,
            base_piece_radius=42,
        ))
        self._host_grid_preference: Optional[BoardGrid] = None
        self.grid = self.adb_grid
        self.capture_source = "unknown"
        self.templates_dir = templates_dir
        self.adb_templates = self._load_templates(templates_dir)
        self.host_templates = self._load_templates("templates/tiantian_host")
        self.templates = self.adb_templates

    @staticmethod
    def _load_templates(directory: str) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for side in ("r", "b"):
            for piece in ("k", "a", "b", "n", "r", "c", "p"):
                name = f"{side}_{piece}"
                path = os.path.join(directory, f"{name}.png")
                image = cv2.imread(path)
                if image is None:
                    raise FileNotFoundError(f"Missing required Tiantian template: {path}")
                templates[name] = image
        return templates

    @staticmethod
    def _has_both_kings(result: Dict[str, Any]) -> bool:
        pieces = [piece for row in result["board"] for piece in row if piece]
        return pieces.count("r_k") == 1 and pieces.count("b_k") == 1

    @staticmethod
    def _detect_last_move(image: np.ndarray, points: np.ndarray, radius: int,
                          board: List[List[Optional[str]]]):
        """读取天天象棋的白色来源圈与灰色落点光圈，供中盘启动恢复回合。"""
        source_scores = []
        destination_scores = []
        height, width = image.shape[:2]
        sample_radius = max(16, int(round(radius * 1.2)))
        for row in range(10):
            for col in range(9):
                cx, cy = np.round(points[row, col]).astype(int)
                if (
                    cx - sample_radius < 0 or cy - sample_radius < 0
                    or cx + sample_radius >= width or cy + sample_radius >= height
                ):
                    continue
                patch = image[
                    cy - sample_radius:cy + sample_radius + 1,
                    cx - sample_radius:cx + sample_radius + 1,
                ]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                yy, xx = np.ogrid[-sample_radius:sample_radius + 1,
                                   -sample_radius:sample_radius + 1]
                distance = xx * xx + yy * yy
                source_ring = (
                    (distance > (radius * 0.30) ** 2)
                    & (distance < (radius * 0.65) ** 2)
                )
                destination_ring = (
                    (distance > (radius * 0.80) ** 2)
                    & (distance < (radius * 1.15) ** 2)
                )
                if board[row][col] is None:
                    white = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 205) & source_ring
                    source_scores.append((int(np.count_nonzero(white)), row, col))
                else:
                    gray = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 120) & destination_ring
                    destination_scores.append((int(np.count_nonzero(gray)), row, col))
        if not source_scores or not destination_scores:
            return None
        source_score, sr, sc = max(source_scores)
        destination_score, dr, dc = max(destination_scores)
        if source_score < radius * 4 or destination_score < radius * 5:
            return None
        piece = board[dr][dc]
        if piece is None:
            return None
        previous = [row[:] for row in board]
        previous[sr][sc] = piece
        previous[dr][dc] = None
        legal = is_pseudo_legal_move(previous, sc, sr, dc, dr, piece)
        if not legal:
            previous[dr][dc] = "b_p" if piece[0] == "r" else "r_p"
            legal = is_pseudo_legal_move(previous, sc, sr, dc, dr, piece)
        if not legal:
            return None
        return {
            "side": piece[0],
            "piece": piece,
            "from": {"row": sr, "col": sc},
            "to": {"row": dr, "col": dc},
        }

    def _recognize_grid(self, image: np.ndarray, grid: BoardGrid,
                        templates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        height, width = image.shape[:2]
        self.grid = grid
        self.templates = templates
        points, radius = grid.compute_points(width, height)
        board: List[List[Optional[str]]] = [[None for _ in range(9)] for _ in range(10)]
        details: List[Dict[str, Any]] = []
        search_size = 90

        for row in range(10):
            for col in range(9):
                cx, cy = np.round(points[row, col]).astype(int)
                sample_radius = max(radius, int(round(radius * 1.18)))
                x0, x1 = cx - sample_radius, cx + sample_radius
                y0, y1 = cy - sample_radius, cy + sample_radius
                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    continue
                patch = cv2.resize(
                    image[y0:y1, x0:x1], (search_size, search_size),
                    interpolation=cv2.INTER_AREA if sample_radius * 2 > search_size else cv2.INTER_CUBIC,
                )
                core_hsv = cv2.cvtColor(patch[20:70, 20:70], cv2.COLOR_BGR2HSV)
                red_ink = (
                    ((core_hsv[:, :, 0] < 15) | (core_hsv[:, :, 0] > 165))
                    & (core_hsv[:, :, 1] > 80)
                    & (core_hsv[:, :, 2] < 210)
                )
                red_count = int(np.count_nonzero(red_ink))
                dark_count = int(np.count_nonzero(core_hsv[:, :, 2] < 120))
                if red_count < 700 and dark_count < 430:
                    continue
                side = "r" if red_count >= 700 else "b"
                ranked = []
                for name, template in templates.items():
                    if name.startswith(f"{side}_"):
                        score = float(cv2.minMaxLoc(
                            cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
                        )[1])
                        ranked.append((name, score))
                ranked.sort(key=lambda item: item[1], reverse=True)
                best_name, best_score = ranked[0]
                second_score = ranked[1][1]
                if best_score < 0.55 or best_score - second_score < 0.03:
                    continue
                board[row][col] = best_name
                details.append({
                    "col": col, "row": row, "cx": cx, "cy": cy,
                    "sample_cy": cy, "vertical_offset": 0,
                    "sample_radius": sample_radius, "piece": best_name,
                    "name_zh": PIECE_NAMES_ZH.get(best_name, "?"),
                    "side": best_name[0], "confidence": round(best_score, 3),
                    "wood_ratio": None,
                })

        last_move = self._detect_last_move(image, points, radius, board)
        return {
            "board": board,
            "fen": matrix_to_fen(board, active_color="w"),
            "piece_count": len(details),
            "pieces_detail": details,
            "text_board": board_to_text(board),
            "last_visual_move": last_move,
            "last_move_side": last_move.get("side") if last_move else None,
            "game_id": "tiantian",
            "game_name": "天天象棋",
        }

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        height = image.shape[0]
        host_frame = (
            self.capture_source == "coregraphics"
            or (self.capture_source == "unknown" and height < 2200)
        )
        if not host_frame:
            return self._recognize_grid(image, self.adb_grid, self.adb_templates)

        candidates = [self.host_grid, self.host_machine_grid]
        if self._host_grid_preference in candidates:
            candidates.remove(self._host_grid_preference)
            candidates.insert(0, self._host_grid_preference)
        best = None
        for grid in candidates:
            result = self._recognize_grid(image, grid, self.host_templates)
            if self._has_both_kings(result):
                self._host_grid_preference = grid
                return result
            score = (result["piece_count"], sum(
                item["confidence"] for item in result["pieces_detail"]
            ))
            if best is None or score > best[0]:
                best = (score, result, grid)
        assert best is not None
        self._host_grid_preference = best[2] if best[1]["piece_count"] >= 2 else None
        return best[1]

    def render_debug_image(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        out = self.grid.draw_debug_overlay(image)
        for piece in result["pieces_detail"]:
            color = (0, 0, 220) if piece["side"] == "r" else (40, 40, 40)
            cv2.circle(out, (piece["cx"], piece["cy"]), 20, color, 2)
            cv2.putText(
                out, f"{piece['piece']} {piece['confidence']:.2f}",
                (piece["cx"] - 25, piece["cy"] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )
        return out


class AdaptiveBoardRecognizer:
    """保持 JJ 识别器独立，同时按捕获层识别的平台切换轻量配置。"""

    def __init__(self, jj_grid_config: Optional[BoardConfig] = None):
        self.jj = BoardRecognizer(grid_config=jj_grid_config)
        self.tiantian = TiantianRecognizer()
        self.game_id = "unknown"
        self.last_game_id = "unknown"

    def set_capture_source(self, source: Optional[str]) -> None:
        normalized = source or "unknown"
        self.tiantian.capture_source = normalized

    def set_game(self, game_id: Optional[str]) -> bool:
        normalized = game_id if game_id in ("jj", "tiantian") else "unknown"
        changed = (
            normalized != "unknown"
            and self.last_game_id != "unknown"
            and normalized != self.last_game_id
        )
        if normalized != "unknown":
            self.game_id = normalized
            self.last_game_id = normalized
        return changed

    @staticmethod
    def _has_both_kings(result: Dict[str, Any]) -> bool:
        pieces = [piece for row in result["board"] for piece in row if piece]
        return pieces.count("r_k") == 1 and pieces.count("b_k") == 1

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        if self.game_id == "tiantian":
            return self.tiantian.recognize(image)
        if self.game_id == "jj":
            result = self.jj.recognize(image)
            result.update(game_id="jj", game_name="JJ象棋")
            return result

        # Mock、手工截图或捕获层暂未识别平台时，仅先试一次高区分度的天天模板。
        candidate = self.tiantian.recognize(image)
        if candidate["piece_count"] >= 2 and self._has_both_kings(candidate):
            self.game_id = self.last_game_id = "tiantian"
            return candidate
        result = self.jj.recognize(image)
        result.update(game_id="jj", game_name="JJ象棋")
        self.game_id = self.last_game_id = "jj"
        return result
