"""JJ 象棋与天天象棋的自动识别配置。"""

import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.board_calibrator import BoardConfig, BoardGrid
from core.fen import PIECE_NAMES_ZH, board_to_text, matrix_to_fen
from core.recognizer import BoardRecognizer
from core.position_validation import is_structurally_valid_board
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
        # 应用宝/天天象棋新版将棋盘整体下移了 49px（以
        # 1440x2560 原始画面为基准）。保留旧网格以兼容旧客户端，
        # 识别时以双将完整性自动选择。
        self.adb_shifted_grid = BoardGrid(BoardConfig(
            base_width=1440,
            base_height=2560,
            x_min=109.0,
            x_max=1331.0,
            y_min=580.0,
            y_max=1958.0,
            base_piece_radius=76,
        ))
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
        self._adb_grid_preference: Optional[BoardGrid] = None
        self._host_grid_preference: Optional[BoardGrid] = None
        self.grid = self.adb_grid
        self.capture_source = "unknown"
        self.templates_dir = templates_dir
        self.adb_templates = self._load_templates(templates_dir)
        self.host_templates = self._load_templates("templates/tiantian_host")
        self.mini_program_templates = self._load_templates(
            "templates/tiantian_miniprogram"
        )
        self.templates = self.adb_templates
        self._dynamic_grid: Optional[BoardGrid] = None
        self._line_grid: Optional[BoardGrid] = None
        self._dynamic_shape = None
        self._dynamic_misses = 0
        self._verified_grid: Optional[BoardGrid] = None
        self._verified_templates_id: Optional[int] = None
        self._verified_templates: Optional[Dict[str, np.ndarray]] = None
        self._verified_profile: Optional[str] = None
        self._verified_piece_count: Optional[int] = None
        self._cell_cache_key = None
        self._cell_patch_cache: Dict[Any, np.ndarray] = {}
        self._cell_result_cache: Dict[Any, Optional[Dict[str, Any]]] = {}
        self._cache_generation = 0

    def reset_tracking(self) -> None:
        """清空网格与逐格识别缓存，下一帧从原始画面完整重建。"""
        self._adb_grid_preference = None
        self._host_grid_preference = None
        self._dynamic_grid = None
        self._line_grid = None
        self._dynamic_shape = None
        self._dynamic_misses = 0
        self._verified_grid = None
        self._verified_templates_id = None
        self._verified_templates = None
        self._verified_profile = None
        self._verified_piece_count = None
        self._cell_cache_key = None
        self._cell_patch_cache = {}
        self._cell_result_cache = {}
        self._cache_generation = 0

    @staticmethod
    def _cluster_circles(circles: np.ndarray, axis: int,
                         tolerance: float) -> List[List[np.ndarray]]:
        groups: List[List[np.ndarray]] = []
        for circle in sorted(circles, key=lambda item: float(item[axis])):
            if not groups:
                groups.append([circle])
                continue
            center = float(np.mean([item[axis] for item in groups[-1]]))
            if abs(float(circle[axis]) - center) <= tolerance:
                groups[-1].append(circle)
            else:
                groups.append([circle])
        return groups

    @staticmethod
    def _fit_axis_origin(values: List[float], step: float, count: int,
                         limit: int, target_center: float) -> Optional[float]:
        best = None
        tolerance = step * 0.24
        for value in values:
            for index in range(count):
                origin = value - index * step
                end = origin + (count - 1) * step
                if origin < -tolerance or end > limit + tolerance:
                    continue
                residuals = []
                matched_indices = set()
                for sample in values:
                    matched = int(round((sample - origin) / step))
                    if not 0 <= matched < count:
                        continue
                    residual = abs(sample - (origin + matched * step))
                    if residual <= tolerance:
                        residuals.append(residual)
                        matched_indices.add(matched)
                if not residuals:
                    continue
                center_penalty = abs(origin + (count - 1) * step / 2 - target_center) / step
                score = (
                    len(residuals), len(matched_indices),
                    # 稀疏残局里，同一组连续棋子可能对应多种相差整整
                    # 一列/一行的网格。先选更接近画面中心的完整棋盘，再用
                    # 亚像素残差破平局，避免整盘被错误平移一个格。
                    -center_penalty, -float(np.median(residuals)),
                )
                if best is None or score > best[0]:
                    best = (score, origin)
        return None if best is None else float(best[1])

    def _detect_piece_grid(self, image: np.ndarray) -> Optional[BoardGrid]:
        """Infer the 9x10 board lattice from circular piece centers."""
        height, width = image.shape[:2]
        detection_scale = min(1.0, 1080.0 / width)
        if detection_scale < 1.0:
            working = cv2.resize(
                image, None, fx=detection_scale, fy=detection_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        detection_width = working.shape[1]
        detection_height = working.shape[0]
        landscape = detection_width > detection_height
        if landscape:
            distance_floor = min(detection_width, detection_height) * 0.040
            radius_floor = min(detection_width, detection_height) * 0.018
            radius_ceiling = min(detection_width, detection_height) * 0.090
        else:
            # Preserve the tuned native/ADB detector for the original portrait
            # clients. The relaxed short-edge geometry is only needed by the
            # landscape WeChat mini-program.
            distance_floor = detection_width * 0.072
            radius_floor = detection_width * 0.026
            radius_ceiling = detection_width * 0.070
        found = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            # 微信小程序横屏窗口会在棋盘两侧保留大块区域。半径与间距
            # 必须按短边估计，不能再假设棋盘几乎占满窗口宽度。
            minDist=max(18, int(distance_floor)),
            param1=120, param2=32,
            minRadius=max(8, int(radius_floor)),
            maxRadius=max(20, int(radius_ceiling)),
        )
        if found is None or len(found[0]) < 6:
            return None
        circles = found[0].astype(np.float64) / detection_scale
        median_radius = float(np.median(circles[:, 2]))
        circles = circles[
            (circles[:, 2] >= median_radius * 0.72)
            & (circles[:, 2] <= median_radius * 1.28)
        ]
        if len(circles) < 6:
            return None

        groups = self._cluster_circles(circles, 1, max(6.0, median_radius * 0.68))
        row = max(groups, key=lambda group: (len(group), np.ptp([item[0] for item in group])))
        if len(row) < 4:
            return None
        row_x = sorted(float(item[0]) for item in row)
        expected_step = median_radius * 2.8
        units = []
        for left, right in zip(row_x, row_x[1:]):
            gap = right - left
            multiple = max(1, int(round(gap / expected_step)))
            unit = gap / multiple
            if median_radius * 1.8 <= unit <= median_radius * 3.6:
                units.append(unit)
        if len(units) < 3:
            return None
        step_x = float(np.median(units))
        origin_x = self._fit_axis_origin(row_x, step_x, 9, width, width / 2.0)
        if origin_x is None:
            return None

        x_aligned = []
        for circle in circles:
            column = int(round((float(circle[0]) - origin_x) / step_x))
            if (
                0 <= column < 9
                and abs(float(circle[0]) - (origin_x + column * step_x)) <= step_x * 0.24
            ):
                x_aligned.append(circle)
        if len(x_aligned) < 6:
            return None

        vertical_units = []
        by_column = {}
        for circle in x_aligned:
            column = int(round((float(circle[0]) - origin_x) / step_x))
            by_column.setdefault(column, []).append(float(circle[1]))
        for y_values in by_column.values():
            y_values.sort()
            for top, bottom in zip(y_values, y_values[1:]):
                gap = bottom - top
                multiple = max(1, int(round(gap / step_x)))
                unit = gap / multiple
                if abs(unit - step_x) <= step_x * 0.24:
                    vertical_units.append(unit)
        step_y = float(np.median(vertical_units)) if vertical_units else step_x
        origin_y = self._fit_axis_origin(
            [float(circle[1]) for circle in x_aligned],
            step_y, 10, height, height / 2.0,
        )
        if origin_y is None:
            return None

        matched = 0
        matched_rows = set()
        matched_columns = set()
        for circle in x_aligned:
            row_index = int(round((float(circle[1]) - origin_y) / step_y))
            column = int(round((float(circle[0]) - origin_x) / step_x))
            if not 0 <= row_index < 10:
                continue
            if abs(float(circle[1]) - (origin_y + row_index * step_y)) > step_y * 0.24:
                continue
            matched += 1
            matched_rows.add(row_index)
            matched_columns.add(column)
        if matched < 6 or len(matched_rows) < 2 or len(matched_columns) < 3:
            return None

        radius = max(
            12,
            int(round(median_radius * 1.30)),
            int(round(min(step_x, step_y) * 0.50)),
        )
        return BoardGrid(BoardConfig(
            base_width=width,
            base_height=height,
            x_min=origin_x,
            x_max=origin_x + step_x * 8,
            y_min=origin_y,
            y_max=origin_y + step_y * 9,
            base_piece_radius=radius,
        ))

    @staticmethod
    def _cluster_line_coordinates(entries, tolerance: float):
        """Merge the two edges and short fragments of one rendered grid line."""
        groups = []
        for coordinate, weight in sorted(entries):
            if not groups or coordinate - groups[-1][-1][0] > tolerance:
                groups.append([(coordinate, weight)])
            else:
                groups[-1].append((coordinate, weight))
        clustered = []
        for group in groups:
            total = sum(weight for _coordinate, weight in group)
            coordinate = sum(
                coordinate * weight for coordinate, weight in group
            ) / max(total, 1.0)
            clustered.append((coordinate, max(weight for _coordinate, weight in group)))
        return clustered

    @staticmethod
    def _fit_line_axis(values, count: int, limit: int, target_center: float,
                       min_step: float, max_step: float):
        """Fit an evenly spaced board axis to Hough line fragments."""
        best = None
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                delta = values[right_index][0] - values[left_index][0]
                for slot_distance in range(1, count):
                    step = delta / slot_distance
                    if not min_step <= step <= max_step:
                        continue
                    for first_slot in range(count - slot_distance):
                        origin = values[left_index][0] - first_slot * step
                        end = origin + (count - 1) * step
                        tolerance = max(3.0, step * 0.06)
                        if origin < -tolerance or end > limit + tolerance:
                            continue
                        assigned = {}
                        for coordinate, weight in values:
                            slot = int(round((coordinate - origin) / step))
                            if not 0 <= slot < count:
                                continue
                            residual = abs(coordinate - (origin + slot * step))
                            if residual > tolerance:
                                continue
                            candidate = (weight, -residual, coordinate)
                            current = assigned.get(slot)
                            if current is None or candidate > current:
                                assigned[slot] = candidate
                        if len(assigned) < max(5, count - 3):
                            continue
                        coverage = max(assigned) - min(assigned)
                        weight_score = sum(
                            item[0] for item in assigned.values()
                        ) / max(step, 1.0)
                        residual = sum(
                            -item[1] for item in assigned.values()
                        ) / len(assigned)
                        center_penalty = abs(
                            origin + (count - 1) * step / 2 - target_center
                        ) / step
                        score = (
                            len(assigned), coverage, weight_score,
                            -center_penalty, -residual,
                        )
                        if best is None or score > best[0]:
                            best = (score, origin, step)
        return best

    def _detect_line_grid(self, image: np.ndarray) -> Optional[BoardGrid]:
        """Locate the 9x10 lattice from board lines, even in a sparse endgame.

        Piece-circle fitting remains the fastest normal path.  The line fitter
        is the resize-safe fallback required by the WeChat mini-program: its
        landscape window can be made much wider than the centered board, and
        a late endgame may not contain enough pieces to recover the lattice.
        """
        original_height, original_width = image.shape[:2]
        scale = min(1.0, 1080.0 / max(original_width, original_height))
        if scale < 1.0:
            working = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        height, width = working.shape[:2]
        minimum_dimension = min(height, width)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 45, 135)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=max(35, int(minimum_dimension * 0.06)),
            minLineLength=max(45, int(minimum_dimension * 0.13)),
            maxLineGap=max(12, int(minimum_dimension * 0.035)),
        )
        if lines is None:
            return None

        horizontal = []
        vertical = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            length = math.hypot(dx, dy)
            if abs(dy) <= max(2.0, abs(dx) * 0.025):
                horizontal.append(((float(y1) + float(y2)) / 2.0, length))
            if abs(dx) <= max(2.0, abs(dy) * 0.025):
                vertical.append(((float(x1) + float(x2)) / 2.0, length))

        cluster_tolerance = max(3.0, minimum_dimension * 0.006)
        horizontal = self._cluster_line_coordinates(
            horizontal, cluster_tolerance,
        )
        vertical = self._cluster_line_coordinates(vertical, cluster_tolerance)
        step_min = minimum_dimension * 0.045
        step_max = minimum_dimension * 0.13
        y_fit = self._fit_line_axis(
            horizontal, 10, height, height / 2.0, step_min, step_max,
        )
        x_fit = self._fit_line_axis(
            vertical, 9, width, width / 2.0, step_min, step_max,
        )
        if x_fit is None or y_fit is None:
            return None
        _x_score, x_origin, step_x = x_fit
        _y_score, y_origin, step_y = y_fit
        if not 0.84 <= step_x / step_y <= 1.18:
            return None
        x_center = x_origin + step_x * 4
        y_center = y_origin + step_y * 4.5
        if (
            abs(x_center - width / 2.0) > width * 0.28
            or abs(y_center - height / 2.0) > height * 0.25
        ):
            return None

        inverse_scale = 1.0 / scale
        x_origin *= inverse_scale
        y_origin *= inverse_scale
        step_x *= inverse_scale
        step_y *= inverse_scale
        radius = max(12, int(round(min(step_x, step_y) * 0.55)))
        return BoardGrid(BoardConfig(
            base_width=original_width,
            base_height=original_height,
            x_min=x_origin,
            x_max=x_origin + step_x * 8,
            y_min=y_origin,
            y_max=y_origin + step_y * 9,
            base_piece_radius=radius,
        ))

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
    def _is_structurally_valid(result: Dict[str, Any]) -> bool:
        return is_structurally_valid_board(result["board"])

    @classmethod
    def _is_complete_initial_candidate(cls, result: Dict[str, Any]) -> bool:
        """Cold-start boards must classify every detected occupied cell."""
        return (
            cls._is_structurally_valid(result)
            and result.get("occupied_count", result["piece_count"]) == result["piece_count"]
        )

    @staticmethod
    def _offset_grids(grid: BoardGrid, width: int, height: int) -> List[BoardGrid]:
        """Make adjacent lattice hypotheses for sparse endgames.

        Piece centres alone are periodic, so a sparse position can fit equally
        well one whole file/rank away.  These alternatives are only evaluated
        when the normal candidate set fails Xiangqi structural validation.
        """
        points, radius = grid.compute_points(width, height)
        x0, y0 = (float(value) for value in points[0, 0])
        step_x = float(np.median(np.diff(points[0, :, 0])))
        step_y = float(np.median(np.diff(points[:, 0, 1])))
        grids = []
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                shifted_x = x0 + col_offset * step_x
                shifted_y = y0 + row_offset * step_y
                end_x = shifted_x + step_x * 8
                end_y = shifted_y + step_y * 9
                tolerance = min(step_x, step_y) * 0.25
                if (
                    shifted_x < -tolerance or shifted_y < -tolerance
                    or end_x > width + tolerance or end_y > height + tolerance
                ):
                    continue
                grids.append(BoardGrid(BoardConfig(
                    base_width=width,
                    base_height=height,
                    x_min=shifted_x,
                    x_max=end_x,
                    y_min=shifted_y,
                    y_max=end_y,
                    base_piece_radius=radius,
                )))
        return grids

    @staticmethod
    def _accept_piece_match(best_score: float, second_score: float, side: str,
                            red_count: int, dark_count: int,
                            highlight_red_min: int = 750,
                            highlight_dark_min: int = 480) -> bool:
        score_margin = best_score - second_score
        highlight_tolerant = (
            # The grey destination halo is strongest on the compact "馬"
            # glyph: a real captured frame measured 0.483 vs 0.410.  Keep a
            # substantial ink requirement and unique lead, but do not demand
            # the normal unhighlighted margin.
            best_score >= 0.47
            and score_margin >= 0.06
            and (
                red_count >= highlight_red_min
                if side == "r" else dark_count >= highlight_dark_min
            )
        )
        return (best_score >= 0.55 or highlight_tolerant) and score_margin >= 0.03

    def _profile_for_templates(self, templates: Dict[str, np.ndarray]) -> str:
        if templates is self.mini_program_templates:
            return "wechat_miniprogram"
        if templates is self.host_templates:
            return "desktop_host"
        return "adb"

    def _ink_thresholds(self, templates: Dict[str, np.ndarray]):
        if templates is self.mini_program_templates:
            # The mini-program renders finer dark glyphs and slightly lighter
            # red advisors than the native/ADB skin. Template confidence still
            # guards occupancy, so these lower pre-filter limits remain strict.
            return 600, 260
        return 700, 430

    def _accept_occupancy_match(
        self,
        best_score: float,
        templates: Dict[str, np.ndarray],
    ) -> bool:
        """Reject board texture before treating a cell as occupied.

        The mini-program's reddish wood can pass the red-ink prefilter on an
        empty edge cell.  Its template confidence stays below a real piece,
        including a halo-covered destination, so use a stricter occupancy
        floor for that skin.  Occupancy intentionally remains independent of
        piece-type margin: the debouncer only needs to know that a piece is
        present while a highlight obscures its glyph.
        """
        floor = 0.42 if templates is self.mini_program_templates else 0.36
        return best_score >= floor

    @staticmethod
    def _resolve_uncertain_occupancy(
        board: List[List[Optional[str]]],
        uncertain: Dict[Any, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Complete occupied cells under global Xiangqi constraints.

        A single halo-covered glyph can make 車/卒 or 馬/車 almost tied.  The
        whole board is much less ambiguous: piece-count limits and palace/
        elephant/pawn coordinates usually leave one legal assignment.
        """
        if not uncertain or len(uncertain) > 6:
            return []
        limits = {"k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5}
        counts = Counter(piece for row in board for piece in row if piece)
        cells = []
        for cell, info in uncertain.items():
            ranked = info["ranked"]
            best_score = ranked[0][1]
            choices = [
                (name, score) for name, score in ranked
                if score >= 0.28 and score >= best_score - 0.24
            ]
            if not choices:
                return []
            cells.append((cell, info, choices))
        cells.sort(key=lambda item: len(item[2]))

        solutions = []
        assignment: Dict[Any, tuple] = {}

        def search(index: int, total_score: float) -> None:
            if len(solutions) > 128:
                return
            if index == len(cells):
                candidate = [row[:] for row in board]
                for (row, col), (piece, _score) in assignment.items():
                    candidate[row][col] = piece
                if is_structurally_valid_board(candidate):
                    solutions.append((total_score, dict(assignment)))
                return
            cell, _info, choices = cells[index]
            for piece, score in choices:
                if counts[piece] >= limits[piece[2]]:
                    continue
                counts[piece] += 1
                assignment[cell] = (piece, score)
                search(index + 1, total_score + score)
                assignment.pop(cell, None)
                counts[piece] -= 1

        search(0, 0.0)
        if not solutions:
            return []
        solutions.sort(key=lambda item: item[0], reverse=True)
        # If two genuinely different whole-board explanations are essentially
        # tied, wait for the next frame rather than invent a piece.
        if len(solutions) > 1 and solutions[0][0] - solutions[1][0] < 0.025:
            if solutions[0][1] != solutions[1][1]:
                return []

        resolved = []
        for cell, (piece, score) in solutions[0][1].items():
            row, col = cell
            board[row][col] = piece
            info = uncertain[cell]
            resolved.append({
                "col": col, "row": row,
                "cx": info["cx"], "cy": info["cy"],
                "sample_cy": info["cy"], "vertical_offset": 0,
                "sample_radius": info["sample_radius"],
                "piece": piece,
                "name_zh": PIECE_NAMES_ZH.get(piece, "?"),
                "side": piece[0],
                "confidence": round(score, 3),
                "wood_ratio": None,
                "constraint_inferred": True,
            })
        return resolved

    @staticmethod
    def _detect_last_move(image: np.ndarray, points: np.ndarray, radius: int,
                          board: List[List[Optional[str]]],
                          profile: str = "desktop_host"):
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
        source_floor = (
            radius * radius * 0.055
            if profile == "wechat_miniprogram" else radius * 4.0
        )
        if source_score < source_floor or destination_score < radius * 5:
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
        occupancy = [[False for _ in range(9)] for _ in range(10)]
        occupied_sides: List[List[Optional[str]]] = [
            [None for _ in range(9)] for _ in range(10)
        ]
        details: List[Dict[str, Any]] = []
        uncertain: Dict[Any, Dict[str, Any]] = {}
        search_size = 90
        cache_enabled = (
            grid is self._verified_grid
            or (self._verified_grid is None and grid is self._dynamic_grid)
        )
        cache_key = (id(grid), width, height, id(templates), radius)
        if not cache_enabled or cache_key != self._cell_cache_key:
            previous_patches = {}
            previous_results = {}
        else:
            previous_patches = self._cell_patch_cache
            previous_results = self._cell_result_cache
        next_patches: Dict[Any, np.ndarray] = {}
        next_results: Dict[Any, Optional[Dict[str, Any]]] = {}
        red_ink_min, dark_ink_min = self._ink_thresholds(templates)

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
                cell = (row, col)
                if cache_enabled:
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    next_patches[cell] = gray_patch
                    previous = previous_patches.get(cell)
                    if previous is not None and cell in previous_results:
                        difference = cv2.absdiff(gray_patch, previous)
                        mean_delta = float(np.mean(difference))
                        changed_ratio = float(np.count_nonzero(difference >= 10)) / difference.size
                        # Even perfectly static cells are stagger-audited.  A
                        # transition frame must never leave a cached empty cell
                        # stuck until some unrelated later move changes it.
                        audit_due = (
                            (row * 9 + col + self._cache_generation) % 4 == 0
                        )
                        if mean_delta < 1.0 and changed_ratio < 0.01 and not audit_due:
                            cached = previous_results[cell]
                            next_results[cell] = cached
                            if cached is not None:
                                board[row][col] = cached["piece"]
                                occupancy[row][col] = True
                                occupied_sides[row][col] = cached["side"]
                                details.append({
                                    "col": col, "row": row, "cx": cx, "cy": cy,
                                    "sample_cy": cy, "vertical_offset": 0,
                                    "sample_radius": sample_radius,
                                    **cached,
                                })
                            continue
                core_hsv = cv2.cvtColor(patch[20:70, 20:70], cv2.COLOR_BGR2HSV)
                red_ink = (
                    ((core_hsv[:, :, 0] < 15) | (core_hsv[:, :, 0] > 165))
                    & (core_hsv[:, :, 1] > 80)
                    & (core_hsv[:, :, 2] < 210)
                )
                red_count = int(np.count_nonzero(red_ink))
                dark_count = int(np.count_nonzero(core_hsv[:, :, 2] < 120))
                if red_count < red_ink_min and dark_count < dark_ink_min:
                    if cache_enabled:
                        next_results[cell] = None
                    continue
                side = "r" if red_count >= red_ink_min else "b"
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
                score_margin = best_score - second_score
                # Occupancy deliberately has a much looser threshold than
                # glyph classification.  The move tracker can infer the piece
                # from its legal source square even when a halo obscures the
                # destination character.  White source rings score far below
                # this cutoff and therefore remain empty.
                if self._accept_occupancy_match(best_score, templates):
                    occupancy[row][col] = True
                    occupied_sides[row][col] = side
                # 天天象棋会在最后落点外画一圈半透明灰色光晕。光晕会把
                # 整图模板分数从正常的 0.6 左右压到约 0.5，但字形相对其它
                # 同色棋子仍有很大的领先幅度。对“墨色足够 + 唯一高置信
                # 字形”的格子放宽绝对阈值，避免刚落下的棋子被当成空格。
                if not self._accept_piece_match(
                    best_score, second_score, side, red_count, dark_count,
                    highlight_red_min=red_ink_min,
                    highlight_dark_min=dark_ink_min,
                ):
                    # Strong ink with a marginal glyph is a transition/highlight
                    # candidate.  Do not cache it as empty: retry next frame.
                    if occupancy[row][col]:
                        uncertain[cell] = {
                            "ranked": ranked,
                            "cx": cx, "cy": cy,
                            "sample_radius": sample_radius,
                        }
                    continue
                board[row][col] = best_name
                cached_result = {
                    "piece": best_name,
                    "name_zh": PIECE_NAMES_ZH.get(best_name, "?"),
                    "side": best_name[0], "confidence": round(best_score, 3),
                    "wood_ratio": None,
                }
                # Marginal highlighted matches are deliberately rechecked on
                # every frame; only normal high-confidence glyphs are cached.
                if cache_enabled and best_score >= 0.55 and score_margin >= 0.08:
                    next_results[cell] = cached_result
                details.append({
                    "col": col, "row": row, "cx": cx, "cy": cy,
                    "sample_cy": cy, "vertical_offset": 0,
                    "sample_radius": sample_radius, **cached_result,
                })

        details.extend(self._resolve_uncertain_occupancy(board, uncertain))

        if cache_enabled:
            self._cell_cache_key = cache_key
            self._cell_patch_cache = next_patches
            self._cell_result_cache = next_results

        last_move = self._detect_last_move(
            image, points, radius, board,
            profile=self._profile_for_templates(templates),
        )
        return {
            "board": board,
            "fen": matrix_to_fen(board, active_color="w"),
            "piece_count": len(details),
            "occupied_count": sum(sum(1 for cell in row if cell) for row in occupancy),
            "pieces_detail": details,
            "text_board": board_to_text(board),
            "last_visual_move": last_move,
            "last_move_side": last_move.get("side") if last_move else None,
            "occupancy": occupancy,
            "occupied_sides": occupied_sides,
            "game_id": "tiantian",
            "game_name": "天天象棋",
        }

    def _recognize_candidates(self, image: np.ndarray,
                              candidates: List[BoardGrid],
                              template_profiles):
        best_any = None
        best_valid = None
        for profile, templates in template_profiles:
            for grid in candidates:
                result = self._recognize_grid(image, grid, templates)
                confidence = sum(
                    item["confidence"] for item in result["pieces_detail"]
                )
                structurally_valid = self._is_complete_initial_candidate(result)
                score = (
                    1 if structurally_valid else 0,
                    result["piece_count"],
                    1 if grid in (self._dynamic_grid, self._line_grid) else 0,
                    confidence,
                )
                candidate = (score, result, grid, templates, profile)
                if best_any is None or score > best_any[0]:
                    best_any = candidate
                if structurally_valid and (
                    best_valid is None or score > best_valid[0]
                ):
                    best_valid = candidate
        assert best_any is not None
        return (best_valid or best_any), best_valid is not None

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        self._cache_generation += 1
        height, width = image.shape[:2]
        host_frame = (
            self.capture_source in ("coregraphics", "wechat_coregraphics")
            or (self.capture_source != "adb" and height < 2200)
        )
        mini_program_frame = (
            self.capture_source == "wechat_coregraphics"
            or (host_frame and width > height)
        )
        if mini_program_frame:
            # Windows may now be either the native 天天 client or the WeChat
            # mini-program. Cold-start both skins, then pin the winner.
            template_profiles = [
                ("wechat_miniprogram", self.mini_program_templates),
                ("desktop_host", self.host_templates),
            ]
        elif host_frame:
            template_profiles = [("desktop_host", self.host_templates)]
        else:
            template_profiles = [("adb", self.adb_templates)]
        frame_shape = image.shape[:2]
        if self._dynamic_shape != frame_shape:
            self._dynamic_grid = None
            self._line_grid = None
            self._dynamic_shape = frame_shape
            self._dynamic_misses = 0
            self._verified_grid = None
            self._verified_templates_id = None
            self._verified_templates = None
            self._verified_profile = None
            self._verified_piece_count = None
            self._cell_cache_key = None
            self._cell_patch_cache = {}
            self._cell_result_cache = {}
            self._cache_generation = 1

        # 已验证网格只跑一次 90 格识别；棋子缓存又会跳过未变化格子的
        # 模板匹配。若将帅丢失或棋子数突然异常下降，再自动回到全候选校验。
        if (
            self._verified_grid is not None
            and self._verified_templates is not None
            and self._verified_templates_id == id(self._verified_templates)
        ):
            result = self._recognize_grid(
                image, self._verified_grid, self._verified_templates,
            )
            count_ok = (
                self._verified_piece_count is None
                or result["piece_count"] >= max(2, self._verified_piece_count - 4)
            )
            if self._is_structurally_valid(result) and count_ok:
                self.grid = self._verified_grid
                self._verified_piece_count = result["piece_count"]
                if self._verified_grid is self._dynamic_grid:
                    result["grid_source"] = "piece_lattice"
                elif self._verified_grid is self._line_grid:
                    result["grid_source"] = "board_lines"
                else:
                    result["grid_source"] = "fixed_fallback"
                result["recognition_profile"] = self._verified_profile
                result["recognition_valid"] = True
                return result
            self._verified_grid = None
            self._verified_templates_id = None
            self._verified_templates = None
            self._verified_profile = None
            self._verified_piece_count = None
            self._cell_cache_key = None
            self._cell_patch_cache = {}
            self._cell_result_cache = {}

        if self._dynamic_grid is None:
            self._dynamic_grid = self._detect_piece_grid(image)
        if mini_program_frame and self._line_grid is None:
            self._line_grid = self._detect_line_grid(image)

        if host_frame:
            fixed_candidates = [self.host_grid, self.host_machine_grid]
            preference = self._host_grid_preference
        else:
            fixed_candidates = [self.adb_grid, self.adb_shifted_grid]
            preference = self._adb_grid_preference
        candidates = []
        if self._dynamic_grid is not None:
            candidates.append(self._dynamic_grid)
        if self._line_grid is not None:
            candidates.append(self._line_grid)
        candidates.extend(fixed_candidates)
        if preference in candidates:
            candidates.remove(preference)
            candidates.insert(0, preference)

        best, valid = self._recognize_candidates(
            image, candidates, template_profiles,
        )
        # If a sparse board was fitted one whole cell away, recover the absolute
        # files/ranks by testing its immediate lattice neighbours.  This is a
        # cold/recovery path only; the verified grid remains the fast path.
        if not valid and self._dynamic_grid is not None:
            shifted_candidates = self._offset_grids(
                self._dynamic_grid, image.shape[1], image.shape[0]
            )
            if shifted_candidates:
                shifted_best, shifted_valid = self._recognize_candidates(
                    image, shifted_candidates, template_profiles,
                )
                if shifted_best[0] > best[0]:
                    best = shifted_best
                valid = valid or shifted_valid
                if shifted_valid and shifted_best[2] is not self._dynamic_grid:
                    self._dynamic_grid = shifted_best[2]
        selected = best[2]
        if host_frame:
            self._host_grid_preference = selected if best[1]["piece_count"] >= 2 else None
        else:
            self._adb_grid_preference = selected if best[1]["piece_count"] >= 2 else None
        self.grid = selected
        if valid:
            self._verified_grid = selected
            self._verified_templates = best[3]
            self._verified_templates_id = id(best[3])
            self._verified_profile = best[4]
            self._verified_piece_count = best[1]["piece_count"]
            self._dynamic_misses = 0
        else:
            self._dynamic_misses += 1
            if self._dynamic_misses >= 12:
                self._dynamic_grid = None
                self._dynamic_misses = 0
        if selected is self._dynamic_grid:
            best[1]["grid_source"] = "piece_lattice"
        elif selected is self._line_grid:
            best[1]["grid_source"] = "board_lines"
        else:
            best[1]["grid_source"] = "fixed_fallback"
        best[1]["recognition_profile"] = best[4]
        best[1]["recognition_valid"] = valid
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

    def reset_tracking(self) -> None:
        """强制下一帧重新定位棋盘并重新识别全部 90 个交叉点。"""
        self.tiantian.reset_tracking()

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
