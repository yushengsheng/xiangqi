"""漏采追帧、视觉回合锚点和搜索取消的确定性回归。"""

import socket
import struct
import subprocess
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from capture.mac_capture import MacCapture
from capture.mock_capture import MockCapture
from core.adaptive_recognizer import TiantianRecognizer
from core.ai_advisor import AIAdvisor
from core.debouncer import BoardDebouncer
from core.engine import SearchResult
from core.fen import fen_to_matrix, matrix_to_fen
from core.pikafish_engine import PikafishEngine, rotate_board_180
from core.position_validation import is_structurally_valid_board
from core.xiangqi import START_FEN, apply_move, generate_legal_moves, is_legal_move, parse_uci
from server.sync_server import SyncServer
from server.dashboard import WEB_DASHBOARD_HTML


def moved(board, uci):
    move = parse_uci(board, uci)
    assert move is not None and is_legal_move(board, move, move.piece[0]), uci
    return apply_move(board, move)


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("等待状态超时")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.start = fen_to_matrix(START_FEN)[0]

    def tracker(self, board, side=None):
        tracker = BoardDebouncer(2, 0)
        tracker.update(matrix_to_fen(board), board)
        if side:
            tracker.active_side = side
        return tracker

    def confirm(self, tracker, board, hint=None):
        tracker.update(matrix_to_fen(board), board, hint)
        return tracker.update(matrix_to_fen(board), board, hint)

    def test_real_visual_turn_hints(self):
        recognizer = TiantianRecognizer()
        cases = [
            ("yyb_adb", "tiantian_adb.png", None),
            ("yyb_adb", "tiantian_selected_cannon.png", "b"),
            ("coregraphics", "tiantian_host_probe.png", "b"),
            ("coregraphics", "tiantian_battle_current.png", "r"),
        ]
        for source, name, side in cases:
            with self.subTest(name=name):
                recognizer.capture_source = source
                image = cv2.imread("test_images/" + name)
                self.assertIsNotNone(image)
                result = recognizer.recognize(image)
                self.assertEqual(result["last_move_side"], side)
                tracker = BoardDebouncer(2, 0)
                tracker.update(result["fen"], result["board"], side)
                if side:
                    self.assertEqual(tracker.active_side, "unknown")
                    event = tracker.update(result["fen"], result["board"], side)
                    self.assertEqual(event["event_type"], "turn_confirmed")
                    self.assertEqual(tracker.active_side, "b" if side == "r" else "r")
                else:
                    self.assertEqual(tracker.active_side, "r")

    def test_wechat_miniprogram_resizes_keep_full_board_and_move_hint(self):
        image = cv2.imread("test_images/tiantian_wechat_miniprogram.png")
        self.assertIsNotNone(image)
        expected = (
            "rn1akabnr/9/1c2b2c1/p1p1p1p1p/9/6P2/"
            "P1P1P3P/1C5C1/9/RNBAKABNR"
        )
        background = (55, 43, 31)
        variants = {
            "small": cv2.resize(
                image, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA,
            ),
            "large": cv2.resize(
                image, None, fx=1.30, fy=1.30, interpolation=cv2.INTER_CUBIC,
            ),
            "wide": cv2.copyMakeBorder(
                image, 0, 0, 500, 500, cv2.BORDER_CONSTANT,
                value=background,
            ),
            "narrow": image[:, 400:-400],
        }
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        for name, frame in variants.items():
            with self.subTest(name=name, shape=frame.shape[:2]):
                result = recognizer.recognize(frame)
                self.assertTrue(result["recognition_valid"])
                self.assertEqual(result["recognition_profile"], "wechat_miniprogram")
                self.assertEqual(result["piece_count"], 32)
                self.assertEqual(result["occupied_count"], 32)
                self.assertEqual(result["fen"].split()[0], expected)
                self.assertEqual(result["last_visual_move"], {
                    "side": "b", "piece": "b_b",
                    "from": {"row": 0, "col": 2},
                    "to": {"row": 2, "col": 4},
                })

    def test_highlighted_destination_piece_is_not_dropped(self):
        # Windows 天天象棋的最后落点灰环会降低绝对模板分，但正确字形
        # 仍远高于其它同色字形。来源白圈虽然红色像素很多，却没有字形分数。
        self.assertTrue(TiantianRecognizer._accept_piece_match(
            0.518, 0.303, "b", red_count=41, dark_count=541,
        ))
        self.assertTrue(TiantianRecognizer._accept_piece_match(
            0.483, 0.410, "b", red_count=44, dark_count=634,
        ))
        self.assertFalse(TiantianRecognizer._accept_piece_match(
            0.161, 0.148, "r", red_count=816, dark_count=0,
        ))

    def test_miniprogram_wood_texture_is_not_occupied(self):
        recognizer = TiantianRecognizer()
        # Measured from the empty lower-right cell in a live mini-program
        # frame: red-brown wood passes the ink prefilter, but the best glyph
        # match is only 0.398. A real halo-covered piece remains above 0.42.
        self.assertFalse(recognizer._accept_occupancy_match(
            0.3985, recognizer.mini_program_templates,
        ))
        self.assertTrue(recognizer._accept_occupancy_match(
            0.483, recognizer.mini_program_templates,
        ))
        self.assertTrue(recognizer._accept_occupancy_match(
            0.36, recognizer.host_templates,
        ))

    def test_grey_highlighted_black_knight_is_kept(self):
        image = cv2.imread("test_images/tiantian_highlighted_knight.png")
        self.assertIsNotNone(image)
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        result = recognizer.recognize(image)
        self.assertTrue(result["recognition_valid"])
        self.assertEqual(result["piece_count"], 25)
        self.assertEqual(result["board"][2][8], "b_n")
        self.assertEqual(result["board"][5][3], "b_n")

    def test_real_highlighted_black_knight_commits_in_first_frame(self):
        image = cv2.imread("test_images/tiantian_highlighted_black_knight_live.png")
        self.assertIsNotNone(image)
        # 该样本包含左侧网页；右侧 1015px 是原始天天象棋窗口画面。
        game = image[:, 2060:]
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        result = recognizer.recognize(game)
        self.assertTrue(result["recognition_valid"])
        self.assertEqual(result["board"][2][8], "b_n")
        self.assertEqual(result["last_visual_move"], {
            "side": "b", "piece": "b_n",
            "from": {"row": 1, "col": 6},
            "to": {"row": 2, "col": 8},
        })

        previous = [row[:] for row in result["board"]]
        previous[2][8] = None
        previous[1][6] = "b_n"
        tracker = self.tracker(previous, side="r")  # 故意模拟已漂移的回合锁。
        event = tracker.update(
            result["fen"], result["board"], result["last_move_side"],
            result["occupancy"], result["occupied_sides"],
            result["last_visual_move"],
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "move")
        self.assertEqual(event["move"]["piece"], "b_n")
        self.assertEqual(event["move"]["uci"], "g8i7")
        self.assertEqual(tracker.get_stable_state()[1], result["board"])
        self.assertEqual(tracker.active_side, "r")

    def test_shifted_sparse_board_is_rejected_by_absolute_piece_rules(self):
        valid = [row[:] for row in self.start]
        self.assertTrue(is_structurally_valid_board(valid))
        shifted = [[None] * 9 for _ in range(10)]
        for row in range(10):
            for col in range(1, 9):
                shifted[row][col - 1] = valid[row][col]
        self.assertFalse(is_structurally_valid_board(shifted))

    def test_current_sparse_endgame_keeps_black_knight_and_absolute_files(self):
        image = cv2.imread("test_images/tiantian_sparse_shift_regression.png")
        self.assertIsNotNone(image)
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        result = recognizer.recognize(image)
        self.assertTrue(result["recognition_valid"])
        self.assertEqual(result["board"][0][4], "b_k")
        self.assertEqual(result["board"][2][8], "b_n")
        self.assertEqual(result["board"][9][4], "r_k")

    def test_global_constraints_restore_ambiguous_corner_rooks(self):
        image = cv2.imread("test_images/tiantian_ambiguous_rooks.png")
        self.assertIsNotNone(image)
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        result = recognizer.recognize(image)
        self.assertTrue(result["recognition_valid"])
        self.assertEqual(result["piece_count"], 32)
        self.assertEqual(result["occupied_count"], 32)
        self.assertEqual(result["board"][0][0], "b_r")
        self.assertEqual(result["board"][0][8], "b_r")

    def test_general_move_stall_screenshot_cold_reads_complete_board(self):
        image = cv2.imread("test_images/tiantian_general_stall_regression.png")
        self.assertIsNotNone(image)
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        result = recognizer.recognize(image)
        self.assertTrue(result["recognition_valid"])
        self.assertEqual(result["piece_count"], 25)
        self.assertEqual(
            result["fen"].split()[0],
            "2RA1A2c/1C2K4/9/P3P3P/1NPn5/3N2p2/p1p1p3p/2n1b4/4a4/2bak2r1",
        )

    def test_highlighted_red_general_move_uses_occupancy_even_when_frame_is_invalid(self):
        image = cv2.imread("test_images/tiantian_red_king_move_regression.png")
        self.assertIsNotNone(image)
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        raw = recognizer.recognize(image)
        self.assertFalse(raw["recognition_valid"])
        self.assertEqual(raw["occupied_count"], 26)

        final = fen_to_matrix(
            "1cc2ABNR/4K4/B6C1/P3P1P1P/4C4/3n5/"
            "p5pNp/9/5r3/1rbakab2 w - - 0 1"
        )[0]
        previous = [row[:] for row in final]
        previous[1][4] = None
        previous[0][4] = "r_k"
        tracker = self.tracker(previous, "r")
        tracker.update(
            raw["fen"], raw["board"], raw["last_move_side"],
            raw["occupancy"], raw["occupied_sides"],
        )
        event = tracker.update(
            raw["fen"], raw["board"], raw["last_move_side"],
            raw["occupancy"], raw["occupied_sides"],
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["move"]["piece"], "r_k")
        self.assertEqual(event["move"]["uci"], "e9e8")
        self.assertEqual(tracker.last_stable_board, final)

    def test_stable_frames_required_before_first_board_is_published(self):
        tracker = BoardDebouncer(2, 0)
        tracker.initial_frames = 3
        tracker.initial_seconds = 0
        self.assertIsNone(tracker.update(START_FEN, self.start))
        self.assertIsNone(tracker.update(START_FEN, self.start))
        event = tracker.update(START_FEN, self.start)
        self.assertEqual(event["event_type"], "initial")
        self.assertEqual(tracker.last_stable_board, self.start)

    def test_standard_start_repairs_a_temporarily_hidden_piece(self):
        tracker = BoardDebouncer(2, 0)
        tracker.initial_frames = 2
        tracker.initial_seconds = 0
        observed = [row[:] for row in self.start]
        observed[9][1] = None
        occupancy = [[piece is not None for piece in row] for row in observed]
        sides = [[piece[0] if piece else None for piece in row] for row in observed]

        self.assertIsNone(tracker.update(
            matrix_to_fen(observed), observed, occupancy=occupancy,
            occupied_sides=sides,
        ))
        event = tracker.update(
            matrix_to_fen(observed), observed, occupancy=occupancy,
            occupied_sides=sides,
        )
        self.assertEqual(event["event_type"], "initial")
        self.assertEqual(tracker.last_stable_board, self.start)

    def test_unchanged_dynamic_grid_reuses_cell_recognition(self):
        image = cv2.imread("test_images/tiantian_host_probe.png")
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "coregraphics"
        with patch(
            "core.adaptive_recognizer.cv2.matchTemplate", wraps=cv2.matchTemplate,
        ) as matcher:
            first = recognizer.recognize(image)
            first_calls = matcher.call_count
            second = recognizer.recognize(image.copy())
            second_calls = matcher.call_count - first_calls
        self.assertEqual(first["board"], second["board"])
        self.assertGreater(first_calls, 0)
        self.assertGreater(second_calls, 0)
        self.assertLess(second_calls, first_calls)

    def test_marginal_glyph_is_not_stuck_in_empty_cell_cache(self):
        image = cv2.imread("test_images/tiantian_sparse_shift_regression.png")
        recognizer = TiantianRecognizer()
        recognizer.capture_source = "windows_graphics_capture"
        recognizer.recognize(image)
        grid = recognizer._verified_grid
        self.assertIsNotNone(grid)
        with patch.object(TiantianRecognizer, "_accept_piece_match", return_value=False):
            recognizer._recognize_grid(image, grid, recognizer.host_templates)
        # A visible piece rejected during a transition is intentionally absent
        # from the reusable result cache, so the next frame must classify it.
        self.assertNotIn((0, 3), recognizer._cell_result_cache)
        recovered = recognizer._recognize_grid(image, grid, recognizer.host_templates)
        self.assertEqual(recovered["board"][0][3], "b_r")

    def test_occupancy_infers_unique_legal_move_when_glyph_is_hidden(self):
        tracker = self.tracker(self.start, "r")
        final = moved(self.start, "a3a4")
        observed = [row[:] for row in final]
        observed[4][0] = None
        occupancy = [[piece is not None for piece in row] for row in final]
        sides = [[piece[0] if piece else None for piece in row] for row in final]
        tracker.update(matrix_to_fen(observed), observed, None, occupancy, sides)
        event = tracker.update(matrix_to_fen(observed), observed, None, occupancy, sides)
        self.assertIsNotNone(event)
        self.assertEqual(event["move"]["uci"], "a3a4")
        self.assertEqual(tracker.last_stable_board, final)

    def test_dashboard_always_reenters_live_follow_mode(self):
        self.assertIn("post('/api/ai/follow',{})", WEB_DASHBOARD_HTML)
        self.assertIn("实时只读 · 合法走子跟踪", WEB_DASHBOARD_HTML)
        self.assertNotIn('id="btn-play"', WEB_DASHBOARD_HTML)
        self.assertIn("setInterval(refreshSnapshot,600)", WEB_DASHBOARD_HTML)
        self.assertIn("revision<latestSessionRevision", WEB_DASHBOARD_HTML)
        self.assertIn("server_instance_id", WEB_DASHBOARD_HTML)
        self.assertIn("cache:'no-store'", WEB_DASHBOARD_HTML)
        self.assertIn('id="btn-reset"', WEB_DASHBOARD_HTML)
        self.assertIn("post('/api/live/reset',{})", WEB_DASHBOARD_HTML)

    def test_hint_does_not_override_known_turn(self):
        middle = moved(self.start, "a3a4")
        tracker = self.tracker(middle, "b")
        for _ in range(5):
            tracker.update(matrix_to_fen(middle), middle, "b")
        self.assertEqual(tracker.active_side, "b")

    def test_unstable_hint_does_not_anchor(self):
        middle = moved(self.start, "a3a4")
        tracker = self.tracker(middle)
        for side in ("b", None, "r", "b", None):
            tracker.update(matrix_to_fen(middle), middle, side)
        self.assertEqual(tracker.active_side, "unknown")

    def test_two_plies_and_recapture_both_orientations(self):
        for first, second in (("a3a4", "a6a5"), ("h2h9", "i9h9")):
            final = moved(moved(self.start, first), second)
            for flipped in (False, True):
                with self.subTest(first=first, flipped=flipped):
                    before = rotate_board_180(self.start) if flipped else self.start
                    after = rotate_board_180(final) if flipped else final
                    tracker = self.tracker(before)
                    event = self.confirm(tracker, after)
                    self.assertEqual(event["event_type"], "catchup")
                    self.assertEqual(len(event["moves"]), 2)
                    self.assertEqual(tracker.active_side, "r")
                    self.assertEqual(tracker.last_stable_board, after)
                    self.assertEqual(tracker.session_revision, 1)

    def test_unknown_order_requires_evidence(self):
        middle = moved(self.start, "h2e2")
        final = moved(moved(middle, "a3a4"), "a6a5")
        tracker = self.tracker(middle)
        self.assertIsNone(self.confirm(tracker, final))
        self.assertEqual(tracker.last_stable_board, middle)
        event = self.confirm(tracker, final, "b")
        self.assertEqual(event["event_type"], "catchup")
        self.assertEqual(tracker.active_side, "r")

    def test_same_side_move_cannot_bypass_via_32_piece_reset(self):
        middle = moved(self.start, "a3a4")
        illegal = moved(middle, "c3c4")
        tracker = self.tracker(middle, "b")
        tracker.resync_seconds = 0
        for _ in range(8):
            self.assertIsNone(tracker.update(matrix_to_fen(illegal), illegal))
        self.assertEqual(tracker.last_stable_board, middle)
        self.assertEqual(tracker.session_revision, 1)

    def test_confirmed_last_mover_does_not_invent_three_ply_cycle(self):
        middle = moved(self.start, "a3a4")
        final = moved(middle, "a6a5")
        tracker = self.tracker(middle, "b")

        event = None
        for _ in range(3):
            event = tracker.update(matrix_to_fen(final), final, "b") or event

        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "move")
        self.assertEqual(event["move"]["uci"], "a6a5")
        self.assertEqual(tracker.last_stable_board, final)
        self.assertEqual(tracker.active_side, "r")

    def test_confirmed_visual_move_repairs_wrong_turn_lock(self):
        middle = moved(self.start, "a3a4")
        tracker = self.tracker(self.start, "b")
        tracker.update(matrix_to_fen(middle), middle, "r")
        event = tracker.update(matrix_to_fen(middle), middle, "r")
        self.assertIsNotNone(event)
        self.assertEqual(event["move"]["uci"], "a3a4")
        self.assertEqual(tracker.active_side, "b")

    def test_confirmed_general_move_repairs_wrong_turn_lock(self):
        moved_general = moved(self.start, "e9e8")
        tracker = self.tracker(self.start, "r")
        tracker.update(matrix_to_fen(moved_general), moved_general, "b")
        event = tracker.update(matrix_to_fen(moved_general), moved_general, "b")
        self.assertIsNotNone(event)
        self.assertEqual(event["move"]["uci"], "e9e8")
        self.assertEqual(tracker.active_side, "r")

    def test_manual_reset_clears_all_tiantian_tracking_caches(self):
        recognizer = TiantianRecognizer()
        recognizer._dynamic_shape = (100, 100)
        recognizer._verified_grid = recognizer.host_grid
        recognizer._cell_patch_cache[(0, 0)] = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer._cell_result_cache[(0, 0)] = {"piece": "r_r"}
        recognizer._cache_generation = 7
        recognizer.reset_tracking()
        self.assertIsNone(recognizer._dynamic_shape)
        self.assertIsNone(recognizer._verified_grid)
        self.assertEqual(recognizer._cell_patch_cache, {})
        self.assertEqual(recognizer._cell_result_cache, {})
        self.assertEqual(recognizer._cache_generation, 0)

    def test_manual_reanchor_completes_when_board_is_unchanged(self):
        tracker = self.tracker(self.start, "r")
        original_revision = tracker.session_revision
        tracker.request_reanchor()
        self.assertIsNone(tracker.update(START_FEN, self.start))
        event = tracker.update(START_FEN, self.start)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "manual_reanchor")
        self.assertEqual(tracker.last_stable_board, self.start)
        self.assertEqual(tracker.active_side, "r")
        self.assertEqual(tracker.session_revision, original_revision + 1)

    def test_verified_undo_returns_to_recorded_board_and_turn(self):
        first = moved(self.start, "a3a4")
        second = moved(first, "a6a5")
        tracker = self.tracker(self.start)
        self.assertEqual(self.confirm(tracker, first)["event_type"], "move")
        self.assertEqual(self.confirm(tracker, second)["event_type"], "move")
        tracker.resync_frames = 3
        tracker.resync_seconds = 0

        event = None
        for _ in range(4):
            event = tracker.update(matrix_to_fen(first), first) or event
        self.assertEqual(event["event_type"], "undo")
        self.assertEqual(tracker.last_stable_board, first)
        self.assertEqual(tracker.active_side, "b")
        self.assertEqual(tracker.session_revision, 2)

        alternative = moved(first, "c6c5")
        self.assertEqual(self.confirm(tracker, alternative)["event_type"], "move")
        self.assertEqual(tracker.active_side, "r")

    def test_reversible_rook_undo_is_not_misread_as_a_new_rook_move(self):
        first = moved(self.start, "a3a4")
        second = moved(first, "a9a8")
        tracker = self.tracker(self.start)
        self.confirm(tracker, first)
        self.confirm(tracker, second)
        tracker.resync_frames = 3
        tracker.resync_seconds = 0

        self.assertIsNone(tracker.update(matrix_to_fen(first), first, "b"))
        self.assertIsNone(tracker.update(matrix_to_fen(first), first, "b"))
        event = tracker.update(matrix_to_fen(first), first, "b")
        self.assertEqual(event["event_type"], "undo")
        self.assertEqual(tracker.active_side, "b")

    def test_undo_then_alternative_move_can_be_proven_without_rollback_frame(self):
        first = moved(self.start, "a3a4")
        second = moved(first, "a6a5")
        alternative = moved(first, "c6c5")
        tracker = self.tracker(self.start)
        self.confirm(tracker, first)
        self.confirm(tracker, second)
        tracker.resync_frames = 3
        tracker.resync_seconds = 0
        visual = {
            "side": "b", "piece": "b_p",
            "from": {"row": 3, "col": 2},
            "to": {"row": 4, "col": 2},
        }

        event = None
        for _ in range(4):
            event = tracker.update(
                matrix_to_fen(alternative), alternative,
                last_move_side="b", last_visual_move=visual,
            ) or event
        self.assertEqual(event["event_type"], "undo_branch")
        self.assertEqual(event["move"]["uci"], "c6c5")
        self.assertEqual(tracker.last_stable_board, alternative)
        self.assertEqual(tracker.active_side, "r")

    def test_undo_requires_complete_stable_observation(self):
        first = moved(self.start, "a3a4")
        second = moved(first, "a6a5")
        tracker = self.tracker(self.start)
        self.confirm(tracker, first)
        self.confirm(tracker, second)
        tracker.resync_frames = 3
        tracker.resync_seconds = 0

        for _ in range(8):
            self.assertIsNone(tracker.update(
                matrix_to_fen(first), first, complete_observation=False,
            ))
        self.assertEqual(tracker.last_stable_board, second)

    def test_manual_reset_after_undo_uses_historical_turn_not_stale_marker(self):
        first = moved(self.start, "a3a4")
        second = moved(first, "a6a5")
        tracker = self.tracker(self.start)
        self.confirm(tracker, first)
        self.confirm(tracker, second)
        tracker.request_reanchor()

        event = self.confirm(tracker, first, "r")
        self.assertEqual(event["event_type"], "manual_reanchor")
        self.assertEqual(tracker.last_stable_board, first)
        self.assertEqual(tracker.active_side, "b")

    def test_manual_reanchor_cannot_commit_isolated_piece_loss(self):
        middle = moved(self.start, "a3a4")
        tracker = self.tracker(middle, "b")
        tracker.request_reanchor()
        missing = [row[:] for row in middle]
        missing[9][1] = None

        for _ in range(12):
            self.assertIsNone(tracker.update(matrix_to_fen(missing), missing))

        self.assertEqual(tracker.last_stable_board, middle)
        self.assertEqual(tracker.session_revision, 1)
        self.assertIn("孤立少子", tracker.last_rejection_reason)

    def test_persistent_missing_piece_never_reanchors_same_session(self):
        tracker = self.tracker(self.start)
        tracker.resync_frames = 2
        tracker.resync_seconds = 0
        missing = [row[:] for row in self.start]
        missing[9][1] = None

        for _ in range(20):
            event = tracker.update(matrix_to_fen(missing), missing, "b")
            self.assertIsNone(event)

        self.assertEqual(tracker.last_stable_board, self.start)
        self.assertIn("r_n", [piece for row in tracker.last_stable_board for piece in row])

    def test_unproved_two_cell_relocation_never_uses_whole_board_reanchor(self):
        tracker = self.tracker(self.start)
        tracker.resync_frames = 2
        tracker.resync_seconds = 0
        corrupted = [row[:] for row in self.start]
        corrupted[9][1] = None
        corrupted[5][1] = "r_n"  # 马不能从 (1, 9) 一步到 (1, 5)

        for _ in range(20):
            event = tracker.update(matrix_to_fen(corrupted), corrupted, "r")
            self.assertIsNone(event)

        self.assertEqual(tracker.last_stable_board, self.start)

    def test_empty_capture_gap_never_reanchors_an_arbitrary_midgame(self):
        tracker = self.tracker(moved(self.start, "a3a4"), "b")
        tracker.resync_frames = 2
        tracker.resync_seconds = 0
        before = [row[:] for row in tracker.last_stable_board]
        for _ in range(3):
            tracker.note_unstable_source(0)
        wrong = [row[:] for row in before]
        wrong[9][1] = None
        wrong[5][1] = "r_n"  # Cannot be reached by one legal horse move.

        for _ in range(12):
            self.assertIsNone(tracker.update(matrix_to_fen(wrong), wrong, "r"))

        self.assertEqual(tracker.last_stable_board, before)
        self.assertEqual(tracker.session_revision, 1)

    def test_partial_two_ply_frame_is_not_committed_as_one(self):
        first = moved(self.start, "a3a4")
        final = moved(first, "a6a5")
        tracker = self.tracker(self.start)
        tracker.update(matrix_to_fen(first), first)
        # 红方两格已经双帧稳定、黑方两格只有一帧，不能先提交红步后丢失黑步。
        self.assertIsNone(tracker.update(matrix_to_fen(final), final))
        event = tracker.update(matrix_to_fen(final), final)
        self.assertEqual(event["event_type"], "catchup")
        self.assertEqual(tracker.active_side, "r")

    def test_four_ply_lag_is_proven_before_catchup(self):
        tracker = self.tracker(self.start)
        tracker.resync_seconds = 0
        final = self.start
        for uci in ("a3a4", "a6a5", "c3c4", "c6c5"):
            final = moved(final, uci)
        event = None
        for _ in range(4):
            event = tracker.update(matrix_to_fen(final), final, "b") or event
        self.assertEqual(event["event_type"], "catchup")
        self.assertEqual(len(event["moves"]), 4)
        self.assertEqual(tracker.active_side, "r")
        self.assertEqual(tracker.session_revision, 1)

    def test_four_ply_catchup_proves_real_captures(self):
        tracker = self.tracker(self.start, "r")
        final = self.start
        for uci in ("h2h9", "i9h9", "a3a4", "a6a5"):
            final = moved(final, uci)

        event = self.confirm(tracker, final, "b")
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "catchup")
        self.assertEqual(len(event["moves"]), 4)
        self.assertEqual(sum(piece is not None for row in final for piece in row), 30)
        self.assertEqual(tracker.last_stable_board, final)

    def test_persistent_false_addition_never_creates_ghost_piece(self):
        middle = moved(self.start, "a3a4")
        tracker = self.tracker(middle, "b")
        corrupted = [row[:] for row in middle]
        corrupted[5][8] = "r_n"

        for _ in range(20):
            self.assertIsNone(tracker.update(
                matrix_to_fen(corrupted), corrupted, "r",
            ))

        self.assertEqual(tracker.last_stable_board, middle)

    def test_pseudo_legal_move_exposing_flying_kings_is_rejected(self):
        board = [[None] * 9 for _ in range(10)]
        board[0][4] = "b_k"
        board[5][4] = "r_r"
        board[9][4] = "r_k"
        tracker = self.tracker(board, "r")
        illegal = [row[:] for row in board]
        illegal[5][4] = None
        illegal[5][5] = "r_r"

        for _ in range(6):
            self.assertIsNone(tracker.update(matrix_to_fen(illegal), illegal, "r"))

        self.assertEqual(tracker.last_stable_board, board)

    def test_simulation_reanchors_recovered_board_and_turn(self):
        middle = moved(self.start, "a3a4")
        advisor = AIAdvisor(enabled=False, engine_kind="builtin")
        try:
            advisor.on_live_position(middle, "unknown", 1)
            advisor.start_local_game()
            advisor.on_live_position(middle, "b", 1)
            self.assertEqual(advisor.snapshot()["to_move"], "b")
            final = moved(moved(middle, "a6a5"), "c3c4")
            advisor.on_live_position(final, "b", 1)
            self.assertEqual(advisor.snapshot()["board"], final)
            self.assertEqual(advisor.snapshot()["to_move"], "b")
            self.assertNotIn("等待盘面稳定", advisor.snapshot()["status"])
        finally:
            advisor.stop()

    def test_cancelled_mode_search_cannot_write_back(self):
        for destination_mode in ("play", "coach"):
            with self.subTest(mode=destination_mode):
                started = threading.Event()
                release = threading.Event()
                calls = []

                def search(engine, board, side, **kwargs):
                    calls.append(side)
                    if len(calls) == 1:
                        started.set()
                        if not release.wait(3):
                            raise TimeoutError("测试未释放搜索")
                    move = generate_legal_moves(board, side)[0]
                    return SearchResult(move, 0, 1, 1, 1, [move])

                with patch("core.ai_advisor.XiangqiEngine.search", search):
                    advisor = AIAdvisor(enabled=False, engine_kind="builtin")
                    try:
                        advisor.on_live_position(self.start, "r", 1)
                        if destination_mode == "coach":
                            advisor.start_local_game()
                        advisor.configure(enabled=True)
                        self.assertTrue(started.wait(2))
                        if destination_mode == "play":
                            advisor.start_local_game()
                        else:
                            advisor.follow_live()
                        release.set()
                        wait_for(lambda: not advisor.snapshot()["thinking"])
                        snap = advisor.snapshot()
                        self.assertEqual(len(calls), 2)
                        self.assertEqual(snap["mode"], destination_mode)
                        self.assertEqual(len(snap["history"]), 1 if destination_mode == "play" else 0)
                        self.assertEqual(snap["to_move"], "b" if destination_mode == "play" else "r")
                    finally:
                        release.set()
                        advisor.stop()
                        if advisor._worker:
                            advisor._worker.join(3)

    def test_pikafish_stop_cannot_overtake_go(self):
        engine = PikafishEngine(threads=1, hash_mb=16)
        entering_go = threading.Event()
        release_go = threading.Event()
        commands = []
        errors = []
        send = engine._send

        def delayed_send(command):
            if command.startswith("go "):
                entering_go.set()
                if not release_go.wait(3):
                    raise TimeoutError("测试未释放 go")
            commands.append(command.split()[0])
            send(command)

        def search():
            try:
                engine.search(self.start, "r", time_ms=3000)
            except Exception as exc:
                errors.append(exc)

        engine._send = delayed_send
        worker = threading.Thread(target=search)
        stopper = threading.Thread(target=engine.request_stop)
        worker.start()
        try:
            self.assertTrue(entering_go.wait(3))
            stopper.start()
            self.assertTrue(engine._cancel_requested.wait(1))
            release_go.set()
            stopper.join(1)
            worker.join(2)
            self.assertFalse(worker.is_alive(), "stop 未终止当前搜索")
            self.assertEqual(errors, [])
            self.assertLess(commands.index("go"), commands.index("stop"))
        finally:
            release_go.set()
            engine.close()
            worker.join(3)
            if stopper.ident:
                stopper.join(3)

    def test_search_exception_exits_thinking(self):
        with patch("core.ai_advisor.XiangqiEngine.search", side_effect=RuntimeError("回归故障")):
            advisor = AIAdvisor(engine_kind="builtin")
            try:
                advisor.on_live_position(self.start, "r", 1)
                wait_for(lambda: not advisor.snapshot()["thinking"])
                self.assertIn("回归故障", advisor.snapshot()["error"])
                self.assertIsNone(advisor.snapshot()["suggestion"])
            finally:
                advisor.stop()

    def capture(self):
        with patch.object(MacCapture, "_load_quartz"), patch.object(MacCapture, "_discover_yyb_adb"):
            return MacCapture()

    def test_adb_display_cache_refreshes(self):
        capture = self.capture()
        capture._adb_display_id = "old"
        capture._display_checked_at = time.monotonic() - 3
        output = ("DisplayViewport{type=EXTERNAL, valid=true, isActive=false, uniqueId='local:11'} "
                  "DisplayViewport{type=EXTERNAL, valid=true, isActive=true, uniqueId='local:22'}")
        capture._adb = Mock(return_value=subprocess.CompletedProcess([], 0, output))
        self.assertEqual(capture._get_adb_display_id(), "22")
        self.assertEqual(capture._get_adb_display_id(), "22")
        self.assertEqual(capture._adb.call_count, 1)
        capture._display_checked_at = time.monotonic() - 3
        capture._adb.return_value = subprocess.CompletedProcess([], 0, "")
        self.assertIsNone(capture._get_adb_display_id())
        self.assertIsNotNone(capture.last_error)

    def test_adb_decode_failure_invalidates_display(self):
        capture = self.capture()
        capture._adb_display_id = "22"
        capture._display_checked_at = time.monotonic()
        capture._adb = Mock(return_value=subprocess.CompletedProcess([], 0, struct.pack("<4I", 10, 10, 1, 0)))
        self.assertIsNone(capture._capture_tiantian_adb())
        self.assertIsNone(capture._adb_display_id)
        self.assertIn("解码失败", capture.last_error)

    def test_tiantian_adb_does_not_request_screen_permission(self):
        capture = self.capture()
        capture.game_id = "tiantian"
        capture._detect_yyb_game = Mock(return_value="tiantian")
        capture._adb_path = "/existing/adb"
        capture._adb_device = "emulator-5554"
        capture._quartz = Mock()
        self.assertTrue(capture.request_screen_permission())
        self.assertEqual(capture.permission_state, "not_required")
        capture._quartz.CGRequestScreenCaptureAccess.assert_not_called()

    def test_window_reopen_discards_old_source_cache(self):
        capture = self.capture()
        capture.target_wid = 1
        capture._adb_display_id = "old"
        capture._tiantian_source = "adb"
        window = {"id": 2, "app": "腾讯应用宝", "title": "天天象棋",
                  "display_name": "腾讯应用宝 - 天天象棋", "bounds": (0, 0, 440, 818)}
        capture.list_windows = Mock(return_value=[window])
        self.assertEqual(capture.find_target_window()["id"], 2)
        self.assertIsNone(capture._adb_display_id)
        self.assertEqual(capture._tiantian_source, "auto")
        capture.list_windows.return_value = []
        self.assertIsNone(capture.find_target_window())
        self.assertIsNone(capture.target_window_info)

    def test_host_failure_fallback_and_adb_stability(self):
        capture = self.capture()
        capture.game_id = "tiantian"
        capture._detect_yyb_game = Mock()
        capture._quartz = Mock()
        capture._quartz.CGPreflightScreenCaptureAccess.return_value = True
        capture.find_target_window = Mock(return_value={"id": 2})
        capture._cgimage_to_bgr = Mock(return_value=None)
        self.assertIsNone(capture.capture())
        self.assertEqual(capture._tiantian_source, "adb")
        frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) * 20
        capture._capture_tiantian_adb = Mock(return_value=frame)
        capture._last_window_probe = time.monotonic()
        self.assertIs(capture.capture(), frame)
        window_capture_calls = capture._cgimage_to_bgr.call_count
        # ADB 可持续识别时不应定时切回几何已变化的宿主窗口。
        capture._last_window_probe = time.monotonic() - 3
        self.assertIs(capture.capture(), frame)
        self.assertEqual(capture._cgimage_to_bgr.call_count, window_capture_calls)
        capture.capture_source = "yyb_adb"  # 真实 _capture_tiantian_adb 会设置来源。
        capture.report_recognition(32, True)
        self.assertEqual(capture._tiantian_source, "adb")

    def test_adb_requires_consecutive_bad_frames_before_window_fallback(self):
        capture = self.capture()
        capture.game_id = "tiantian"
        capture._tiantian_source = "adb"
        capture.capture_source = "yyb_adb"
        capture._quartz = Mock()
        capture._quartz.CGPreflightScreenCaptureAccess.return_value = True
        for _ in range(3):
            capture.report_recognition(18, False)
            self.assertEqual(capture._tiantian_source, "adb")
        capture.report_recognition(18, False)
        self.assertEqual(capture._tiantian_source, "window")

    def test_capture_pipeline_hint_and_live_overlay(self):
        middle = moved(self.start, "a3a4")

        class Recognizer:
            def recognize(self, image):
                return {"fen": matrix_to_fen(middle), "board": middle,
                        "piece_count": 32, "last_move_side": "r"}

        sockets = [socket.socket(), socket.socket()]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        server = SyncServer(
            MockCapture(["test_images/board_test_01.png"]), recognizer=Recognizer(),
            ws_port=ports[0], http_port=ports[1], interval=0.02, idle_timeout=0,
            ai_enabled=False,
        )
        server.start()
        try:
            wait_for(lambda: server.get_latest_state().get("side_to_move") == "b")
            state = server.get_latest_state()
            self.assertFalse(state["recognition_pending"])
            self.assertEqual(state["fen"].split()[1], "b")
            self.assertEqual(state["ai"]["to_move"], "b")
            server.advisor.start_local_game(to_move="r")
            state = server.publish_ai()
            self.assertEqual(state["side_to_move"], "b")
            self.assertEqual(state["ai"]["to_move"], "r")
            self.assertEqual(state["board"], middle)
        finally:
            server.stop()

    def test_chat_overlay_keeps_last_board_and_recovers_automatically(self):
        start = [row[:] for row in self.start]
        after = moved(start, "a3a4")

        class OverlayRecognizer:
            def __init__(self):
                self.mode = "board"
                self.board = start

            def recognize(self, image):
                if self.mode == "overlay":
                    empty = [[None] * 9 for _ in range(10)]
                    return {
                        "fen": matrix_to_fen(empty), "board": empty,
                        "piece_count": 0, "recognition_valid": False,
                    }
                return {
                    "fen": matrix_to_fen(self.board), "board": self.board,
                    "piece_count": sum(piece is not None for row in self.board for piece in row),
                    "recognition_valid": True,
                    "last_move_side": "r" if self.board == after else None,
                }

        recognizer = OverlayRecognizer()
        sockets = [socket.socket(), socket.socket()]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        server = SyncServer(
            MockCapture(["test_images/board_test_01.png"]), recognizer=recognizer,
            ws_port=ports[0], http_port=ports[1], interval=0.01,
            idle_timeout=0, ai_enabled=False,
        )
        server.start()
        try:
            wait_for(lambda: server.get_latest_state().get("piece_count") == 32)
            stable_before = server.get_latest_state()["board"]
            recognizer.mode = "overlay"
            wait_for(lambda: server.get_latest_state().get("raw_piece_count") == 0)
            covered = server.get_latest_state()
            self.assertEqual(covered["board"], stable_before)
            self.assertEqual(covered["piece_count"], 32)
            self.assertTrue(covered["recognition_pending"])
            self.assertIn("聊天或弹窗遮挡", covered["recognition_rejection"])

            # 即使用户在遮挡期间点“重新读取”，也必须保留锚点，并在短暂
            # 超时后自动退出重读状态，不能永久卡在“再次尝试”。
            original_revision = covered["session_revision"]
            server._manual_reset_timeout = 0.15
            resetting = server.request_live_reset()
            self.assertEqual(resetting["board"], stable_before)
            self.assertTrue(resetting["manual_reset_pending"])
            wait_for(
                lambda: not server.get_latest_state().get("manual_reset_pending"),
                timeout=2,
            )
            timed_out = server.get_latest_state()
            self.assertEqual(timed_out["board"], stable_before)
            self.assertEqual(timed_out["session_revision"], original_revision)

            recognizer.board = after
            recognizer.mode = "board"
            wait_for(lambda: server.get_latest_state().get("board") == after)
            recovered = server.get_latest_state()
            self.assertFalse(recovered["recognition_pending"])
            self.assertEqual(recovered["piece_count"], 32)
        finally:
            server.stop()

    def test_manual_reset_rejects_partially_classified_frame(self):
        start = [row[:] for row in self.start]
        degraded = [row[:] for row in start]
        degraded[9][1] = None
        occupied = [[piece is not None for piece in row] for row in start]

        class ResetRecognizer:
            def __init__(self):
                self.degraded = False

            def recognize(self, _image):
                board = degraded if self.degraded else start
                return {
                    "fen": matrix_to_fen(board),
                    "board": board,
                    "piece_count": 31 if self.degraded else 32,
                    "occupied_count": 32,
                    "occupancy": occupied,
                    "occupied_sides": [
                        [piece[0] if piece else None for piece in row]
                        for row in start
                    ],
                    "recognition_valid": True,
                    "last_move_side": None,
                }

        recognizer = ResetRecognizer()
        sockets = [socket.socket(), socket.socket()]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        server = SyncServer(
            MockCapture(["test_images/board_test_01.png"]), recognizer=recognizer,
            ws_port=ports[0], http_port=ports[1], interval=0.01,
            idle_timeout=0, ai_enabled=False,
        )
        server.start()
        try:
            wait_for(lambda: server.get_latest_state().get("piece_count") == 32)
            original_revision = server.get_latest_state()["session_revision"]
            recognizer.degraded = True
            server.request_live_reset()
            wait_for(lambda: server.get_latest_state().get("raw_piece_count") == 31)
            held = server.get_latest_state()
            self.assertEqual(held["board"], start)
            self.assertEqual(held["piece_count"], 32)
            self.assertTrue(held["manual_reset_pending"])
            self.assertIn("连续完整盘面", held["recognition_rejection"])

            recognizer.degraded = False
            wait_for(
                lambda: not server.get_latest_state().get("manual_reset_pending"),
                timeout=2,
            )
            restored = server.get_latest_state()
            self.assertEqual(restored["board"], start)
            self.assertEqual(restored["piece_count"], 32)
            self.assertEqual(restored["session_revision"], original_revision + 1)
        finally:
            server.stop()

    def test_server_commits_general_move_from_degraded_occupancy_frame(self):
        final = fen_to_matrix(
            "1cc2ABNR/4K4/B6C1/P3P1P1P/4C4/3n5/"
            "p5pNp/9/5r3/1rbakab2 w - - 0 1"
        )[0]
        previous = [row[:] for row in final]
        previous[1][4] = None
        previous[0][4] = "r_k"
        image = cv2.imread("test_images/tiantian_red_king_move_regression.png")
        real = TiantianRecognizer()
        real.capture_source = "windows_graphics_capture"
        degraded = real.recognize(image)
        self.assertFalse(degraded["recognition_valid"])

        class SequenceRecognizer:
            def __init__(self):
                self.mode = "previous"

            def recognize(self, _image):
                if self.mode == "degraded":
                    return degraded
                board = final if self.mode == "final" else previous
                return {
                    "fen": matrix_to_fen(board), "board": board,
                    "piece_count": 26, "recognition_valid": True,
                    "last_move_side": "r" if self.mode == "final" else None,
                }

        recognizer = SequenceRecognizer()
        sockets = [socket.socket(), socket.socket()]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        server = SyncServer(
            MockCapture(["test_images/board_test_01.png"]), recognizer=recognizer,
            ws_port=ports[0], http_port=ports[1], interval=0.01,
            idle_timeout=0, ai_enabled=False,
        )
        server.start()
        try:
            wait_for(lambda: server.get_latest_state().get("board") == previous)
            recognizer.mode = "degraded"
            wait_for(lambda: server.get_latest_state().get("board") == final)
            moved_state = server.get_latest_state()
            self.assertEqual(moved_state["last_move"]["piece"], "r_k")
            self.assertEqual(moved_state["last_move"]["uci"], "e9e8")
            recognizer.mode = "final"
            wait_for(lambda: not server.get_latest_state().get("recognition_pending"))
        finally:
            server.stop()

    def test_capture_and_recognition_exceptions_recover_without_losing_board(self):
        start = [row[:] for row in self.start]
        after = moved(start, "a3a4")

        class FaultyCapture(MockCapture):
            def __init__(self):
                super().__init__(["test_images/board_test_01.png"])
                self.raise_error = False

            def capture(self):
                if self.raise_error:
                    raise RuntimeError("temporary capture failure")
                return super().capture()

        class FaultyRecognizer:
            def __init__(self):
                self.board = start
                self.raise_error = False
                self.reset_calls = 0

            def recognize(self, _image):
                if self.raise_error:
                    raise RuntimeError("temporary recognition failure")
                return {
                    "fen": matrix_to_fen(self.board),
                    "board": self.board,
                    "piece_count": 32,
                    "recognition_valid": True,
                    "last_move_side": "r" if self.board == after else None,
                }

            def reset_tracking(self):
                self.reset_calls += 1

        sockets = [socket.socket(), socket.socket()]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        capture = FaultyCapture()
        recognizer = FaultyRecognizer()
        server = SyncServer(
            capture, recognizer=recognizer,
            ws_port=ports[0], http_port=ports[1], interval=0.02,
            idle_timeout=0, ai_enabled=False,
        )
        server.start()
        try:
            wait_for(lambda: server.get_latest_state().get("board") == start)

            capture.raise_error = True
            wait_for(lambda: server.get_latest_state().get("capture_status") == "error")
            self.assertEqual(server.get_latest_state()["board"], start)
            self.assertTrue(server._capture_thread.is_alive())
            capture.raise_error = False
            wait_for(lambda: server.get_latest_state().get("capture_status") == "ok")

            recognizer.raise_error = True
            wait_for(
                lambda: server.get_latest_state().get("recognition_rejection")
                == "识别遇到临时异常，正在自动重建"
            )
            self.assertEqual(server.get_latest_state()["board"], start)
            self.assertTrue(server._capture_thread.is_alive())
            recognizer.board = after
            recognizer.raise_error = False
            wait_for(lambda: server.get_latest_state().get("board") == after)
            self.assertGreater(recognizer.reset_calls, 0)
            self.assertFalse(server.get_latest_state()["recognition_pending"])
        finally:
            server.stop()

    def test_adaptive_polling_slows_only_stable_frames(self):
        server = SyncServer(
            MockCapture(["test_images/board_test_01.png"]),
            ai_enabled=False, idle_timeout=0, interval=0.12,
        )
        self.assertEqual(
            server._effective_poll_interval({"source": "mock"}, has_stable_board=True),
            0.12,
        )
        self.assertEqual(
            server._effective_poll_interval({"source": "coregraphics"}, has_stable_board=True),
            0.30,
        )
        self.assertEqual(
            server._effective_poll_interval({"source": "coregraphics"}),
            0.5,
        )
        self.assertEqual(
            server._effective_poll_interval({"source": "yyb_adb"}, has_stable_board=True),
            0.75,
        )
        self.assertEqual(
            server._effective_poll_interval(
                {"source": "yyb_adb"}, has_stable_board=True, recognition_pending=True,
            ),
            0.12,
        )
        stable = [[None] * 9 for _ in range(10)]
        stable[3][0] = "r_p"
        moved_board = [row[:] for row in stable]
        moved_board[3][0] = None
        moved_board[4][0] = "r_p"
        noisy_board = [row[:] for row in stable]
        noisy_board[3][0] = None
        self.assertTrue(server._looks_like_single_move(stable, moved_board))
        self.assertFalse(server._looks_like_single_move(stable, noisy_board))


def run_recovery_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("实盘追帧与搜索取消回归失败")


if __name__ == "__main__":
    unittest.main()
