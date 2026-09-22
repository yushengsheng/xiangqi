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
from core.xiangqi import START_FEN, apply_move, generate_legal_moves, is_legal_move, parse_uci
from server.sync_server import SyncServer


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

    def test_resync_uses_confirmed_hint(self):
        tracker = self.tracker(self.start)
        tracker.resync_seconds = 0
        final = self.start
        for uci in ("a3a4", "a6a5", "c3c4", "c6c5"):
            final = moved(final, uci)
        event = None
        for _ in range(4):
            event = tracker.update(matrix_to_fen(final), final, "b") or event
        self.assertEqual(event["event_type"], "session_reset")
        self.assertEqual(tracker.active_side, "r")

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
