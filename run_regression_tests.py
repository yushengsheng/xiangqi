"""
完整自动化回归测试套件 (Full Automated Regression Test Suite)
全面验证:
1. FEN 解析与矩阵双向转换
2. 棋盘网格标定与多分辨率自适应
3. 棋子 14 种标准模板与图像识别精度 (100% 匹配真实残局)
4. 走子去抖与吃子状态机 (含 UCI 走法输出)
5. 指定窗口捕获器与 Mock 模拟层
6. WebSocket 实时流与 HTTP API 端到端通信
"""

import sys
import time
import json
import hashlib
import urllib.request
import urllib.error
import asyncio
import socket
import tempfile
import subprocess
import threading
import os
from pathlib import Path
import numpy as np
import cv2
import websockets

from core.fen import matrix_to_fen, fen_to_matrix, board_to_text, PIECE_TO_FEN
from core.board_calibrator import BoardGrid, BoardConfig
from core.recognizer import BoardRecognizer
from core.adaptive_recognizer import TiantianRecognizer
from core.debouncer import BoardDebouncer
from core.xiangqi import (
    START_FEN, FLIPPED_START_FEN, apply_move, generate_legal_moves, infer_bottom_side,
    is_legal_move, move_to_chinese, parse_uci,
)
from core.engine import XiangqiEngine
from core.ai_advisor import AIAdvisor
from core.pikafish_engine import (
    PIKAFISH_BINARY_SHA256, PIKAFISH_NNUE_SHA256,
    PikafishEngine, rotate_board_180,
)
from capture.mock_capture import MockCapture
from capture.mac_capture import MacCapture
from capture.factory import create_capture, list_available_windows
from server.sync_server import SyncServer
from tests.test_sync_recovery import run_recovery_tests
from tests.test_launcher_app import (
    test_app_bundle_structure,
    test_cold_start_and_service,
    test_runtime_communication,
    test_clean_stop
)


def test_fen_conversion():
    print("[Test 1/6] 正在回归 FEN 解析与双向转换...")
    start_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    matrix, color, half, full = fen_to_matrix(start_fen)
    assert len(matrix) == 10 and all(len(r) == 9 for r in matrix), "矩阵维度不是 10x9"
    assert color == "w"
    
    fen_back = matrix_to_fen(matrix, color, half, full)
    assert fen_back == start_fen, f"FEN 回转不匹配: {fen_back}"

    # 测试残局 FEN
    test_fen = "3k5/6R1C/3c5/6N2/6b2/9/9/9/2r1p1p2/5K3 w - - 0 1"
    m2, _, _, _ = fen_to_matrix(test_fen)
    fen_back_2 = matrix_to_fen(m2, "w", 0, 1)
    assert fen_back_2 == test_fen, "残局 FEN 回转不匹配"
    print("  ✓ FEN 解析与双向转换测试通过")


def test_board_calibrator():
    print("[Test 2/6] 正在回归棋盘网格定位与自适应缩放...")
    grid = BoardGrid()
    # 原始基准分辨率 918x1704
    pts_1 = grid.get_points(918, 1704)
    assert len(pts_1) == 90, "基准分辨率交叉点数不等于 90"
    assert pts_1[0] == (0, 0, 60, 449)
    assert pts_1[-1] == (8, 9, 858, 1315)

    # 缩放分辨率 (模拟器Retina或缩小窗口 459x852)
    cols_2, rows_2, r_2 = grid.compute_grid(459, 852)
    assert len(cols_2) == 9 and len(rows_2) == 10
    assert abs(cols_2[0] - 30.0) < 1.0
    assert abs(rows_2[-1] - 657.5) < 1.0
    print("  ✓ 棋盘网格自适应缩放测试通过")


def test_recognizer_accuracy():
    print("[Test 3/6] 正在回归核心图像识别引擎与全盘匹配精度...")
    recognizer = BoardRecognizer(templates_dir="templates/pieces")
    assert len(recognizer.templates) == 14, f"模板数量不足: {len(recognizer.templates)}"

    img = cv2.imread("test_images/board_test_01.png")
    assert img is not None, "未找到 test_images/board_test_01.png"

    result = recognizer.recognize(img)
    expected_fen = "3k5/6R1C/3c5/6N2/6b2/9/9/9/2r1p1p2/5K3 w - - 0 1"
    
    assert result["piece_count"] == 10, f"识别棋子数期望 10，实际为 {result['piece_count']}"
    assert result["fen"] == expected_fen, f"FEN 识别不一致:\n实际: {result['fen']}\n期望: {expected_fen}"
    print(f"  ✓ 真实盘面 10 颗棋子全部 100% 精确识别 (FEN: {result['fen']})")

    # macOS CoreGraphics 实际窗口帧：先验证黑边裁切，再验证完整的 19 子残局。
    raw_live = cv2.imread("test_images/live_coregraphics.png")
    assert raw_live is not None, "未找到真实 CoreGraphics 窗口帧"
    trimmed_live = MacCapture._trim_empty_border(raw_live)
    assert trimmed_live.shape == (1704, 918, 3), f"实时窗口黑边裁切失败: {trimmed_live.shape}"
    live_result = recognizer.recognize(trimmed_live)
    expected_live_fen = "3k4C/2P1a4/3n1a1RN/7r1/7n1/7p1/7p1/7pC/1crp1p3/4K4 w - - 0 1"
    assert live_result["piece_count"] == 19, f"实际窗口棋子数错误: {live_result['piece_count']}"
    assert live_result["fen"] == expected_live_fen, f"实际窗口 FEN 错误: {live_result['fen']}"
    print("  ✓ CoreGraphics 实际 JJ 窗口帧：裁切后 19 子与 FEN 均正确")

    selected_img = cv2.imread("test_images/per_cell_live.png")
    assert selected_img is not None, "未找到选中特效真实帧"
    selected_result = recognizer.recognize(selected_img)
    assert selected_result["piece_count"] == 19, f"选中特效帧漏子: {selected_result['fen']}"
    selected_rook = [p for p in selected_result["pieces_detail"] if p["row"] == 2 and p["col"] == 7]
    assert selected_rook and selected_rook[0]["piece"] == "r_r"
    assert selected_rook[0]["vertical_offset"] < 0, "未通过上浮位置识别选中红车"
    print("  ✓ 真实选中特效帧：上浮红车仍识别为 19 子")

    resized_img = cv2.imread("test_images/red_missing_live.png")
    assert resized_img is not None and resized_img.shape[:2] == (1954, 1058)
    perf_started = time.perf_counter()
    resized_result = recognizer.recognize(resized_img)
    resized_elapsed = time.perf_counter() - perf_started
    expected_resized = "5kc2/3r3R1/5a3/4p4/8C/7C1/9/9/3p1pp2/4K4 w - - 0 1"
    assert resized_result["piece_count"] == 12, f"放大窗口漏子: {resized_result['fen']}"
    assert resized_result["fen"] == expected_resized
    red_count = sum(p["side"] == "r" for p in resized_result["pieces_detail"])
    assert red_count == 4, f"放大窗口红棋应为 4，实际 {red_count}"
    assert resized_elapsed < 0.30, f"放大窗口单帧识别过慢: {resized_elapsed:.3f}s"
    print(f"  ✓ 1058x1954 放大窗口完整识别，单帧 {resized_elapsed*1000:.0f}ms")

    full_board_path = "test_images/full_board_live.png"
    assert os.path.exists(full_board_path), "缺少真实32子全盘回归图"
    full_board_img = cv2.imread(full_board_path)
    full_started = time.perf_counter()
    full_result = recognizer.recognize(full_board_img)
    full_elapsed = time.perf_counter() - full_started
    assert full_result["piece_count"] == 32, (
        f"完整32子盘面漏识别: {full_result['piece_count']}/32，{full_result['fen']}"
    )
    assert full_result["fen"] == START_FEN, full_result["fen"]
    piece_types = [item["piece"] for item in full_result["pieces_detail"]]
    expected_counts = {
        "r_r": 2, "r_n": 2, "r_b": 2, "r_a": 2, "r_k": 1, "r_c": 2, "r_p": 5,
        "b_r": 2, "b_n": 2, "b_b": 2, "b_a": 2, "b_k": 1, "b_c": 2, "b_p": 5,
    }
    assert {name: piece_types.count(name) for name in expected_counts} == expected_counts
    assert min(item["confidence"] for item in full_result["pieces_detail"]) >= 0.90
    assert full_elapsed < 0.30, f"32子全盘单帧识别过慢: {full_elapsed:.3f}s"
    for scale in (0.65, 1.50):
        scaled = cv2.resize(
            full_board_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        scaled_result = recognizer.recognize(scaled)
        assert scaled_result["piece_count"] == 32 and scaled_result["fen"] == START_FEN, (
            scale, scaled_result["piece_count"], scaled_result["fen"]
        )
    print(f"  ✓ 真实完整开局32子逐枚识别正确，最低置信度>0.90，单帧 {full_elapsed*1000:.0f}ms，0.65x~1.50x 缩放通过")

    # 天天象棋：ADB 人机盘与 CoreGraphics 真人盘是两个独立画面层。
    tiantian = TiantianRecognizer()
    tiantian.capture_source = "yyb_adb"
    tt_ai = tiantian.recognize(cv2.imread("test_images/tiantian_adb.png"))
    assert tt_ai["piece_count"] == 32 and tt_ai["fen"] == START_FEN
    tt_selected = tiantian.recognize(cv2.imread("test_images/tiantian_selected_cannon.png"))
    assert tt_selected["piece_count"] == 32
    assert tt_selected["board"][6][1] == "b_c" and tt_selected["board"][2][1] is None

    # 2026-09 应用宝画面更新后棋盘整体下移 49px；新旧布局应自动选择。
    shifted_adb = cv2.warpAffine(
        cv2.imread("test_images/tiantian_adb.png"),
        np.float32([[1, 0, 0], [0, 1, 49]]),
        (1440, 2560),
    )
    tt_shifted = tiantian.recognize(shifted_adb)
    assert tt_shifted["piece_count"] == 32 and tt_shifted["fen"] == START_FEN

    tiantian.capture_source = "coregraphics"
    tt_machine_host = tiantian.recognize(cv2.imread("test_images/tiantian_host_probe.png"))
    expected_machine_host = "1rbakabnr/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/4C2C1/9/RNBAKABNR w - - 0 1"
    assert tt_machine_host["piece_count"] == 32 and tt_machine_host["fen"] == expected_machine_host
    tt_live = tiantian.recognize(cv2.imread("test_images/tiantian_battle_current.png"))
    expected_tt_live = "2BAKA3/9/4CCN1B/c6RP/4P2R1/2r3p2/p3p3p/2n1b1n2/9/2baka2r w - - 0 1"
    assert tt_live["piece_count"] == 26 and tt_live["fen"] == expected_tt_live
    tt_flipped = tiantian.recognize(cv2.imread("test_images/tiantian_battle_black_bottom.png"))
    assert tt_flipped["piece_count"] == 32 and infer_bottom_side(tt_flipped["board"]) == "b"
    for scale in (0.8, 1.2):
        scaled = cv2.resize(
            cv2.imread("test_images/tiantian_battle_current.png"), None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        scaled_result = tiantian.recognize(scaled)
        assert scaled_result["piece_count"] == 26 and scaled_result["fen"] == expected_tt_live
    print("  ✓ 天天象棋人机/真人、红黑翻面、选中特效与窗口缩放识别通过")


def test_debouncer_and_moves():
    print("[Test 4/6] 正在回归走子去抖、常规走法与吃子判定状态机...")
    debouncer = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.05)
    fen_init = "3k5/6R1C/3c5/6N2/6b2/9/9/9/2r1p1p2/5K3 w - - 0 1"
    fen_move = "3k5/6R1C/3c5/6N2/6b2/9/9/9/2r1p1p2/4K4 b - - 1 1"
    fen_pre_capture = "3k5/6R1C/3c5/6n2/6b2/9/9/9/2r1p1p2/4K4 b - - 1 1"
    fen_capture = "3k5/8C/3c5/6R2/6b2/9/9/9/2r1p1p2/4K4 w - - 2 2"

    # 1. 初始帧
    ev1 = debouncer.update(fen_init)
    assert ev1 is not None and ev1["event_type"] == "initial"

    # 2. 动画瞬态杂音
    ev_noise = debouncer.update("9/9/9/9/9/9/9/9/9/9 w - - 0 1")
    assert ev_noise is None, "单帧杂音不应触发更新"

    # 3. 选中动画：某棋子短暂被识别为空。即使持续多帧也必须保持旧稳定盘面。
    animated_board, _, _, _ = fen_to_matrix(fen_init)
    animated_board[1][6] = None  # 原盘面中红俥被选中特效遮扰
    fen_missing_piece = matrix_to_fen(animated_board, "w", 0, 1)
    ev_missing = None
    for _ in range(8):
        ev_missing = debouncer.update(fen_missing_piece)
        time.sleep(0.01)
    assert ev_missing is None, "选中特效造成的持续单子消失不应触发盘面更新"
    assert debouncer.last_stable_fen == fen_init, "持续缺子不应覆盖稳定盘面"

    # 原位字形被光圈误分类，也不能改变该格的稳定棋种。
    misclassified = [row[:] for row in animated_board]
    misclassified[1][6] = "r_n"
    misclassified_fen = matrix_to_fen(misclassified, "w", 0, 1)
    for _ in range(6):
        debouncer.update(misclassified_fen, misclassified)
        time.sleep(0.01)
    assert debouncer.last_stable_fen == fen_init, "原位误分类不应改写稳定棋子"

    # 特效误分类可能伪造完整 from→to；不符合棋种走法时也必须拒绝。
    illegal_board, _, _, _ = fen_to_matrix(fen_init)
    illegal_board[9][5] = None
    illegal_board[7][5] = "r_k"  # 帅一次跳两格，非法
    fen_illegal_jump = matrix_to_fen(illegal_board, "w", 0, 1)
    debouncer.update(fen_illegal_jump)
    time.sleep(0.06)
    ev_illegal = debouncer.update(fen_illegal_jump)
    assert ev_illegal is None, "非法伪走子不应触发盘面更新"
    assert debouncer.last_stable_fen == fen_init

    # 已选中并持续缺失多帧后真正落子：来源与目标应原子更新。
    selection_move = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.01)
    selection_move.update(fen_init)
    selected_board, _, _, _ = fen_to_matrix(fen_init)
    selected_board[1][8] = None
    selected_fen = matrix_to_fen(selected_board)
    for _ in range(6):
        selection_move.update(selected_fen, selected_board)
        time.sleep(0.01)
    landed_board = [row[:] for row in selected_board]
    landed_board[5][8] = "r_c"
    landed_fen = matrix_to_fen(landed_board)
    selection_move.update(landed_fen, landed_board)
    time.sleep(0.02)
    selected_move_event = selection_move.update(landed_fen, landed_board)
    assert selected_move_event and selected_move_event["move"]["uci"] == "i8i4"

    # 若服务启动时棋子正被特效遮住，取消选中后应逐格补回。
    startup_recovery = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.01)
    startup_recovery.update(fen_missing_piece, animated_board)
    recovered_event = None
    recovered_board = fen_to_matrix(fen_init)[0]
    for _ in range(5):
        event = startup_recovery.update(fen_init, recovered_board)
        recovered_event = event or recovered_event
        time.sleep(0.01)
    assert startup_recovery.last_stable_fen == fen_init
    assert recovered_event is not None and recovered_event["event_type"] == "recovered"

    # 补回被遮挡棋子不能把当前行棋方错误重置成红方。
    turn_recovery = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.0)
    turn_recovery.update(fen_missing_piece, animated_board)
    turn_recovery.active_side = "b"
    for _ in range(turn_recovery.recovery_frames + 1):
        turn_recovery.update(fen_init, recovered_board)
    assert turn_recovery.last_stable_fen.split()[1] == "b"

    # 4. 常规走法 (帅移步)
    debouncer.update(fen_move)
    time.sleep(0.06)
    ev_m = debouncer.update(fen_move)
    assert ev_m is not None and ev_m["event_type"] == "move"
    assert ev_m["move"]["uci"] == "f0e0"

    # 5. 吃子走法 (俥吃黑马)：单独建立合法的吃子前盘面。
    capture_debouncer = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.05)
    capture_debouncer.update(fen_pre_capture)
    capture_debouncer.update(fen_capture)
    time.sleep(0.06)
    ev_c = capture_debouncer.update(fen_capture)
    assert ev_c is not None and ev_c["event_type"] == "move"
    assert ev_c["move"]["uci"] == "g8g6"
    assert ev_c["move"]["captured"] == "b_n"

    # 6. 退出棋盘时的空画面不能清盘；新局稳定出现后必须整盘切换并广播。
    session_debouncer = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.01)
    session_debouncer.resync_frames = 3
    session_debouncer.resync_seconds = 0.03
    session_debouncer.update(fen_init)
    empty_fen = "9/9/9/9/9/9/9/9/9/9 w - - 0 1"
    for _ in range(6):
        session_debouncer.update(empty_fen)
        time.sleep(0.01)
    assert session_debouncer.last_stable_fen == fen_init, "菜单/空画面不应清空盘面"

    standard_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    reset_event = None
    for _ in range(5):
        reset_event = session_debouncer.update(standard_fen) or reset_event
        time.sleep(0.015)
    assert reset_event and reset_event["event_type"] == "session_reset"
    assert session_debouncer.last_stable_fen == standard_fen
    assert session_debouncer.session_revision == 2

    another_game = "3k5/6R1C/3c5/6n2/6b2/9/9/9/2r1p1p2/4K4 w - - 0 1"
    for _ in range(3):
        session_debouncer.update(empty_fen)
        time.sleep(0.01)
    reopen_event = None
    for _ in range(5):
        reopen_event = session_debouncer.update(another_game) or reopen_event
        time.sleep(0.015)
    assert reopen_event and reopen_event["event_type"] == "session_reset"
    assert session_debouncer.last_stable_fen == another_game
    assert session_debouncer.session_revision == 3

    # 菜单空画面不能成为初始稳定盘；从中盘启动时先等待首个真实走子锚定回合。
    guarded = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.0)
    empty = [[None] * 9 for _ in range(10)]
    assert guarded.update(matrix_to_fen(empty), empty) is None
    assert guarded.get_stable_state() == (None, None)
    middle = fen_to_matrix(START_FEN)[0]
    middle[6][0] = None
    middle[5][0] = "r_p"
    guarded.update(matrix_to_fen(middle), middle)
    assert guarded.active_side == "unknown"
    black_move = [row[:] for row in middle]
    black_move[3][0] = None
    black_move[4][0] = "b_p"
    guarded.update(matrix_to_fen(black_move), black_move)
    event = guarded.update(matrix_to_fen(black_move), black_move)
    assert event and event["move"]["piece"] == "b_p" and guarded.active_side == "r"
    illegal_second_black = [row[:] for row in black_move]
    illegal_second_black[3][2] = None
    illegal_second_black[4][2] = "b_p"
    for _ in range(3):
        assert guarded.update(matrix_to_fen(illegal_second_black), illegal_second_black) is None
    assert guarded.last_stable_board == black_move, "同一方连续走子必须拒绝"

    # 采集频率低于双方落子速度时，可从上一稳定布局推演两步并恢复到 AI 回合。
    catchup = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.0)
    start_board = fen_to_matrix(START_FEN)[0]
    catchup.update(START_FEN, start_board)
    two_plies = [row[:] for row in start_board]
    two_plies[6][0] = None
    two_plies[5][0] = "r_p"
    two_plies[3][0] = None
    two_plies[4][0] = "b_p"
    two_fen = matrix_to_fen(two_plies)
    catchup.update(two_fen, two_plies)
    caught = catchup.update(two_fen, two_plies)
    assert caught and caught["event_type"] == "catchup" and len(caught["moves"]) == 2
    assert catchup.active_side == "r" and catchup.last_stable_board == two_plies
    print("  ✓ 逐格防抖、双步追帧、回合锁定、空画面保护与新局/重开同步全部通过")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_window_capture_layer():
    print("[Test 5/6] 正在回归指定窗口捕获层与 Mock 模式...")
    # 1. 枚举可见窗口
    windows = list_available_windows()
    assert isinstance(windows, list), "窗口列表返回值必须为列表"
    print(f"  • 系统可见应用窗口数: {len(windows)}")

    # 2. Mock 捕获测试
    mock_cap = MockCapture(["test_images/board_test_01.png"])
    assert mock_cap.is_available()
    frame = mock_cap.capture()
    assert frame is not None and frame.shape == (1704, 918, 3)

    # 3. 指定关键词工厂
    cap_auto = create_capture(mode="auto", target_keywords=["天天象棋", "腾讯应用宝"])
    # 是否存在真实游戏窗口取决于测试机；工厂不得静默切换为 mock。
    assert cap_auto.get_info().get("mode") != "mock"
    print("  ✓ 捕获层不会将生产模式静默伪装成 Mock 模式")


def test_port_fallback_with_occupied_defaults():
    print("[Test 7/7] 正在模拟代理占用 8765/8766 并验证自动端口避让...")
    blockers = []
    process = None
    try:
        # 模拟 Shadowrocket 或其他本地代理已监听默认服务端口。
        for port in (8765, 8766):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", port))
                sock.listen(1)
                blockers.append(sock)
            except OSError:
                # 若测试机本身已有代理占用，正好使用真实冲突环境验证避让。
                sock.close()

        with tempfile.TemporaryDirectory() as directory:
            runtime_file = os.path.join(directory, "runtime.json")
            process = subprocess.Popen(
                [sys.executable, "main.py", "--mode", "mock", "--interval", "0.1", "--runtime-file", runtime_file],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            runtime = None
            for _ in range(50):
                if os.path.exists(runtime_file):
                    with open(runtime_file, encoding="utf-8") as handle:
                        runtime = json.load(handle)
                    break
                time.sleep(0.1)
            assert runtime is not None, "端口避让未生成 runtime.json"
            assert runtime["fallback_ports"] is True
            assert runtime["ws_port"] not in (8765, 8766)
            assert runtime["http_port"] not in (8765, 8766)

            state = {}
            for _ in range(30):
                try:
                    with urllib.request.urlopen(runtime["http_url"] + "/status", timeout=2.0) as response:
                        state = json.loads(response.read().decode("utf-8"))
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    # Windows 上 HTTP 线程刚绑定端口时，第一次连接可能被
                    # 系统重置；这不代表端口避让或服务启动失败。
                    time.sleep(0.1)
                    continue
                if state.get("piece_count") == 10:
                    break
                time.sleep(0.1)
            assert state.get("piece_count") == 10

            async def fallback_ws_check():
                async with websockets.connect(runtime["ws_url"], open_timeout=2) as ws:
                    state = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    assert state["piece_count"] == 10
            asyncio.run(fallback_ws_check())
            print(f"  ✓ 默认端口被占用时改用 WS={runtime['ws_port']} / HTTP={runtime['http_port']}")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        for sock in blockers:
            sock.close()


def test_session_reset_websocket():
    print("[Test 9/9] 正在验证新开局通过 WebSocket 实时刷新网页...")
    old_fen = "3k5/6R1C/3c5/6N2/6b2/9/9/9/2r1p1p2/5K3 w - - 0 1"
    new_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    old_board = fen_to_matrix(old_fen)[0]
    new_board = fen_to_matrix(new_fen)[0]

    class SequenceRecognizer:
        def __init__(self):
            self.frames = 0
        def recognize(self, image):
            self.frames += 1
            board = old_board if self.frames <= 12 else new_board
            fen = old_fen if self.frames <= 12 else new_fen
            return {
                "board": board, "fen": fen,
                "piece_count": sum(piece is not None for row in board for piece in row),
                "text_board": board_to_text(board), "pieces_detail": [],
            }

    ws_port, http_port = _free_tcp_port(), _free_tcp_port()
    while http_port == ws_port:
        http_port = _free_tcp_port()
    debouncer = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.01)
    debouncer.resync_frames = 3
    debouncer.resync_seconds = 0.05
    server = SyncServer(MockCapture(["test_images/board_test_01.png"]), recognizer=SequenceRecognizer(),
                        debouncer=debouncer, ws_port=ws_port, http_port=http_port,
                        interval=0.03, idle_timeout=0)
    server.start()
    time.sleep(0.1)
    try:
        async def receive_reset():
            async with websockets.connect(f"ws://127.0.0.1:{ws_port}") as ws:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    if message.get("event_type") == "session_reset":
                        return message
                raise AssertionError("未收到 session_reset WebSocket 事件")
        reset = asyncio.run(receive_reset())
        assert reset["fen"] == new_fen
        assert reset["piece_count"] == 32
        assert reset["session_revision"] == 2
        print("  ✓ 新局整盘通过 session_reset 事件实时推送到网页")
    finally:
        server.stop()


def test_dashboard_idle_shutdown():
    print("[Test 8/8] 正在验证看板关闭后的自动资源释放...")
    ws_port, http_port = _free_tcp_port(), _free_tcp_port()
    while http_port == ws_port:
        http_port = _free_tcp_port()
    server = SyncServer(MockCapture(["test_images/board_test_01.png"]), ws_port=ws_port, http_port=http_port,
                        interval=0.05, idle_timeout=2.0)
    server.start()
    time.sleep(0.15)  # 等待 WebSocket 监听线程完成 bind
    try:
        async def hold_dashboard_connection():
            async with websockets.connect(f"ws://127.0.0.1:{ws_port}") as ws:
                initial = {}
                for _ in range(3):
                    initial = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    if initial.get("piece_count") == 10:
                        break
                assert initial.get("piece_count") == 10
                await asyncio.sleep(0.8)
                assert server.running, "看板仍连接时服务不应退出"
        asyncio.run(hold_dashboard_connection())

        for _ in range(40):
            if not server.running:
                break
            time.sleep(0.1)
        assert not server.running, "看板关闭后服务未按空闲超时退出"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{http_port}/status", timeout=0.2)
            raise AssertionError("自动退出后 HTTP 端口仍可访问")
        except Exception:
            pass
        print("  ✓ 看板连接期间服务保持运行；关闭后自动停止并释放端口")
    finally:
        server.stop()


def test_sync_server_end_to_end():
    print("[Test 6/6] 正在回归 WebSocket 广播与 HTTP REST 接口端到端通信...")
    mock_cap = MockCapture(["test_images/board_test_01.png"])
    ws_port, http_port = _free_tcp_port(), _free_tcp_port()
    while http_port == ws_port:
        http_port = _free_tcp_port()
    ai_settings_path = os.path.join(tempfile.gettempdir(), f"xiangqi-ai-settings-{os.getpid()}.json")
    try:
        os.unlink(ai_settings_path)
    except FileNotFoundError:
        pass
    server = SyncServer(
        capture=mock_cap,
        ws_port=ws_port,
        http_port=http_port,
        interval=0.1,
        ai_settings_path=ai_settings_path,
    )
    server.start()

    try:
        # 等待识别首帧真实完成，不使用固定睡眠造成慢机器伪失败。
        ready_state = {}
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/status", timeout=0.5) as resp:
                    ready_state = json.loads(resp.read().decode("utf-8"))
                if ready_state.get("piece_count") == 10:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        assert ready_state.get("piece_count") == 10, f"同步服务首帧未就绪: {ready_state}"

        # 1. 测试 HTTP /fen
        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/fen", timeout=2.0) as resp:
            fen_body = resp.read().decode("utf-8")
            assert "3k5/6R1C" in fen_body, f"HTTP FEN 异常: {fen_body}"

        # 2. 测试 HTTP /status
        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/status", timeout=2.0) as resp:
            status_data = json.loads(resp.read().decode("utf-8"))
            assert status_data.get("piece_count") == 10
            assert status_data.get("ai", {}).get("bottom_side") in ("r", "b")

        # 3. 测试 HTTP / (Web 看板)
        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/", timeout=2.0) as resp:
            html_body = resp.read().decode("utf-8")
            assert "象棋盘面同步" in html_body
            assert "楚 河" in html_body and "最新 FEN" not in html_body
            assert "AI 执下方" in html_body and "实时只读 · 自动跟随实盘" in html_body
            assert "棋盘模拟" not in html_body and "回退一步" not in html_body
            assert "AI 强度" in html_body and "清理缓存并重新读取对局" in html_body
            assert "/api/live/reset" in html_body
            assert "节能" in html_body and "普通" in html_body and "进阶" in html_body and "高级" in html_body
            assert "Pikafish" in html_body

        # 4. 手动恢复接口会先清空展示，再由捕获线程完整重建盘面。
        reset_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/live/reset",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(reset_request, timeout=2.0) as resp:
            reset_state = json.loads(resp.read().decode("utf-8"))
            assert reset_state["event_type"] == "reset_requested"
            assert reset_state["capture_status"] == "resetting"
            assert reset_state["piece_count"] == 10
            assert reset_state["board"], "后台重读期间必须保留当前稳定盘面"
        for _ in range(40):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/api/status", timeout=2.0
            ) as resp:
                recovered_state = json.loads(resp.read().decode("utf-8"))
            if recovered_state.get("piece_count") == 10:
                break
            time.sleep(0.05)
        assert recovered_state.get("piece_count") == 10

        # 5. AI 回退接口存在并返回结构化状态。
        undo_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/ai/undo",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(undo_request, timeout=2.0) as resp:
            undo_state = json.loads(resp.read().decode("utf-8"))
            assert "ai" in undo_state and "can_undo" in undo_state["ai"]

        config_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/ai/config",
            data=json.dumps({
                "time_ms": 1500, "max_depth": 60, "engine_kind": "pikafish",
                "engine_threads": 2, "engine_hash_mb": 64,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(config_request, timeout=2.0) as resp:
            configured = json.loads(resp.read().decode("utf-8"))["ai"]
            assert configured["time_ms"] == 1500 and configured["max_depth"] == 60
            assert configured["engine_kind"] == "pikafish"
            assert configured["engine_threads"] == 2 and configured["engine_hash_mb"] == 64
        with open(ai_settings_path, encoding="utf-8") as handle:
            persisted = json.load(handle)
        assert persisted == {
            "engine_kind": "pikafish", "time_ms": 1500, "max_depth": 60,
            "engine_threads": 2, "engine_hash_mb": 64,
        }
        bad_config = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/ai/config",
            data=b'{"max_depth":"bad"}',
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(bad_config, timeout=2.0)
            raise AssertionError("非法 AI 参数应返回 HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        builtin_config = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/ai/config",
            data=b'{"engine_kind":"builtin"}',
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(builtin_config, timeout=2.0)
            raise AssertionError("发布接口不应允许切换到内置 AI")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        hostile = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/api/ai/config",
            data=b'{"time_ms":30000}',
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
            method="POST",
        )
        try:
            urllib.request.urlopen(hostile, timeout=2.0)
            raise AssertionError("外部网页不应能控制本地 AI")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403

        # 5. 测试 WebSocket 客户端
        async def client_test():
            async with websockets.connect(f"ws://127.0.0.1:{ws_port}") as ws:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg_data = json.loads(raw_msg)
                assert "fen" in msg_data
                assert msg_data["piece_count"] == 10

        asyncio.run(client_test())
        async def hostile_ws_test():
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{ws_port}", origin="https://evil.example",
                ):
                    raise AssertionError("外部网页不应能读取本地 WebSocket 盘面")
            except websockets.exceptions.InvalidStatus:
                pass
        asyncio.run(hostile_ws_test())
        print("  ✓ HTTP/WebSocket、AI设置持久化与本地来源保护全部通过")
    finally:
        server.stop()
        try:
            os.unlink(ai_settings_path)
        except FileNotFoundError:
            pass


def test_xiangqi_engine_and_bottom_ai():
    print("[Test 10/10] 正在回归象棋规则、下方 AI 与翻面几何...")
    start_board = fen_to_matrix(START_FEN)[0]
    assert infer_bottom_side(start_board) == "r"
    start_moves = generate_legal_moves(start_board, "r")
    assert len(start_moves) == 44, f"开局着法数应为 44，实际 {len(start_moves)}"
    cannon = parse_uci(start_board, "h2e2")
    assert cannon is not None and is_legal_move(start_board, cannon, "r")
    assert move_to_chinese(start_board, cannon) == "炮二平五"
    horse = parse_uci(start_board, "b0c2")
    assert horse is not None and move_to_chinese(start_board, horse) == "马八进七"

    flipped = fen_to_matrix(FLIPPED_START_FEN)[0]
    assert infer_bottom_side(flipped) == "b"
    black_pawn = None
    for move in generate_legal_moves(flipped, "b"):
        if move.piece == "b_p" and move.sc == 0 and move.dc == 0:
            black_pawn = move
            break
    assert black_pawn is not None and black_pawn.dr == black_pawn.sr - 1, "翻面后下方黑卒应向上走"

    hanging_fen = "3k5/9/9/9/9/9/9/6Rr1/9/4K4 w - - 0 1"
    hanging = fen_to_matrix(hanging_fen)[0]
    engine = XiangqiEngine()
    result = engine.search(hanging, "r", time_ms=250, max_depth=3)
    assert result.move is not None and result.move.uci == "g2h2", f"应吃掉悬空车，实际 {result.move}"

    advisor = AIAdvisor(time_ms=80, max_depth=2, enabled=True, engine_kind="builtin")
    advisor.on_live_position(start_board, "r")
    snap = advisor.snapshot()
    assert snap["bottom_side"] == "r" and snap["top_side"] == "b"
    advisor.start_local_game(start_board, "b")
    play = advisor.snapshot()
    assert play["play_mode"] is True
    assert play["to_move"] == "b"
    # 上方黑卒 7 路进 1（视觉行 3→4）
    moved = advisor.play_move(3, 0, 4, 0)
    assert moved["error"] is None, moved.get("error")
    assert moved["to_move"] == "r"
    advisor.stop()

    # 回合变为待确认/上方时必须立即取消旧搜索，旧线程不能把界面改回“思考中”。
    cancellation = AIAdvisor(time_ms=1000, max_depth=60, enabled=True, engine_kind="pikafish")
    cancellation.on_live_position(start_board, "r")
    assert cancellation.snapshot()["thinking"] is True
    cancellation.on_live_position(start_board, "unknown")
    assert cancellation.snapshot()["thinking"] is False
    time.sleep(0.15)
    cancelled = cancellation.snapshot()
    assert not cancelled["thinking"] and cancelled["suggestion"] is None
    assert cancelled["status"] == "等待首个真实走子确认回合"
    cancellation.stop()

    # 标准象棋始终红先；黑方在下时必须等待上方红方，不能让 AI 抢走。
    turn_tracker = BoardDebouncer(required_stable_frames=2, min_stable_seconds=0.01)
    turn_tracker.update(FLIPPED_START_FEN, flipped)
    assert turn_tracker.active_side == "r"

    # 本地对弈支持回退完整回合，并在改走后保留原线为变招分支。
    branch = AIAdvisor(time_ms=80, max_depth=2, enabled=False, engine_kind="builtin")
    strength = branch.configure(time_ms=2500, max_depth=14)
    assert strength["time_ms"] == 2500 and strength["max_depth"] == 14
    branch.on_live_position(start_board, "b")
    branch.start_local_game(start_board, "b")
    first = branch.play_move(3, 0, 4, 0)
    assert first["history"][-1]["to"] == {"row": 4, "col": 0}
    undone = branch.undo()
    assert undone["can_undo"] is False and undone["ply"] == 0
    alternative = branch.play_move(3, 2, 4, 2)
    assert alternative["history"][-1]["to"] == {"row": 4, "col": 2}
    assert alternative["variation_count"] == 1
    assert alternative["variations"][0]["moves"][0]["to"] == {"row": 4, "col": 0}
    branch.stop()

    # 连续实盘：AI走子→实盘追平→对手走子→AI必须自动再应手。
    flowing = AIAdvisor(time_ms=80, max_depth=2, enabled=True, engine_kind="builtin")
    flowing.on_live_position(start_board, "r", session_revision=1)
    flowing.start_local_game(start_board, "r")
    for _ in range(40):
        flow = flowing.snapshot()
        if len(flow["history"]) >= 1 and not flow["thinking"]:
            break
        time.sleep(0.03)
    assert flow["history"][-1]["actor"] == "ai", flow
    ai_board = flow["board"]
    # 实盘旧帧只是尚未执行 AI 着法，不得回滚 AI 时间线。
    flowing.on_live_position(start_board, "r", session_revision=1)
    assert len(flowing.snapshot()["history"]) == 1
    # 用户在 JJ 执行 AI 着法后，实盘追上。
    flowing.on_live_position(ai_board, "b", session_revision=1)
    top_move = generate_legal_moves(ai_board, "b")[0]
    opponent_board = apply_move(ai_board, top_move)
    flowing.on_live_position(opponent_board, "r", session_revision=1)
    for _ in range(50):
        flow = flowing.snapshot()
        if len(flow["history"]) >= 3 and not flow["thinking"]:
            break
        time.sleep(0.03)
    assert [item["actor"] for item in flow["history"][-3:]] == ["ai", "opponent_live", "ai"], flow
    assert flow["to_move"] == "b"
    flowing.stop()

    # 正式主引擎：真实 Pikafish UCI、红下/黑下坐标转换及停止搜索。
    pika = PikafishEngine(threads=2, hash_mb=64)
    assert pika.is_available(), "缺少打包的 Pikafish 或 NNUE"
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    assert sha256(pika.binary_path) == PIKAFISH_BINARY_SHA256
    assert sha256(pika.nnue_path) == PIKAFISH_NNUE_SHA256
    pika_red = pika.search(start_board, "r", time_ms=120, max_depth=60)
    assert pika_red.engine == "pikafish" and pika_red.move is not None
    assert is_legal_move(start_board, pika_red.move, "r")
    red_open = parse_uci(start_board, "h2e2")
    assert red_open is not None
    flipped_after_red = rotate_board_180(apply_move(start_board, red_open))
    pika_black = pika.search(flipped_after_red, "b", time_ms=120, max_depth=60)
    assert pika_black.move is not None and is_legal_move(flipped_after_red, pika_black.move, "b")
    old_process = pika._process
    pika.configure(threads=1, hash_mb=32)
    assert pika._process is None and old_process.poll() is not None, "强度切换后应重启引擎以释放旧 Hash 内存"
    reconfigured = pika.search(start_board, "r", time_ms=80, max_depth=60)
    assert reconfigured.move is not None and pika._process is not old_process

    cancelled = {}
    def long_search():
        try:
            cancelled["result"] = pika.search(start_board, "r", time_ms=3000, max_depth=60)
        except Exception as exc:
            cancelled["error"] = str(exc)
    search_thread = threading.Thread(target=long_search)
    search_thread.start()
    for _ in range(50):
        if pika._searching.is_set():
            break
        time.sleep(0.01)
    pika.request_stop()
    search_thread.join(timeout=2.0)
    assert not search_thread.is_alive(), "Pikafish stop 未及时停止搜索"
    pika.close()

    # 主引擎缺失时必须自动回退，不能让整套 AI 停摆。
    fallback = AIAdvisor(
        time_ms=80, max_depth=2, enabled=True, engine_kind="pikafish",
        allow_builtin_fallback=True,
    )
    fallback.pikafish.binary_path = Path("/definitely/missing/pikafish")
    fallback.on_live_position(start_board, "r", session_revision=1)
    for _ in range(60):
        fallback_state = fallback.snapshot()
        if fallback_state["suggestion"] and not fallback_state["thinking"]:
            break
        time.sleep(0.03)
    assert fallback_state["active_engine"] == "builtin-fallback", fallback_state
    assert fallback_state["engine_error"], fallback_state
    fallback.stop()
    print("  ✓ 回合/变招连续应手、Pikafish红黑翻面、停止搜索与故障回退全部通过")


def test_windows_launcher_structure():
    print("[Launcher Windows] 验证无黑框启动器与 Pikafish 固定配置...")
    vbs_path = "启动象棋同步.vbs"
    cmd_path = "start_windows.cmd"
    assert os.path.isfile(vbs_path) and os.path.isfile(cmd_path)
    with open(vbs_path, encoding="utf-8") as handle:
        vbs = handle.read()
    with open(cmd_path, encoding="utf-8") as handle:
        cmd = handle.read()
    assert "shell.Run command, 0, False" in vbs
    assert "/api/ai/follow" in vbs and "OpenDashboard url" in vbs
    assert "--ai-engine pikafish" in cmd and "--ai-engine builtin" not in cmd
    assert all(line.strip().lower() != "pause" for line in cmd.splitlines())
    assert PikafishEngine().is_available()
    print("  ✓ VBS 隐藏启动、自动跟随实盘、自动打开网页并固定使用 Pikafish")


def run_all():
    print("=" * 65)
    print("      开始执行中国象棋盘面同步程序完整回归测试套件      ")
    print("=" * 65)
    t0 = time.perf_counter()

    test_fen_conversion()
    test_board_calibrator()
    test_recognizer_accuracy()
    test_debouncer_and_moves()
    test_xiangqi_engine_and_bottom_ai()
    test_window_capture_layer()
    test_sync_server_end_to_end()
    test_port_fallback_with_occupied_defaults()
    test_dashboard_idle_shutdown()
    test_session_reset_websocket()
    run_recovery_tests()
    
    if sys.platform == "darwin":
        print("\n--- 启动器与 .app 自动化测试 ---")
        test_app_bundle_structure()
        test_cold_start_and_service()
        test_runtime_communication()
        test_clean_stop()
    elif sys.platform == "win32":
        print("\n--- Windows 无黑框启动器自动化测试 ---")
        test_windows_launcher_structure()

    t1 = time.perf_counter()
    print("=" * 65)
    print(f"🎉 核心功能、平台启动器与追帧/捕获恢复回归测试 100% 通过！(总耗时: {t1-t0:.2f}s)")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
