#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

mkdir -p logs
PID_FILE="$DIR/logs/service.pid"
RUNTIME_FILE="$DIR/logs/runtime.json"

# 发布包自带 Python 运行时，避免依赖用户安装 Xcode Command Line Tools。
BUNDLED_PYTHON="$DIR/runtime/Python3.framework/Versions/3.9/bin/python3"
if [ -x "$BUNDLED_PYTHON" ] && [ -d "$DIR/runtime/site-packages" ]; then
    PYTHON_BIN="$BUNDLED_PYTHON"
    PYTHON_PATH="$DIR/runtime/site-packages"
else
    PYTHON_BIN="$DIR/venv/bin/python"
    PYTHON_PATH=""
fi

run_python() {
    if [ -n "$PYTHON_PATH" ]; then
        PYTHONPATH="$PYTHON_PATH" "$PYTHON_BIN" "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

# 仅供自动化回归使用；正常双击不会设置它，因此默认始终尝试实时窗口捕获。
EXTRA_ARGS=()
if [ -n "${XIANGQI_SYNC_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<< "$XIANGQI_SYNC_ARGS"
fi

runtime_http_port() {
    run_python - "$RUNTIME_FILE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        print(int(json.load(f)["http_port"]))
except Exception:
    sys.exit(1)
PY
}

open_existing_service() {
    local port
    port="$(runtime_http_port)" || return 1
    if curl -s -m 0.5 "http://127.0.0.1:${port}/status" | grep -q '"capture_info"'; then
        open "http://127.0.0.1:${port}/"
        return 0
    fi
    return 1
}

# 无论使用默认端口还是避让端口，只要是本同步器实例就复用。
if open_existing_service; then
    osascript -e 'display notification "同步服务已在运行，已打开看板。" with title "象棋同步器"' 2>/dev/null &
    exit 0
fi
rm -f "$RUNTIME_FILE"

# 清理失效 PID；端口是否被代理占用无需在这里判断，main.py 会自动避让。
if [ -f "$PID_FILE" ]; then
    PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        # PID 仍在但没有有效同步 HTTP 服务，不能再启动第二个实例。
        osascript -e 'display notification "已有同步进程正在初始化，请稍后再试。" with title "象棋同步器"' 2>/dev/null &
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# 双击 .app 时，录屏权限归属于启动器链路，而不是用户的终端。
# 已授权时立即返回；未授权才请求一次系统授权。
if [ -z "${XIANGQI_SYNC_ARGS:-}" ]; then
    if ! run_python "$DIR/main.py" --request-screen-permission > "$DIR/logs/permission.log" 2>&1; then
        osascript -e 'display notification "请允许“象棋盘面同步/Python”使用屏幕录制权限，然后再次点击启动。" with title "象棋同步器需要授权"' 2>/dev/null &
        exit 1
    fi
fi

if [ -n "$PYTHON_PATH" ]; then
    nohup env PYTHONPATH="$PYTHON_PATH" "$PYTHON_BIN" -u "$DIR/main.py" "${EXTRA_ARGS[@]}" > "$DIR/logs/app.log" 2>&1 &
else
    nohup "$PYTHON_BIN" -u "$DIR/main.py" "${EXTRA_ARGS[@]}" > "$DIR/logs/app.log" 2>&1 &
fi
PID=$!
echo "$PID" > "$PID_FILE"

READY=0
for i in {1..40}; do
    if open_existing_service; then
        READY=1
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 0.1
done

if [ "$READY" -eq 1 ]; then
    osascript -e 'display notification "同步服务已启动；看板已自动打开。" with title "象棋同步器"' 2>/dev/null &
else
    rm -f "$PID_FILE" "$RUNTIME_FILE"
    osascript -e 'display notification "同步服务启动失败，请查看 logs/app.log。" with title "象棋同步器"' 2>/dev/null &
    exit 1
fi
