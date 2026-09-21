#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/logs/service.pid"
RUNTIME_FILE="$DIR/logs/runtime.json"
STOPPED=0

if [ -f "$PID_FILE" ]; then
    PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        COMMAND="$(ps -p "$PID" -o command= 2>/dev/null)"
        # PID 来自本启动器；再确认其仍是同步器 main.py，避免 PID 被系统复用后误杀。
        # 不用完整路径比较，以兼容 macOS 在进程列表中转义中文目录名的情况。
        if [[ "$COMMAND" == *"main.py"* ]]; then
            kill -TERM "$PID" 2>/dev/null
            for i in {1..20}; do
                kill -0 "$PID" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "$PID" 2>/dev/null; then
                kill -KILL "$PID" 2>/dev/null
            fi
            STOPPED=1
        fi
    fi
    rm -f "$PID_FILE"
fi
rm -f "$RUNTIME_FILE"

# 主进程若曾被强制终止，Pikafish 子进程可能来不及收到 quit。
# 仅清理本项目绝对路径下的引擎，避免误杀用户另行启动的 Pikafish。
PIKAFISH_BIN="$DIR/engines/pikafish/pikafish"
while IFS= read -r ENGINE_PID; do
    [ -n "$ENGINE_PID" ] || continue
    kill -TERM "$ENGINE_PID" 2>/dev/null || true
    STOPPED=1
done < <(pgrep -f "$PIKAFISH_BIN" 2>/dev/null || true)

if [ "$STOPPED" -eq 1 ]; then
    MESSAGE="象棋盘面同步服务已停止。"
else
    MESSAGE="未发现由象棋同步器启动的服务。"
fi
osascript -e "display notification \"$MESSAGE\" with title \"象棋同步器\"" 2>/dev/null &
echo "$MESSAGE"
