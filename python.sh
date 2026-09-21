#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/runtime/Python3.framework/Versions/3.9/bin/python3"
if [ -x "$PYTHON" ] && [ -d "$DIR/runtime/site-packages" ]; then
    exec env PYTHONPATH="$DIR/runtime/site-packages" "$PYTHON" "$@"
fi
if [ -x "$DIR/venv/bin/python" ]; then
    exec "$DIR/venv/bin/python" "$@"
fi
echo "未找到可用的 Python 运行时。请重新下载完整发布包。" >&2
exit 1
