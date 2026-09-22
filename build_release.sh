#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="${1:-1.1.0-rc1}"
NAME="Xiangqi-Sync-v${VERSION}"
DIST="$DIR/dist"
STAGE="$DIST/$NAME"
ZIP="$DIST/$NAME.zip"

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"

ITEMS=(
  README.md RELEASE_NOTES.md requirements.txt requirements-macos.lock python.sh launcher.sh stop.sh main.py run_regression_tests.py
  AppIcon.icns .gitignore capture core server templates test_images tests engines runtime packaging
  "象棋盘面同步.app" "停止服务.app"
)
for item in "${ITEMS[@]}"; do
  [ -e "$DIR/$item" ] || { echo "缺少发布文件: $item" >&2; exit 1; }
  rsync -a --exclude='tiantian_battle_desktop.png' --exclude='tiantian_battle_live_now.png' \
    --exclude='tiantian_battle_red_top.png' "$DIR/$item" "$STAGE/"
done

find "$STAGE" -name '.DS_Store' -delete
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete
mkdir -p "$STAGE/logs" "$STAGE/config"
# 不改启动器签名身份，避免使已有 TCC 授权失效。发布版本记录在包名和说明中。
codesign --verify --deep --strict "$STAGE/象棋盘面同步.app"
codesign --verify --deep --strict "$STAGE/停止服务.app"

# 发布阻塞项：打包运行时、引擎和主入口必须能从任意目录工作。
(
  cd "$STAGE"
  ./python.sh -c 'import cv2, numpy, websockets, Quartz; from core.pikafish_engine import PikafishEngine; assert PikafishEngine().is_available()'
  ./python.sh -m py_compile main.py core/*.py capture/*.py server/*.py
  bash -n launcher.sh stop.sh python.sh
  plutil -lint "象棋盘面同步.app/Contents/Info.plist" "停止服务.app/Contents/Info.plist" >/dev/null
)

# macOS ditto 能正确保存 .app、可执行位和 Unicode 文件名。
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP"
shasum -a 256 "$ZIP" > "$ZIP.sha256"

echo "发布包已生成: $ZIP"
cat "$ZIP.sha256"
