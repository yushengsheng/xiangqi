# 中国象棋实时盘面同步器

自动识别 JJ 象棋或天天象棋窗口，经本地视觉识别生成中国象棋 FEN，并通过 HTTP 与 WebSocket 推送。程序不读游戏内存、不注入游戏进程。

发布版支持 **Apple Silicon + macOS 14.0 或更新版本**，已自带 Python、依赖与 Pikafish，无需额外安装开发环境。版本说明见 `RELEASE_NOTES.md`。

> 当前模板覆盖真实 JJ 完整开局/残局，以及天天象棋人机盘、真人盘、红黑翻面、窗口缩放和选中特效。游戏平台更换棋子皮肤或整体美术资源后需重新采集模板。
>
> **使用与授权提示：** 仅用于本地学习、复盘、自建对局及平台明确允许的场景。请遵守游戏平台规则，禁止用于违规在线对局辅助。打包的 Pikafish NNUE 权重禁止未经许可的商业使用，详情见 `engines/pikafish/NNUE-License.md`。

## 从 GitHub 源码运行

Git 仓库包含源码、识别模板、回归图片、原生启动器及 Pikafish；不包含 `runtime/`、虚拟环境、`dist/`、运行日志和个人配置。源码运行需自行准备 Python（建议 3.11 或更新），在 macOS 上执行：

```bash
git clone https://github.com/yushengsheng/xiangqi.git
cd xiangqi
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt pyobjc-framework-Quartz
./python.sh main.py
```

源码启动自动使用 `venv/`；自包含发布包优先使用随包的 `runtime/`。`build_release.sh` 和完整启动器回归依赖自包含运行时，源码仓库本身不提供它；不要将虚拟环境或个人配置提交到仓库。

## 运行方式

在项目目录中执行：

```bash
./python.sh main.py --list-windows
```

Window ID 在客户端重开后会变化；正常双击启动会按标题自动重新查找。需要手动验证时先用 `--list-windows` 获取当前 ID：

```bash
./python.sh main.py --window-id <当前ID> --capture-once logs/manual_capture.png
```

打开保存的图片，必须能看到当前象棋盘面。若命令报告未授予录屏权限，请在 macOS「系统设置 → 隐私与安全性 → 屏幕录制」中允许“象棋盘面同步”；授权后重新启动。天天象棋人机盘可直接使用应用宝画面通道，真人对战使用 macOS 宿主窗口画面，因此真人盘必须保留录屏授权。

确认抓帧正确后，首次为当前 UI 标定棋盘。弹出的窗口中按 **左上、右上、右下、左下** 的顺序点击棋盘最外层四个交叉点，按 Enter 保存（R 重置，Esc 取消）：

```bash
./python.sh main.py --window-id <当前ID> --calibrate
```

标定会保存到 `config/board_calibration.json`，后续自动加载。若窗口 UI、方向或棋盘区域改变，需要重新标定。

启动实时服务。当前 `.app` 为 ad-hoc 签名，首次从网络下载后如被 Gatekeeper 拦截，请在 Finder 中右键选择“打开”；正式公开分发建议使用 Apple Developer ID 签名并公证。

```bash
./python.sh main.py --window-id <当前ID>
# 或双击「象棋盘面同步.app」自动识别 JJ/天天象棋及应用宝窗口
```

- Web 看板：`http://127.0.0.1:8766/`
- 最新 FEN：`http://127.0.0.1:8766/fen`
- 状态/捕获错误：`http://127.0.0.1:8766/status`
- WebSocket：`ws://127.0.0.1:8765`

看板或 `/status` 中的 `capture_status` 必须是 `ok`；`error` 时请查看 `capture_info.last_error`，不要使用旧 FEN。

## 下方 AI 对战

默认使用打包的本地 **Pikafish 2026-09-06 + NNUE**，始终执画面下方的棋子（红黑按将/帅位置自动识别），与上方对战。不会点击或注入游戏窗口；Pikafish 不可用时自动回退到内置轻量引擎。

- 实盘助手：识别到上方走子后，在看板上用箭头给出下方着法。
- 棋盘模拟：点击「棋盘模拟」后只在网页棋盘中对弈，你走**上方**棋子，AI 自动走下方；真实对局请保持“跟随实盘”，避免把模拟盘误当实时盘。
- 若实盘是执黑翻面，下方变成黑方，AI 仍然执下；标准开局始终红先。天天中盘可用连续两帧确认的来源/落点标记恢复回合，没有可靠标记时等待真实走子，不猜测先手。
- 捕获漏掉中间帧时，从上一稳定布局推演最多两个合法半回合（含吃子），整批追上实盘；顺序影响回合且无可靠证据时保持等待。
- 看板提供“普通 / 进阶 / 高级”三档 AI 强度，选择后立即生效；命令行仍可精细设置时间、深度、线程和 Hash。
- 黑方位于下方时，程序会把盘面和 Pikafish 着法双向旋转 180°，引擎始终接收标准象棋 FEN。

```bash
./python.sh main.py --window-id <当前ID>      # 默认 Pikafish：2线程 / 64MB Hash
./python.sh main.py --mode mock --ai-time 1.2 --ai-threads 2 --ai-hash 64
./python.sh main.py --mode mock --ai-engine builtin
./python.sh main.py --mode mock --no-ai      # 只要同步、不要 AI
```

相关接口：`GET /api/ai`，`POST /api/ai/config`、`/api/ai/start`、`/api/ai/play`、`/api/ai/follow`、`/api/ai/turn`、`/api/ai/undo`。Pikafish 许可证、来源和校验值见 `engines/pikafish/README.md`。

## 离线开发

Mock 模式只能显式启用，避免把静态测试图误认为实时盘面：

```bash
./python.sh main.py --mode mock
./python.sh run_regression_tests.py
```

回归包含原有 14 项功能/启动器测试及 17 项追帧、搜索取消和捕获恢复专项测试。服务测试使用独立随机端口；启动器测试会停止当前服务并显式启动 Mock，结束后需重新打开生产应用。

## 接入格式

WebSocket 会发送 JSON，包含 `fen`、`text_board`、`piece_count`、`capture_status`、`last_frame_at`、`side_to_move`。单步为 `event_type: "move"`；漏采补齐为 `catchup`，带按顺序排列的 `moves`；视觉回合确认是 `turn_confirmed`。HTTP `/fen` 返回当前 FEN 纯文本。

主状态 `board` / `fen` / `side_to_move` 始终对应稳定实盘；模拟盘仅在 `ai.board` / `ai.fen` / `ai.to_move`。回合未确认时 `side_to_move` 为 `unknown`；FEN 因格式限制暂用 `w`，下游必须检查该字段，不得据此启动搜索。
