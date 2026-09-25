# 中国象棋实时盘面同步器

自动识别 JJ 象棋、天天象棋客户端或微信「天天象棋」小程序窗口，经本地视觉观测和象棋规则跟踪生成中国象棋 FEN，并通过 HTTP 与 WebSocket 推送。程序不读游戏内存、不注入游戏进程。

## Windows 快速运行

当前文件夹已提供 Windows 启动入口：

1. 先打开 JJ 象棋、天天象棋、微信「天天象棋」小程序或常用安卓模拟器中的棋局窗口。
2. 双击 `启动象棋同步.vbs`。启动过程完全在后台运行，首次运行会自动创建 `venv` 并安装依赖。
3. 服务就绪后会自动打开默认浏览器并进入看板，不需要手动输入网址。关闭看板约 20 秒后，后台服务会自动退出。

如果只想确认程序与网页棋盘是否正常，双击 `启动模拟棋盘.bat`。也可以在命令行执行 `start_windows.cmd --list-windows` 查看可捕获的窗口。Windows 启动入口固定使用随项目安装的官方 Pikafish Windows 通用版与完整 NNUE；引擎出错时会直接显示错误，不会改用内置 AI。

Windows 会优先使用系统的 Graphics Capture 接口直接读取所选窗口的图形表面，因此其他窗口遮挡棋盘时不会被一并截入；只有系统或目标程序不支持该接口时才回退到传统窗口截图。天天象棋会同时利用棋子圆心和棋盘横竖线拟合 9×10 棋盘网格；微信小程序横屏窗口即使随时放大、缩小或改变两侧留白，也会重新定位棋盘。棋子太少的残局由棋盘线定位兜底。

实时状态采用**规则驱动的合法走子跟踪**：标准新局直接锚定已知 32 子开局；同一局中，视觉结果只能通过唯一合法走子或最多四个半回合的精确合法路径改变内部盘面。悔棋只能回到本次运行期间记录过的历史局面；若跳过回退画面直接改走，还必须有稳定落点标记和完整合法走法。单独少子、多子、原位变色和未经走法证明的整盘跳变都会被拒绝并保留上一正确盘面。自动重开只接受标准开局；中盘断线重接若无法用合法走子追上，可在确认画面完整后按“清理缓存并重新读取对局”。为获得最高稳定性，建议先启动程序再开始新对局；中盘接入时会先等待完整稳定盘面和首个可靠回合证据。

发布版支持 **Apple Silicon + macOS 14.0 或更新版本**，已自带 Python、依赖与 Pikafish，无需额外安装开发环境。版本说明见 `RELEASE_NOTES.md`。

> 当前模板覆盖真实 JJ 完整开局/残局，以及天天象棋客户端和微信小程序的人机盘、真人盘、红黑翻面、窗口缩放和选中特效。游戏平台更换棋子皮肤或整体美术资源后需重新采集模板。
>
> **使用与授权提示：** 仅用于本地学习、复盘、自建对局及平台明确允许的场景。请遵守游戏平台规则，禁止用于违规在线对局辅助。打包的 Pikafish NNUE 权重禁止未经许可的商业使用，详情见 `engines/pikafish/NNUE-License.md`。

## 从 GitHub 源码运行

Git 仓库包含源码、识别模板、回归图片、原生启动器及 Pikafish；不包含 `runtime/`、虚拟环境、`dist/`、运行日志和个人配置。源码运行需自行准备 Python（建议 3.11 或更新），在 macOS 上执行：

```bash
git clone https://github.com/yushengsheng/xiangqi.git
cd xiangqi
python3.11 -m venv venv
./venv/bin/python -m pip install --upgrade "pip>=26.2.1" "setuptools>=84.0.0"
./venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-macos.lock
./python.sh main.py
```

`requirements-macos.lock` 锁定 macOS/Python 3.11 依赖及发布包哈希，避免重装时静默换用其他构建；`requirements.txt` 保留为直接依赖清单。

源码启动自动使用 `venv/`；自包含发布包优先使用随包的 `runtime/`。`build_release.sh` 和完整启动器回归依赖自包含运行时，源码仓库本身不提供它；不要将虚拟环境或个人配置提交到仓库。

### macOS 微信小程序（代码适配，待 Mac 实机验收）

Mac 版现在会识别「微信 / WeChat / Weixin」窗口，并把选中的微信窗口固定走 CoreGraphics 采集与微信小程序棋子模板；即使应用宝同时运行，也不会把微信棋局误接到应用宝 ADB 画面。微信窗口需要 macOS「屏幕录制」授权，不能沿用应用宝 ADB 的免授权通道。横屏、窄窗和纵向拉长窗口会重新尝试定位棋盘。

如果同时打开应用宝或多个微信窗口，先用 `./python.sh main.py --list-windows` 找到棋局 Window ID；首次使用先执行 `./python.sh main.py --window-id <当前ID> --request-screen-permission` 并按系统提示授权、重启启动器或终端，再执行 `./python.sh main.py --window-id <当前ID> --capture-once logs/wechat_check.png`，确认图片确为当前棋盘。之后用相同 `--window-id` 启动实时服务。也可用 `--window "微信"` 按窗口名选择。检查 `/api/status` 中的 `capture_info.window_kind` 为 `wechat`、`capture_info.source` 为 `wechat_coregraphics`，且 `capture_status` 为 `ok`。Window ID 在微信重开后可能变化，程序会优先重找微信窗口而非跳到应用宝。

这些路径已通过隔离的窗口模拟和微信棋盘图片回归，但尚未在真实 Mac 微信窗口完成授权、抓帧与连续走棋验收；不能把代码级通过当成实机通过。GitHub 源码仍需先安装上方所列的 Python 依赖，不是可直接双击的自包含发布包。

## 运行方式

在项目目录中执行：

```bash
./python.sh main.py --list-windows
```

Window ID 在客户端重开后会变化；正常双击启动会按标题自动重新查找。需要手动验证时先用 `--list-windows` 获取当前 ID：

```bash
./python.sh main.py --window-id <当前ID> --capture-once logs/manual_capture.png
```

打开保存的图片，必须能看到当前象棋盘面。若命令报告未授予录屏权限，请在 macOS「系统设置 → 隐私与安全性 → 屏幕录制」中允许“象棋盘面同步”；授权后重新启动。天天象棋优先使用应用宝 ADB 本地画面通道，不主动申请录屏权限；JJ 象棋或 ADB 不可用时的窗口捕获才需要该权限。

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
稳定棋盘会自动降低截图频率，发现候选走子后会立即恢复快速双帧确认；无需手动调低 `--interval`。

## 下方 AI 对战

默认使用打包的本地 **Pikafish 2026-09-06 + NNUE**，始终执画面下方的棋子（红黑按将/帅位置自动识别），与上方对战。不会点击或注入游戏窗口；Windows 正常启动模式禁止自动回退，Pikafish 不可用时会直接显示错误。

- 实盘助手：识别到上方走子后，在看板上用箭头给出下方着法。
- 看板采用只读实盘模式，每次打开都会自动回到实时棋局，模拟状态不会覆盖识别盘面。
- 若实盘是执黑翻面，下方变成黑方，AI 仍然执下；标准开局始终红先。天天中盘可用连续两帧确认的来源/落点标记恢复回合，没有可靠标记时等待真实走子，不猜测先手。
- 捕获漏掉中间帧时，从上一稳定布局推演最多四个合法半回合（含吃子）；三至四步追赶必须同时取得稳定的最后落点/行棋方证据。找不到唯一精确路径时保持上一盘面，不做整盘猜测覆盖。
- 看板提供“节能 / 普通 / 进阶 / 高级”四档 AI 强度，默认节能档只使用 1 个搜索线程；命令行仍可精细设置。
- 黑方位于下方时，程序会把盘面和 Pikafish 着法双向旋转 180°，引擎始终接收标准象棋 FEN。

```bash
./python.sh main.py --window-id <当前ID>      # 默认节能档：1线程 / 32MB Hash
./python.sh main.py --mode mock --ai-time 1.2 --ai-threads 2 --ai-hash 64
./python.sh main.py --mode mock --ai-engine builtin
./python.sh main.py --mode mock --no-ai      # 只要同步、不要 AI
```

相关接口：`GET /api/ai`，`POST /api/ai/config`、`/api/ai/follow`。Pikafish 许可证、来源和校验值见 `engines/pikafish/README.md`。

## 离线开发

Mock 模式只能显式启用，避免把静态测试图误认为实时盘面：

```bash
./python.sh main.py --mode mock
./python.sh run_regression_tests.py
```

回归包含完整功能/启动器测试及 57 项规则跟踪、窗口缩放、搜索取消和捕获恢复专项测试。服务测试使用独立随机端口；启动器测试会停止当前服务并显式启动 Mock，结束后需重新打开生产应用。

## 接入格式

WebSocket 会发送 JSON，包含 `fen`、`text_board`、`piece_count`、`capture_status`、`last_frame_at`、`side_to_move`、`tracking_mode` 和 `tracking_status`。`tracking_mode` 为 `legal_moves`；`tracking_status` 为 `waiting`、`verifying`、`holding` 或 `synced`。单步为 `event_type: "move"`；漏采补齐为 `catchup`，带按顺序排列的 `moves`；悔棋为 `undo`，跳过回退画面的改走为 `undo_branch`；视觉回合确认是 `turn_confirmed`。HTTP `/fen` 返回当前 FEN 纯文本。

主状态 `board` / `fen` / `side_to_move` 始终对应稳定实盘；模拟盘仅在 `ai.board` / `ai.fen` / `ai.to_move`。回合未确认时 `side_to_move` 为 `unknown`；FEN 因格式限制暂用 `w`，下游必须检查该字段，不得据此启动搜索。
