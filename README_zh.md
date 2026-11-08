# MacWhisper

[English](README.md) | [中文](README_zh.md)

一款轻量级 macOS 菜单栏工具，用于语音输入和会议记录，基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)，通过 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上本地运行，中英文混说识别效果好。

**按住快捷键，说话，松开 — 文字立即出现。** 无需云端 API，无需订阅，一切在本地完成。

## 功能

- **按住录音** — 按住右 Option 键录音，松开自动转录并粘贴到光标位置
- **实时字幕** — 浮动窗口实时显示已确认（白色）和未确认（灰色）的转录文字
- **会议模式** — 持续录制，自动分段，捕获麦克风 + 系统音频，导出 Markdown/JSON/SRT。每行标注 **Mic**（你）或 **Remote**（通话另一方）；扬声器回声会被去掉，字幕窗口每方一行。停止后可选的终稿转录会按轨道重新转录（`meeting_final_pass`，默认开启）
- **中英双语** — 支持中文、英文、中英混合语音识别；英文原样保留（不会被翻译）
- **Qwen3-ASR 引擎** — Qwen3-ASR 0.6B（默认，快）或 1.7B（最准），随时切换
- **自动纠错** — 你的术语词表（作为热词发给 Qwen3）、纠错规则、语音教学（⇧ + Push to Talk）、用本地 LLM 每日学习新词、**Review Suggestions…**
- **可选 AI 润色** — 本地 LLM 可以改错字、让文字更通顺、整理格式，或按你的自定义规则处理（`make llm-setup`）
- **听写不丢失** — 粘贴失败会提示，**Copy Last Transcript** 可恢复最新一条听写
- **权限设置向导** — 逐步引导开启麦克风、辅助功能、输入监控（缺权限时启动即弹出，或菜单 → **Permissions…**）
- **菜单栏状态** — 🎙 就绪 / 🔴 录音中 / 💬 转录中
- **快捷键可自定义** — 默认：按住右 ⌥ 说话，`⌃⌥M` 或 `⌃⇧F8` 会议，`⌃⇧M` 切换模型，`⌃⇧S` 实时字幕；都可以在 **Settings…** 里修改
- **设置窗口** — 所有选项在一页里（菜单栏 → Settings…，`⌘,`）
- **设置持久化** — 模型选择和所有开关重启后自动恢复
- **完全本地** — 全部在 Apple Silicon GPU 上处理，数据不离开你的电脑

## 系统要求

- Apple Silicon 芯片的 Mac（M1 及以上），不支持 Intel Mac
- macOS 14 Sonoma 或更新版本
- Python 3.10+（运行 App），以及 [uv](https://docs.astral.sh/uv/) 或 Python 3.12（运行语音引擎）
- Command Line Tools：`xcode-select --install`
- 约 4 GB 可用磁盘空间（千问 Qwen3-ASR 0.6B 模型约 2.9 GB）。1.7B 模型另需约 7.6 GB，只在你选择它时才下载。

## 安装

> **请从源码安装。** Homebrew 配方目前还是 v0.6.1，那是旧的 Whisper 版本，会在下一次发版时更新。

### 1. 检查系统要求

- Apple Silicon Mac（其他架构 `install.sh` 会直接停止）
- PATH 中有 `python3` 3.10 或更高版本（`python3 --version`）
- 语音引擎需要 [uv](https://docs.astral.sh/uv/) **或** PATH 中的 `python3.12`（`brew install uv`）
- Command Line Tools（`xcode-select --install`）：`clang` 用来编译 App 启动器，`swift` 用来编译 Meeting Mode 的辅助程序
- 约 4 GB 可用磁盘空间（见[系统要求](#系统要求)）

### 2. 下载代码并运行安装脚本

```bash
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
bash install.sh
```

`install.sh` 会显示六个步骤：

| 步骤 | 做什么 | 装到哪里 |
|---|---|---|
| 1/6 | 创建 App 的 Python 环境（已存在则跳过） | 本文件夹的 `venv/` |
| 2/6 | 安装 `requirements.txt`（需要几分钟） | `venv/` |
| 3/6 | 编译原生启动器 App（`scripts/build_app.sh`），并创建数据文件夹 | `/Applications/MacWhisper.app`、`~/.macwhisper/logs`、`~/.macwhisper/audio` |
| 4/6 | 在独立环境中安装 Qwen3-ASR 语音引擎，并下载 0.6B 模型（约 2.9 GB，最耗时的一步） | `.venv-qwen/`；模型在 `~/.cache/huggingface/` |
| 5/6 | 编译系统音频辅助程序，让 Meeting Mode 能听到其他参会者 | `swift/SystemAudioHelper/.build/release/SystemAudioHelper` |
| 6/6 | 检查 App 的依赖能否导入 | — |

第 4 或第 5 步失败时，安装会继续并显示警告。解决原因后（uv / Python 3.12，或 Command Line Tools），再运行 `make qwen-setup` 或 `make helper`。没有辅助程序时，会议只录你的麦克风。

`MacWhisper.app` 只是一个启动器：它用 `venv` 运行这个文件夹里的代码。安装后不要移动或删除这个文件夹（移动后请运行 `make app`）。

### 3. 第一次启动

1. 从 `/Applications`（或 Spotlight）打开 **MacWhisper**。
2. 如果缺少权限，设置引导会自动打开，逐项处理——见[权限](#权限)。
3. 最后点击 **Restart Now**，让 Accessibility 和 Input Monitoring 生效。

### 4. 确认能用

1. 菜单栏出现麦克风图标（🎙）。
2. 点进任意输入框（例如备忘录），按住 **右 ⌥** 说一句话，松开。录音时图标变为 🔴，转写时为 💬，然后文字被粘贴进去。
3. 没有反应时，见[故障排查](#故障排查)。

### 更新

```bash
git pull
```

然后退出并重新打开 MacWhisper，App 始终运行这个文件夹里的代码。只有移动了文件夹或重建了 `venv` 时才需要运行 `make app`；如果 `swift/` 有改动，运行 `make helper`。

### 可选：本地 AI 文字润色

**设置 → What gets pasted** 可以用本地模型（Qwen3-4B，约 4.2 GB，第一次使用时下载）整理文字，默认关闭。要使用它：

```bash
make llm-setup
```

### 卸载

```bash
rm -rf /Applications/MacWhisper.app     # App 本身
rm -rf ~/.macwhisper                    # 设置、日志、词表、会议记录
```

再删掉克隆下来的文件夹即可。下载的模型保存在 `~/.cache/huggingface/`，需要的话手动删除。

## 权限

Push to Talk 需要三项 macOS 权限，Meeting Mode 还需要第四项。都在 **System Settings → Privacy & Security** 中。

| 权限 | Privacy & Security 中的位置 | 用途 | 没有它会怎样 |
|---|---|---|---|
| **Microphone** | Microphone | 录你的声音 | 什么都录不到 |
| **Accessibility** | Accessibility | 把文字粘贴到你正在输入的 App | 文字不会被粘贴 |
| **Input Monitoring** | Input Monitoring | 识别右 ⌥ 和其他快捷键 | 按住右 ⌥ 没有反应 |
| **Screen & System Audio Recording**（仅 Meeting Mode） | Screen & System Audio Recording（旧版 macOS：Screen Recording） | 录下会议中其他人的声音（系统音频） | 辅助程序只收到静音；没有 Remote 行 |

### 设置引导

启动时 MacWhisper 会检查三项必需权限。缺少任何一项，逐步设置引导会自动打开。也可以随时从菜单栏打开：**Permissions…**。每一项权限：

1. 先说明为什么需要。点击 **Allow**（macOS 弹出自己的授权提示）或 **Open Settings**（打开对应的 System Settings 页面）。**Skip** 跳过。
2. 在 System Settings 里打开 **MacWhisper**。
3. 如果仍未打开，对话框会提供 **Check Again**、**Open Settings** 和 **Skip**。
4. 最后，如果刚授予了 Accessibility 或 Input Monitoring，点击 **Restart Now**：macOS 只有在 App 重启后才对正在运行的 App 应用这两项权限。

手动设置：Accessibility 和 Input Monitoring 点击 **+**，选择 `/Applications` 中的 `MacWhisper`，再打开开关。

### 该打开哪一项

- 从 `/Applications/MacWhisper.app` 启动：打开 **MacWhisper**。启动器内嵌了 Python，所以权限属于这个 App，而不是 Python 或 Terminal。
- 用 `./run.sh` 启动：需要给你的**终端 App**（Terminal、iTerm、Warp……）授权。

想确认结果，可以在 `~/.macwhisper/logs/macwhisper.log` 里找这一行：`Permissions (pid …): accessibility=True input_monitoring=True`。`False` 表示那项权限还没开。（这一行里显示的路径是内嵌的 Python，从 App 启动时也是这样；要打开的是 **MacWhisper**，不是 Python。）

## 故障排查

**按住右 ⌥ 没反应，或者文字没有粘贴。** 从菜单打开 **Permissions…**，检查 Input Monitoring（按键）和 Accessibility（粘贴），然后重启 MacWhisper。如果开关看起来是开的但日志里是 `False`：重新编译 App 后，旧的权限记录可能和新 App 不再匹配。退出 MacWhisper，运行 `make reset-permissions`，重新打开并再次授权。

**会议里没有 Remote 行。** 确认辅助程序已编译（`make helper`；需要 `xcode-select --install`），并且 MacWhisper 的 **Screen & System Audio Recording** 已打开，然后重启。

**没有看到"Meeting saved"横幅。** 通知会进入通知中心。要看到横幅，打开 System Settings → Notifications → MacWhisper，选择 **Banners** 或 **Alerts**。如果你使用专注模式，把 MacWhisper 加入允许的 App。

**公司电脑上 App 打不开。** `MacWhisper.app` 是在你的 Mac 上编译的，没有 Apple 开发者 ID 签名。受管理的 Mac 上的安全软件可能会拦截它。请联系 IT 部门。

**日志。** `~/.macwhisper/logs/macwhisper.log`。

## 使用

**方式 A** — 在"应用程序"里双击 `MacWhisper`（或从 Launchpad / Spotlight 打开）。想开机自动启动，打开 **Settings… → General → Launch MacWhisper at login**。

**方式 B** — 从终端启动（开发用；这时权限要授给终端）：
```bash
./run.sh
```

菜单栏出现麦克风图标（🎙），即可使用。

同一时间只会运行一个 MacWhisper。已经在运行时再打开，只会弹出一条通知。想在 App 和 `./run.sh` 之间切换，先退出正在运行的那个（菜单栏 → Quit）；`./run.sh` 会自动停掉正在运行的那个。

### 操作方式

| 操作 | 方式 |
|---|---|
| **录音转录** | 按住 **右 ⌥**，说话，松开 |
| **会议模式** | **⌃⌥M** 或 **⌃⇧F8**、菜单，或会议字幕窗口里的 **■ Stop** |
| **切换模型** | **⌃⇧M** 或菜单 |
| **实时字幕** | **⌃⇧S** 或 Settings |
| **所有选项 / 修改快捷键** | 菜单栏 → **Settings…**（`⌘,`） |
| **退出** | 点菜单栏图标 → Quit |

以上都是默认快捷键，可以在 Settings 里修改：点一下快捷键，按下新的组合（Esc 取消）。Push to Talk 可以用右 ⌥/⌘/⌃/⇧ 或 F1–F20。会议模式的快捷键从 ⌃⇧R（v0.7.0 及以前）改成了 ⌃⌥M / ⌃⇧F8。

### 菜单栏图标

| 图标 | 状态 |
|---|---|
| 🎙 | 就绪 |
| 🔴 | 录音中 |
| 💬 | 转录中 |

## 模型

| 菜单名称 | 模型 | 说明 |
|---|---|---|
| **Qwen3 0.6B (Fast, zh+en)** | `mlx-community/Qwen3-ASR-0.6B-bf16` | 默认 |
| **Qwen3 1.7B (Best, zh+en)** | `mlx-community/Qwen3-ASR-1.7B-bf16` | 更准，较慢 |

**固定版本。** 上面的模型和 Qwen3-4B LLM 都按 `model_revisions.py` 里固定的 Hugging Face commit 加载，上游改动不会悄悄生效。你自己指定的模型（`MACWHISPER_QWEN_MODEL` 或 `qwen_model`）加载最新文件并记录一条警告。要更新固定版本，把 `~/.cache/huggingface/hub/models--<org>--<name>/refs/main` 里的新 commit 复制到 `MODEL_REVISIONS`。

用 **⌃⇧M** 或菜单切换。Qwen3 在独立进程中运行（`.venv-qwen`）；可选的 AI 润色和词表学习使用另一个本地 LLM（`.venv-llm`）。

Whisper 已在 0.8.0 移除。旧的 `"engine": "whisper"` 设置会自动迁移（Whisper Large → Qwen3 1.7B，其他 → 0.6B）。部分以前的 Qwen3 与 Whisper 对比数据在 [更新日志](CHANGELOG.md) 的 0.7.3 里（历史数据）。

## 配置

设置保存在 `~/.macwhisper/config.json`（大多数也在 **Settings…** 里）。部分配置项：

```json
{
  "qwen_model": "mlx-community/Qwen3-ASR-0.6B-bf16",
  "live_mode": false,
  "save_audio": false,
  "log_level": "WARNING",
  "qwen_strip_fillers": true,
  "qwen_tight_spacing": true,
  "collapse_repeats": true,
  "paste_style": "original",
  "meeting_final_pass": true
}
```

- `qwen_strip_fillers` — 去掉单独的 嗯 / 呃 / um（不会去掉词里的字，比如 额外）。
- `qwen_tight_spacing` — 中英文之间不加空格（`这个commit的情况`）。
- `collapse_repeats` — 去掉口吃式重复；有意的重复会保留，只把失控的循环（6 次及以上）缩成一次。
- `paste_style` — `original`，或 AI 风格：`light`、`medium`、`heavy`、`custom`。
- `meeting_final_pass` — 停止后按轨道重新转录会议（只在内存中）。运行期间 Push to Talk 暂停。

## 技术栈

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon 机器学习框架
- [Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)（通过 [mlx-audio](https://github.com/Blaizzy/mlx-audio)）— 在 Apple GPU 上运行的语音识别
- [rumps](https://github.com/jaredks/rumps) — Python macOS 菜单栏应用框架
- [pynput](https://github.com/moses-palmer/pynput) — 全局键盘监听
- [sounddevice](https://python-sounddevice.readthedocs.io/) — 通过 PortAudio 进行音频输入

## 许可证

MIT
