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

### 1. 下载代码并运行安装脚本

```bash
brew install uv                 # 如果 uv 和 Python 3.12 都没有
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
./install.sh
```

`install.sh` 会做五件事：
1. 创建 App 的运行环境 `venv` 并安装依赖
2. 生成 `/Applications/MacWhisper.app`，它运行的就是这个文件夹里的代码
3. 在独立环境 `.venv-qwen` 里装好千问 Qwen3-ASR 语音引擎，并下载模型（第一次安装时最耗时的一步）
4. 编译系统声音小程序，会议模式靠它录到对方的声音
5. 检查所有依赖都能正常导入

如果第 3 或第 4 步失败，安装会继续往下走并提示你；之后运行 `make qwen-setup` 或 `make helper` 补上即可。没有系统声音小程序，会议只能录到你自己的麦克风。

### 2. 授予权限，然后启动

从 `/Applications` 打开 **MacWhisper**。第一次启动时，它会一步步带你完成三个必需权限：逐个向 macOS 申请、打开对应的系统设置页面、确认已经打开，最后提示重启（辅助功能和输入监控要重启后才生效）。之后随时可以从菜单 **Permissions…** 再打开。想手动设置的话，详细说明见下文。

### 必需权限（重要！）

打开 **系统设置 → 隐私与安全性**，为 `MacWhisper` 打开：

| 权限 | 用途 |
|---|---|
| **麦克风** | 录你的声音（第一次使用时系统会询问） |
| **辅助功能** | 把文字粘贴到你正在输入的 App |
| **输入监控** | 识别右 Option 键和其他快捷键 |
| **录屏与系统录音** | 仅会议模式需要：录到对方的声音 |

辅助功能和输入监控需要点 **+**，在 `/Applications` 里选 `MacWhisper`，再打开开关。

> **从终端运行？** 如果你用 `./run.sh` 而不是 App，需要给你的**终端程序**（Terminal、iTerm、Warp 等）授予这些权限。

> **权限没生效？** 授权后退出并重新打开 MacWhisper。在 `~/.macwhisper/logs/macwhisper.log` 里查找 `Permissions (...): accessibility=True input_monitoring=True`。如果开关看起来已打开但日志显示 `False`，退出 MacWhisper，运行 `make reset-permissions`，再重新授权这两项。
>
> **没看到"会议已保存"的横幅？** 通知会进通知中心。想看到横幅，打开 系统设置 → 通知 → MacWhisper，选择 **横幅** 或 **提醒**。如果你开了专注模式，请把 MacWhisper 加入允许的 App。

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
