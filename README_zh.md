# MacWhisper

[English](README.md) | [中文](README_zh.md)

一款轻量级 macOS 菜单栏语音转录工具，基于 [OpenAI Whisper](https://github.com/openai/whisper)，通过 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上本地运行。

**按住快捷键，说话，松开 — 文字立即出现。** 无需云端 API，无需订阅，一切在本地完成。

## 功能

- **按住录音** — 按住右 Option 键录音，松开自动转录并粘贴到光标位置
- **实时字幕** — 浮动窗口实时显示已确认（白色）和未确认（灰色）的转录文字
- **会议模式** — 持续录制，自动分段，支持麦克风 + 系统音频双声道捕获，导出 Markdown/JSON/SRT
- **中英双语** — 支持中文、英文、中英混合语音识别
- **翻译模式** — 一键切换，将任何语言翻译成英文输出
- **三种模型** — 随时在 Small、Medium、Large 之间切换
- **菜单栏状态** — 🎙 就绪 / 🔴 录音中 / 💬 转录中 / 🌐 翻译模式
- **快捷键可自定义** — 默认：按住右 ⌥ 说话，`⌃⌥M` 或 `⌃⇧F8` 会议，`⌃⇧M` 切换模型，`⌃⇧T` 翻译，`⌃⇧S` 实时字幕；都可以在 **Settings…** 里修改
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

`install.sh` 会做四件事：
1. 创建 App 的运行环境 `venv` 并安装依赖
2. 生成 `/Applications/MacWhisper.app`，它运行的就是这个文件夹里的代码
3. 在独立环境 `.venv-qwen` 里装好千问 Qwen3-ASR 语音引擎，并下载模型（第一次安装时最耗时的一步）
4. 检查所有依赖都能正常导入

如果第 3 步失败，安装会继续往下走；之后运行 `make qwen-setup` 补上即可。

### 2. 编译系统声音小程序（会议模式需要）

会议模式同时录你的麦克风和对方的声音（系统声音）。录系统声音需要一个 Swift 小程序，`install.sh` 目前还不会自动编译它：

```bash
cd swift/SystemAudioHelper && swift build -c release && cd ../..
```

没有它，会议只能录到你自己的麦克风。按住说话（Push to Talk）不需要它。

### 3. 授予权限，然后启动

从 `/Applications` 打开 **MacWhisper**，按下面的说明授予权限。授权后要退出再重新打开，macOS 只在启动时读取权限。

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

然后退出并重新打开 MacWhisper，App 始终运行这个文件夹里的代码。只有移动了文件夹或重建了 `venv` 时才需要运行 `make app`；如果 `swift/` 有改动，重新做一次第 2 步。

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
| **翻译开关** | **⌃⇧T** 或 Settings |
| **所有选项 / 修改快捷键** | 菜单栏 → **Settings…**（`⌘,`） |
| **退出** | 点菜单栏图标 → Quit |

以上都是默认快捷键，可以在 Settings 里修改：点一下快捷键，按下新的组合（Esc 取消）。Push to Talk 可以用右 ⌥/⌘/⌃/⇧ 或 F1–F20。会议模式的快捷键从 ⌃⇧R（v0.7.0 及以前）改成了 ⌃⌥M / ⌃⇧F8。

### 菜单栏图标

| 图标 | 状态 |
|---|---|
| 🎙 | 就绪（转录模式） |
| 🌐 | 就绪（翻译模式） |
| 🔴 | 录音中 |
| 💬 | 转录中 |

## 模型对比

我们用同一段中英混合的测试台词，对比了三个模型的准确率：

> 我上周在San Francisco参加了一个conference，speaker是一个叫做Andrew Ng的professor。他讲了about artificial intelligence and machine learning。会议是在March 15th，大概有2000 people参加。The ticket price was $299 per person。

### 测试结果

| 测试项 | 预期结果 | Small | Medium | Large |
|---|---|---|---|---|
| San Francisco | San Francisco | ❌ 三方西斯与庆祝 | ✅ San Francisco | ✅ San Francisco |
| conference | conference | ❌ 订阅会 | ✅ conference | ✅ conference |
| Andrew Ng | Andrew Ng | ❌ AndroidNG | ❌ Andre NG | ✅ Andrew N.G. |
| professor | professor | ❌ 教授 | ✅ professor | ✅ professor |
| March 15th | March 15th | ⚠️ march 15 | ⚠️ March 15 | ✅ March 15th |
| 2000 people | 2000 people | ❌ 2000批会 | ⚠️ 2000人 | ✅ 2000 people |
| ticket price | ticket price | ❌ taking surprise | ✅ ticket price | ✅ ticket price |
| $299 | $299 | ❌ 299 | ⚠️ 299 | ✅ $299 |

### 总结

| 模型 | 大小 | 首次加载 | 准确率 | 适用场景 |
|---|---|---|---|---|
| **Small** | ~460 MB | ~5秒 | 低 — 中文中夹杂英文时容易出错 | 快速草稿、纯单语使用 |
| **Medium** | ~1.5 GB | ~10秒 | 良好 — 双语处理好，偶尔专有名词出错 | 日常使用（推荐） |
| **Large** | ~3 GB | ~30秒 | 优秀 — 专有名词、数字、格式全部正确 | 重要会议、正式文档 |

> **注意：** "首次加载"指 app 启动后第一次使用某个模型的 JIT 编译时间，之后同一模型的转录几乎是即时的。

## 配置

设置保存在 `~/.macwhisper/config.json`：

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx",
  "live_mode": false,
  "save_audio": false,
  "log_level": "WARNING"
}
```

## 技术栈

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon 机器学习框架
- [Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)（通过 [mlx-audio](https://github.com/Blaizzy/mlx-audio)）— 在 Apple GPU 上运行的语音识别
- [rumps](https://github.com/jaredks/rumps) — Python macOS 菜单栏应用框架
- [pynput](https://github.com/moses-palmer/pynput) — 全局键盘监听
- [sounddevice](https://python-sounddevice.readthedocs.io/) — 通过 PortAudio 进行音频输入

## 许可证

MIT
