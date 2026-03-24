# MacWhisper

[English](README.md) | [中文](README_zh.md)

一款轻量级 macOS 菜单栏语音转录工具，基于 [OpenAI Whisper](https://github.com/openai/whisper)，通过 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上本地运行。

**按住快捷键，说话，松开 — 文字立即出现。** 无需云端 API，无需订阅，一切在本地完成。

## 功能

- **按住录音** — 按住右 Option 键录音，松开自动转录并粘贴到光标位置
- **中英双语** — 支持中文、英文、中英混合语音识别
- **翻译模式** — 一键切换，将任何语言翻译成英文输出
- **三种模型** — 随时在 Small、Medium、Large 之间切换
- **菜单栏状态** — 🎙 就绪 / 🔴 录音中 / 💬 转录中 / 🌐 翻译模式
- **快捷键** — `Ctrl+Shift+M` 切换模型，`Ctrl+Shift+T` 切换翻译模式
- **设置持久化** — 模型选择和翻译模式重启后自动恢复
- **完全本地** — 全部在 Apple Silicon GPU 上处理，数据不离开你的电脑

## 系统要求

- macOS + Apple Silicon（M1/M2/M3/M4）
- Python 3.10+

## 安装

### 方式一：Homebrew（推荐）

```bash
# 第一步 — 添加 tap（只需一次）
brew tap sfc-gh-jiyan/macwhisper

# 第二步 — 安装
brew install macwhisper-mlx

# 第三步 — 创建 MacWhisper.app 到 /Applications
macwhisper-mlx-install
```

### 方式二：从源码安装

```bash
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
./install.sh
```

安装脚本会自动完成：
1. 创建 Python 虚拟环境并安装依赖
2. 在 `/Applications` 中创建 `MacWhisper.app`
3. 预下载默认 Whisper 模型（约 1.5 GB）
4. 验证所有依赖是否正常

### 必需权限（重要！）

MacWhisper 需要 **三项** macOS 权限才能正常运行。**不授权这些权限，应用将无法工作。**

打开 **系统设置 → 隐私与安全性**：

**1. 麦克风（Microphone）**
- 路径：隐私与安全性 → 麦克风
- 将 `MacWhisper` 的开关打开
- *用途：录制麦克风音频*

**2. 辅助功能（Accessibility）**
- 路径：隐私与安全性 → 辅助功能
- 点击 **+** 按钮，从 `/Applications` 中找到并添加 `MacWhisper`
- 将 `MacWhisper` 的开关打开
- *用途：模拟 Cmd+V 自动粘贴转录文字*

**3. 输入监控（Input Monitoring）**
- 路径：隐私与安全性 → 输入监控
- 点击 **+** 按钮，从 `/Applications` 中找到并添加 `MacWhisper`
- 将 `MacWhisper` 的开关打开
- *用途：全局监听右 Option 键作为快捷键*

> **从终端启动？** 如果你使用 `macwhisper-mlx` 命令或 `./run.sh` 而不是 .app，还需要将你的**终端应用**（Terminal、iTerm、Warp 等）也添加到以上**三项**权限中。

> **权限不生效？** 授权后可能需要退出并重新启动 MacWhisper。某些情况下，输入监控权限需要重启系统才能激活。

## 使用

**方式 A** — 从"应用程序"文件夹打开 `MacWhisper`（Launchpad 或 Spotlight 搜索）

**方式 B** — 从终端启动：
```bash
./run.sh
```

菜单栏出现麦克风图标（🎙），即可使用。

### 操作方式

| 操作 | 方式 |
|---|---|
| **录音转录** | 按住 **右 Option** 键，说话，松开 |
| **切换模型** | 点菜单栏图标，或 **Ctrl+Shift+M** |
| **翻译开关** | 点菜单栏图标，或 **Ctrl+Shift+T** |
| **退出** | 点菜单栏图标 → Quit |

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

设置保存在 `~/.macwhisper_config.json`：

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx"
}
```

## 技术栈

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon 机器学习框架
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — 针对 Apple GPU 优化的 Whisper
- [rumps](https://github.com/jaredks/rumps) — Python macOS 菜单栏应用框架
- [pynput](https://github.com/moses-palmer/pynput) — 全局键盘监听
- [sounddevice](https://python-sounddevice.readthedocs.io/) — 通过 PortAudio 进行音频输入

## 许可证

MIT
