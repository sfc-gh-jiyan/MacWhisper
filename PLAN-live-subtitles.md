# MacWhisper 实时字幕 — 实现计划

**分支：** `feature/live-subtitles`
**修改文件：** `app.py`（唯一）
**新增依赖：** 无（PyObjC 已通过 rumps 间接安装）

---

## 你要的东西（一句话）

开启实时字幕 → 说话 → 浮窗实时显示文字（预览）→ 停止 → 浮窗消失 → 完整音频重新转录 → 自动粘贴到输入框。

## 流程图

```
开启实时字幕（菜单 或 Ctrl+Shift+S）
     │
     ▼
浮窗出现（透明无边框，屏幕底部）
麦克风持续录音
     │
     ├──→ 每 4 秒分块 → Whisper 推理 → 浮窗显示文字（预览）
     │
     └──→ 同时把完整音频存到缓冲区
     │
     ▼
停止实时字幕（菜单 或 Ctrl+Shift+S）
     │
     ▼
浮窗立即消失
     │
     ▼
完整音频缓冲区 → Whisper 重新转录（更准确）
     │
     ▼
结果自动粘贴到当前输入框（和按住录音一样的行为）
```

## 按住录音 vs 实时字幕 共存

```
实时字幕运行中 + 按住 Right Option：
  → 实时字幕暂停
  → 切换到按住录音模式
  → 松开 Right Option → 按住录音的结果正常粘贴
  → 实时字幕自动恢复
```

只有一个 InputStream 始终运行（单流架构），通过切换"帧目标"来区分两种模式。

## 具体改动清单

### 1. 录音图标改色
- 🔴 → 🟠（橙色，与 macOS 麦克风隐私指示器一致）

### 2. 菜单栏图标状态
```
🎙  待机（转录模式）
🌐  待机（翻译模式）
🟠  正在录音（按住 Right Option 或 实时字幕运行中）
💬  正在转录
```
实时字幕开启后图标为 🟠（因为麦克风一直在录音）。推理时短暂变 💬，推理完切回 🟠。
实时字幕的开/关状态只在菜单项里体现，不占用菜单栏图标。

### 3. 新增菜单项
```
现有菜单：
  Status: Ready
  ──────────
  ✅ Small (Fast)
     Medium (Accurate)
     Large (Best)
  ──────────
     Translate to English
  ──────────
+ 新增：
     Live Subtitles: Off          ← 点击切换开/关
  ──────────
  Switch Model: Ctrl+Shift+M
  Toggle Translate: Ctrl+Shift+T
+ Live Subtitles: Ctrl+Shift+S   ← 新增快捷键提示
  Hold Right Option to record
  ──────────
  Quit
```

### 4. 新增快捷键
- `Ctrl+Shift+S` → 切换实时字幕开/关

### 5. 浮窗视觉规格
```
类型:       NSPanel (PyObjC)
样式:       无边框 (NSWindowStyleMaskBorderless)
背景:       黑色半透明 rgba(0,0,0,0.88)
层级:       始终置顶 (NSFloatingWindowLevel)
交互:       点击穿透 (setIgnoresMouseEvents_(True))
位置:       屏幕底部居中，距底部 40px
宽度:       min(屏幕宽度 × 80%, 960px)
高度:       动态，随文字累积自动增高（Zoom 风格）
字体:       SF Pro Text (系统字体), 18pt
行高:       1.4
文字颜色:   白色 #FFFFFF, 透明度 95%
文字阴影:   无（黑底上不需要阴影）
初始状态:   空白（不显示"正在聆听"）
```

### 6. 音频管线（单流架构）

```
麦克风 InputStream（持续运行）
    │
    │  音频回调（C 线程）
    │  无锁，直接 list.append
    │
    ├──→ live_frames[]     （实时字幕的帧）
    │     每 4 秒原子快照：
    │     snapshot = self.live_frames
    │     self.live_frames = []
    │     → 送入推理队列 → 结果显示在浮窗
    │
    └──→ full_audio_buffer[] （完整音频缓冲区）
          持续追加，停止时用于重新转录

按住 Right Option 时：
    帧目标从 live_frames 切换到 record_frames
    松开后切回 live_frames
```

### 7. 推理

- 实时字幕推理：无 language 参数、无 initial_prompt → Whisper 自动检测语言
- 重新转录推理：同上
- 推理队列：`queue.Queue(maxsize=2)`，满时丢弃最旧的块

### 8. 配置持久化

`~/.macwhisper_config.json` 新增字段：
```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-small-mlx",
  "live_mode": false
}
```

### 9. 错误处理

| 场景 | 处理 |
|---|---|
| 无麦克风 | 拒绝启动 + rumps 通知 |
| 麦克风中途断开 | 停止实时模式 + 通知 |
| 推理队列满 | 丢弃最旧块 + print 警告 |
| Large 模型推理慢 | 启动时 rumps 通知提醒延迟 |
| NSPanel 创建失败 | 禁用实时模式 + 通知 |

## 不做的东西（明确排除）

- ❌ 会议记录模式（系统音频捕获）— 第二步
- ❌ 说话人标记（Person A/B）— 第三步
- ❌ VAD 智能分段 — 后续优化
- ❌ 双语翻译字幕 — 后续
- ❌ 分块大小 UI 配置 — 先硬编码 4 秒
- ❌ 自动隐藏（静音后）— 后续
- ❌ 多显示器支持 — 先只支持主屏幕
- ❌ 窗口拖拽定位 — 先固定位置
- ❌ "正在聆听..."提示文字 — 不要
- ❌ 状态点指示器 — 不要

## 新增方法清单（app.py）

```python
# 浮窗管理
_create_overlay()          # 创建 NSPanel
_destroy_overlay()         # 销毁 NSPanel
_update_overlay(text)      # 主线程调度更新浮窗文字

# 实时字幕生命周期
_toggle_live_mode(_)       # 菜单回调：开/关
_start_live_mode()         # 启动持续录音 + 分块定时器 + 浮窗
_stop_live_mode()          # 停止录音 + 关浮窗 + 触发重新转录

# 分块推理
_live_chunk_timer()        # 每 4 秒触发一次分块快照
_live_transcription_worker()  # 实时字幕的推理工作线程
_do_live_transcribe(chunk) # 单个块的推理（无 prompt，结果送浮窗）

# 重新转录
_do_retranscribe(full_audio)  # 完整音频二次推理 → 自动粘贴
```

## 预计代码量

- 新增 ~200 行
- 修改 ~20 行（图标改色、菜单、快捷键、帧目标切换）
- 总计 app.py 从 298 行 → ~500 行
