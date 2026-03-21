import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile, os
from faster_whisper import WhisperModel

print("正在加载模型（首次运行会下载约 769MB，请稍候）...")
# medium = 769MB，够用且速度快；想更准换 large-v3（1.5GB）
model = WhisperModel("medium", device="cpu", compute_type="int8")
print("模型加载完成。\n")

SAMPLE_RATE = 16000

while True:
    input("【按回车开始录音，录完再按回车】")
    print("录音中...")

    frames = []

    def callback(indata, frame_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    print("转录中...")
    audio = np.concatenate(frames, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio)
        tmp_path = f.name

    segments, info = model.transcribe(tmp_path, beam_size=5)
    os.unlink(tmp_path)

    print(f"\n检测语言：{info.language}（置信度 {info.language_probability:.0%}）")
    print("─" * 40)
    for seg in segments:
        print(seg.text)
    print("─" * 40 + "\n")
