# Homebrew Formula for MacWhisper (MLX)
#
# To use this formula:
#   1. Create a GitHub repo: <your-user>/homebrew-macwhisper
#   2. Copy this file to Formula/macwhisper-mlx.rb
#   3. Update the url and sha256 after running build.sh + release.sh
#   4. Users install via: brew tap <your-user>/macwhisper && brew install macwhisper-mlx
#

class MacwhisperMlx < Formula
  desc "macOS menu bar real-time speech transcription powered by MLX Whisper"
  homepage "https://github.com/sfc-gh-jiyan/MacWhisper"
  url "https://github.com/sfc-gh-jiyan/MacWhisper/releases/download/v0.4.0/MacWhisper-0.4.0.tar.gz"
  sha256 "1742f2ce7bee79bc357373c4866f8aff6104e0afd09be55ed889190e5704a954"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64
  depends_on "python@3.11"

  def install
    libexec.install Dir["*"]

    # Create virtualenv and install Python deps
    venv = libexec/"venv"
    system "python3.11", "-m", "venv", venv.to_s
    system venv/"bin/pip", "install", "--upgrade", "pip"
    system venv/"bin/pip", "install", "-r", libexec/"requirements.txt"

    # Wrapper script
    (bin/"macwhisper-mlx").write <<~EOS
      #!/bin/bash
      cd "#{libexec}"
      exec "#{venv}/bin/python3" app.py "$@"
    EOS

    # Install script for .app bundle
    (bin/"macwhisper-mlx-install").write <<~EOS
      #!/bin/bash
      cd "#{libexec}"
      exec bash install.sh
    EOS
  end

  def caveats
    <<~EOS
      MacWhisper requires Apple Silicon (M1/M2/M3/M4).

      To create the .app bundle in /Applications:
        macwhisper-mlx-install

      Required macOS permissions (System Settings → Privacy & Security):
        1. Microphone
        2. Accessibility
        3. Input Monitoring
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{libexec}/venv/bin/python3 -c 'exec(open(\"#{libexec}/app.py\").readline()); exec(open(\"#{libexec}/app.py\").readlines()[8]); print(__version__)'")
  end
end
