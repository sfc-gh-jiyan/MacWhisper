# Homebrew Formula for MacWhisper (MLX)
#
# NOTE: The url and sha256 in the tap repo (sfc-gh-jiyan/homebrew-macwhisper)
# are automatically updated by CI when a v*.*.0 tag is pushed.
# See .github/workflows/ci.yml (update-homebrew job).
#
# Manual install: brew tap sfc-gh-jiyan/macwhisper && brew install macwhisper-mlx
#

class MacwhisperMlx < Formula
  desc "macOS menu bar real-time speech transcription powered by MLX Whisper"
  homepage "https://github.com/sfc-gh-jiyan/MacWhisper"
  # NOTE: url and sha256 are auto-updated by CI on release.
  # These values reflect the latest tagged release for local reference.
  url "https://github.com/sfc-gh-jiyan/MacWhisper/releases/download/v0.6.0/MacWhisper-0.6.0.tar.gz"
  sha256 "PLACEHOLDER_UPDATED_BY_CI"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64
  depends_on "python@3.12"

  def install
    libexec.install Dir["*"]

    # Create virtualenv and install Python deps
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv.to_s
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
