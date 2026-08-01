#!/usr/bin/env bash
set -euo pipefail

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

uv venv --python 3.12

if [ "$OS" = "linux" ] || [ "$OS" = "darwin" ]; then
    source .venv/bin/activate
    ACTIVATE_CMD="source .venv/bin/activate"
elif echo "$OS" | grep -q mingw; then
    .venv/Scripts/activate
    ACTIVATE_CMD=".venv\\Scripts\\activate"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

uv pip install -r requirements.txt

mkdir -p models piper interviews questions

PIPER_URL=""
PIPER_FILE=""
if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz"
    PIPER_FILE="piper_linux_x86_64.tar.gz"
elif [ "$OS" = "darwin" ] && [ "$ARCH" = "x86_64" ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/latest/download/piper_macos_x86_64.tar.gz"
    PIPER_FILE="piper_macos_x86_64.tar.gz"
elif [ "$OS" = "darwin" ] && [ "$ARCH" = "arm64" ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/latest/download/piper_macos_aarch64.tar.gz"
    PIPER_FILE="piper_macos_aarch64.tar.gz"
fi

if [ -n "$PIPER_URL" ]; then
    wget -q "$PIPER_URL" -O "/tmp/$PIPER_FILE"
    tar xzf "/tmp/$PIPER_FILE" -C piper/ --strip-components=1
    chmod +x piper/piper
fi

wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -O piper/en_US-lessac-medium.onnx
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -O piper/en_US-lessac-medium.onnx.json

echo ""
echo "Setup complete!"
echo "Activate: $ACTIVATE_CMD"
echo ""
echo "Next:"
echo "  1. Start the llama.cpp server (README §5.3):"
echo "     llama-server -m models/<gemma gguf> -ngl 999 --host 127.0.0.1"
echo "  2. Run: .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"
echo ""
echo "Note: Kokoro 82M and faster-whisper auto-download on first use."
echo "To keep them project-local: export HF_HOME=\"\$PWD/models/hf-cache\""
