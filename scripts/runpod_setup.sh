#!/bin/bash
# =============================================================================
# RunPod Setup Script for Novel Writer Studio
#
# Usage:
#   1. Create a RunPod GPU Pod (A100 40GB recommended, ~$1.10/hr)
#      - Template: "RunPod Pytorch 2.x"
#      - Expose port 7860 (HTTP) in pod settings
#      - Disk: 50GB container + 20GB volume (/workspace is persistent)
#
#   2. Open the pod's web terminal or SSH in, then run:
#        curl -sSL https://raw.githubusercontent.com/<your-user>/Novel_Writer/main/scripts/runpod_setup.sh | bash
#      Or clone first:
#        git clone https://github.com/<your-user>/Novel_Writer.git /workspace/Novel_Writer
#        bash /workspace/Novel_Writer/scripts/runpod_setup.sh
#
#   3. Upload your LoRA adapter (see UPLOAD section below)
#
#   4. Open the Gradio URL shown in the terminal
# =============================================================================

set -e

echo "============================================"
echo "  Novel Writer Studio — RunPod Setup"
echo "============================================"

# ---- Paths ----
WORK_DIR="/workspace"
APP_DIR="${WORK_DIR}/Novel_Writer"
MODELS_DIR="${APP_DIR}/models"

# ---- Clone repo if not present ----
if [ ! -d "$APP_DIR" ]; then
    echo "[1/4] Cloning Novel_Writer repository..."
    cd "$WORK_DIR"
    git clone https://github.com/$(git config user.name 2>/dev/null || echo "YOUR_USER")/Novel_Writer.git 2>/dev/null || {
        echo ""
        echo "ERROR: Could not clone repo. Please clone manually:"
        echo "  git clone https://github.com/YOUR_USER/Novel_Writer.git ${APP_DIR}"
        echo ""
        exit 1
    }
else
    echo "[1/4] Repository already exists at ${APP_DIR}"
    cd "$APP_DIR" && git pull --ff-only 2>/dev/null || true
fi

cd "$APP_DIR"

# ---- Install Python dependencies ----
echo "[2/4] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    gradio \
    google-genai \
    openai \
    torch \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    sentencepiece \
    loguru

echo "    Dependencies installed."

# ---- Check for LoRA adapter ----
echo "[3/4] Checking for LoRA model..."
mkdir -p "$MODELS_DIR"

ADAPTER_FOUND=false
for dir in "$MODELS_DIR"/*/; do
    if [ -f "${dir}adapter_config.json" ]; then
        ADAPTER_NAME=$(basename "$dir")
        echo "    Found LoRA adapter: ${ADAPTER_NAME}"
        ADAPTER_FOUND=true
        break
    fi
done

if [ "$ADAPTER_FOUND" = false ]; then
    echo ""
    echo "    No LoRA adapter found in ${MODELS_DIR}/"
    echo ""
    echo "    ===== UPLOAD YOUR LORA ADAPTER ====="
    echo "    Option A: From Hugging Face Hub"
    echo "      huggingface-cli download YOUR_USER/qwen3_32b_novel_lora --local-dir ${MODELS_DIR}/qwen3_32b_novel_lora"
    echo ""
    echo "    Option B: From your local machine (run on YOUR Mac):"
    echo "      scp -r models/qwen3_32b_novel_lora root@<POD_IP>:${MODELS_DIR}/"
    echo ""
    echo "    Option C: Using runpodctl (run on YOUR Mac):"
    echo "      runpodctl send models/qwen3_32b_novel_lora"
    echo "      Then on RunPod: runpodctl receive --dir ${MODELS_DIR}/"
    echo ""
    echo "    After uploading, re-run this script or start manually:"
    echo "      cd ${APP_DIR} && python3 scripts/webui.py --port 7860 --share"
    echo ""
fi

# ---- Launch ----
echo "[4/4] Launching Novel Writer Studio..."
echo ""
echo "============================================"
echo "  Starting on port 7860"
echo "  Cloud API mode is the default (no model load needed)"
echo "  Switch to 'Local Model' + load adapter for LoRA writing"
echo "============================================"
echo ""

cd "$APP_DIR"
python3 scripts/webui.py --port 7860 --share
