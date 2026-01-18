#!/usr/bin/env bash
set -e

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y tmux netcat-openbsd build-essential

echo "=== Installing bd cli ==="
go install github.com/steveyegge/beads/cmd/bd@latest

echo "=== Installing Claude Beads Plugin ==="
claude plugin marketplace add steveyegge/beads
claude plugin install beads

# echo "=== Setting up ML cache directory ==="
# # Use persistent volume for HuggingFace cache
# sudo mkdir -p /ml-cache/huggingface /ml-cache/pip
# sudo chown -R vscode:vscode /ml-cache
# mkdir -p ~/.cache
# ln -sf /ml-cache/huggingface ~/.cache/huggingface
# export PIP_CACHE_DIR=/ml-cache/pip
#
# echo "=== Setting up Python virtual environment ==="
# cd /workspaces/joco
# python3 -m venv .venv
# source .venv/bin/activate
#
# echo "=== Installing Python ML packages ==="
# pip install --upgrade pip
# if [ -f requirements-finetune.txt ]; then
#     # Install CPU-only PyTorch first (smaller than GPU version)
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
#     pip install -r requirements-finetune.txt
# fi
#
# echo "=== Cloning llama.cpp for GGUF conversion ==="
# if [ ! -d ~/llama.cpp ]; then
#     git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/llama.cpp
#     cd ~/llama.cpp
#     pip install -r requirements.txt
# fi
#
# echo "=== Setup complete ==="
# echo "To activate the ML environment: source .venv/bin/activate"
# echo "To convert models to GGUF: python ~/llama.cpp/convert_hf_to_gguf.py"
