#!/bin/bash
# Setup script for GPU VM (Vast.ai / Lambda Labs / RunPod)
# Run this ONCE after uploading the code.
#
# Usage:
#   chmod +x setup_gpu_vm.sh
#   ./setup_gpu_vm.sh
#   python run_all_gpu.py

set -e

echo "=== Installing Java (for BM25/Pyserini) ==="
apt-get update && apt-get install -y default-jdk
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "JAVA_HOME=$JAVA_HOME"

echo "=== Installing Python dependencies ==="
pip install -r requirements_gpu.txt

echo "=== Verifying setup ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import langchain; print(f'LangChain {langchain.__version__}')"
python -c "from llama_cpp import Llama; print('llama-cpp OK')"
python -c "from pyserini.search.lucene import LuceneSearcher; print('Pyserini OK')"

echo ""
echo "=== Setup complete! ==="
echo "Now run:  python run_all_gpu.py"
echo ""
