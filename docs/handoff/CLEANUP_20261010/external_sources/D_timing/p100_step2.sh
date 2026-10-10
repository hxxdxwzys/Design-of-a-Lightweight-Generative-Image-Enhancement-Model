#!/bin/bash
# P100 step2: 确保 huggingface_hub + 启动权重镜像下载 (2026-09-02)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"

echo "=== [1/3] huggingface_hub ==="
python -c "import huggingface_hub; print('HF_HUB', huggingface_hub.__version__)" 2>/dev/null || \
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub

echo "=== [2/3] 确保 p100_dl.py 运行 ==="
if pgrep -f "p100_dl[.]py" > /dev/null 2>&1; then
  echo DL_ALREADY_RUNNING
else
  nohup python "$HOME/p100/tools/p100_dl.py" > "$HOME/p100/logs/dl.log" 2>&1 &
  echo DL_STARTED_PID_$!
fi

echo "=== [3/3] 当前下载日志 ==="
sleep 5
tail -5 "$HOME/p100/logs/dl.log" 2>/dev/null || echo "(日志尚未生成)"
echo STEP2_DONE
