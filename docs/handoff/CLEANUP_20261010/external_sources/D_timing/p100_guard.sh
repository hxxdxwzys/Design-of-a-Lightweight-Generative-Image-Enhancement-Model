#!/bin/bash
# P100 综合守护 (2026-09-03): 训练/下载崩溃自动恢复 + 资源巡检
# 部署: nohup bash ~/p100/tools/p100_guard.sh > /dev/null 2>&1 &
# 防双实例: 单例锁 (pid 文件)
P="$HOME/p100"
LOG="$P/logs/guard.log"
LOCK="$P/.guard.pid"
TARGET=6000
POLL=240

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 单例锁
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "guard 已在运行 (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
log "[boot] guard 启动 pid=$$ poll=${POLL}s target=${TARGET}"

# 训练 env (与 p100_resume.sh 一致)
export PATH="$HOME/miniconda3/bin:$PATH"
export VOSR_DIR=$P/VOSR
export VOSR_CKPT=$P/weights/VOSR_0.5B_os
export VOSR_AE=$P/weights/sd21base
export VOSR_LWDEC=$P/weights/sd21_lwdecoder.pth
export VOSR_CACHE=$P/weights/torch_cache
export VOSR_DATA="$P/data/flickr2k"
export VOSR_OUT=$P/runs
export EMP_DEG=$P/tools/degradation_analysis.json
export EMP_JPEG=$P/tools/jpeg_blockiness.json
export PAUSE_FLAG=$P/PAUSE.flag

# variant -> gpu / w_clip
resume_train() {
  local v=$1 gpu=$2 wclip=$3
  if pgrep -f "train_vosr_lora[.]py.*${v}" > /dev/null 2>&1; then
    return  # 已在跑
  fi
  local newest step
  newest=$(ls -t "$P/runs/$v"/lora_s*.pt 2>/dev/null | head -1)
  if [ -z "$newest" ]; then
    log "[$v] 无存档, 跳过自动恢复"
    return
  fi
  step=$(basename "$newest" | grep -oE 's[0-9]+' | tr -d 's')
  if [ "$step" -ge "$TARGET" ]; then
    log "[$v] 最新存档 step=$step 已达目标, 无需重启"
    return
  fi
  # GPU 空闲检查 (防教学/其他占用)
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' MiB')
  if [ -n "$mem" ] && [ "$mem" -gt 1024 ]; then
    log "[$v] GPU$gpu 显存被占用(${mem}MiB), 跳过自动恢复"
    return
  fi
  log "[$v] 崩溃恢复: 从 $newest (step $step) 续训至 $TARGET"
  CUDA_VISIBLE_DEVICES=$gpu nohup python "$P/tools/train_vosr_lora.py" \
      --variant "$v" --w-clip "$wclip" --steps "$TARGET" --save-every 500 \
      --patch 768 --half --resume-from "$newest" \
      >> "$P/logs/${v}_resume.log" 2>&1 &
  log "[$v] 恢复进程 PID=$!"
}

dl14_running() { pgrep -f "p100_dl_14b[.]py" > /dev/null 2>&1; }

while true; do
  # 1) 训练守护 (双卡)
  resume_train p768_w20 0 2.0
  resume_train p768_w15 1 1.5
  # 2) 1.4B 下载守护
  if ! dl14_running && ! grep -q DL14_DONE "$P/logs/dl_14b.log" 2>/dev/null; then
    log "[dl14] 下载进程不在且未完成 -> 重启"
    nohup "$HOME/miniconda3/bin/python" "$P/tools/p100_dl_14b.py" >> "$P/logs/dl_14b.log" 2>&1 &
    log "[dl14] PID=$!"
  fi
  # 3) 巡检 (每 5 轮记一次)
  local_cnt=$((local_cnt + 1))
  if [ $((local_cnt % 5)) -eq 0 ]; then
    log "[watch] $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ') | disk $(df -h / | tail -1 | awk '{print $4}')"
  fi
  sleep $POLL
done
