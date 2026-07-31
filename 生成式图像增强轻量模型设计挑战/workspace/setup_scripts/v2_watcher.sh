#!/bin/bash
# v2 训练完成后自动: 评测(step6000 + final) + 打包提交件。无人值守。
LOG=/data/train_v2.log
echo "watcher: waiting for v2 to finish..."
for i in $(seq 1 300); do
  if grep -q "training done" "$LOG" 2>/dev/null || [ -f /data/proj/runs/v2/unet_final.pth ]; then
    echo "watcher: v2 finished (iter $i)"; break
  fi
  sleep 60
done
sleep 15

echo "watcher: eval step6000"
if [ -f /data/proj/runs/v2/unet_step6000.pth ]; then
  bash /data/proj/eval_ckpt.sh /data/proj/runs/v2/unet_step6000.pth > /data/v2_step6000_eval.txt 2>&1
  echo "step6000 eval done" >> /data/v2_step6000_eval.txt
fi

echo "watcher: eval final"
if [ -f /data/proj/runs/v2/unet_final.pth ]; then
  bash /data/proj/eval_ckpt.sh /data/proj/runs/v2/unet_final.pth > /data/v2_final_eval.txt 2>&1
  echo "final eval done" >> /data/v2_final_eval.txt
fi

echo "watcher: package submission"
bash /data/proj/package_submission.sh /data/proj/runs/v2/unet_final.pth /data/submit_v2.zip > /data/v2_package.txt 2>&1

echo "ALL_DONE" > /data/v2_pipeline.status
echo "watcher: ALL_DONE"
