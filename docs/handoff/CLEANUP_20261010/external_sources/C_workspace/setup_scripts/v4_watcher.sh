#!/bin/bash
# v4 训练完成后自动: 评测(step5000+final) + 打包提交件
LOG=/data/train_v4.log
echo "v4_watcher waiting"
for i in $(seq 1 360); do
  if grep -q "v4 training done" "$LOG" 2>/dev/null || [ -f /data/proj/runs/v4/v4_final.pth ]; then
    echo "v4 finished (iter $i)"; break
  fi
  sleep 60
done
sleep 15
echo "v4_watcher: eval step5000"
[ -f /data/proj/runs/v4/v4_step5000.pth ] && bash /data/proj/eval_ckpt_v4.sh /data/proj/runs/v4/v4_step5000.pth > /data/v4_step5000_eval.txt 2>&1
echo "v4_watcher: eval final"
[ -f /data/proj/runs/v4/v4_final.pth ] && bash /data/proj/eval_ckpt_v4.sh /data/proj/runs/v4/v4_final.pth > /data/v4_final_eval.txt 2>&1
echo "v4_watcher: package"
bash /data/proj/package_submission_v4.sh /data/proj/runs/v4/v4_final.pth /data/submit_v4.zip > /data/v4_package.txt 2>&1
echo ALL_DONE > /data/v4_pipeline.status
echo "v4_watcher ALL_DONE"
