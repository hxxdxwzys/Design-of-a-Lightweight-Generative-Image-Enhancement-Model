#!/bin/bash
# 预下载 OSEDiff 教师权重: SD21 unet+text_encoder(safetensors) + RAM (供潜在蒸馏用)
export HF_TOKEN=$(cat /data/hf_token)
/data/env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Manojb/stable-diffusion-2-1-base', local_dir='/data/weights/sd21base',
    allow_patterns=['unet/*.safetensors','unet/config.json',
                    'text_encoder/*.safetensors','text_encoder/config.json',
                    'tokenizer/*','scheduler/*'],
    ignore_patterns=['*.bin','*.fp16.*'], max_workers=4)
print('SD21_unet_textenc_done')
PY
if [ ! -f /data/weights/ram_swin_large_14m.pth ]; then
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o /data/weights/ram_swin_large_14m.pth \
    "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
fi
echo TEACHER_DL_DONE > /data/teacher_dl.status
ls -lh /data/weights/sd21base/unet/ /data/weights/sd21base/text_encoder/ 2>/dev/null | tail -8
