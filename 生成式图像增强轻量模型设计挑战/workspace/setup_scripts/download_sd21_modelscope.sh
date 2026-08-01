#!/bin/bash
# 从魔搭 (ModelScope, 国内快) 下载 SD2.1 base 到 D:\models\sd21base
# 用法: bash download_sd21_modelscope.sh
set -u
DEST="/d/models/sd21base"
BASE="https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v2-1/resolve/master"
mkdir -p "$DEST/vae" "$DEST/unet" "$DEST/text_encoder" "$DEST/tokenizer" "$DEST/scheduler" "$DEST/feature_extractor"

dl() {
  local rel="$1"
  if [ -s "$DEST/$rel" ]; then echo "skip $rel"; return; fi
  curl -L -C - --retry 3 --retry-all-errors --connect-timeout 20 \
    -o "$DEST/$rel" "$BASE/$rel" 2>/dev/null
  local sz=$(stat -c%s "$DEST/$rel" 2>/dev/null || echo 0)
  if [ "$sz" -gt 1000 ]; then echo "OK  $rel ($sz B)"; else echo "FAIL $rel"; fi
}

# 小文件优先 (学生训练必需: VAE + tokenizer + scheduler)
dl model_index.json
dl feature_extractor/preprocessor_config.json
dl scheduler/scheduler_config.json
dl tokenizer/tokenizer_config.json
dl tokenizer/vocab.json
dl tokenizer/merges.txt
dl tokenizer/special_tokens_map.json
dl vae/config.json
dl vae/diffusion_pytorch_model.safetensors
dl text_encoder/config.json
dl text_encoder/model.safetensors
# 大文件最后 (SD 蒸馏教师用, 慢慢下)
dl unet/config.json
dl unet/diffusion_pytorch_model.safetensors
echo SD21_DONE
ls -lh "$DEST/vae" "$DEST/text_encoder" "$DEST/unet" 2>/dev/null
