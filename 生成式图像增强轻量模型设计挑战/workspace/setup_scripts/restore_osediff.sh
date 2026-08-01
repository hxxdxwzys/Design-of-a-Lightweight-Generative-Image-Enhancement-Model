#!/bin/bash
# 恢复 OSEDiff 教师依赖: 仓库(含 osediff.pkl/DAPE.pth) + RAM 权重
# 上海网络: github TLS 不稳 -> 多路重试; HF 走 hf-mirror
set -x
mkdir -p /data/weights /data/osediff

# ---------- 1) OSEDiff 仓库 ----------
# 1a) 清理失败的 clone
rm -rf /data/osediff_tmp /data/osediff/OSEDiff
# 1b) 先试 git clone (浅层, 跳过 LFS 内容)
if [ ! -d /data/osediff/OSEDiff/.git ]; then
  export GIT_LFS_SKIP_SMUDGE=1
  git config --global http.postBuffer 104857600
  git config --global http.lowSpeedLimit 1000
  git config --global http.lowSpeedTime 300
  for i in 1 2 3 4 5; do
    echo "clone attempt $i"
    timeout 240 git clone --depth 1 https://github.com/cswry/OSEDiff.git /data/osediff/OSEDiff && break
    rm -rf /data/osediff/OSEDiff
    sleep 10
  done
fi
# 1c) git 不行就 tarball (不含 LFS 内容, 但代码完整; 权重靠第2步单独下)
if [ ! -f /data/osediff/OSEDiff/osediff.py ]; then
  rm -rf /data/osediff_tmp
  for i in 1 2 3; do
    echo "tarball attempt $i"
    curl -L --retry 3 --retry-all-errors -o /tmp/osediff.tar.gz \
      https://github.com/cswry/OSEDiff/archive/refs/heads/main.tar.gz && \
      mkdir -p /data/osediff_tmp && tar xzf /tmp/osediff.tar.gz -C /data/osediff_tmp && break
    sleep 10
  done
  [ -d /data/osediff_tmp/OSEDiff-main ] && mv /data/osediff_tmp/OSEDiff-main /data/osediff/OSEDiff
  rm -rf /data/osediff_tmp /tmp/osediff.tar.gz
fi
ls /data/osediff/OSEDiff/ 2>/dev/null

# ---------- 2) LFS 权重 (osediff.pkl / DAPE.pth) ----------
# 2a) git lfs pull (若仓库是 git 且权重是 LFS 指针)
if [ -d /data/osediff/OSEDiff/.git ]; then
  cd /data/osediff/OSEDiff
  if command -v git-lfs >/dev/null 2>&1; then
    for i in 1 2 3 4 5; do
      echo "lfs pull attempt $i"
      timeout 240 git lfs pull && break
      sleep 10
    done
  else
    echo "git-lfs NOT INSTALLED - fallback to media URL download"
  fi
fi
# 2b) 兜底: 直接按 LFS 指针文件内容解析 media URL 下载
for f in /data/osediff/OSEDiff/preset/models/osediff.pkl /data/osediff/OSEDiff/preset/models/DAPE.pth; do
  if [ -f "$f" ] && [ $(stat -c%s "$f") -lt 1000000 ]; then
    echo "LFS pointer found: $f"
    OID=$(grep -oP 'oid sha256:\K[0-9a-f]{64}' "$f" | head -1)
    if [ -n "$OID" ]; then
      URL="https://media.githubusercontent.com/media/cswry/OSEDiff/main/preset/models/$(basename $f)"
      for i in 1 2 3 4 5; do
        echo "media download attempt $i: $(basename $f)"
        timeout 600 curl -L --retry 3 --retry-all-errors -o "$f.real" "$URL" && mv "$f.real" "$f" && break
        sleep 10
      done
    fi
  fi
  [ -f "$f" ] && echo "  -> $(stat -c%s "$f") bytes"
done

# ---------- 3) RAM 权重 (hf-mirror 多路) ----------
if [ ! -f /data/weights/ram_swin_large_14m.pth ]; then
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_TOKEN=$(cat /data/hf_token 2>/dev/null)
  /data/env/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
import os
ok = False
for repo in ['xinyu1205/recognize-anything', 'cswry/OSEDiff']:
    try:
        p = hf_hub_download(repo_id=repo, filename='ram_swin_large_14m.pth',
                            local_dir='/data/weights')
        print('RAM OK from', repo, p, flush=True); ok = True; break
    except Exception as e:
        print('RAM FAIL', repo, repr(e)[:200], flush=True)
assert ok
PY
fi
ls -lh /data/weights/*.pth /data/osediff/OSEDiff/preset/models/ 2>/dev/null
echo RESTORE_DONE
