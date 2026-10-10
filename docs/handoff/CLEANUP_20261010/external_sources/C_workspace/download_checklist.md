# 用户手动下载清单 (OSEDiff 官方环境复现)

> **原则（用户 2026-08-02 明确要求）**：Claude 自己下载不了/下不动时，**不硬下、不回避**，把下载需求事无巨细列在这里交给用户。用户有代理且愿意手动下载。

**背景**: OSEDiff 官方代码需要 diffusers 0.25 + transformers 4.28 + peft 0.9, 仅支持 Python 3.11。
本地 Python 3.13 装不了 (tokenizers 编译失败 + huggingface_hub API 冲突)。
**torch 需要 cp311 版**（本地 wheel 是 cp313，Python 3.11 用不了；download.pytorch.org 直连自动下载 hash 校验失败 → 必须用户手动下）。

## 唯一需要下载的

| # | 文件 | 大小 | 链接 |
|---|---|---|---|
| 1 | **Python 3.11.9 安装包** | ~30MB | https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe |

## 安装步骤 (装完 Python 后告诉我, 我来跑 pip)

```bash
# 1. 安装 Python 3.11.9 (勾选 "Add to PATH")
# 2. 创建独立 venv (Python 3.11):
C:/Users/Yuki Nagato/AppData/Local/Programs/Python/Python311/python.exe -m venv D:/venv_osediff311

# 3. 装依赖 (torch 用本地 2.7 wheel, 其他用清华源) — 我来执行
#    torch: D:/models/torch-2.7.0+cu126-cp313...whl  (注意: 需 cp311 版! 这个 cp313 不兼容)
```

⚠️ **注意**: 本地 torch wheel 是 cp313 (Python 3.13) 的, **Python 3.11 不能用**。
所以 torch 也得下 cp311 版 — 见下方备选。

## 备选方案 (如果不想装 Python 3.11)

**方案 B: 装 Python 3.11 + 用 cu126 的 torch 2.7 (cp311)**
| # | 文件 | 大小 | 链接 |
|---|---|---|---|
| 1 | Python 3.11.9 | ~30MB | https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe |
| 2 | torch 2.7.0+cu126 (cp311) | ~2.8GB | https://download.pytorch.org/whl/cu126/torch-2.7.0%2Bcu126-cp311-cp311-win_amd64.whl |
| 3 | torchvision 0.22.0+cu126 (cp311) | ~6MB | https://download.pytorch.org/whl/cu126/torchvision-0.22.0%2Bcu126-cp311-cp311-win_amd64.whl |

**方案 C: 最稳 (完全官方环境)**
| # | 文件 | 大小 | 链接 |
|---|---|---|---|
| 1 | Python 3.11.9 | ~30MB | https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe |
| 2 | torch 2.0.1+cu118 (cp311) | ~2.5GB | https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp311-cp311-win_amd64.whl |
| 3 | torchvision 0.15.2+cu118 (cp311) | ~2MB | https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp311-cp311-win_amd64.whl |

## 为什么必须装
- OSEDiff 的 `osediff.pkl` 是 LoRA 权重, vae LoRA 在 diffusers 0.25 的 add_adapter 行为下正常
- diffusers 0.39 的 add_adapter 只给 target_modules 建 LoRA, 导致 vae decoder LoRA 缺失 → 输出压平 (std 12 vs 官方 50)
- 本地已验证: 官方 benchmark 输出 std=50 (正常), 我的实现 std=12 (压平) — 确认为版本差异
