# 2026-10-10 清理前保存的项目证据

FACT：用户授权深度清理 C/D 盘项目相关产物，同时要求 GitHub 保留充分内容。范围由两盘根目录发现，再只检查已识别的项目目录；没有扫描其它用户项目或系统目录。

## 保存内容

- `external_sources/`：仓库外工具、配置、台账、训练/评测/失败日志的字节副本。一级目录名对应 `file_inventory.json::root`，二级以下保持原相对路径。它们是历史快照，不代表当前推荐实现。
- `file_inventory.json`：清理前文件的路径、大小、时间和保存方式。文本及小权重记录 SHA-256；ZIP 记录整包 SHA-256。没有散列字段的图像/大数组仅记录元数据，不能据此宣称其内容已完整备份。
- `checkpoint_index.json`：较小权重的原路径、SHA-256 和归档 blob 对应关系。实际权重在 GitHub Release 的 `research-small-checkpoints.zip` 中，不直接放入 Git。
- `historical_zip_members.json`：历史 ZIP 身份和成员集合。除单独保存的原件外，旧 ZIP 的输出图和大权重没有全部保存；该清单不能重建原 ZIP 字节。
- `cleanup_plan.json`：精确清理目标；`cleanup_execution.json`（执行后生成）记录删除/跳过结果。
- `release_assets.json` / `release_upload_receipt.json`：原件、复核包和小权重归档的本地 SHA-256，以及 GitHub 上传返回的 SHA-256 核验结果。

敏感文本只保存在 `D:/project_archive_20261010/private_sources/`，不会推送到 GitHub。其保存方式在清单中标为 `private_local_only`。

## 正式原件与研究权重

Release 固定名 `project-archive-20261010`：

- V3 原提交 ZIP；V4t1 原 ZIP；V3 正式复核 ZIP；V4t1 EdgeFix 复核 ZIP。
- 较小的历史/研究权重，按内容哈希去重。研究失败、未验收、未提交的状态不因归档而改变，恢复使用前必须查对应报告。

A2 的完整已评分 ZIP（42.2145）另存 `D:/project_archive_20261010/Track2_A2_fp16_v1_42.2145.zip`，SHA-256 为 `49bd0e7e2df18cbad6e490998b76ad69031d43e33afc913843a912db424a1a67`；它目前是本地保留项，不在此次 Release 中。

## 恢复路径

将 `external_sources/<root>/<relative_path>` 复制回其原根即可恢复该文本文件。完整复制后先按清单比对 SHA-256。源代码中的旧绝对路径没有重写，恢复后需按实际环境调整配置。

恢复小权重时，下载 `research-small-checkpoints.zip`，按其中 `checkpoint_index.json` 将 `blobs/<sha>.<ext>` 复制到所需 `source`，再核对 SHA-256。不要反序列化来源未知的权重。

原始 DIV2K/Flickr2K、VOSR 基座、LoRA、当前提交目录及正式复核工程保留在本机。派生训练缓存、旧实验输出和旧环境可再生成，清理后不会假装其仍位于历史路径。Python 环境冻结清单保存在 `external_sources/environment/`，仅记录版本，不保证其它机器可逐位复现。
