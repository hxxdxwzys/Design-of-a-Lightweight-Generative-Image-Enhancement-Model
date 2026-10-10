# 项目地图（project_map）

> **2026-10-10资产增量**：仓库外工具/配置/台账/运行日志的文本快照在 `docs/handoff/CLEANUP_20261010/external_sources/`，按原根名和相对路径定位；V3/V4t1原包、两份正式复核包、740个去重小权重blob在私有GitHub Release `project-archive-20261010`。原始数据、VOSR基座、LoRA、当前upload与复核工程仍保留；A2原ZIP迁至 `D:/project_archive_20261010/Track2_A2_fp16_v1_42.2145.zip`。下方旧`D:/tmp_timing/`、旧`C:/models/exports/`及历史提交包路径属于已清理历史，恢复须查 `handoff/CLEANUP_20261010_磁盘清理与归档.md`，不直接当作当前可用资产。

> **2026-09-26复核增量**：当前正式V3复核工程在 `D:/models/review_V3_20260926/V3_Review_Final/`，正式ZIP为同级 `Track2_TinyRefV3_Review_Final.zip`，执行证据在 `evidence/`；新增构建/验证源码在Git根 `workspace/review_v3_20260926/`（不是内层workspace）。工程通过两样例双进程精确复现，正式一页PPT已补齐；本地材料就绪，待用户提交。详细身份唯一见 `handoff/REVIEWV3_20260926_正式材料收口.md`。同级 `V3_Review/` 是此前缺PPT草稿，旧 `D:/models/finals_20260927/` 是V4t1材料，均保留不覆盖，不能混为当前交付。

> Owner：GPT-6 Astra；最终决策权：用户。核验基线：2026-09-16。
> FACT 限于已核验文件/代码；存在不代表运行通过。当前任务和授权只见 current_task / decisions。
> 2026-09-18增量：内层`workspace/pipeline_baseline/`为已验收同主体eager研究基线，小证据在其`evidence/PIPE01RB_20260917/`，大产物在`D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01/`。内层`workspace/export_baseline/`为导出/交付实现；**A2已完成100图、本地ZIP、用户提交与评分反馈**，最新文件身份及改名后绝对路径唯一见progress §22；运行资产在`C:/models/exports/DELIV02_20260918_073810/`，小证据在`export_baseline/evidence/DELIV02A2FINAL_20260918_114638/`。原`EXPORT01_RUN_20260917_135055/`是失败历史，不是当前提交包。限定验收见review/D33，下一任务草案见current_task §0G。

## 1. 项目身份

FACT：天池赛道二“生成式图像增强轻量模型设计挑战”的参赛工程及实验记录，目标是同分辨率图像增强。包含官方材料、100张4K测试图、3对验证图、历史自训代码、外置 VOSR/LoRA 资产、离线评测与提交包。不是已封装完毕的单一应用；主力源码和权重大量未纳入本仓库 Git。

DECISION：未来质量图与测速件必须为同一主体，详见 decisions D17/D18。历史高分组合不代表现行路线。规则来源及无法确认的赛事现状见 review，不在地图维护动态榜单。

## 2. 仓库内定位

以下相对路径均以 Git 根 C:/生成式图像增强轻量模型设计挑战 为基准。

| 位置 | 当前内容 / 用途 |
|---|---|
| AGENTS.md / README.md | 协作规则 / 项目入口；README未纳入本轮修改范围，涉及旧策略时结合 D17 与 review 勘误阅读 |
| docs/ | 六份现行文档；另有 handoff 17、reference 6、archive 4 个文件（均含 INDEX，核验时合计33份） |
| 生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战.txt | 本地赛题文本；不能替代后来公告原件 |
| 生成式图像增强轻量模型设计挑战/runner.py | 官方样例接口，不是当前主力 runner |
| 生成式图像增强轻量模型设计挑战/赛题二/ | 测试/验证数据；测试图不入库 |
| 生成式图像增强轻量模型设计挑战/workspace/ | 历史主线代码，核验62个Python文件；主力外置脚本不在这里 |
| 提交历史/ | 18个本地归档ZIP；其中五包有结构/图像核验记录，见 progress |
| DreamLite-main/、neural-enhance-master/ | 第三方资料；后者与当前主线的调用关系 UNKNOWN |

workspace 内：models/light_model.py 是旧 LightLatentDiff；train*.py 是历史自训/蒸馏实验；data/dataset.py、data/degradation.py 管理训练数据与同分辨率退化；infer.py、export*.py、submit/、speed_submit/ 是历史推理/导出入口；eval.py 实现本地八指标评估；results/ 保存历史 JSON。ssh*/srv.py 是历史运维，不证明云端当前状态。源码有 smoke 检查，但不能据此声称当前 VOSR 端到端受测覆盖。

## 3. 仓库外关键位置（FACT：本机资产，非可检出依赖）

| 资产 | 位置及限定 |
|---|---|
| VOSR 0.5B 基座 | C:/models/finals/preset/ckpts/VOSR_0.5B_os，含 args.json、checkpoints/ |
| VAE/解码/DINO资源 | 基座父目录下 stable-diffusion-2-1-base、sd21_lwdecoder.pth、torch_cache/；精确运行配置见 architecture |
| VOSR源码 | D:/models/finals/VOSR/ |
| 离线出图、训练、评测、打包及台账工具 | D:/tmp_timing/；关键文件索引见 review 证据定位表 |
| LoRA | D:/models/vosr_lora/；v5文件为 v5_s1000.pt…v5_s4000.pt；v6/v7/v8为 lora_s*.pt，已见终点分别5000/6000/9000 |
| v1/v2/v3候选目录 | 上述LoRA目录内只见日志/指标，未见权重；其他备份 UNKNOWN，不能凭日志创建“存在”的候选 |
| 现有M24研究/包资产 | D:/models/final_compliant/{model_dir,output_dir}；目录名不是合规结论，当前代码差异见 architecture |
| Tiny/PixB包 | D:/models/pkg_Tiny*、D:/models/pkg_PixB 等；不能搭配VOSR图作新提交 |
| 工作流目录 | D:/models/upload/、D:/models/已提交待评测/、D:/models/不提交_空模型/、D:/models/待判_v11b/；名称不证明平台状态 |
| 数据 | D:/datasets/{div2k,flickr2k,distill256,distill_hypir,holdout_v2,tiny_patches}；训练唯一场景数/去重和分割 manifest 尚缺 |
| 外部候选源 | D:/models/newsrc/、D:/models/hypir/ |

FACT：核验终端默认 python 与 D:/Python/python.exe 均为3.13.0且可找到 numpy；不再使用“默认Python没有numpy”的旧断言。依赖版本读数归 progress；本地环境不等于锁定的可复现安装方案。

## 4. 六文档职责与历史索引

| 唯一维护处 | 保留内容 |
|---|---|
| project_map | 路径、资产、模块定位、文档入口 |
| architecture | 当前实现、数据/接口关系、与目标约束的差距 |
| decisions | 用户选择、技术决策、有效期、替代关系、标签规范 |
| current_task | 唯一现行任务、允许改动、先验判据、停止条件 |
| progress | 已执行工作与测试读数，不把计划写作进行中 |
| review | TR01–TR60完整调查、历史勘误、审查与未关闭问题 |

历史入口为 docs/handoff/INDEX.md、docs/archive/INDEX.md、docs/reference/INDEX.md。这些目录的名称或索引“有效/失效”概括不能证明每一段内容都有效/失效：

- handoff：按当时视角保存的档案，不继续执行旧提示词。
- archive：方案已不作为现行指令，仍可能含有效读数。
- reference：规则转述、评分记录、研究和旧runbook；逐项查来源和日期，不当作最新授权。
- 本轮不迁移、不删除、不改写原文。勘误索引统一见 review；未来清理须另行授权。
