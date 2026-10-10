# 当前架构（architecture）

> Owner：GPT-6 Astra；最终决策权：用户。2026-09-16。
> §1–4记录旧外置实现（历史FACT）；§5为约束及其原时点缺口，不能混读。当前研究/交付实现见§0；原始对账见review TR01–TR60。2026-09-18已补A2实际交付增量。

## 0. 新增已验收研究基线（FACT，2026-09-17）

内层workspace/pipeline_baseline已形成共享core + tensor runner/离线文件两个薄入口；监督进程派生worker，worker持有加载/调用预算并落盘，监督者读取最新记录合并判定。固定fp32、局部seed42、mode后验、原VOSR/LoRA/分块/完整wavelet链路，有限三样本的两独立实例输出与完整身份验证通过。验收和剩余资源核算/字段问题见review的PIPE-01RB结论。

这是本地eager研究实现，仍依赖外置权重/源码，不是已封装JIT或fp16提交模型。下文infer2与旧M24 runner差异仍是旧资产事实，不能套到新共享入口；目标部署与历史资产等价性仍未验证。

2026-09-17导出历史（HISTORICAL）：初版C1失败及分块/finite缺陷见review EXP-R01–R07；不能把该快照当成最新A2代码现状。

2026-09-18 A2交付实现（FACT）：内层workspace/export_baseline形成单一ScriptModule容器及trace子模块（VAE 512/1024、DINO、DiT一步、Decoder 64/128、张量wavelet）；Python runner负责编排、分块融合、显式局部噪声与接口检查。**实际混合精度**为VAE编码器/LightDecoder FP32、DiT/DINO FP16、融合/wavelet FP32；编码出口存在FP16边界cast，不能把简称“A2_fp16”解释为所有数值全FP16。原始权重、转换/配方、交付件及100图通过manifest关联；包内runner只需torch与标准库。`models_export.json`和交付runner定义实际配方，不用研究实现的默认配置代替。

当前已经生成100张JPEG并形成本地ZIP，用户已提交及收到总分；身份、路径、成绩唯一见progress §22。固定样本同配方/同件跨加载的通过读数与旧FP32迁移FAIL披露并存，限定验收见D33/review。官方实际runner、完整环境与资格未知；旧M24和新A2不是已证明等价的同一产物。下一任务只是SCORE-01规划，不以新架构重写代替性能测量。

## 1. 实际系统边界

项目由离线研究脚本、外置权重/源码、图像文件、JSON/日志和提交ZIP协作，不是完整可检出的部署工程。当前主力是 VOSR 0.5B + LoRA；仓库内 LightLatentDiff 为历史主线。未见业务数据库或在线服务。文件定位归 project_map。

| 模块 | 输入 → 输出 | 当前关系 |
|---|---|---|
| 离线VOSR推理 | LQ、基座、LoRA/scale、配置 → PNG | D:/tmp_timing/vosr_lora_infer2.py；承担高分辨率研究出图 |
| 数据/质量评估 | 预测图、可选GT、指标配置 → JSON/日志 | workspace/eval.py及外部measure/q脚本；并非完全同一口径 |
| M24 runner | 图像tensor、普通权重字典 → float32图像tensor | D:/models/final_compliant/model_dir/runner.py；与离线路径存在实质差异 |
| Tiny runner | 图像tensor、JIT模型 → 图像tensor | 独立模型家族；部分含CUDA Graph；不能替代VOSR主体 |
| 打包/门禁 | model_dir、output_dir → ZIP/检查报告 | 外置工具；已有10项门禁不充分，失败传播缺陷未修 |
| 成绩/拟合 | 人工记录、测速/质量读数 → 台账、经验拟合 | 输入关联/参数版本有问题，不能作为官方公式 |

历史上质量文件与测速文件分别进入评测；两份高分包确实混用主体。DECISION：此历史结构不是今后允许的组合方式，D17已禁止。

## 2. VOSR 当前离线路径

FACT：VAE 将 LQ 编码为1/8空间分辨率、4通道潜码；DINO vitb14条件（输入448，所选索引8即第9个block）与独立随机z输入DiT；时间 t=1、r=0，一步 z−u，再由 LightDecoder 解码。r不是CFG scale。0.5B配置包括 depth28、dim1024、heads16、patch2；不是旧小UNet的 t=500 链路。

FACT：infer2 接收基座目录，先查 checkpoints、再用 clean_weights 回退；基座参数与CLI会合并，非None的CLI默认值可能覆盖args.json。DINO已有绝对缓存路径补丁，但缺失时仍可能远程回退；不能宣称完全离线。当前CWD操作习惯不等于代码仍必然依赖旧相对路径。

| 差异 | 离线 infer2 / tiled_vae | 现有M24 runner |
|---|---|---|
| 精度 | 默认fp32，无统一half/autocast | half、eval |
| VAE后验 | TILE默认mode=True | sample |
| 随机数 | 入口设置torch/CUDA/numpy seed，仍有独立z | 随机采样路径；尚无与离线成对复现的验收 |
| 空间处理 | VAE/DiT分块、高斯融合 | padding至8倍数，完整前向 |
| 后处理 | 可选颜色对齐，默认wavelet，输出PNG | 解码后clamp、返回float32；无相同wavelet后处理 |
| 权重载体 | 基座 + LoRA/scale | torch.load字典；不是JIT |

FACT：DiT tile512像素；CLI overlap32经实现下限换算后有效overlap为64像素（latent64/overlap8，步长448像素）。VAE tile1024的有效overlap128像素，步长896像素。不同tile上的DINO条件、全图共享z与融合共同影响输出。不能用每512列采样的旧seam统计作为正确分块边界证明。

UNKNOWN：跨两入口逐像素一致性、同设备确定性和导出数值误差上限，未实际运行验证。“有随机z”不证明不可复现；“设seed”也不证明已经可复现。

## 3. 核心接口与数据语义

FACT：交付入口形态为 Runner(model_dir).infer(image_tensor, prompt="")；M24将输入转到所需设备、padding、推理、裁回，并返回clamp后的float32输出。官方样例/本地文本的接口与格式要求需独立核验，不以自定义模拟器代替官方环境。

FACT：离线PNG转交付JPEG涉及不同数据阶段；指标必须记录格式、样本ID、数量、裁切/缩放、分辨率、指标配置与来源。现有eval可能部分指标失败后继续、计数范围混淆、重复case覆盖及静默resize；TR32/33未修。现有Q筛查也存在缺文件/部分样本，不能称全协议扫描完结。

FACT：台账/拟合是辅助分析层，不是质量或官方成绩真值源。score_pred2与v3.1参数不同，Q拟合存在样本-分数错配，GT路径glob错误；见TR35–39、TR52。任何“上限”“有参权重为零”尚不足以作为架构约束。

## 4. 历史代码与当前验证边界

FACT：LightLatentDiff使用固定t=500；部分注释仍写999。train_local载入权重后新建优化器/调度器是warm start；是否意图完整resume UNKNOWN，不将其单独定为训练失败的已证根因。

FACT：69文件AST解析、五包图像核验属于已有静态检查（读数在progress），不是当前VOSR全链路集成测试。Tiny graph fallback的宽度检查不足以概括支持任意形状；raw x0越界与最终clamp返回值需分开。现有门禁可忽略子进程非零返回码，打印成功不构成通过。

UNKNOWN：全依赖锁定、训练数据独立性、所有资产许可证/来源、官方当前审核与计时细节。无证据不以文档补写成功。

## 5. 下一阶段认可的约束（尚未实现）

DECISION：采用 decisions D17/D18 的同主体证据链；JIT要求未豁免；离线出图与测速不得换模型或跳过主体。现存 M24 的普通字典、精度/采样/后处理差异意味着尚未验收该目标。

DECISION：RCON-01仓库内工作副本已通过CPU范围验收，原外置门禁及生产入口未切换。当前任务PIPE-01准备建立统一eager研究入口与小样本成对验证；上文现有离线/runner差异仍是实际代码现状，不能用新任务参数覆盖。JIT、性能优化和候选比较尚未实施，没有“格式门禁通过即可上传”的自动下一步。
