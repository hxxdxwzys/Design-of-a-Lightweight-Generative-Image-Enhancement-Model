# 当前审查与项目事实对账（review）

## 2026-09-27 17:22 V3-FIDELITY-27收口

FACT：用户授权最后一次V3保真微调，已执行。新3000步候选在本轮源图隔离holdout32上PSNR+1.166dB、LPIPS改善约13.46%；完整官方3对4K上PSNR三图均提高、均值+0.328dB，LPIPS两升一降、均值改善0.569%。有有限保真收益，但未达到运行前2%交付门槛。

DECISION：保留研究候选，不批准正式交付，不将局部进步写成全面超越V3或官方分数提升。没有启动test30、时延和100图生产。原V3/EdgeFix不改动。

FACT/UNKNOWN：最终文件在两个新进程中，零/seed42随机/固定真实图块三输入8位输出精确一致；参数与固定计算图检查通过。额外增加的“内存注入参考对保存文件”检查两次不一致，原因UNKNOWN；第一次将其归因于CUDA Graph更新顺序的解释未获验证，已撤回单因解释。原协议§7是最终文件两次独立加载，直接检验该项通过；没有把额外FAIL改记为PASS，不能声称eager等价。

证据、方法限制与资源增补见 `handoff/V3FIDELITY27_20260927_保真微调结案.md`。本轮GPU任务已结束，无自动续训。

---

## 2026-09-27 QUALITY-27结论

本节优先于下方历史时点。用户已授权8小时内由Architect亲自训练尝试并制作有证据支持的新版材料。

- **REJECTED_FOR_DELIVERY：原图条件细化器新权重。** 完成6000步，留出选择2000步；完整3张4K验证LPIPS均值0.405961，V4t1为0.401877，V3为0.397719（同一修正版出图口径，越低越好）。预登记质量门槛FAIL。默认导出检查及一次限定执行模式复查亦未达全部门槛，不放宽数值标准交包。
- **ACCEPTED（仅工程修复）：V4t1 EdgeFix。** 保留V4t1模型和runner字节，修正外缘零权重与uint8截断。两完整样例跨独立进程JPEG及编码前数组均精确一致，编码前外圈不再强制全黑；排除16像素外缘后与同一模型块旧融合最大差1色阶。10项随包CPU检查、正式ZIP隔离身份验证通过。
- **不作出的结论**：未证明新的学习能力/整体画质/官方成绩提高；未重出100图；不能由3张验证图断言V3全面优于V4t1。官方是否接受赛后修订UNKNOWN。
- **保留失败及原件**：V3正式复核ZIP哈希再次一致；初次CPU夹具/融合测试失败、导出FAIL、资源上限的执行诊断增补、首次ZIP时间戳失败均在新报告披露，旧证据未改写。

事实、资源和交付指纹见 `handoff/QUALITY27_20260927_训练试验与边缘修订交付.md`。不自动开展下一轮训练或上传。

2026-09-27 16:52恢复后只读复核：正式新ZIP与V3保底ZIP摘要均和收据一致，解压工程身份校验再次exit0。目视检查两张新完整样例；花叶示例仍有明显合成纹理/锐化观感，边缘修复没有消除父模型的这些局限，不能称整体画质更好。未把缩放预览目测当作逐像素证明。

---

## 2026-09-26 最新复核结论：V3本地材料ACCEPTED，待用户提交

本节优先于下方9/18历史“当前”结论。用户已要求Architect亲自落实复核整改。

- FACT：V3原包模型与runner不变，新增便携入口沿用历史归一化、512/32分块、羽化、uint8截断和JPEG参数。两个原始样例由原ZIP直接提取。
- FACT：16项CPU回归通过。两独立GPU进程各运行case3、case17，四次JPEG都与原ZIP逐字节一致、解码RGB差0。预留262/300次，17.95/900秒，失败不退款规则未触发失败重跑。
- DECISION：工程接受范围是所测两样例、本地记录环境、源文件身份；不扩展到100图重推理、官方环境或官方人工合规认可。历史黑边未悄悄修复。
- DECISION：用户明确授权本次不用slidemind，使用本地演示文稿工具直接制作。
- FACT：V3正式一页PPT已生成并完成结构/字体/可编辑文本/单页和最终渲染版面检查。工程、原权重、requirements及PPT齐备，正式ZIP独立解压后身份检查和16项CPU回归通过。状态 **READY_FOR_USER_SUBMISSION**，不等于官方认证。旧DRAFT_NO_PPT不再是当前交付。
- UNKNOWN：平台是否允许赛后替换模型、精确分数与平台提交编号截图对应。用户确认约70分属于V3；精确70.1021为仓库记录，不冒充本轮平台独立核验。

工程证据见 `handoff/REVIEWV3_20260926_复核工程交付.md`；正式材料身份与PPT收口见 `handoff/REVIEWV3_20260926_正式材料收口.md`。旧V3/V4t1包最终哈希与审查前一致。本次补PPT不改变任何已验数值文件、不新增GPU调用。未改依赖、未训练/重导出、未上传、未git add/commit。

---

> Owner：GPT-6 Astra；最终决策权：用户。最新审查：2026-09-18，SCORE-01报告REQUEST_CHANGES，维持不出新包；一次有限收口授权见D34/current_task §0H。DELIV-02限定ACCEPTED与已披露迁移FAIL不变，A2仍为已出分工程基线。
> 本文件有三个时点：本节是当前审查；其后9/15自审和9/16原始对账报告均为**有明确边界的历史快照**。旧快照的“待裁决/尚无审查/不得开发”不得覆盖本节和current_task的最新范围。

## 当前结论

DECISION：用户已明确停止未来解耦风险、测速件和图片同主体、Architect决定技术细节、先统一现行六文档。正式生效内容唯一维护在 decisions D17–D20，不重写旧决定来伪造“从未解耦”。

FACT：本轮统一六文档现行摘要；完整60项对账及证据原样保留在下方。未修改源码或历史档案。当前代码与未来约束仍有差距，不能称项目已完成合规整改。

当前结论（2026-09-18）：**ACCEPTED：DELIV-02本地交付及评分反馈里程碑，含用户路线A的披露例外。** 不等于旧矩阵全PASS或官方合规认证；当前身份与用户反馈唯一记录progress §22。以下旧REQUEST_CHANGES是历史时点，不应继续阻断已完成的交付。

## SCORE-01审查与最终交接（2026-09-18，当前）

**判决：REQUEST_CHANGES（测量与结论）；接受保留A2、不出新包的处置。** 不要求为勘误再单独递交报告审批，按D34/§0H一次修正实际入口、有限复测、通过才交包。本轮只读报告、profile/bench/v2源码、原始bench JSON、gate与ledger；独立从20次原始数组重算两窗口中位数均值、核验bench/v2 SHA。未GPU、未重跑9项或74项CPU套件，不把自报测试当独立复跑。

| ID | 已核对事实及判断 | 处置 |
|---|---|---|
| SC-R01 | **FACT：score01_bench.py实际调用infer(..., timing={})；v2只有timing is None才关闭六处计时同步。** 当前bench SHA `e557e66e10c4fef6bc5b6d0ab5326dda1fee7803414669993b747411ca667679`与ledger的cand_p2/base_p3/cand_p3一致，因此至少第二窗口确定没有测试该主要优化分支。v2 SHA `cbbbac92cae3a73a0c7cc40313f065079cd35c5be0fa4a07573d19d8a1c0d0ca`与报告一致；第一窗口另一源码哈希，不以当前源推定其全部细节。 | 修正bench实际入口并做参数spy测试，再测同一候选；不能由当前负结果宣布优化路线已证伪。 |
| SC-R02 | FACT：profile阶段和与wall差约1ms；但阶段内部含Python编排、张量构造、finite检查/隐式同步、算子启动及等待。空队列连续200次同步均值8.52µs不等于有未完成GPU工作的同步成本；计数7包括外部收尾，6×8.52≈51而非7次。 | 撤回“所有可去除开销≤0.2%/其余99.8%是计算”的上界。68–71%仅是所测encode+decode阶段墙钟占比，支持优先关注，不证明唯一杠杆。 |
| SC-R03 | FACT：重算两窗口基线453.605/526.903667ms、候选518.209167/531.803167ms，改善−14.2424%/−0.9299%，均不足10%；第二窗口p90_ok实际true而非报告所称两轮false。第一窗口两进程相隔8分20秒，第二窗口温度也不同；基线+16.16%是观察，不能据此证明热/功耗是全部原因或第二窗口无混杂。 | 保留原数值和门槛FAIL；限定结论为本次条件未证明收益，不声称真实效果恰0或未来不可能达10%。新复测按预登记邻近ABBA/BAAB和漂移有效性判据。 |
| SC-R04 | FACT：所读7份bench JSON均frozen_flags=false、deterministic_algorithms=false、cudnn_allow_tf32=true；bench的--frozen默认关闭，注释称“贴近平台”。这不是A2生产冻结模式，官方实际配置UNKNOWN。 | 不将该结果与旧冻结读数混为同口径，也不由此认定平台运行方式；后续对照显式统一冻结条件，隔离公开入口另验。 |
| SC-R05 | FACT：score01_profile.py的3次timed调用均在infer完成后才reserve；其CPU替身段因Generator设备错误失败，不能把未完成的CPU开销测量视为证实上界。原9次GPU调用有记录，未发现据此超总额度，但先扣后跑不成立。 | 修正真实profile入口并CPU证明预算不足0调用；旧FAIL保留，不为补上界再专门跑长诊断。 |
| SC-R06 | FACT：差异表引用D:/models/final_compliant/model_dir/runner.py，只能证明该runner无wavelet/未见分块，不能证明历史100张M24提交图也按此链路生成；历史离线infer2另有wavelet记录，来源等价性曾为UNKNOWN。 | 表述限定为旧runner，图像生成配方继续按原manifest缺失边界记录；不得把A2与M24分差归因于单个后处理或后验选择。 |

**不关闭的事实边界**：SCORE-01的负结果不使A2失效；A2实际官方反馈仍见progress §22。新候选GPU保输出/4K/100图尚未验证，不能称新版本可提交。官方质量/时延分项、当前排名/截止状态、混合精度/历史训练最终资格仍UNKNOWN，未访问平台。

**Architect自评与战略裁决**：此前工作使同主体JIT、来源、可复现出图和提交闭环具备工程依据，也修复了真实实现缺陷；但未提高已记录比赛成绩。前期过细切分预算/任务、跨实现与跨精度标准层次不够清晰，造成过多往返；不能把全部投入包装成必需或成绩进步。当前应完成一次可信、有限的已有候选测试，而不是在测量问题未处理前再开VAE精度/训练支线。新的精度路线不是永远否决，但仅512降精度与4K保留旧路线不符合D17/D18，且跨精度质量损失不能用原纯性能“保像素”判据替代。

**关于分数**：现有证据不能给出可靠的新分数区间或上限；追回并超过历史M24只能是目标，不是预测。阶段加速比也不能直接换算成总分。未获得新的同主体官方读数之前，不承诺45/50或80+，不复活旧解耦公式。

## DELIV-02最终交付与首个官方反馈（2026-09-18，上一阶段）

本轮为Architect定向审查：读取最终报告、合并readings、phase_c、supervisor、generation_manifest、C2/ZIP/隔离记录、实际交付runner/配方及历史分数来源；独立计算当前已改名ZIP的完整SHA256和四份C输出NPY哈希，均与记录一致。未GPU、未复跑CPU全套、未重新计算所有数组差异或复核100张JPEG；数值结论明确来源于已落盘读数。未修改源码/模型/历史档案。

### 成绩对照与能得出的结论

当前分数见progress §22（FACT：用户转述官方反馈；本轮未访问平台）。下表旧分来自reference/新世界模型分数对应表.md §一/§一·G及外部score_ledger；它们是历史观测，不是同时同条件的对照试验。

| 参照 | 历史分数 | 本次相对变化 | 可比性限定 |
|---|---:|---:|---|
| M24 v3 | 43.6951 | −1.4806（−3.3885%） | 最接近的历史父系参考；旧包是权重字典、配置/采样/runner不同，不是同一已认证JIT版本 |
| M24 v2 | 43.6731 | −1.4586 | 文档记为v3复现前版本；两次相差0.0220，不构成稳定方差估计 |
| M24 v4 | 43.5844 | −1.3699 | 文档记为相同图片、冷启动加固runner；不把全部差异归为噪声 |
| TinyV7gan | 41.7346 | +0.4799 | 历史自身模型/图像记录；架构和速度/质量折中不同，不宣称重新完成其资格审核 |
| TinyV9b | 40.5757 | +1.6388 | 同上；旧“待评测”文字已被后面的出分记录取代 |
| M24图+V7gan/V4g测速 | 94.7249 / 105.6401 | 不列可用路线差额 | 不同主体；D17已否决，不能以高分证明路线可用 |

- **FACT**：本次不是历史总分新高，是首个经本轮可追溯交付流程完成并收到评分反馈的A2基线。约1.48分落差值得查，但不能将某历史“平台区间”直接当作统计噪声带。
- **UNKNOWN**：分差由实际512稳态时延、图像内容/精度/后验/seed/后处理差异、平台波动各贡献多少。只有总分不能反解这些变量；不能认定迁移max导致扣分，也不能认定质量没有退步。
- **FACT / 限定**：本地赛题文本明确测速输入输出512×512。现有隔离前向infer_s=1.1847是一次读数；历史M24约353–366ms的测试条件不同，不能直接宣称A2慢三倍。100图约79分钟、平均4K 44.60s是生产数据，不是官方计分时延。
- **Recommendation**：先做同条件512全链profile。现行runner即使timing=None仍调用多次阶段同步，存在可测试的非数值开销；收益尚未验证。4K encode+decode约占77.3%仅说明当前生产瓶颈，不能替代512测量。
- 历史Slow8同图更慢而分数下降、TinyV2g同图更快而分数上升，支持“时延值得测”；TinyV7gan相对V5d的历史改善也说明不能一概宣称质量轴无空间。多因素、不同时间的旧读数不证明官方公式或给出可承诺的涨分幅度。旧48.0603及更早机制/解耦成绩不作为当前策略标尺。

### 未关闭问题与非阻塞勘误

| ID | 结论与处置 |
|---|---|
| SCORE-01 | UNKNOWN：提交ID、官方分项/时延及实际执行路径未取得。用户确认总分足以记录已出分，不足以认证所有环境/审核要求；有附加页面信息再追加，不阻断本地规划。 |
| SCORE-02 | FACT：A2实际为FP32 VAE编码器/LightDecoder + FP16 DiT/DINO，不能简称后推“所有参数纯FP16”。官方样例注释与混合精度资格、历史v7 IQA训练资格仍为外部UNKNOWN，出分不自动关闭。 |
| SCORE-03 | FACT：两张4K迁移max保留FAIL_DISCLOSED。报告826/37,748,736与151/37,748,736按RGB通道元素统计却称像素；case1此前独立复算为391/12,582,912个任一通道超限像素。新勘误区分单位，旧档不改；稀疏分布不证明无害/算子根因。 |
| SCORE-04 | FACT：旧supervisor_success=false、child_exit_code=1、ledger_consistent=false等确实存在，不能只因“旧口径”就删去。合并readings、独立phase_c和C2/隔离记录为后续补充证据，不是旧attempt变PASS。下一任务须按现行schema汇总新证据并关联各attempt账本；不得继承一个含False的ALL作为已通过门禁。 |
| SCORE-05 | UNKNOWN：FP32 DINO非重复的具体算子因果、全部数值/环境交互；现有后端探针只覆盖已测条件。暂停该研究，不作为A2后续保输出优化的前置，也不宣称已排除所有导出语义问题。 |

**Architect结论**：交付里程碑关闭，完整合规认定未关闭。下一包优先“保住现有图像与主体→测准512→一个低风险提速候选”；通过才再出100图/ZIP，不为了获得另一个版本号提交无收益修改。训练/蒸馏或重选权重是后续独立决策，不由本次评分自动授权。任务范围/拟议预算唯一见current_task §0G。

## DELIV-02整包续跑/迁移失败审查（2026-09-18，历史时点）

审查报告、numeric_escalation、readings实际顺序/成对记录、参考路径、runner/core相关路径及ledger；CPU分块读取既有case1_full_B和PIPE01RB参考NPY复算全图。未GPU、未修改源代码或模型。

| ID | 核对事实与边界 |
|---|---|
| DEL-M01 | FACT：迁移max=0.3137254715、mean=0.0006250350，确认原判据失败。按任一通道统计，>0.05像素391（0.0031074%），>0.1为60，>0.2为5；峰值c/y/x=(1,1566,1128)，63个256网格有>0.05像素。分布稀疏是事实；“只一个接缝/只是无害舍入”为ASSUMPTION。 |
| DEL-M02 | FACT：readings.pairs仅case1_full，A/B max0.007843256、mean2.19294e-7、u8max1、974像素不同（7.74066e-5）通过；case7未执行完、C未通过，不能说全4K/全部保真度已达标。阶段原始文件主要是512，不能据此签收全4K量化前每tile。 |
| DEL-M03 | UNKNOWN：首次分歧来自精度、输入/噪声cast、tile融合、参考条件或后处理哪一项，尚无成对中间读数证明。代码可见export以FP32生成noise再cast、原基线按latent dtype生成；不据同seed或代码相似推断实际张量已核对。允许CPU来源审查和有限同输入层级探针。 |
| DEL-M04 | FACT：本包limits169/141/128，used19/3/93，加历史22/4/136=总已用41/7/229。D31原component本包应122而非128；未见实际used超过原122，但配置不符须勘误。新D32组件上限明确变为含历史352，不追认旧配置。 |

**DECISION：先A，不接受B改判方式。** same-recipe导出保真与对原FP32迁移是两项不同检查，前者通过不能替后者。选择有证据的局部修复，或在有限诊断后回退到同父资产全FP32，均属于预授权技术分支；根因解释可有UNKNOWN，完整验收与来源不可缺。保留迁移0.05/0.002及其它矩阵；最多两份新数值候选、固定样本，不为速度更快或视觉“似乎没问题”擅自接受失败。

Implementer在§0F完整分支内自主决定探针/修法/证据复用，已知迁移异常允许用于定位而不反复暂停；新严重数值/身份问题、保守回退仍失败或硬资源不足再升级。交付承诺仍仅为本地技术候选，官方环境/历史资格UNKNOWN单列。

## DELIV-02R续跑核对与调度调整（2026-09-18，前次）

本轮读取续跑报告、continuation_escalation、readings/readings_resume、_d02_ctl、实际总账及续跑脚本；未GPU或重跑全CPU套件。现行总账本包9/1/58，加历史22/4/136=top31/4K5/component194，与申报一致。

- FACT：有A四次及B_512_1的局部记录，仍未完成4K/C/P2/P3；旧readings总体FAIL须保留，局部判据复算/脚本错误不能直接变为整个实验PASS。
- FACT：`_d02_ctl.json`10个passed=true中，有两项明确写“无历史数组可查（NOT_RUN）”；不能称10项真实执行通过。应局部修正路径与结果标记，不另开审批轮。
- FACT：当前`d02_resume_worker.py`仍将patch返回list赋给merged，并重复merge，与“已修”自报不一致；且量化前检查if文件存在才执行，缺文件可静默略过。续跑路径应保留4K迁移与C前置，锁/终态协调须由实际入口测试覆盖。全部列入Implementer本包自主收口，不由Architect代写代码。
- FACT：所谓剩余11次= A两次4K + B五次 + C四次，而非5+4=11；续跑新B实例不得冒充旧实例重复。更大的整包余量避免再为这些必需补测请求1次许可。
- DECISION：接受用户调度要求，批准D31，不跳过验证，不再逐次补额度。正确模型SHA仍为任务单完整98fe49b99e92…串，escalation缩写/笔误不得作为身份依据。失败完整保留，范围内已知脚本错误修复后CPU验证并自主续跑；实质性风险和硬上限仍停。

## DELIV-02阶段审查（2026-09-18，前次）

只读检查报告、_d02_cpu.json、实际ledger及8个attempt、component_gate、phase_c、worker/ledger/verify源码；未GPU/全套CPU复跑、未修改业务代码。模型身份为交付证据记录，未在本轮重新读取1.5GB计算SHA；下次加载仍必须强制校验。

| ID | 核对结果 | 处置 |
|---|---|---|
| DEL02-01 | FACT：当前ledger本包used=top3/component29/4K0，含历史25/165/4；A2组件记录enc512/1024、DINO为0，dec64 max4.0382e-6、dec128 max4.6492e-6，均达原阈值。中断记录为额度/TypeError/重复ID，不是新数值失败。 | 支持继续固定候选，不重导出；阶段未验收。 |
| DEL02-02 | FACT：CPU JSON验证像素转换，但真实数组比较是A1↔A2；后处理是随机64输入调用同一WaveletFix自比，并非真实decode_raw独立重放；未见其证明任务要求的真实控制子进程用例。 | 补真实入口正负例及重放，报告不能把像素PASS扩成全控制PASS。 |
| DEL02-03 | FACT：C在加载runner之前没有A/B成功及B文件完整哈希检查；attempt_c3实际在ab3失败后启动并因缺B_512_r1文件失败。ledger缺文件/字段仍有默认值、只比对top/component、deadline按attempt重新计算。DELIV02 worker仍有缺模型自动build和配方默认FP16路径，实际reserve多处仍固定ID。 | 一次补齐已要求的fail-closed与累计控制，CPU真实入口证明；不允许只是改ID继续。 |
| DEL02-04 | FACT：当前worker没有参考复用分支，仍扣加载1+参考9+检查18+验证加载1≈29组件；报告18建立在尚未实现的复用假设。r_verify量化前阶段主要记录hash/finite，未见全链对应严格数值断言。 | top+3之外主动补组件12避免下一轮再次额度不足；补全链阶段断言，不放宽数值矩阵。 |

**DECISION：选择A，不选择B跳过C。** 本次批准的是预算变更与有前置条件的执行，不是ACCEPTED交付。精确top19/component60/4K6（本包）及新增20分钟监督、原P2/P3连续交付授权唯一维护current_task §0F补充。允许CPU局部收口自主迭代，完成后一次GPU验证到ZIP；真实数值/控制失败或触顶停止。用户“争取下一轮出图”是目标，不写成结果保证。

## DELIV-01R审查（2026-09-17，上一阶段）

本轮读取报告、A3/A2/checker/ledger/runner源码与落盘证据；用CPU独立复算现有NPY（最终A/B文件SHA与readings一致），无GPU、无业务代码修改。

| ID | FACT / 判断 | 后续 |
|---|---|---|
| DEL-R01 | A3只在同一进程新trace六个片段，未加载失败的.pt，未显式使用optimized_execution(False)，未包含整个posterior mode/scale与原解码scale包装，也未测两个大尺寸。因此exact_equal只覆盖所测新片段，不排除保存/加载或整模块差异。 | 不作为最终图验收；补真实保存件对照。 |
| DEL-R02 | A2组件recipe_dtypes为空，checker默认FP16，结果FAIL/无效；另起a2_verify_worker没有组件成功前置仍进行了完整infer。supervisor的model_hash_expected=null却model_binding_ok=true。 | 这是门禁失效，不是单纯少一组数字；加载前强制配方/身份/阶段证据。 |
| DEL-R03 | 多份ledger各自used起算；r_ledger.reserve使用构造对象self.limits，不以磁盘limits为准，prior仅记录未加入检查。ledger_verify2磁盘component=0但默认对象仍可允48；component累计98/96未被阻断。12top/4K4有预留非执行，不能称真实4K已跑。 | 不接受“补2次额度即可”；唯一预算与跨worker负例先修，旧消耗保守留账。 |
| DEL-R04 | 独立CPU复算512：encode/features/noise/step完全一致；decode_raw max=3.0994415e-6、mean=1.0276191e-7，均在原浮点阈值内；post max=0.007843255997、mean=4.1086628e-7。按最近色阶重建38像素差1级；**按实际to_u8再次截断则101279/262144像素不同（38.6349%），131669通道不同、max2级**。 | 报告“最终仅少数像素1级”不成立。波前处理输出已量化为色阶，末端截断引入二次映射差异；修正并检验网格恢复，不只放宽mean。 |
| DEL-R05 | 实际VAEEncode在模块出口转FP16，再由runner升FP32融合；不同于§0E“编码输出保留FP32至融合”的文字。A2相对旧fp16的6–10倍改善也未提供同条件成对归因。4K、迁移和跨加载还未通过。 | 明确登记实际cast，本次限定允许验证该现成配方，不回写历史合规；不从512外推4K或断言精度提升根因。 |

**DECISION / 自我纠偏：** 先前把跨实现的量化后输出也要求max1e-5与PNG完全相等，混淆了连续数值保真和离散边界翻转；这不等于证明候选无错，但不应为抹平少量色阶边界差异持续重建模型。PyTorch不保证数学等价路径逐位相同（[2.7数值精度说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)）；具体新容差是本项目工程决定，不是官方标准或该文档提供的合规认证。

批准D29/§0F：保持量化前严格阈值和同产物跨加载像素一致；跨实现的最终量化结果采用max1色阶、不同像素≤0.1%、mean≤1e-6等合取，新GPU前冻结。CPU先核验已量化浮点结果的网格恢复（旧A/B网格残差max约1.8954e-5/7.5698e-6色阶），不改wavelet内部floor、模型或旧产物。新协议不能倒推旧FAIL为PASS。

未关闭：DEL-R01真实组件保真、DEL-R02/R03控制与来源、DEL-R04全尺寸后处理证据、DEL-R05实际配方4K/迁移；100图/ZIP及官方环境资格仍未完成。此次不再发独立“几像素探针”任务，前置CPU修复和完整有限验证合并，成功直达P2/P3。

## DELIV-01 P1审查（2026-09-17，当前）

本轮只读审查报告、实际checker/worker/frozen_mode及component_gate/supervisor/预登记；未GPU复跑、未全套CPU复跑。认可正确停止，不将P0声明作为全包完成验收。

| ID | 判断与证据 | 处置 |
|---|---|---|
| DEL-P01 | FACT：记录的执行开关匹配，enc两尺寸max0.00390625、dec两尺寸max0.006591796875/0.0084228515625，均超阈值；step三组为0、wavelet非零但达标。根因UNKNOWN。 | 旧FAIL不变；分段定位/限定新候选后按原阈值验证。 |
| DEL-P02 | FACT：make_inputs为enc与dec各造独立随机输入，dec没有接收本次enc输出；“随后被LightDecoder放大”不是该实验结论。重复读数仅含step三组，不含enc/dec。 | 撤销传播因果表述；在新实验中分别重放相同中间输入，补enc/dec重复。1 ULP也不能从最大差普遍推出。 |
| DEL-P03 | FACT：cmp未保存检查侧原始数组，只留差异摘要；输入比对未显式校验完整集合/dtype/shape/t/r；flags记录未包括warn_only，JIT只记录检查侧上下文。现有失败可记录，但不足以独立复算/签收完整保真度。 | 合并补齐证据与负例，不单独开启文档返修审批轮；不得伪补旧NPY为原运行产物。 |
| DEL-P04 | FACT：supervisor.wall_s=59.224、worker.elapsed=51.059；component22/top0/4K0。剩余监督墙钟不是2648.9秒；实际ledger.disk_gib=8与§0D20GiB不一致（当前为更严格，并非已超授权）。 | 新记录绑定累计，续跑监督≤2640秒，组件≤74；修正预算配置但不重写旧账。 |

**DECISION：拒绝A1替换验收对象；批准A3+明确分支的修复/A2一次任务。** 同一个错误导出件也可能跨进程稳定，故稳定性不是eager→导出语义保真证明。浮点不同执行路径不保证逐位相同（[PyTorch数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)）；该一般事实既不能证明本项目是舍入误差，也不能证明1e-5结构性不可达。保留严格工程阈值，A2仅为需验证的限定配方试验。

仍未关闭：新配方/导出的组件及全链保真度、512/4K迁移与实际资源、100图来源和隔离ZIP；官方环境/历史资格UNKNOWN独立保留。精确授权只维护current_task §0E；通过即可直接出图打包，真实失败才再升级。

## EXPORT-01R3审查与调度纠偏（2026-09-17，当前判决）

**DECISION：REQUEST_CHANGES；批准A的协议修复，但不再只派一次组件补跑。** 用户明确指出反复裁决影响出图交付，Architect将允许范围内的局部错误处理下放Implementer，终点改为本地100图ZIP；不以提高效率为由降低数值/同主体要求。

本轮重点只读核对报告、r3_component_gate.py、component_gate.json与supervisor，未代修代码、未GPU或全套CPU复跑：

- FACT：检查子进程设置了JIT上下文，却未设置或记录完整的进程内确定性/cuDNN/TF32开关；另将不同STEP_PAIRS的结果与first比较并影响最终ok。不同输入差0.65/0.68不能证明重复性失败。
- FACT（落盘读数）：三个step各自对eager和DINO为0，wavelet max1.19209e-7；VAE/Decoder在本次不对称协议下超阈值。全新进程已执行原报错step路径，可认可**该测试范围内设备异常已消除**；不据step通过替VAE/Decoder或全链签收。
- ASSUMPTION：漏开关导致全部VAE/Decoder差异。**尚未完成修正协议对照，不能写“非导出语义错误”或“算法选择不同就是根因”。** 0.00390625=2^-8是数值事实，是否恰为各误差元素的1 ULP须按对应数值间距分析，单看最大差不能普遍推出。
- FACT：supervisor wall_s=**73.430秒**、exit=1；报告66.346秒不是完整监督口径。top0、4K0、component26为本批申报/记录；旧FAIL及扣款保留，不因新授权改写。

本次不再增设独立纯文档返修关：执行模式对齐、同输入重复、准确因果标签及旧时间勘误合并到DELIV-01；具体字段和代码修复由DeepSeek负责。旧关口没过仍不能直接出100图，但一旦按既定协议通过即可在同包继续生成/打包，不再等待Architect中间签字。

未关闭：匹配协议下VAE/Decoder及全链512/4K等价性；最终100图生成来源；隔离本地部署；官方目标环境/历史v7资格（外部UNKNOWN，不伪称PASS）。新预算与局部修复/升级边界唯一见current_task §0D。

## EXPORT-01R2正式审查（2026-09-17，上一交付判决）

**DECISION：REQUEST_CHANGES；选择A，不选B。** 接受本批如实FAIL和有范围的合并对照，不签收C1；按current_task §0C交DeepSeek一次完成局部重导出及验证，不逐函数报审。

本轮为Architect重点抽审，未重跑GPU/全套CPU、未代修实现：

- FACT：以zipfile只读实际 `C:/models/exports/EXPORT01R2_20260917_200903/model_dir/your_model.pt` 内代码（不unpickle/执行模型），TimestepEmbedder确有CPU arange→与CONSTANTS.c6/c7运算→硬编码cuda:0的路径，与readings异常栈吻合。设备错误路径已定位，常量在保存/重载/执行器中的具体传播机制仍UNKNOWN；切回优化器不是已证修复。
- FACT：CPU独立读取两份合并对照NPY，max=**1.862645149230957e-6**、mean=**3.336815552756889e-7**，双方finite，原判据通过。独立复算A_512_r1/r2/r3，两组重复max均0，shape=[1,3,512,512]、float32/finite/[-1,1]；另有A奇数尺寸输出。B首调用异常、4K=0，无完整成对/4K迁移通过证据。
- FACT：监督退出码1、未超时、ALL=False；top5、4K0、component2。**监督wall_s=109.669秒**，不是报告102.565秒；预算按监督口径。53项CPU/5场景为Implementer记录，本轮不冒称独立复跑。
- 勘误：7b815202…件来自EXPORT-01R；原EXPORT-01的1bdafc4c…是不同文件。“同一.pt曾在EXP-01内存态成功”缺同hash/同条件证据，不能作FACT；trace或CPU load成功不证明保存后CUDA首调用成功。

| ID | 当前判断 / 合并任务要求 |
|---|---|
| EXP-R13 | P1 / FACT：保存图时间嵌入设备路径失败。允许在export_baseline局部适配t/r嵌入、常量承载和必要导出边界，不改外置VOSR。保留原频率公式、fp32计算和MLP权重/CastIn语义；“数学形式相同”不等于数值已验证相同。 |
| EXP-R14 | P1 / FACT：r_verify.check_required_set对phaseA值直接执行`'call' in v`，真实记录含`released=True`，走到末端将TypeError；A循环未即时检查自身重复，migration仍在两张4K后统一检查且4K缺参考可跳过；失败readings仅异常、未保留此前结构化partial。须在本包CPU真实驱动路径收口，不只单测helper。 |
| EXP-R15 | 证据/资源：监督109.669秒、旧5次top/2次component扣款保留；新图新SHA/new bundle。旧A只能诊断，不能代替新版本成对证据。报告R10/R11编号混用按review原编号解释，不另建事项。 |

DECISION：保留optimized_execution(False)、原精度和阈值。先做保存后全新进程的嵌入/DiT组件首次与第二次调用验证，再进入完整12次协议。§0C允许一个有上限的组件级局部修复回合，非任意失败续跑。

交付距离：还欠①重载JIT在512/奇数/4K的成对、重复及迁移验证，②最终bundle的100图/manifest/隔离ZIP，③目标环境、历史v7训练资格等外部UNKNOWN及用户提交授权。不以本地技术包代替官方确认，不承诺评分耗时或百分比进度。

## EXPORT-01R正式审查（2026-09-17，上一交付判决）

**DECISION：REQUEST_CHANGES。认可原attempt_01以FAIL停止，不能签收“R0全部收口”或直接按现有READY启动。批准一份合并收口+条件复验任务EXPORT-01R2，唯一授权见current_task §0B；不要求CPU收口后再单独交一轮报告。** 用户已澄清此前“有正式训练请求”为误判，本次报告也没有训练请求，不据此改变任务路线。

本轮采用Architect重点抽审：未代修源码，未重跑GPU或全套CPU。读取账本、监督记录、gate、bundle和真实实现；gate列出的15份源码现行哈希全部一致，但**其中不含r_merge_probe.py**。独立读取实际模型SHA-256：`7b8152029e3d96cf90e0787b42949e680cdb3c1bdfce6f6be99452444283ccc6`，与bundle吻合，大小1,343,344,555字节。bundle_manifest在仓库evidence而不在旧model_dir，不把路径差异视作缺失模型。

FACT：旧账仅一个FAIL attempt，open_attempt=null；top=0、4K=0、component=10（含资产加载及未完成探针的保守预留，**不是10次成功组件前向**）。失败栈确认缺models导入，当前探针已补路径；没有成对/重复结果，不能推断数值通过或失败。worker66.78秒，**supervisor71.083秒**，预算扣后者。旧失败数不退款、不重写。25/45项等历史测试结果按各次报告区分，本次45项为Implementer报告，Reviewer未独立复跑。

| ID | 判断与处置（本包一次处理） |
|---|---|
| EXP-R08 | **P1 / FACT，解码语义回归。** 原`tiled_vae.py:86–90,175–181`逐块decode先clamp再融合；现`delivery_runner.py:193–201`在finite检查后融合未clamp的pix，最后才clamp。原始输出检查应保留，但之后须**逐块clamp再融合**；clamp与加权平均不交换。添加超过[-1,1]的重叠块反例，不能只有范围内输入。可在runner修复，原无clamp的JIT decode无需因此重导出。 |
| EXP-R09 | **P1 / FACT，冻结≠强制验证。** gate未列刚修的r_merge_probe.py；make_r_prereg没有完整源码/资产/bundle绑定，worker/supervisor仅读文件/记哈希，未在加载前强制核对。补完整可执行依赖清单、冻结快照和启动校验；篡改探针、runner、模型、输入、判据或预算应加载前失败。新gate和证据用新目录，不覆盖旧gate。 |
| EXP-R10 | **P1 / FACT，判据收口不足。** r_verify的A三次未即时检查自身重复；merge对照只看error，不检查finite/差值；migration只追加、不按阈值停；supervisor对pairs/B512用all，记录数量却不要求数量和身份精确，4K双方NPY未加入核验。要求原12次完整协议的唯一调用集合、A/B重复、奇数尺寸、两对4K及双方NPY、三项fp32诊断，缺项/重复/空列表即FAIL；阈值失败当场落盘并停。异常时不得丢掉此前部分读数。 |
| EXP-R11 | **P1 / FACT，进程/实例控制缺口。** r_merge_probe独立进程不继承父进程内torch确定性开关；supervisor终止仅针对worker，实际worker→drive→role有孙进程，CREATE_NEW_PROCESS_GROUP并不兑现整棵树退出。统一各模型子进程的执行模式并记录；监督截止时间传递，超时终止本批整个进程树并确认，不杀无关进程。另r_verify在B512之后仍持有b_runner/rec_b，再加载4K A；需先释放并以弱引用证明，不以empty_cache冒充。CPU真实父子孙超时与释放负例验证。 |
| EXP-R12 | **P2 / FACT，旧账用词及复用边界。** 原申报22不能在缺完整调用记录时称数学上已证的“保守上界”；保留UNKNOWN及已知事件。复用仅表示无需再构建相同.pt，不表示bundle已通过C1。修正runner后需新files/runner哈希和bundle_id，保留旧转换来源及旧bundle；不得原地改写历史。 |

此前EXP-R01重复更新的主要修复和EXP-R03原始finite检查方向认可；不因此关闭以上回归/执行控制问题，也不签收EXP-R02/R04/R07已全部解决。复验仍维持固定精度、原输入、原阈值，无训练/量化/候选扫描。C2/C3继续阻断。

## EXPORT-01正式审查（2026-09-17，上一交付判决）

### 独立核验与边界

- FACT：隔离副本运行 `D:/Python/python.exe -B -m unittest discover -s tests -v`，**25 tests / 0 failures / 0 errors，exit=0**，测试耗时2.869秒。但现有用例未覆盖下述分块更新与量化前非有限值问题，不能据测试数签收C0完整放行门槛。
- FACT：重新流式核对实际JIT文件SHA-256为 `1bdafc4cc7e69911323e8f7235015965832e848c58b8c7fe426ac6974c6743a9`；prereg为 `c7c3ae9e871d52116f109be0c34f8715c00dc3dafffef90a9a16cddf6cc82564`。chain / delivery_runner / build_export / export_config四份现行源码与source_hashes记录相同。只证明本次读取一致，不凭mtime证明过去未改。
- FACT：独立CPU反例执行现有 `DitStep` 和 `Runner._one_step`：零速度场应保留固定噪声；latent 64×64最大误差0，64×65最大误差**4.220885276794434**，后者输出绝对最大仅4.76837158203125e-7。无需GPU即可证明EXP-R01。
- FACT：独立CPU读取源参考NPY与已验收fp32基线：case1/7的max均2，mean分别 **0.2497825325 / 0.4002472162**。报告0.2497780919 / 0.4002537429实际取自重载侧比较记录，不是它所标注的源参考侧。
- FACT：C1记录还包含511×513失败（max0.0235294104 / mean0.0005536967）；512首次成对0，但同一重载实例重复max0.0470588207。正式目录没有保存重载侧和重复侧原始NPY，因此本轮不能独立重算这些成对数字，只能核对记录与实现；不重跑GPU补证。
- FACT：本轮只读检查代码/日志/资产，在临时隔离目录做CPU测试与反例；未修改实现、原证据或历史handoff，未GPU、训练、导出、打包或Git操作。诊断脚本位于 `C:/Users/Yuki Nagato/AppData/Local/Temp/export01-review-dbfd102322fd4e0b9cf1f80115f945be/audit_cpu.py`。

### 未关闭问题（按阻塞程度）

| ID | 等级 / 判断 | 证据及必须处理的内容 |
|---|---|---|
| EXP-R01 | P1 / FACT，阻塞C1 | `chain.py:202` 的step已返回z−dt·u；`delivery_runner.py:152–158`把该结果作为u再执行z−dt·blend，dt=1时近似变成blend(u)，不再是基线z−blend(u)。单块直接返回step，故512通过不能排除4K共同错误；eager源参考也共用错误runner。修正契约并以独立算式验证零/常数/非均匀速度场、单块/重叠块/奇数尺寸。**不据此声称解释了全部C1差异。** |
| EXP-R02 | P1 / FACT + UNKNOWN，阻塞C1 | `verify_c1.py:234–247`最终合取未纳入repeat结果、完整必需样本集合、范围及全部执行控制。发现重复失败仍继续4K；512重复3.293s→31.88s须区分冷/热执行、JIT优化、输入/噪声/状态/中间结果。`determinism_probe.json`的7个fp16组件同输入重复均为0；这不能证明全链稳定，也不能支持“fp16结构性不可达/非导出缺陷”。原因UNKNOWN，不能事后放宽阈值。 |
| EXP-R03 | P1 / FACT，阻塞C1 | 只在runner末端检查finite；`Decode.forward`先clamp，wavelet先uint8化，可把Inf/NaN变成有限像素。CPU独立注入NaN、Inf到wavelet均得到全有限输出。恢复基线量化前拦截原则，并明确覆盖**原始解码输出clamp前、量化前**及关键中间值；不声称旧基线已经检查所有clamp前位置。实际异常必须抛出且不得生成成功记录。仅检查最终out不充分。 |
| EXP-R04 | P1 / FACT + UNKNOWN，阻塞新增GPU | verify没有先扣后跑的共享总账/硬墙钟监督，容错循环失败仍继续；tol取现行config而非强制与prereg绑定；`analyze_c1_failure.py:39–40`把22/8写为常量。目录有6份验证日志（含设备和JSON异常）及FAIL后组件探针，故“按预案首次失败即停”不成立。**22/24、8/8仅为Implementer申报，不是本轮独立确认的完整预算**；不能因此断言一定超额。补事件级跨目录旧账重建，无法确定项保守注明，不删除或修改旧证据；新批强制监督、冻结检查、即停与不可退款记账。 |
| EXP-R05 | P2 / FACT，阻塞精度因果结论 | `probe_determinism.py:56–65`先在共享assets上half，再float，不是重新从原fp32父权重建立对照；时间嵌入包装也残留。`probe_trace_gpu.py:74–90`在merge后才算“未合并”侧，且两侧输入分别为[first4,first4]与原8通道；0.539的所谓merge误差不能解释为真实合并差异。probe2的warn_only=True不是严格确定性保证，且DiT调用在autocast上下文外。新对照必须独立加载、相同输入、只改变预登记变量。 |
| EXP-R06 | P2 / FACT，证据修复 | 保存双方与重复原始NPY/编码像素，逐项绑定prereg/源码/模型/recipe/输入/噪声/执行模式；修正源/重载读数标签。source_hashes中VAE目录只有27个文件计数，无实际加载权重/config内容哈希；conversion仍写CastOut，而当前实现走CastIn。报告勘误写新交付，原档案保持只读。 |
| EXP-R07 | P1（放行C2/C3前）/ FACT | 尚无满足原契约的bundle_id及完整bundle_manifest；generation模板缺run/attempt/call、逐图bundle/seed/配置/NPY文件或明确NOT_PRODUCED。`collect_evidence.py:65`将run内ZIP再加一次，且只枚举本run和临时解包目录，未覆盖其他探针目录；本轮不是“完整峰值资源核算已通过”。make_zip水位6GiB与任务8GiB不符；gen_images只在循环边界计时，加载未计时，没有监督/先扣后跑与C1文件哈希门禁。允许本包合并CPU修复，不因此授权100图/打包。 |

### 三项裁决

1. **DECISION：暂不换精度配方。** 保持fp16学习模块权重/激活、已声明的fp32融合累加/wavelet及原输入噪声协议，准确称为该固定混合数值路径，不称“所有运算纯fp16”。先修EXP-R01/R03；全局autocast、VAE/注意力整体fp32新路线本包不批准。
2. **DECISION：同精度max≤1e-5、mean≤1e-6及PNG解码像素一致不变，重复项也是硬门槛。** fp16↔旧fp32的0.05/0.002仍是内部迁移诊断阈值，非官方质量标准，超限阻断后续交付。不能用本次观测到的重复误差反向定义可接受误差。
3. **DECISION：补发一次有条件的小批复验，而非开放式4K定位。** CPU修复、真实控制流测试、冻结与旧账勘误通过后直接进入EXPORT-01R attempt_01，额度/顺序只见current_task §0A；无需每个局部修复重新往返审查。C2/C3仍不放行。

ASSUMPTION：JIT冷/热执行或图优化可能解释部分重复差异，尚未定位。PyTorch的[可复现性说明（2.7）](https://docs.pytorch.org/docs/2.7/notes/randomness.html)区分随机状态、算法选择和确定性设置，并不支持“fp16天然不可能达到该阈值”；该资料是诊断依据，不是本项目根因证明。

UNKNOWN：目标环境和历史v7训练资格沿用EXP-04/05；旧final_compliant等价性非本包前置。不批准新训练。原EXPORT整包用户批准在Implementer记录中有声明，但本审查可见对话无对应原话，本轮不追认其历史时间与范围；新授权仅按D24和§0A。后续若恢复C2/C3，需给出原批准出处或用户新批准。

## PIPE-01RB正式验收（2026-09-17，已验收研究基线）

**DECISION：ACCEPTED，scope=固定候选、seed42/fp32、原tile/wavelet、三个固定输入的同主体eager闭环。** CPU收口和attempt_01有效结果已足以支撑下一阶段；不把工具的通用鲁棒性或官方合规一起签收。历史REQUEST_CHANGES按各自版本保留，不再作为当前任务状态。

### 独立验收证据

1. FACT：隔离运行机器执行器，**86 tests / failures=0 / errors=0 / skipped=0，exit=0，158.619秒**。副本 `C:/Users/Yuki Nagato/AppData/Local/Temp/pipe01rb-review-20d9334a633449a5a93478b9289e67e9/pipeline_baseline/`。包含真实父子进程CPU集成测试；未GPU复跑，未调用会重写交付证据的make_cpu_gate。
2. FACT：现行7份文件、frozen_sources_RB、cpu_gate及GPU预登记中的实现完整SHA-256一致；旧9份证据与首版公开前缀一致。仓库内run/账本/日志副本与D盘对应原件一致。预登记完整哈希 `795d24efbaa83ecac658af7d423cf707b8bc2c3ecd4564607438f86a2d8b8d09` 与账本/manifest/监督记录一致。gate记录07:24:02、预登记07:28:11、监督开始07:28:21；支持先登记，不证明历史不可篡改。
3. FACT：不依赖交付判定函数，用NumPy/PIL独立读取8份NPY（allow_pickle=False）及PNG：全部float32/finite/[-1,1]，shape为[1,3,512,512]或[1,3,3072,4096]；8份tensor内容/NPY文件/PNG哈希均可复算。三样本A/B **max=0、mean=0、PNG RGB完全同一**，A/B各自重复512亦max=mean=0。三份输入文件及tensor内容哈希吻合，裁切框[1792,1280,2304,1792]。
4. FACT：A/B身份剔除entry后完整相等，身份内实现与资产记录绑定预登记，DINO安全本地加载字段一致。8条成功call与8条reservation逐条ID/顺序/占用序号对应，唯一attempt终态OK、open_attempt=null、锁已释放。未重新加载大模型重算有效参数指纹，不声称本轮独立重做过GPU身份生成。
5. FACT：批次全部实际文件 **680335202字节（648.8182MiB）**，小于4GiB。账本680325938字节少9264字节（末端文件未全纳入）。worker账本313.4736秒、被监督子进程345.3758秒；监督开始至最终合并判决的记录时差约**354秒**。报告“345.4秒含全部监督收尾”不准确，但各口径都远低于1800秒，不改变本批预算通过结论。
6. FACT：改动前快照并非“内容级复原不可得”。上轮Reviewer隔离副本 `C:/Users/Yuki Nagato/AppData/Local/Temp/pipe01ra2-review-4857a726f0c0469793a18da1c9882bae/pipeline_baseline/` 仍存在，7份逐一与R-A2完整哈希匹配。本轮只核验未复制；后续须注明“审查后从既存隔离副本恢复”，不得倒填为改动前已归档。

独立只读核验脚本在本轮临时副本的父目录`audit_outputs.py`，输出AUDIT_OK；无模型导入/执行。全部原交付与历史资料保持不动。

### 关闭与结转

PIPE-RB01本批关闭：生产监督入口真实GPU运行成功。PIPE-RB02本批扣款与终态闭合通过；PIPE-RB03本批身份/协议/产物独立复核通过。四段判决作为工程结构接受，不代替实际记录审查。剩余问题不要求重跑本批，合并入下一包：

| ID | 范围与处理 |
|---|---|
| EXP-01 / P2 | 统一A/B manifest字段；完整墙钟/磁盘包括监督收尾与目录外日志，最终预算实际重算。下一轮CPU修复并重新冻结，旧证据只读 |
| EXP-02 / P2 | 恢复已找到的R-A2快照并注明恢复时间；cpu_gate.open_issues=[]不是流程全覆盖证明。报告“启动失败/写盘失败已测”等概括需按实际测试限定，扩展长任务前补相关负例 |
| EXP-03 / P1（导出门槛） | 当前FP32/eager不是拟提交形态。本地赛题文本明确torch.jit，官方runner.py:46–47明确所有模型用fp16的公平比较要求；需独立精度迁移/导出/重载验证 |
| EXP-04 / UNKNOWN（提交门槛） | v7历史CLIPIQA训练资格/官方接受边界仍未闭合；技术包成功不能签成官方可接受。不删披露、不换名遮掩，本轮不因此擅自新训 |
| EXP-05 / UNKNOWN（部署门槛） | 完整数值链ScriptModule、忠实wavelet张量化和目标解释器/依赖兼容未实测。Py<3.10语法检查不是目标环境运行验证，缺环境不写PASS |

### 报告请求的裁决

- **没有正式训练申请**：§6/§7请求导出、100张推理出图和打包预检，且把训练列入“不做”。出图是推理；无训练方案/数据/损失/预算可供批准，也无新训练授权。
- **拒绝L1 eager权重包作为最终交付捷径**，它与本地赛题原文及D18.4不符。TorchScript不是可选加固项；不能只导DiT或换微模型测速。fp16也不能被“出图不计时”绕过。
- **不原样接受C1预算/判据**：6次512前向不能验证其承诺的两张4K；0.5B DiT的fp32参数约2.0GB（503901760×4字节），尚未含其他模块，新增1GiB不是完整包的有效预算。精度迁移与同精度导出等价性须分开，转换后源码/文件哈希不可能要求逐字等同旧实现。
- **不接受100+2失败补偿**：首次真实失败停，不自动切配方/重试凑100张。150分钟可作资源上限，不以本次两张样本外推保证工期。
- **不直接运行旧共享目录预检**：harness_sim_zip会删除重建`D:/models/_harness_zip_sim`且执行真实前向，不是纯CPU操作。后续用本任务独立目录的审过副本，不自动接管RCON门禁/写upload。
- 推荐一次批准EXPORT-01整体，内部导出/精度与重载、100图、打包预检顺序通过后直接推进。**本轮用户请求验收，报告中的更大GPU预算和打包提案不等于用户授权；任务单保持DRAFT，需用户一次确认。**

本轮Architect仅更新现行六文档的验收、状态及整包草案；未修业务源码/依赖，未改旧档案/运行数据/Git索引，未GPU、训练、导出、打包、上传。

## PIPE-01R2 / R-A2正式审查（2026-09-17，前一轮判决）

**DECISION：REQUEST_CHANGES。** 不否定已经完成的修复，也不把80项通过升级为默认入口可运行。用户要求减少往返，本轮不再把每个局部修复拆成一次审查；通过下述客观CPU门槛后，Implementer可在同一工作包内进入唯一一次R-B验证，不必再次请示。不能自改当前交付的审查判决为ACCEPTED；新交付最终统一验收。

### 独立证据

- FACT：隔离副本运行`D:/Python/python.exe -B evidence/PIPE01R2_20260917/run_tests_machine.py tests`，**80 tests / failures=0 / errors=0 / skipped=0，exit=0，25.749秒**。副本：`C:/Users/Yuki Nagato/AppData/Local/Temp/pipe01ra2-review-4857a726f0c0469793a18da1c9882bae/pipeline_baseline/`；禁用CUDA可见设备及pycache，无GPU前向。NaN转uint8的测试警告为预期负例，xFormers缺失警告未构成失败。
- FACT：本报告7份交付完整SHA-256均匹配；source_snapshot_preRA2的7份与上一轮R-A完整哈希匹配（测试快照使用扁平名`tests__test_pipe01_cpu.py`）；旧9份PIPE-01证据与首版公布的16位哈希前缀匹配。不能把前缀核对说成首版提供过完整哈希。报告§0“上一轮7份未覆盖”准确含义是旧版本已由快照保留，工作副本当然已变更。
- FACT：RA01测试执行了真实Assets.__init__主体，仅替换重依赖；dino_info及module_files缺导入处理已改。RA02断言失败后只两次调用、B身份不等时B零前向、首个超阈值后B仅一次。RA03释放顺序有事件断言、最终A弱引用失效；弱引用断言发生在整次测试结束，而非B构造瞬间，不过源码确已丢弃A引用再构造B，原强引用缺陷已修。
- FACT：RA06两入口现在都把CPU张量送入共享core，搬运位于共同计时区间，首尾有同步调用；事件测试不能单独证明GPU同步行为。RA07八次成功调用的内容哈希和NPY哈希可独立复算，前向失败缺产物测试成立。原RA01/RA02/RA03/RA06/RA07的指定CPU修正认可。
- FACT：RA04的O_EXCL、PID三态、正常累计磁盘与run_supervised真实启动/终止CPU子进程均有测试；RA05原图/裁切/环境/总账绑定确有校验。但端到端DriverHarness均使用`--no-supervise`；两个监督测试只执行`print(1)`或`sleep`，**未覆盖默认CLI→监督者→真实worker控制流→总账终态**。因此RA04未关闭，RA05尚有fail-closed完整性缺口。
- FACT：Reviewer用隔离副本及临时合成预登记运行默认监督入口，**exit=1、WORKER_FAILED、forwards_reserved=0**，worker日志为“拒绝复用非空run目录…['worker_log.txt']”。没有加载模型/读真实图。诊断脚本位于上述临时副本的父目录`audit_default_entry.py`，落盘fixture为`C:/Users/Yuki Nagato/AppData/Local/Temp/pipe01ra2-supervisor-audit-677do4ku/`。这是确定可复现的集成阻塞，不是对GPU效果的推测。

### 收敛后的未关闭问题（不继续拆成多轮零碎任务）

代码位置针对本轮pair_verify.py哈希`b6d5dafb397c8b3e64caf146811d40371649276154d598250a856541c9c329cc`。

| ID / 级别 | 已确认问题 | 同一工作包内必须达到的结果 |
|---|---|---|
| PIPE-RB01 / P1（承接RA04） | `_supervise:604–618`父进程持锁、begin_attempt并创建worker_log；`_run_worker:640–644`把此日志当残留拒绝。即便放过日志，652行worker仍再抢父进程持有的锁，679行还再次begin_attempt。注释“worker不再抢锁”不符合实现 | 统一父子进程的run所有权、锁/attempt生命周期；默认入口能穿过真实子进程完成CPU假核心8次流程，且保留拒绝其他任务并发/旧run的能力。不得用`--no-supervise`绕过来运行GPU |
| PIPE-RB02 / P1（承接RA04） | 父子各持有总账内存副本；父在621行end_attempt之前不重读worker写入的reservation，单修锁后仍会以陈旧数据覆盖扣款。worker的finally只清模型，后续compare_tag/compare_repeat、JSON写入等若抛错会绕过823–825行记账/解锁；supervisor启动失败亦无终态。end_attempt仍可抛IO异常；最终磁盘/墙钟布尔值是在最终文件写入之前算的 | 确定单一记账责任，父不得覆盖子扣款；完成/失败/超时/启动失败/后处理或写证据失败均有可恢复非成功终态，失败扣款不退。最终状态必须合并监督结果、最新账本与最终产物，不凭worker的早期true宣称通过。CPU真子进程测试验证扣款跨进程保留、超时后无存活子进程、释放仅本次持有的锁 |
| PIPE-RB03 / P1（承接RA05及汇总判定） | verify_protocol只遍历已提供samples，缺/空samples可跳过样本核对；原图缺哈希也跳过。10项合取中timing事件两边都缺失仍可相等；no_unsafe_load实际只查A的decoder布尔值，不能代表全部加载安全。10项没有证明监督完成、账本无回退、最终文件预算满足 | 必需字段/三样本集合/原图哈希/裁切来源/固定阈值预算逐项校验，缺项fail closed。保留10项作worker子结果，但最终通过还必须有监督终态成功、账本及8条reservation/调用对应一致、最终产物/预算核验；安全加载与事件须校验明确预期而非仅相等。以源码审查+行为证据验证释放/停止，不无限新增布尔标签 |

非阻塞但必须准确披露：失败发生在写PNG/哈希时可能已有部分NPY/PNG，不能一律称NOT_PRODUCED；应登记PARTIAL/UNKNOWN并列出现存产物，不能算成功，也不需为此删除证据。异常同步被吞掉不证明同步成功，新的收口应让真实执行错误进入失败路径。陈旧锁接管仍有并发安全局限；本批禁止自动接管不确定锁/重启失败attempt，跨主机、恶意并发及通用生产级恢复不是本次扩展目标。

### 对五项裁决请求的答复

1. R-A2整体未达门槛，主要剩余是监督与worker的连接，不是要求重新设计模型。已认可的数值核心不重新开启架构讨论。
2. **10项合取不充分，也不要求为了整齐把它改成某个固定数量。** 最终工程判据为`worker_checks_pass AND supervisor_success AND ledger_consistent AND final_artifacts_verified`；每项须有落盘证据。最终还要核对输出float32/finite/正确H/W/范围、两实例完整身份、三组差值和两次重复、PNG像素、真实完整流程。安全标志不能替代安全加载代码与实际身份。
3. **R-B attempt_01条件授权**：只在current_task的新CPU门槛全部满足、最终源码冻结后启用。GPU上限仍8次/0预热/30分钟/4GiB，失败停且不准attempt_02；不把旧批次或CPU模拟的次数混入，不刷新本批总账。条件未满足时GPU额度仍为0。无需中途再发一份小返修报告等Architect回复。
4. 硬墙钟从监督批次开始计入加载/前向/证据处理；调用边界检查加监督进程的deadline终止共同执行。必须记录起止/绝对deadline、超时触发/terminate/kill请求、确认子进程退出、实际墙钟、最后reservation与call状态。超过1800秒即预算失败；OS调度/终止延迟如实记为overrun，不能静默增加额度。不要为验证超时故意跑一个GPU挂起实验；CPU超时集成测试足够，R-B记录其实际终止方式。
5. 导出阶段的强制来源格式见current_task的“后续导出manifest契约”。这是提前冻结交付要求，不是提前授权导出/打包。旧final_compliant等价性仍UNKNOWN且不作为此批门槛，官方JIT接受范围亦未获确认。

### 效率与边界

DECISION：本次采用一个工作包、两个内部阶段门槛、一次最终汇报。局部修复/CPU迭代/证据补齐由DeepSeek自主完成；仅任务范围改变、数值路线/权重/依赖变化、条件无法满足或GPU失败才升级。成功后下一包拟按“完整导出与重载验证→同主体出图→交付准备”整体规划，不继续逐函数派单；此意向不授权现阶段制作model.zip。

本轮Architect只审查/诊断并维护review/current_task/progress；未代修业务代码、未改交付证据/历史档案/索引，未GPU、导出、打包或提交。80项测试与诊断使用临时副本，原件再次核对无漂移。

## PIPE-01R R-A正式审查（2026-09-17，上一轮判决）

**DECISION：REQUEST_CHANGES。R-B不放行。** 不以测试数代替覆盖判断；下列修复由Implementer执行，Architect本轮不代改代码。

### 独立复核与认可范围

- FACT：隔离副本执行 `D:/Python/python.exe -B evidence/PIPE01R_20260917_013000/run_tests_machine.py tests`，**tests=49、failures=0、errors=0、skipped=0，exit=0**，6.839秒。禁用CUDA可见设备和pycache；未进行GPU前向或完整模型前向。副本位于 `C:/Users/Yuki Nagato/AppData/Local/Temp/pipe01ra-review-14cb9ece32b343b6b527eb200000a937/pipeline_baseline/`。这是独立执行器输出，不把副本内原有Implementer日志冒称本轮日志。
- FACT：返修报告§1的7份文件**完整SHA-256全部一致**；9份PIPE-01旧证据均与首版报告公开的**前16位哈希**一致。后者支持未见漂移，不能冒称首版报告提供了完整哈希或证明历史从未变动。
- FACT：覆盖矩阵的测试名对应实际测试。R01有独立参考数值对照和旧错误回归护栏；R02安全/离线路径确有修正；R05量化前拦截有真实小张量测试。这些CPU层修正予以认可，修正后的真实模型输出仍UNKNOWN。
- FACT：R03测试确实证明“即使输出相同，身份不等也不能最终PASS”；R04证明reserve先持久化、失败不退额度。但**不证明身份不符/前向失败后立即停止**，亦不证明锁竞争、完整加载路径和资源释放正确。端到端测试替换了整个`core.Assets`，故没有覆盖下面RA01。
- DECISION：报告§5关于script/trace、wavelet当前实现与算法能力的区别、旧预登记副本及旧汇总器副作用的勘误与上轮裁决一致。旧失败次数维持**8次确认+1次保守占用=9/10记账**；不能升级成“实测准确执行9次”。报告对新总账“必然计入/无法判定PID必拒绝”的断言仍受RA04限制。

### 未关闭项与CPU返修验收要求

以下代码位置以本次7份交付哈希为准，均位于内层`workspace/pipeline_baseline/`。不要求增加GPU试跑来证明这些控制流问题。

| ID / 级别 | FACT：证据与影响 | 返修验收结果要求 |
|---|---|---|
| PIPE-RA01 / P1 | `core.py:392`的`n_dino_keys`只在`load_dinov2_local`局部定义，函数409行只返回enc；`Assets.__init__`在513行引用该未定义名字。真实加载走到此处必然NameError，49项mock测试没走这段 | 修复真实组装路径；用轻量依赖替身执行真实Assets构造控制流并验证load_report，不能继续把整个Assets换掉来证明构造正常；不要求CPU大模型前向 |
| PIPE-RA02 / P1 | `pair_verify.py:286–291`捕获前向异常、记录FAILED后继续循环；A失败仍可进入B。A/B身份比较389行、差值比较394行以后均在8次调用之后，已具备可比结果时仍不会即停 | 首次前向失败记录且保留扣款后，不再启动后续调用/实例；B加载后身份不符必须在B首个infer前阻断；每个可比较样本/重复项及时判定，超阈值后不得继续。驱动级负测试断言调用次数和无后续产物，不只断言最终FAIL |
| PIPE-RA03 / P1 | `pair_verify.py:246,303,364`保留并返回A的assets，main的`_a`强引用存活到B加载；`Runner.release()`仅清自身字段，不能释放被外部持有的模型/缓存 | B构造前A模型及缓存不再被驱动持有；补CPU生命周期顺序测试（包含失败清理），不把empty_cache当作释放活对象的证据 |
| PIPE-RA04 / P1 | `BudgetLedger`先读总账再抢锁；91–106行用exists后write_text而非排他创建，存在竞争与陈旧计数覆盖。OpenProcess失败直接返回False，不区分“无法确认”。零额度仍先加载A；墙钟只在调用边界检查，无运行中超时监督；磁盘只算当前run。374行finally里的check_disk可再次抛错，跳过475–476行终态记账/解锁；最后两份JSON写后也没再检查磁盘 | 真正排他且持锁读取最新总账；不确定状态fail closed；无余额/时间不可加载；监督本任务自建工作进程以兑现硬墙钟；累计本批所有attempt输出；超额/异常也保留终态与已占用次数、可靠释放本次锁。用CPU假时钟/假工作进程/小文件验证竞争、挂起、失败收尾和跨attempt预算，无需GPU |
| PIPE-RA05 / P1 | `make_evidence.py:36`仍生成10次预算，R-B草案为8次、预热0。预登记samples只有名称，未冻结原图哈希/尺寸与裁切坐标；`verify_prereg:179–191`仅比四类指纹，CLI可换case路径或crop（甚至仍标crop512）；驱动也未核验总账保存的prereg_sha256，未把env/判据/样本协议变动纳入阻断 | R-B专用预登记固定8/0/30分钟/4GiB和原样本、坐标、阈值、环境、源码；运行前校验实际输入/协议与该批授权，总账绑定同一预登记。漂移在加载前阻断；不要修改旧10次登记或复用干跑总账。明确区分CPU合成测试fixture和R-B真实协议 |
| PIPE-RA06 / P2 | `pair_verify.py:267`先把A输入搬到设备再计core.infer_s；B由`offline.py:48`传CPU张量，搬运发生在`core.py:694`计时范围内；core注释还称不含搬运，且没有明确共同CUDA同步边界 | 按原§6.5兑现A/B相同的完整infer计时边界，包含实际传输、随机数、共享后处理，读写图另列；以CPU事件顺序测试证明边界，不为此新增GPU测速 |
| PIPE-RA07 / P2 | A的`tensor_sha256`是NPY文件哈希（pair_verify.py:280），B同名字段是裸tensor字节哈希（offline.py:63–64）。同一manifest字段不同协议，容易把不同含义哈希当作生成来源证据 | 分开命名/定义tensor内容哈希与NPY文件哈希，标记dtype/shape/字节序和协议版本，A/B各自可复算；补manifest语义测试。失败项无产物则明确缺失，不能填成功 |

归并原问题：PIPE-R01/R02/R05的指定CPU修正认可；PIPE-R03仍受RA02/RA05约束；PIPE-R04仍受RA02–RA05约束；PIPE-R06仍受RA06约束；PIPE-R08已有schema但受RA07及失败收尾约束。不是“R01–R06全部关闭”。

### 技术裁决及授权边界

1. **新身份链方向接受，但不足以放行R-B**。实现源码指纹进预登记、加载前比对是必要改进；还需上述输入/协议冻结及运行时身份即停。普通本地哈希不是历史不可篡改证明。零差异仍仅为固定条件数值判据，不单独证明同主体；本轮不追加seed、精度或加载顺序扫描。
2. **JIT方向维持完整数值链的可保存/重载ScriptModule目标**，允许内部拆子模块，不接受仅DiT+Python其余学习模块为最终交付。公开runner保持薄入口。优先研究忠实张量化wavelet并纳入可执行链；张量化验收必须单独预登记中间浮点误差、最终RGB像素、舍入/clamp、边界/padding、核大小、奇数尺寸和save/load后成对读数。不能通过删除wavelet或事后放宽误差过关；若不能忠实保持则升级裁决。Python侧wavelet只有在两入口一致、完整计时且存在官方明确依据后才可讨论最终边界；目前UNKNOWN。**没有批准导出或张量化实施**，本轮不预设未经试验的导出容差。
3. **生成manifest强制**。R-B先证明修正eager源码/资产→每次输入输出。后续导出阶段必须包含父基座/LoRA/各学习资产的哈希与谱系、scale/是否合并、转换脚本哈希/命令/参数/版本/时间、源与导出有效参数、导出文件哈希、runner/辅助源码/配置哈希、bundle ID和save/load验证记录；每张图绑定同一bundle及输入输出哈希、seed、shape、后处理。未发生转换填NOT_APPLICABLE并解释，不造转换记录；缺必需来源不得标交付就绪。包哈希待另行获准打包后增加，不能为了补字段自行打包。
4. 历史`final_compliant`等价性仍UNKNOWN且非本任务门槛，不授权历史资产验证/改动。官方接受形式、修正后GPU输出均UNKNOWN。
5. **当前只允许原范围CPU返修（R-A2）并重新送审；R-B GPU配额为0。** 后续候选协议保持原三输入、seed42/fp32、A/B各crop→case1→case7→crop、合计8次、预热0、累计30分钟/4GiB、失败占用且触顶停止。尚不新建真实GPU attempt。返修通过后再单独放行一个新批次的attempt_01，失败立即停并送审，不自动开attempt_02，也不刷新总账。判据max_abs≤1e-5、mean_abs≤1e-6、PNG解码RGB一致与完整身份/输出契约不变。

本轮仅更新review/current_task/progress审查状态；未修改源码、测试、交付证据、历史handoff、外部运行目录或Git索引，未执行GPU/导出/训练/扫描/量化/真实门禁/打包上传/提交。

## PIPE-01正式审查（2026-09-17，前一轮判决；返修状态见上）

**REQUEST_CHANGES。** 接受“共享入口已实现、有限样本最终输出一致”的有界事实，不接受“固定参考基线已正确复现、身份/安全/预算全部通过”的总体结论。PIPE-01既有GPU许可冻结；下一任务仅为current_task的PIPE-01R草案，GPU/JIT不自动启动。

### 独立核验结果

1. **CPU 23/23，exit=0**。审查前发现test_make_evidence_refuses_existing_run_dir会调用生成器，生成器在检查run存在之前已改写evidence/hashes.json与key_contract.json。为保全交付证据，将交付目录复制到隔离临时目录执行同一测试文件/命令，并设置CUDA_VISIBLE_DEVICES为空、禁止pycache；未编辑测试。用时12.640秒。原工作区16文件前缀哈希均与报告一致（报告只列SHA-256前16位，不冒称比过报告未提供的全值）。完整真实模型未加载、GPU未复跑。
2. **预登记**：最终三个副本SHA-256均为a60c0085943b80fc8e3f1f7fbba6f00824540e1fc4eb3f69c4b4797770bcf548，run JSON引用同值。最终副本mtime 23:52:15早于第三次日志23:52:20及输出。另找到两次原始失败目录，其预登记哈希分别a694df61540ae720…、ed28ed60858c0c46…；逐字段比较三者只差generated_at，阈值/预算/资产/参考源码声明相同。证据支持“各次登记早于对应产物、未见阈值放宽”，**不支持“同一文件在整个GPU过程从未改过”**；普通时间戳/副本不能证明历史不可篡改。预登记还未绑定core/runner/驱动自己的源码版本。
3. **成对输出可复算**：独立mmap读取A/B六个NPY（allow_pickle=False），分块差值计算及PNG解码比较。crop512、case1原图、case7原图均max_abs=0、mean_abs=0、PNG像素同一；全部float32/finite/[-1,1]，shape分别[1,3,512,512]和[1,3,3072,4096]×2；PNG哈希与记录一致。末次run有8条记录及forwards=8，实际文件673350636字节。重复crop的first/second哈希相同，但首张NPY被第二次同名文件覆盖，无法独立重算首张数组，只能核对日志哈希与保留的末张。
4. **criteria_all_pass只计算数值/PNG/repeat条件**，不计算A/B身份、全部预算、参考预处理、安全加载等。因此JSON里true不能复算成“任务全部通过”。
5. **失败没有被删除**：两个原始目录PIPE01_20260916_234741、PIPE01_20260916_235905仍在；最终目录仅attempt2_failed下嵌入第一份失败JSON，不是报告概括的两个并列attempt目录。三次elapsed相加322.9341185秒，约5.38分钟（进程内部计时，不包含所有shell启动/分析时间）。第一次在load记账处失败，0前向与日志相符。第二次报Ledger.tick不存在；当前修复版tick调用位于infer之后，而计数仅在tick成功时增加，所以**不能用forwards=0证明未运行前向**。根据执行位置高置信推测第二次已消耗1次，即总数可能9；缺原始失败源码快照/traceback时保持ASSUMPTION，不能伪造精确历史读数。保守预算至少为该次预留一次，不再沿用“0+0+8”的已证断言。

### 未关闭问题（以源码/数据为证据）

下列源码路径均在仓库内层workspace/pipeline_baseline/。

| ID / 优先级 | 证据与影响 | 必须处理 |
|---|---|---|
| PIPE-R01 / P1 | core.py:417–422构造0..255的raw后直接clip(0,1)，漏了参考inference_vosr_onestep.py:131–135的除255。灰色x=0的DINO归一化输入实际[2.248908,2.428571,2.64]，参考应[0.065502,0.196429,0.417778]，最大差2.232143 | 固定基线不一致。恢复参考数学，补黑/灰/白/梯度与边界CPU数值测试；不得保留错实现只改“参考一致”文案。修正后的输出必须另批验证，不覆盖旧GPU证据 |
| PIPE-R02 / P1 | core.py:398对LightDecoder显式weights_only=False，违反§5；本轮用weights_only=True安全读取该文件已成功，返回config/global_step/model_state_dict。DINO:319仅source=local且未传pretrained=False/已核验ckpt，local hub仍调用load_state_dict_from_url | 修复所有非安全权重加载；本地无预训练构造后显式安全加载指定checkpoint并strict核对，不能只检查文件存在。网络禁用/缺缓存/错误ckpt都需负测试。本轮未观察到实际下载，不把可联网路径写成已经联网 |
| PIPE-R03 / P1 | pair_verify.py只保存A_identity，B只有加载时长与峰值显存；core.effective_fp漏DINO，资产清单漏实际VAE权重/配置。driver只读取prereg的criteria/budget，未核对当前配置、源码/资产与登记一致 | A/B都保存并比较完整身份；包含DINO、VAE实际资产及PIPE自身源码。身份不一致必须FAIL，即使输出全零也不能PASS。预登记/加载时指纹强制比对，不用事后两个入口“看起来相同”替代 |
| PIPE-R04 / P1 | Budget.check不检查剩余前向次数；tick在infer之后，异常前向可计0，跨进程重置，最后一次后不再check墙钟/磁盘。CPU反例max_forwards=0时check仍放行。failed attempt2正好落在记账缺陷上 | 调用前持久化扣预算，异常也消耗；所有attempt总账，加载时间和末次输出都核验；不得新进程重置配额。补0额度、临界值、infer异常、进程中断、旧run重用的负测试，并如实修订失败次数说明 |
| PIPE-R05 / P1 | core.py:565先wavelet/PIL量化再validate；CPU传入NaN target，出现cast warning但输出finite且validate=True | 在VAE/DiT/decode及量化前关键数值边界检查NaN/Inf，失败不得写成功图/manifest；不能靠uint8/clamp把无效值变成合法像素 |
| PIPE-R06 / P2 | A计时前已x.to(device)，B计时包含读图/PNG/NPY；不是相同infer计时口径。repeat文件同名覆盖；测试/汇总器标“只读”却写证据/summary；runner.model_dir只检查目录存在，实际从全局config加载 | 明确研究入口定位；统一infer计时边界，IO另列，保存每次唯一产物；工具读写行为如实说明。最终model_dir加载契约和生成manifest在后续交付阶段必须落地 |
| PIPE-R07 / P2 | §8若干静态判断过强/不准确，详见下方JIT裁决 | 新勘误逐条区分已观察错误、静态风险、未经验证假说；不改写历史报告 |
| PIPE-R08 / 后续强制 | 当前有部分输入/输出记录，但缺B身份、实际执行源码快照和未来导出包到图片的完整生成manifest | 新验证批次必须产出完整manifest；未来拟交付output_dir必须由拟交付runner/权重生成并记录，不能给旧图补写来源 |
| PIPE-R09 / UNKNOWN，非本轮阻塞 | 新基线与final_compliant/your_model.pt的历史等价性未证 | 不要求作为PIPE验收或前进门槛；本轮不授权运行、转换或改写历史资产。若要比较，另立只读专题，事先取得用户同意并明确口径 |

### 五项开放问题的裁决

1. **零差异不是主体证明的充分条件**：它是当前三个样本、seed42、fp32、最终量化输出的实测事实，且共享同一错误也能零差异。无需临时追加跨精度或多seed才能验收原任务；优先修R01–R05和完整身份。在修正基线的重新验证里沿用原三个样本/阈值，保留重复512。跨精度是另一个实现版本的等价性实验，不能混入本轮。
2. **JIT清单不完整**：TorchScript scripting支持带类型的动态控制流；trace可能固化Python分支/由Python计算的shape，但不必然固定所有shape，更不必然把随机算子的输出变成常量。当前采样式trace有非确定性校验风险，应显式传noise便于成对验证；Python侧生成noise若采用，也必须计入完整infer。LoRA合并是简化路径，不是“否则必定不可导出”的已证定理。构造/加载from_pretrained无需进入forward图。还需检查RoPE/shape转换、容器缓存、副作用、CPU↔GPU与uint8舍入、cv2边界/核大小、运行版本/算子/自包含依赖、save/load后验证与trace重复执行预算。
3. **导出边界**：项目选择保守的“完整数值链可追踪”目标，不接受只导出DiT后把其余主体隐藏在另一个Python模型中作为已合规交付。可研究将VAE编码、DINO、DiT一步和LightDecoder作为单一ScriptModule的子模块/导出方法，再评估可脚本化编排；分模块不等于换主体。仅DiT导出最多算内部阶段实验，不是最终格式通过。当前普通Python/PIL/NumPy/cv2函数不能直接作为纯TorchScript调用，**wavelet算法本身并非不可张量化**。应研究忠实张量实现，并对量化/边界误差验证；不能未经验证删除后处理。若最终要把wavelet保留Python侧，必须两入口同样执行、完整计时且记录，同时获得官方明确接受该边界的依据；工程上自洽不等于官方确认。当前不作“合规”承诺。
4. **旧资产等价性**不是本任务必要条件。未知可以保留，不能反向追认。新图应由新候选生成；本轮不授权历史资产比较的GPU投入或修改。
5. **生成manifest是后续强制交付**：本次返修先补eager运行版本→每次输入/输出的来源链；导出阶段再加父权重、转换记录、导出文件和runner哈希。拟交付每图需关联同一bundle ID、源码/有效参数/配置、seed、输入输出哈希、尺寸及后处理，不能只记目录名。

官方规则本地文本只写torch.jit可执行模型及runner，未明确认可“仅DiT + Python其余主体”。TorchScript技术判断核对了与本机版本对应的官方文档：
- [PyTorch 2.7 trace](https://docs.pytorch.org/docs/2.7/generated/torch.jit.trace.html)：记录张量操作/控制流与shape限制及非确定性说明。
- [PyTorch 2.7语言参考](https://docs.pytorch.org/docs/2.7/jit_language_reference_v2.html)：支持带类型的脚本控制流等语义。
这些文档只支持技术结论，**不提供赛事合规许可**。

### 审查边界

本轮只修改现行审查/状态与任务草案；未修业务代码、未改16项交付证据及历史材料、未GPU前向/训练/导出/平台操作、未暂存/提交。临时测试副本位于本机临时目录，未删除用户材料。对最终报告哈希再次核验，以确保CPU测试隔离没有改动原证据。

## RCON-01 R3验收（当前结论，2026-09-16）

FACT：独立复跑34项CPU测试，全部OK，测试进程exit=0（2.203秒）。R3报告8份文件的字节数及SHA-256全部吻合；31条引用全部命中；manifest内生成器身份与实际文件一致。未运行真实子工具、清单生成器main或模型；没有读取整个真实包的model_dir重做验收。暂存区指纹9d0630790f498736c26d455f371fa74783c29a8d未变。

交付版本：gate=3cc62222957f11ebdaf2eb0e63e3832c4a52dc697e0aa51fb7076d4ee1d1b7c7；tests=60b5700b880d906168ac37814c83287027581dc93dfc642b9cc5ef63af4f0bbb；generator=3f52f338e09a92665293fc84bbcd9c2ac2719a1b92b8fafbcc6132b66b602c7e。

RCON-R2关闭依据：新增ZIP模型文件集合与解压目录集合/逐文件大小/SHA-256比较；同数量不同权重、额外文件、改名、只改mtime、超阈值的5个负测试均阻断③④。这满足上一轮要求的普通本地合成包身份核对，不声称抵御全部恶意ZIP或并发替换。

非阻塞限制（保留，不继续扩展本任务）：

- 哈希预算在_zip_dir_digest读完整个ZIP模型后才判断，是“结果拒绝阈值”而非实际IO预算；磁盘端额外大文件也可能先被完整读取。不能称有严格资源上限，生产化前需要预检/累计限额。这不造成超预算PASS，但可能浪费IO。
- model_dir文件内容相同不证明output_dir由该模型生成；未核对图像内容，也未核验官方规则或恶意路径/重复ZIP成员/符号链接/并发竞争。门禁未获真实执行授权，原外置调用链仍不切换。
- mtime变化不等于目录被重建；现在由内容比较补足普通串包检测，报告中的“重建”只当作实现者措辞，不作独立事实。
- 历史报告后续只用新勘误记录，不再追加修改原handoff正文。现行摘要状态本轮同步，原冻结任务在current_task历史折叠节保留。

DECISION：允许从治理准备进入独立正式工程任务PIPE-01：固定一个候选、共享推理核心、有限成对验证。小规模GPU授权只适用于该任务先通过CPU阶段后的固定验证，不包括真实门禁、训练、调参或提交。基线可用于内部研究不等于现有权重获官方认可。Architect本轮仅发布任务与文档，不代替Implementer开发或运行GPU。

## RCON-01 R2复核：最后的验收门槛（2026-09-16）

FACT：已读R2报告并检查工作副本及测试。独立CPU复跑29/29 OK（1.935秒）；报告列出的8份文件SHA-256及字节数全部一致，manifest的生成器哈希与当前生成器一致，31条代码引用全部命中。沿用下方复跑命令，未运行真实子工具链或清单生成器main。暂存区指纹仍为9d0630790f498736c26d455f371fa74783c29a8d。

关闭范围：RCON-R1的CLI默认不调用工具、stage拒绝立即返回已受测；RCON-R3的成功整行匹配已受测；RCON-R4的R04证据归属、Q11引用、交付哈希已纠正。该结论不批准显式执行真实子工具；run_gate的Python接口仍默认execute_tools=True，调用者不能将它误认成CLI的默认计划模式。

**RCON-R2仍未关闭（P1）**：_sim_binding只检查model_dir存在、目录mtime变化、自报output成员数相同，未验证模型文件或包身份。mtime变化不等于删除重建，100个成员亦无法区分绝大多数提交包。代码没有读取runner/权重内容来区分A/B包；因此“本包本次绑定已证明”仍然过强。

有界CPU反例：对_sim_binding传入package_A.zip和directory_of_package_B，mock is_dir=True、pre_state=100、post_state=101、ZIP成员数100、自报output_dir=100，返回None（表示绑定成功）。此模拟不声称真实机器已发生串包，只证明判定所用信息不足以排除串包；没有创建/读取任何真实模型目录。

最小返修建议（不扩展模型/训练范围）：

- 本阶段可直接拒绝不具备可信产物契约的真实后端，将遮蔽检查置NOT_RUN；不必为了结项重写外置harness。
- 如保留工作副本的执行检查，应核对本次ZIP与目标model_dir的相对文件集合及内容哈希（含runner、权重、依赖，拒绝额外/缺失/不一致文件）；不把mtime/成员数作为身份充分证据。真实并发隔离和独立运行目录可在后续专门立项，未验证时不能声称支持。
- 必须补“成员数相同但runner/权重不同”“只改变目录mtime、内容仍为旧包”的负测试；UNKNOWN/不支持后端必须NOT_RUN，不能PASS。测试只用临时微型合成文件，禁止实际大模型执行。
- 同步当前状态，避免current_task顶部READY/尚未执行与已交付状态并存。生成器会写证据文件，不能称整个复跑列表无副作用；历史报告勘误写新记录，不继续向旧handoff原文追加修改。

Readiness：可以准备正式任务的方案，但本轮未下发实现指令，未启动下一阶段。下一正式工程任务建议是“同主体最小可复现推理闭环”：先固定一个基座/LoRA配置、统一有效权重与前向协议、建立512与4K入口的来源链及成对验收，再谈候选扫描/训练/提交。JIT属于该交付链后续必须验证的一环，不能提前写成已完成。完成上面的窄修复并通过复核后，再发布有明确输入输出、GPU预算和失败停止条件的任务单。

## RCON-01 交付复核（Architect，2026-09-16）

FACT：阅读工作副本、18项测试、清单生成器、落盘清单/哈希记录及交付报告，并追踪实际子工具。以原任务判据复跑CPU测试，18/18 OK，测试进程正常退出；只使用原测试的临时输入与stub，未运行真实门禁、模型或清单生成器main（后者会覆盖证据输出）。

复跑命令（从Git根执行）：

```text
D:/Python/python.exe -B 生成式图像增强轻量模型设计挑战/workspace/tools/reconciliation/tests/test_gate_failure_propagation.py -v
```

核验读数：门禁SHA-256 cd14fb2cc9828dcf1ca047a8614950474a00c2d0dcea6414a39947e6f0621afa；测试65e90d2c5c4e5aa31317c8c673dde1611adbd6d16c41b687ae5fd35dea46078d；当前生成器0203d5551ab469a0519a3f25dde74a9330bf3d38954243dee61a74b7bf1e6fa5。清单中code_citations的30个期望子串均命中；这只验证引用字面位置，不证明所有解释与其他引用正确。暂存区指纹仍为9d0630790f498736c26d455f371fa74783c29a8d。

### 必须修正的审查项

以下路径中GATE/BUILD以本仓库内层workspace/tools/reconciliation为根；原件保持只读。

| ID | 证据与影响 | 判定 / 返修要求 |
|---|---|---|
| RCON-R1 / P1 | GATE main默认SubprocessExecutor调用D:/tmp_timing真实工具。harness_sim_zip.py:17–30会rmtree共享目录、解压、加载runner和infer；audit_official_format.py:90–97也删除/重建共享目录、写_probe_iface.py并运行推理。GATE:267拒绝stage后仍继续执行器，模拟复核记录了compat、harness、format三次调用 | FACT：交付报告§9“可选只读复跑/不删除共享目录”错误；当前CLI不是安全CPU只读入口。不要执行该命令。本阶段默认必须拒绝真实工具执行，stage拒绝须在任何检查/子进程前返回。用mock证明默认入口及拒绝路径零子进程调用；历史报告只加新勘误，不回改 |
| RCON-R2 / P1 | GATE:308用--sim-dir查遮蔽目录，:314只给harness传src；原harness仍硬编码D:/models/_harness_zip_sim。换sim-dir后可检查无关旧目录，或误报无产物；现有测试提前创建sim/model_dir且stub不产出目录 | FACT：注入参数没有与实际产物绑定。应绑定本次包/本次执行的产物身份；本CPU阶段至少拒绝不支持该契约的真实后端。新增“harness成功但产物来自另一目录/另一包”负测试，不能只测失败时旧目录 |
| RCON-R3 / P2 | GATE:254用子串识别终态。CPU直接调用evaluate_tool，rc=0且out=NOT_HARNESS_ZIP_SIM_PASS或HARNESS_ZIP_SIM_PASS_FAILED时均返回PASS | FACT：未严格辨认终态成功标记；这是合成反例，不声称原子工具当前输出该文本。改为按工具输出协议完整行/字段判定，补前后缀伪标记和成功/失败并存测试 |
| RCON-R4 / P2 | BUILD R04把progress§4的100/100证据归给“本地output_dir对归档”；实际旧记录比较的是不同归档包图像。protocol_artifacts的Q11_sample_ids_source写line14，CASES实际在line15，且不纳入30条自校验。报告生成器哈希c3f740…与当前0203d555…不同 | FACT：证据主体误接、引用遗漏、交付版本快照不一致。保留有效n=3抽样但纠正100/100归属；直接记录实际样本ID并校验引用；冻结最终生成器/产物/日志版本并追加勘误。不得把“30条OK”扩成所有证据链闭合 |

补充：current_task仍同时写READY/尚未执行与DONE待审查，progress摘要仍READY；交付报告§1“仅两处文档”未完整列出报告及handoff/INDEX新增导航。下一次交付应更新唯一当前状态、保留历史时点并列全改动，不继续只追加相反状态。报告复跑相对路径从Git根不可直接用；需标明内层工作目录或给出完整路径。

### 对Implementer四点请求的裁决

1. DECISION：本轮无需git add。任务中“受跟踪”意图是进入仓库内可审查范围，不覆盖“不改索引”约束；当前应准确称“仓库内未跟踪工作副本”，等待用户安排暂存/提交。
2. DECISION：注入参数、拒绝标记及缺PIL→NOT_RUN均是原§4明确要求，不是越权偏离；接受意图，但R1/R2实现需返修。阈值不必调整。
3. DECISION：不要求目录内README；现有docs任务/交付文档足够，避免再维护一份摘要。历史报告勘误用新记录。
4. FACT：asset_profiles.json确有Track2_CompliantM24——.zip旧路径；作为TR52关联的资产路径漂移登记，不凭名称相似直接重指向M24_v3，不修改旧台账。

总体：认可已实现的退出码传播、部分失败隔离和UNKNOWN披露；18项PASS不能覆盖上述缺口。建议仅在RCON-01原范围返修并重新送审，不进入下一阶段。本次Architect只更新本审查记录，未修改交付源码、测试、清单、历史报告或暂存区。

## H事项最新状态（取代报告快照的待裁决状态）

| 项目 | 当前处理 |
|---|---|
| H1 / TR03–05 | 决策已解决：D17/D18取代未来解耦；同主体实现与验证尚待后续任务 |
| H2 / TR06–07 | 技术边界已确定：D18区分评估、历史权重审计、新训练和指标攻击；历史IQA训练事实不隐藏，官方认可仍UNKNOWN |
| H3 | 不再等待“要不要冒险”的选择；既有提交的外部处置与官方回复仍UNKNOWN/未授权。本轮不重传、撤回或联系组委会；改注释不能修复既有事实 |
| H4 / TR01、TR56–57 | 六文档基线已落实；标签规则见D19。AGENTS/README/历史原文不在此次修改范围；旧索引或旧口径以本文件勘误限定 |

## 对账项关闭口径

- TR01–60全部保留作为调查ID，不机械批量标“已解决”。**文档纠错、方向裁决、代码修复、运行验证、外部确认是五种不同完成条件。**
- 文档层已纠正或限定：主力路线、环境、文件名/存在性、门禁10项、旧TODO/旧审查状态、warm start用语、有限样本与拟合假说、历史标签。原文存在本身仍可作为历史证据。
- 技术问题未修：TR13–18的交付/前向差异；TR27候选/runner来源链；TR28–40测量覆盖与分析风险；TR42门禁误报；TR44测试覆盖等。RCON-01仅修TR42指定失败传播，其他条目先登记。
- 外部UNKNOWN保留：TR02/08/12等规则/机制、历史提交审核与许可证/凭据事项；不向用户重复索要可由代码核验的答案。
- 旧R1的“只改注释重打风险件”、R2的“合规数学上不可能”、R3的“评分不受影响”、R4的“无复现路径”、R5的“README仍报旧分”均不再是当前认可结论；证据分别见TR03–05/13/15–17/39/43/46。

## 历史索引与勘误使用方式

下方完整报告的C、D、E、I节就是本轮历史索引/勘误与消重建议；handoff/archive/reference原文及其INDEX未修改。旧资料可用于恢复事件/读数，不作当前任务来源。README仍属入口但未在此次范围改写；若与最新策略不同，按用户最新决定及D17执行。

13组标签是调查时的语义归类，不是批准过13套标准；旧扫描出现次数是修改前快照，不是当前六文档计数。当前规范只见D19。报告I的广泛搬迁、台账改造和删除建议未自动获准，也未执行。

---

<details>
<summary>HISTORICAL：2026-09-15 DeepSeek 自审原记录（非当前指令）</summary>

# 审查记录（review）

> **Owner**：GPT-6 Astra ｜ **最终决策权**：用户 ｜ 本文件记录 **Architect 对项目/任务的审查结果**
> 最后更新：2026-09-15 夜（文档整理轮）

---

## 0. 当前审查状态

**⚠️ 尚无 GPT-6 Astra 的正式审查记录。** 本文档此前不存在，本轮只能：
1. 归档已有的**自审材料**（P0 审计，见 §1）；
2. 列出本轮文档整理过程中发现的**必须由 Architect 裁决的问题**（§2）；
3. 给出**待审查清单**（§3）。

凡本文件未标注来源的结论，均为 **DeepSeek 自审（非 Architect 审查）**，不得当作架构裁定。

## 1. 已有自审材料

### 1.1 P0 提交合规审计（2026-09-14，自审，原文见 `docs/handoff/P0_AUDIT_20260914.md`）
逐条核对官方要求 vs 我们的包，结论：
- **六项形式要求全部满足**（100 张 jpg、目录结构、命名、zip ≤10GB、fp16、接口签名）；
- ⚠️ **唯一形式偏差**：我们最好成绩那一份（M24, 43.6951）的 `your_model.pt` 是**权重字典**，靠 `runner.py` 里 `torch.load` 加载，而官方原文要求"**通过 torch.jit 导出**可跨平台执行的模型"（微模型包都是 JIT，✅）。
- 排除的假设："某个技术形式上的取巧能一次补 30 分"——不成立，榜上高分是实打实的模型/评分项差异。
- 相关隐患已修：graph runner 在输入形状不一致时会抛异常 ⇒ 已改为"形状不符自动回落 eager"。

## 2. ⭐ 本轮发现、必须由 Architect / 用户裁决的问题

### R1（**最高优先**）已提交的两发存在**不实的合规声明**
- **事实**：`pkg_TinyV4g/model_dir/runner.py` 与 `pkg_TinyV7gan/model_dir/runner.py` 第 5 行均写有
  「输出与 output_dir 的 4K 出图由【同一个模型主体】产生」——而这两发（**105.6401 / 94.7249**）是解耦件，该声明**不实**。
- **为什么严重**：2026-09-02 公告第三条明写「**将复核 ckpt 与推理代码**」。代码里的不实声明会把"边界解释"变成"虚假陈述"。
- **未做处置**（本轮为文档轮，且动作会影响竞赛结果）⇒ **待用户决定**：是否重打这两发（图逐字节不变、测速路径不变，只改注释措辞并加越界回退）。
- **注意**：另两份（`pkg_TinyV9b`、`final_compliant`）的同一句话是**真的**。

### R2 解耦件与公告第三条的**字面冲突**（战略级）
- 见 decisions D11。合规路线的数学上限（`Q ≤ 43.4` ⇒ 同主体包 ≤ ~44 分）与决赛线 89.71 之间存在结构性缺口。
- **已有结论**：若复核按字面执行，两发作废 ⇒ 掉回 43.6951 ⇒ 出前 10；**惩罚阶梯**：单次=取消该发成绩，多次=取消资格（阈值 UNKNOWN）。
- **待裁决**：是否需要（以及在什么时点）主动向组委会问询判据。

### R3 微模型 runner 的**输出越界**
- 实测 20 张 × 2 口径：v4g 越界像素 0.004%（肉眼不可见）、**v7gan@256 20/20 图 2.5~3.7 万像素**、v9b max`|x0|`=18.7、v11b max=56.7（成片黑斑）。
- 影响：评分不受影响（图像项只算 100 张 jpg），但**复核者若跑一次 runner 会看到"坏图"**。
- **待裁决**：是否在后续版本中加"越界回退输入"的防御（会改动 runner 行为）。

### R4 架构脆弱点（**只记录，未修**，详见 architecture.md §8）
A1 渲染 CLI 依赖 CWD 的相对路径；A2 离线 dinov2 相对路径；A4 渲染无逐像素可复现路径（影响 A/B 实验效力，噪声底 ~1/255）；A5 `workspace/` 存留与主力无关的历史训练代码；A7 协作规则双份且含过期进度（本轮已合并）。

### R5 文档与事实不一致（本轮发现）
- `README.md` 仍写"当前成绩 **256.3045**，暂列第 2（榜首 275.90）"——那是**旧世界**；新世界账号 C 最好 **105.6401**，合规最好 **43.6951**。**本轮未改 README 正文**（属用户入口文档），仅在顶部加状态提示。
- `docs/reference/新世界模型分数对应表.md` 的 "一·G" 小节被追加在文件末尾（顺序错乱）。

## 3. 待 Architect 审查清单（建议顺序）

1. **R2**：解耦策略的风险裁定与"是否问询组委会"。
2. **R1**：是否重打两发以删除不实声明（含改后是否重新过门禁）。
3. **图像项 Q 的本质**：跨家族 maniqa 强相关、家族内失效 ⇒ 下一个最有信息量的实验是什么？（crop 筛选器的指标选择是否需要改口径？）
4. **权重侧路线**：LoRA 家族重选 vs 续训 vs 引入家族外强模型，三者的优先级与资源分配。
5. **A4**：是否值得为可复现性引入固定 latent（会改变现有渲染结果，需用户同意）。
6. **决赛演练**：现场 20 张的运行时间/内存/失败回退方案是否达标。

## 4. 审查结论

**待审查（pending）**。在 GPT-6 Astra 给出审查意见前，DeepSeek 不应单方面改变技术路线、架构或已提交物。

---

</details>

---

## HISTORICAL：2026-09-16 对账调查原始快照（完整保留）

以下报告在用户本轮裁决**之前**完成，其“H1–H4待确认”“尚未获开发批准”和“建议尚未执行”描述的是调查完成时点；现在的效力见本文件顶部、decisions与current_task。证据断言仍可追踪，不将快照结论冒充当前任务。

# Project Truth Report — 2026-09-16

> 作者：Architect / Project Auditor / Knowledge Reconciler。性质：调查结果与待确认基线，不是实现授权或新架构决策。以上 §0–4 保留为 2026-09-15 DeepSeek 自审快照；其“尚无正式审查”不再描述本节之后的状态。本轮只追加本报告，不批量修改其他文档、代码、依赖、权重或历史材料。

## 范围、方法与证据说明

- 本轮重新读取当前治理规则、README、核心文档及索引，结合首轮已读的 handoff/archive/reference 全部文件，进行跨文件检索、代码/配置追踪、Git 与本机资产复核。标签扫描范围为 **36 份 Markdown**：AGENTS、根 README、docs 下 33 份、内层 workspace README。未声称逐行审完全部第三方源码或所有实验脚本。
- Git：main / HEAD `e50000c`；接手时仍有 **43 条状态记录**，含原有文档迁移、删除及上轮审计修改。未提交、暂存、恢复或删除任何文件。
- 本轮只读核验：默认 Python 与 numpy 可用性、候选文件清单、代码分支、已有 JSON、Git 历史、标签字面计数、Q 拟合算术。Q 拟合检查的判据是“原输入能否复算原输出”，不代表输入关联正确或官方规律成立。首次标签扫描在输出 emoji 时遇 GBK 编码错误；显式 UTF-8 重跑成功，无数据文件改动。
- 首轮的 69 文件 AST 解析、五包 JPEG/hash 与权重容器核验作为**已有验证记录引用**，本轮未重跑。见 [progress：首轮核验记录](C:/生成式图像增强轻量模型设计挑战/docs/progress.md)。本轮没有 GPU 前向/训练/测速，也没有在线平台或服务器核实。
- 优先级采用本轮用户指定的顺序：明确用户决定 > 当前代码/验证行为 > Git/演变记录 > 已确认决策 > 当前文档 > 历史文档 > 推测。用户决定可以确定行动方针，不能使一个外部机制假说自动成为事实。
- 分类允许组合：**FACT** 当前可证事实；**DECISION** 已明确接受的行动选择；**HISTORICAL** 只在原时点/条件成立；**ASSUMPTION** 待验证解释；**UNKNOWN** 无法确认；**CONFLICT** 可靠记录存在分歧。对已能通过代码判清的问题，用“CONFLICT → FACT”表示结论已明确，不再交用户猜选。
- “高/中/低”表示对本条判断的证据把握，不是模型质量分，也不是比赛合规认证。历史转述支持“当时有人这样报告”，不当然支持“今天平台仍然如此”。

### 证据定位表

下文短名均指以下绝对路径；章节号/函数名用于定位。Git 无证据、代码不适用或未做运行测试时会明确注明。

| 短名 | 文件 |
|---|---|
| G / R | [AGENTS](C:/生成式图像增强轻量模型设计挑战/AGENTS.md) / [README](C:/生成式图像增强轻量模型设计挑战/README.md) |
| M / A / T | [project_map](C:/生成式图像增强轻量模型设计挑战/docs/project_map.md) / [architecture](C:/生成式图像增强轻量模型设计挑战/docs/architecture.md) / [current_task](C:/生成式图像增强轻量模型设计挑战/docs/current_task.md) |
| D / P / V | [decisions](C:/生成式图像增强轻量模型设计挑战/docs/decisions.md) / [progress](C:/生成式图像增强轻量模型设计挑战/docs/progress.md) / [review](C:/生成式图像增强轻量模型设计挑战/docs/review.md) |
| H | [历史总记录](C:/生成式图像增强轻量模型设计挑战/docs/handoff/HANDOFF.md) |
| RULE / SCORE | [规则台账](C:/生成式图像增强轻量模型设计挑战/docs/reference/规则细则与判例台账.md) / [分数对应表](C:/生成式图像增强轻量模型设计挑战/docs/reference/新世界模型分数对应表.md) |
| RUNBOOK / ANN | [提交手册](C:/生成式图像增强轻量模型设计挑战/docs/reference/SUBMIT_RUNBOOK_20260909.md) / [9月2日公告分析](C:/生成式图像增强轻量模型设计挑战/docs/reference/RULE_UPDATE_20260902.md) |
| TXT / SAMPLE | [本地赛题原文](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战.txt) / [官方样例 runner](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/runner.py) |
| WR / LM / TL / EV | [内层 README](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/README.md) / [light_model](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/models/light_model.py) / [train_local](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/train_local.py) / [eval](C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/eval.py) |
| INF / TILE / TRAIN | [infer2](D:/tmp_timing/vosr_lora_infer2.py) / [tiled_vae](D:/models/finals/VOSR/tiled_vae.py) / [LoRA训练](D:/tmp_timing/train_vosr_lora.py) |
| MR / TR | [M24 runner](D:/models/final_compliant/model_dir/runner.py) / [TinyV4g runner](D:/models/pkg_TinyV4g/model_dir/runner.py) |
| GATE / HAR / FMT | [preupload_gate](D:/tmp_timing/preupload_gate.py) / [harness模拟](D:/tmp_timing/harness_sim_zip.py) / [格式检查](D:/tmp_timing/audit_official_format.py) |
| LED / ASSET | [score_ledger](D:/tmp_timing/score_ledger.json) / [asset_profiles](D:/tmp_timing/asset_profiles.json) |
| SP / CF / QF / QM | [score_pred2](D:/tmp_timing/score_pred2.py) / [ceiling_check](D:/tmp_timing/ceiling_check.py) / [q_fit2](D:/tmp_timing/q_fit2.py) / [q_fit_measure2](D:/tmp_timing/q_fit_measure2.py) |
| Q10 / Q11 | [协议筛查脚本](D:/tmp_timing/q10_protocol_screen.py) / [对齐筛查脚本](D:/tmp_timing/q11_align_screen.py)，各自同名 JSON 保存汇总 |

## A. 当前可以确认的项目事实

1. **FACT**：赛题是 Diffusion 图像增强，测试集 100 张、验证集 3 对、质量图与 512×512 测速有不同交付路径；决赛 PPT 与现场 20 组图见 TXT。精确评分公式不在该文本中。
2. **FACT**：当前主力资产是 VOSR 0.5B、v7_s5000 LoRA/sc0.8 对应的 M24；仓库内 LightLatentDiff 是历史主线，不应与当前 VOSR 混称同一实现。
3. **FACT**：主线运行代码与权重大量位于 Git 外；本机可找到并不等于新机器可以检出后复现。当前主要数据载体是权重/LoRA、图像、JSON、日志与 ZIP，未见业务数据库或在线服务接口。
4. **FACT**：主力 4K 路径为 VAE 潜码 + DINO 条件 + 随机 z → DiT 时间 1→0 一步更新 → LightDecoder → 分块/对齐/PNG。INF 默认 VAE 后验使用 TILE 的 mode；MR 用 sample，且两条路径的精度和后处理并不完全一致。
5. **FACT**：本轮终端默认 python 和显式 `D:/Python/python.exe` 均指向 Python 3.13.0，numpy 可发现。此结论限定本轮终端环境，不覆盖其他应用、Shell 或旧会话。
6. **FACT**：当前文件中 v6 有 s1000–s5000，v7 有 s1000–s6000，v8 有 s1000–s9000；v5 有 `v5_s1000.pt`–`v5_s4000.pt`；指定 v1/v2/v3 目录只有日志/指标，没有权重文件。不能把指标文件名当成现存 checkpoint。
7. **FACT（已有验证）**：18 个归档 ZIP 中，首轮检查五包；两份 M24 高分解耦包及 Slow8 各100张 JPG 与 M24_v3 字节一致，avg2 则全部不同。此证据不等于它们均已通过官方复核。
8. **FACT**：MR 使用普通权重字典；TinyV4g 使用 JIT + CUDA Graph。GATE 有10项，但只覆盖部分技术检查，并非规则认证器。
9. **DECISION（文档记录）**：用户已确定优先真实质量/现场能力、单 GPU 关键任务纪律和用户最终裁决。本轮最新明确决定是仅进行事实对账，不执行开发。
10. **FACT**：既有原生4K LQ/GT基线 JSON 能支撑 README 对应均值；但这些是三对验证图的历史测量，不是测试集质量上限或官方评分上限。

## B. 文档冲突与事实缺口清单

共 **60 项重要对账项**。包含仍待裁决的冲突、已能判清的纯事实错误、已解决但仍被重复传播的旧问题，以及缺少证据的强断言；不是声称存在60个代码 bug。表内“无运行验证”不等于代码正确或错误，仅说明证据边界。

### B1. 规则、权限与交付语义（01–12）

| ID | 涉及来源及各方说法 | 实现 / Git / 测试证据 | 判断与置信度 |
|---|---|---|---|
| TR01 | G §6 只有四类知识标签；本轮要求再区分 HISTORICAL/CONFLICT | 36文件扫描中后两词均为0；代码不决定标签规范 | **FACT**：现有规范不覆盖本輪要求；本报告用六类，不擅改G。高 |
| TR02 | RULE 标“用户转述”的9/2公告又标【原文】；R/A 将若干解释合称官方硬约束 | TXT没有9/20日期、八指标逐一名单或JIT/CUDA Graph许可清单；SAMPLE确有fp16说明；未获得公告原件 | **FACT/UNKNOWN**：必须区分本地原文、样例、转述、团队解释。高 |
| TR03 | R/D3用“出图测速解耦”支持不同模型；RULE/ANN要求同一模型主体 | 两条评测路径确实独立；两高分件的图/runner也确为不同来源（首轮包核验） | **CONFLICT**：评测分离不逻辑蕴含允许换模型。用户风险接受不能被写成官方许可。高；后续策略待裁决 |
| TR04 | D3“生效中、用户接受”；D12 L3/T“仅同主体”；D14转向硬实力 | 已提交件证明过去行为，不证明未来仍授权；代码不能裁定战略优先级 | **CONFLICT/DECISION**：保留既有决定，确认D3对未来新提交的效力。高 |
| TR05 | RULE L3要求“由该runner渲染”；R/D写“同源/同主体”；T用现成MR配新候选 | INF基座+LoRA离线分块，MR合并权重512前向，入口不同 | **CONFLICT**：同入口、同架构、同权重、同主体不是一个概念。技术差异FACT；官方粒度UNKNOWN。高 |
| TR06 | RULE L4“不计算/不读取任何官方指标”；R/T“禁止针对IQA优化” | EV/QM正常计算IQA；TRAIN显式可微CLIPIQA，v7/train.log记w_clip=2.0 | **CONFLICT**：评估、选模型、训练损失、逐图挑图、像素对抗需分别定义。不得自行追溯废弃v7。高 |
| TR07 | RULE L1/L2称主动加的门槛；R/D“红线依据公告”容易读成官方数值阈值 | TXT/公告转述未给lapvar60或clip0.30；GATE10项未实现这两个质量门槛 | **FACT/DECISION**：是团队代理门槛，不是已知官方判据，也不是GATE已覆盖。高 |
| TR08 | RULE §1将A定为出分前自动检查且不累计；§4计数未知；§8假说为批量人工复核 | 同文件记录标签延后出现；无平台执行代码/原始审核记录 | **CONFLICT/ASSUMPTION**：A/B/C是本地解释框架，不能充当已证平台架构；“未取消”不证明永不累计。高 |
| TR09 | RULE称Speed204x未打标、无人因提速被拦；9/13交接§1.5/H §16.100列其欺诈失败 | 两组历史记录冲突；没有代码可判平台标签，未在线核验 | **CONFLICT**：具体该提交状态UNKNOWN，不能继续用作“所有提速都通过”反证。高 |
| TR10 | RULE把MetricNet1105x说成“运行时算指标”；R §10说LPIPS-only NAFNet；H记训练w_clipiqa=0 | 历史训练/模型说明不支持把感知损失网络等同指标计算器；没有对该原始包作新的运行验证 | **CONFLICT/ASSUMPTION**：训练用指标与推理算指标须分开，欺诈标签原因未证明。中 |
| TR11 | TXT命名示例含中文；R同时说纯ASCII与全角——可用；RULE/GATE禁非ASCII；D/9/11交接不带队名电话 | GATE实际检查ASCII；归档包名含——属于历史归档命名；不同时间的用户上传经验不一致 | **FACT/DECISION/HISTORICAL**：上传工作名与归档名分域；当前团队保守规则ASCII，不称TXT明确禁止一切中文字符。高 |
| TR12 | TXT写每周一次、大小“=10GB”；R写约10次+5/≤10GB；9/13交接写平台20000M | 当前GATE按10GB检查；次数/上传控件为用户转述，无在线验证 | **HISTORICAL/UNKNOWN**：区分原始文本、日期化平台观察和当前团队限制，不能给剩余额度或平台上限下定论。高 |

### B2. 实现、环境与模型资产（13–26）

| ID | 涉及来源及各方说法 | 实现 / Git / 测试证据 | 判断与置信度 |
|---|---|---|---|
| TR13 | R/P称43.6951“合规最好”；V/A承认非JIT | TXT要求torch.jit；MR为torch.load普通字典；首轮容器检查支持 | **CONFLICT→FACT**：可称同主体基线，不能称全面合规已确认；形式豁免UNKNOWN。高 |
| TR14 | V/P0称graph形状不符已回退；A首轮发现TinyV4g并非如此 | TR固定128缓冲；仅检查宽度，graph可用时直接copy；本轮未运行非方图 | **CONFLICT→FACT**：不能普遍宣称任意形状回退完成；512方图不能覆盖全部接口。高 |
| TR15 | R/A笼统“强制fp16”；历史称渲染fp16 | INF用默认float32模型/输入且无half/autocast，所调用TILE无精度切换；MR显式half | **FACT/UNKNOWN**：现行离线与测速精度不同；未做运行dtype探针。fp16约束是否覆盖离线生成须按来源确认。高（代码层） |
| TR16 | 首轮A §3笼统称VAE采样；MR确实sample；INF实际不同 | TILE encode_dispatch默认posterior_mode=True，args.json无覆盖，sd2分支mode；MR sample | **CONFLICT→FACT**：更正首轮不精确描述。两条路径的VAE后验策略不同，不擅自统一。高 |
| TR17 | A §8/P称随机latent导致“无逐像素复现路径”；A §3/§10又限定该推断 | INF设置torch/CUDA/numpy seed；文件顺序影响随机数消耗；没有严格重复运行证据 | **ASSUMPTION/UNKNOWN**：随机算法可被确定种子控制；精确历史复现尚未建立，不等于不可能。高 |
| TR18 | R/T说离线化已修；RUNBOOK说工具只有SSH联网 | INF本地hub失败仍remote fallback；HF_HUB_OFFLINE不封torch.hub；未断网试跑 | **CONFLICT→FACT**：仅本地优先，并非完全离线。高 |
| TR19 | R/A旧A1称INF必须CWD因ae_path相对；INF头部说任意CWD | INF的VAE从checkpoint父目录拼，DINO补丁绝对路径；上游旧helper仍相对路径 | **FACT/HISTORICAL**：脚本版本必须入断言范围；不能将旧helper缺陷泛化到当前补丁路径。高 |
| TR20 | M/R说默认python无numpy；旧A说D盘Python3.11；首轮已改3.13 | 本轮默认python实际D:/Python/python.exe 3.13.0，numpy可发现；首轮显式版本亦3.13 | **CONFLICT→FACT**：当前终端警告过时；其他启动环境UNKNOWN。高 |
| TR21 | R/WR快速开始以requirements代表依赖；requirements注释要求云GPU | 文件主要评测依赖，未显式列当前主线peft/diffusers/上游包，也未锁版本；本机装好不等于干净安装可复现 | **FACT/HISTORICAL**：缺当前主线环境清单；本轮不安装依赖。高 |
| TR22 | WR头部“不用云”；技术路线“训练云GPU”；D2称腾讯云/P100均关机 | H记录8/31重开服务器路线及9月P100工作、后续不可达；未连接云端 | **CONFLICT/HISTORICAL**：本地优先是记录的方向；云资源实际已关机UNKNOWN。高 |
| TR23 | WR仍把小VAE+UNet作为当前路线；M/A称VOSR主力 | LM旧线存在；MR实际DINO+VAE+DiT+LightDecoder；主线代码在Git外 | **CONFLICT→FACT**：WR应明确历史模块文档，不能替代项目架构。高 |
| TR24 | LM文件头t999/scheduler.step；代码默认t500/显式公式 | __init__/forward直接证明；无需GPU；历史D1也记t500 | **CONFLICT→FACT**：源码注释过时，不改实现。高 |
| TR25 | D9/M称train_local“resume漏LR对齐的bug”；TL自己写warm start | TL只载权重后新建优化器/调度器；未恢复step/opt状态 | **FACT/ASSUMPTION**：确实不是完整续训；若意图warm start，是否bug取决于原任务。不能从失败分数唯一归因LR。高（行为）/中（原因） |
| TR26 | T/M枚举v1/v2/v3/v5及统一lora_s命名 | 目录递归核验：v1–v3无权重；v5文件名v5_s*.pt；v6–v8存在多档 | **CONFLICT→FACT**：当前候选清单不能照搬。别处是否有备份UNKNOWN，但不要求用户猜本目录内容。高 |

### B3. 渲染、测量与评分证据（27–40）

| ID | 涉及来源及各方说法 | 实现 / Git / 测试证据 | 判断与置信度 |
|---|---|---|---|
| TR27 | T任务B换候选图但固定MR，声称同ckpt零违规 | MR声明固定v7_s5000/sc0.8合并；INF支持变更LoRA/scale；未见任务单绑定相应合并产物 | **CONFLICT**：现任务不能保证同权重；应补证据链，而非擅自选新主体。高 |
| TR28 | T/R“协议轴系统关闭”；Q10仍列六配置；首轮着重引用case12冒烟 | Q10 JSON只有A/C成功、B失败，缺D/E/F/REF；磁盘A/C及三align各6图，D_dit768仅3图 | **FACT/ASSUMPTION**：成功组不是单张实验，而是6张集合；整体扫描并未完整落盘，不能宣称全部旋钮穷尽。高 |
| TR29 | T/R“wavelet就是真实历史配方”；A限定为最接近 | Q11六图汇总MAE wavelet5.124、nofix7.745、adain7.265；比较新PNG与旧JPG，无完整manifest | **ASSUMPTION**：历史配方近似候选，不是精确身份认证；编码/seed/精度等仍混杂。高 |
| TR30 | 竖缝指标被用作分块结案证据 | Q10/Q11 seam固定每512列测差，实际DiT默认stride448像素，VAE默认stride896；只测竖向，且自然边缘也会入统计 | **FACT/ASSUMPTION**：代理不与实际全边界对齐，不足以证明接缝消除/无收益；不修改算法。高 |
| TR31 | R/WR“正式评测必须原生4K”；9/13交接禁止原生LPIPS/DISTS；T引入c1024 | EV支持resize；历史有全图、512中心crop、c1024、分块LPIPS等不同量 | **CONFLICT**：可要求同口径，但不能同时要求所有指标原生且禁止两项原生；后续应指定每项协议，不直接跨表比较。高 |
| TR32 | R/A“8指标闭环”容易被读成结果完整；H曾把null说成均值仍可用 | EV创建失败省略、逐图异常None、均值只算有效值，总n却是pred数；本轮源码检查 | **FACT**：必须记录逐指标有效n/失败原因，缺项均值不可默认为同样本可比。高 |
| TR33 | EV称按case号匹配、原尺寸评测 | collect相同case后者覆盖；尺寸不一致自动缩放pred，随后无参也算缩放后的pred | **FACT**：缺重复键/尺寸严格验收；文档没有充分披露。高 |
| TR34 | “一切指标由落盘JPEG”；INF/Q10/Q11却直接测PNG；QM的test30含义不显式 | INF输出PNG，Q10/Q11测PNG；QM取字典序前30张，不是case1–30 | **CONFLICT→FACT**：PNG/JPG、样本ID列表、排序、裁切均需进协议；不能只写test30。高 |
| TR35 | R/D/T写v3.1且只准SP；SP实际v3 | SP p=.19/C=2.0，CF/QF .198/2.515；LED两版本同时保留 | **CONFLICT→FACT**：文档/代码版本漂移；修正选择属于后续任务，本轮不改SP。高 |
| TR36 | R/A称Q~118.3·maniqa−15.26，LOOCV1.78为强证据 | QF把w2_m0s1_s090与34.0989配对；SCORE/ASSET将34.0989归属SC093_v2；QM读取的是s090目录 | **CONFLICT**：图指标与官方成绩锚点配错身份；回归可复算不等于配对可信。应先对资产，不直接用该回归筛选决策。高 |
| TR37 | D13用GT推Q约30，QM称测GT；R拿测试M24与valGT比较 | QM对通配符*_gt.*用Path.exists而非glob，GT_val n=0；本轮QF不输出GT行；历史GT3图与M24 test30非同样本 | **FACT/ASSUMPTION**：GT基线JSON存在，但当前QF没有GT观测；“完美图打不过生成图”非被证明结论。高 |
| TR38 | SP/LED/SCORE将avg2列“同图/近似同图速度轴” | 首轮100/100 JPG哈希不同；平均多样本本就改变图像。CF四点不含avg2 | **FACT**：avg2不是严格单变量速度对照；不能连带否定未用avg2的四点算术拟合。高 |
| TR39 | R/D/T将113.6x、116.4分、只剩9%写成上限；V说合法路线数学上不可能入围 | CF四点三参数拟合，输入为本地时延+历史成绩；无官方机制或区间外独立验证；首轮复算极限约116.5 | **ASSUMPTION**：这是所选模型下外推，不是官方物理极限。ΔQ=Δ分亦仅近归一化倍率1时近似成立。高 |
| TR40 | SCORE称M24四有参全差、GAN只改一项故有参权重0/负；旧H反说只认LPIPS | 表内LPIPS .4476<.4977实际更好；GAN实验步数/调度也变；多家族/速度混杂 | **FACT（算术纠错）/ASSUMPTION（机制）**：拒绝“官方有参权重已识别”的表述，不因此擅自重开旧路线。高 |

### B4. 状态、测试与历史治理（41–52）

| ID | 涉及来源及各方说法 | 实现 / Git / 测试证据 | 判断与置信度 |
|---|---|---|---|
| TR41 | R/T/D L6、reference索引写9项；历史某阶段8/9项；P/A写10项 | 当前GATE为5基础+兼容+harness+两遮蔽+格式；Git a7b1206已记9/11升10项 | **CONFLICT→FACT**：现行为10；8/9为阶段版本，不是三种同时有效标准。高 |
| TR42 | GATE“全部通过可以上传”；HAR打印infer OK被视作通过 | HAR先打印后assert；GATE忽略rc，FMT有部分独立补救；未覆盖JIT/来源/原生图质量 | **FACT**：存在假阳性子项和覆盖缺口。测试通过范围必须明确，不能扩展为合规认证。高 |
| TR43 | V/P称越界；TR返回clamp后的Tensor；V又称评分不受影响 | raw x0与返回值不同；历史黑白饱和不是返回Tensor必然越界；本轮未重跑20图 | **FACT/HISTORICAL/ASSUMPTION**：区分内部越界、clamp饱和、接口范围、观感；“不影响评分”未获完整官方机制证明。高 |
| TR44 | 旧smoke输出SMOKE_OK、导出打印NaN；资料说冒烟已过 | smoke针对旧LM/v4，不是VOSR；多项打印而非断言；首轮69 AST只查语法 | **FACT**：语法通过、打印完成、前后向正确、端到端/平台通过不可共用一个PASS。高 |
| TR45 | R/V说尚无Architect审查；A/P已有首轮Architect报告 | 当前文件与上轮实际落盘证明已审计，但报告散在A/P；V旧“pending”未更新 | **HISTORICAL→FACT**：审查已发生；是否批准开发是另一状态。本报告补入V，不视作开发批准。高 |
| TR46 | V R5/P §6仍称README当前256分；R正文实际105分 | 读取当前R即可排除；A/P首轮已提醒但旧条仍在 | **HISTORICAL**：已修错误仍被列未修，需状态关闭而非再次修改R成绩。高 |
| TR47 | T同时写A/B/C进行中与“本轮文档整理不改代码”；P续训条件；用户本轮仅对账 | 无扫描完成清单、无当前作业状态证据；旧计划不是活进程 | **HISTORICAL/DECISION**：当前授权仅对账；研究任务是历史计划/待恢复，不据“🔄”启动。高 |
| TR48 | T验收要求一定找到更优者；候选指标排名未定义主指标/权衡；10分钟无边界 | 现有脚本/结果不足以闭合任务；不同seed/样本/设备耗时不可替代计划预算 | **UNKNOWN/Recommendation**：需要确定成功/负结果/超时/OOM/缺权重出口。不是已实现验收。高 |
| TR49 | P说20图脚本待写、约19分钟；H §16.45记20图23.4分钟；T称现场已验证参数 | 旧演练是历史条件；没有证据把旧脚本/权重/精度直接绑定当前INF+现场环境 | **HISTORICAL/UNKNOWN**：不能说从未演练，也不能说当前现场条件已验收。高 |
| TR50 | SCORE待补TinyV9b、TinyV1待评测行；同表后文及P已有40.5757/27.3521 | 本地ZIP/LED的历史读数存在；无在线状态重验 | **HISTORICAL→FACT**：本地已记录出分，旧TODO应标已被后续记录取代。高 |
| TR51 | 9/11交接要求修ASSET四发“待评测”；ASSET现已标失败且含后续13条 | 本轮读取submissions直接确认；历史文件不回写 | **HISTORICAL→FACT**：该数据修正TODO已完成，不再下发。高 |
| TR52 | LED称统一/机器可读台账；根accounts.C.best仍43.6951，entries缺最新三发 | 最新40.5757/94.7249/105.6401只在score_model_v3.new_scores；derived还含旧“速度小轴”与旧在跑任务 | **CONFLICT→FACT**：同一JSON内部有旧快照和新事实，消费者读不同键会得到不同现状。baseline_best也未明确是同主体基线还是账号最佳。高 |

### B5. 证据保存、术语与目录边界（53–60）

| ID | 涉及来源及各方说法 | 实现 / Git / 测试证据 | 判断与置信度 |
|---|---|---|---|
| TR53 | D16称全部删除文档曾入Git可恢复；M旧清单仍含已删项；D2指不存在SERVER档 | 指定NEXT_ROUND_PROMPT_20260916的git log --all无记录；当前目录无SERVER文件；首轮M已纠正部分数量 | **FACT/UNKNOWN**：不能保证全部可从本仓库恢复；外部备份未知。33份docs=6核心+17handoff+6reference+4archive。高 |
| TR54 | handoff索引说内容当时为真且至今有效、排除已证伪；实际收很多已被取代方案；archive把入档等同“此路已死” | NEXT_ROUND旧公式/云计划与D14等不相容；文件存在能证历史文本，不能证其判断当时正确 | **CONFLICT**：目录应表达用途/时效，不自动认证真实性或技术不可行。高 |
| TR55 | reference承诺长期有效；ANN含旧163分/云计划；RUNBOOK按废弃公式自动判读；调研写“模拟”且推荐已弃候选 | 新核心记录已另定方向；本轮未联网复核论文/发布/许可证现状 | **HISTORICAL/UNKNOWN**：源文本与过时分析分开，不能把整篇标长期现行；外部模型结论限定调查日期。高 |
| TR56 | 多处称唯一入口/单一事实源；成绩/规则/公式/任务重复存R/P/D/SCORE/RULE/LED/ASSET | LED与文档实有不一致；A §8旧说法与§10审计反向；首轮追加纠错也留下双版本 | **FACT**：重复维护造成冲突。本轮明确承认首轮“追加纠错”尚未实现统一基线；报告不能再被当新增平行台账。高 |
| TR57 | 同名v3/v7/M24、Q/S/A/B/C混指模型、包、公式、账号、失败类别；R将HYPIR称4x用法 | 目录/代码分别对应自训v3、包v3、评分v3、VOSR-LoRA v7/TinyV7gan；HYPIR历史1x增强与OSEDiff漂移不是同一概念 | **FACT**：强制带命名空间及时间/包ID才可比较；HYPIR泛化描述降为历史待核解释。高 |
| TR58 | R称重JPEG为赛题画像；H由估计说轻JPEG/轻模糊重噪；WR训练写含下采样 | TXT未公布退化流程；旧data/degradation.py保持同尺寸；不同训练器采用不同采样器 | **FACT/ASSUMPTION**：实现的训练退化可证，测试真实退化参数仍是估计；不得把某版本采样器当官方数据生成器。高 |
| TR59 | RULE“全自研/全程开源，无影响”；M列第三方资产；旧资料称凭据另存/工具不入Git | VOSR/DINO/SD等外部组件实际被使用；历史跟踪文件含明文凭据位置；未查许可证、凭据有效性或仓库可见性 | **FACT/UNKNOWN**：自研集成≠全自研，使用开源≠所有许可条件已核实；不删除/回写历史、不暴露凭据值。高 |
| TR60 | 数据称DF2K3450；v7日志记4250；一系列Q/时延/基线混用val3、test30、test100 | TRAIN按文件收集，日志4250，历史另有800+2650描述；本轮未对全部训练图去重。README拿testM24与valGT作“超过GT”比较 | **CONFLICT/UNKNOWN**：文件数≠独立场景数，需数据manifest；样本集不同不能证明生成图优于同一GT。高（差异）/中（成因） |

## C. 历史内容与当前内容混淆

应采用三层时间线，而不是“日期最新的一份整篇为真”：

| 时段/材料 | 可以作为证据的部分 | 不再自动生效的部分 |
|---|---|---|
| 早期自训、云V100、v1–v5 | 代码存在、训练日志、当时提交与故障 | 小UNet仍是主力、当时环境/额度/榜单 |
| 8月VOSR/LoRA、v7c与旧评分推断 | 模型来源、当时实验读数、用户当时指令 | 256分为当前有效成绩、LPIPS唯一权重、旧“终极天花板” |
| 9/2公告转述及9/4–11交接 | 规则变化转述、P100演变、依赖兼容修复过程 | 4包仍待提交、平台仍关闭、ASSET失败标签尚未修 |
| 9/13–15微模型与评分拟合 | 保存的包/图/日志、标有协议的历史测量 | 所有失败等同自动检查、旧速度/质量封盘结论、旧预测工具版本 |
| 9/15文档整理与9/16首轮审计 | 文件迁移事实、已验证差异、审计证据 | “本轮不改代码”永久约束、审查仍未发生、上轮审计推断无须再验 |

明确更正自己的首轮理解：INF的VAE默认mode而非MR的sample；Q10/Q11成功组对应6图而非把case12冒烟当全部实验；“Python默认缺numpy”不能继续沿用；train_local新建调度器是可证行为，但warm start与续训意图需要分开。上一轮A/P的纠错附录与旧正文仍互相冲突，这是尚未完成的知识治理，不以“已经审计过”为理由豁免。

历史文件只记录勘误指针，不回头改写；当前文档中的历史段落应显式标时期和superseded_by。被否定的假说应写“历史假说”，而不是HISTORICAL=当时一定为真。

## D. 标签体系问题及统一建议

### D1. 到底有多少套

没有现存注册表，无法给出一个天然、唯一的“官方体系数量”。按**语义域**归纳，当前至少有 **13组并存的标记约定**；这是本轮分类口径，不是声称项目批准过13套规范。

| 组 | 当前写法 | 重复/冲突 |
|---|---|---|
| 1 知识等级 | FACT/DECISION/ASSUMPTION/UNKNOWN，实测/推断/区间推断/未知/假说 | 实测把“观察、解释、外推”包在一起；缺历史/冲突标签 |
| 2 来源 | 原文、官方、用户口述/转述、截图、模型推断 | 用户转述又标原文；“官方分”混本地录入与当前在线状态 |
| 3 工作状态 | TODO、待办、进行中、待评测、阻塞、完成、[ ]/[x] | 旧任务无日期仍表现活跃；完成缺证据/适用版本 |
| 4 决策生命周期 | 生效中、采纳、待裁决、已取代、放弃、关闭 | 技术假说被放进决策目录即像用户拍板；D3/D14效力未明 |
| 5 实验结论 | 已证伪、封盘、永久关闭、闭合、终局 | 把一次失败等同所有配置不可能，已经失去可靠证据语义 |
| 6 测试结果 | PASS/FAIL/SKIP/done、SMOKE_OK、infer OK、各工具专用标记 | 打印/退出码/断言/覆盖范围不同；SKIP可能被算成功 |
| 7 优先级/风险 | P0/P1/P2、最高优先、红线、⚠️ | 优先级、严重性、合规性和不确定性混在一起 |
| 8 权限/署名 | Owner、用户决定、模型决定、自审、Architect审查 | 维护Owner被误解为最终裁决；自审PASS不等于Architect批准 |
| 9 问题/任务编号 | D1–D16、A1–A7、R1–R5、B1–B4、L1–L6、任务A/B/C | 字母A/B/C另有账号/判负含义；编号只有文件内唯一 |
| 10 版本 | v1/v2/v3/v3.1、v7、M24、s5000/sc0.8/seed8 | 模型、包、预测公式复用v3；M24同时指配方、图片集和提交 |
| 11 时期/来源阶段 | 旧世界/新世界、W_old/new、M_old/new、Phase、波2、链H2等 | 提交时间、模型训练年代、评分机制和当前任务阶段混用 |
| 12 提交结构/审核 | S1–S4、A/B/C、R=出分/fraud/技术失败、合规/退化 | S2+S3可叠加，并非互斥类；A/C主体未证；出分被误写通过 |
| 13 测量协议与展示符号 | native4K/c512/c1024/test30/Q/r；✅❌⛔🔄⏳⏸⭐ | 协议名不含样本/处理；✅既指完成又指已取代；❌可能指失败或禁止；⭐没有稳定优先级含义 |

扫描读数（字面出现次数/涉及文件数，非语义条目数）：FACT25/4、DECISION3/2、ASSUMPTION8/3、UNKNOWN22/7；HISTORICAL与CONFLICT均0。实测179/26、区间推断14/6、未知48/13、已证伪32/15、永久关闭9/2；[ ]9、[x]3；✅178、❌41、⚠️136、⛔15、🔄16、⏳13、⏸6、⬜9、⭐139。部分词计数存在包含关系（如“区间推断”含“推断”），不能把这些数字相加称标签总数。

### D2. Recommendation：分维度标记，而不是再造一个万能状态

建议每条重要记录保留以下独立字段，先应用于文档，不修改代码/JSON接口：

```text
ID: 全局可区分的命名空间编号（例如 DOC-TR-036）
kind: FACT | DECISION | HISTORICAL | ASSUMPTION | UNKNOWN | CONFLICT
scope: 具体文件/模型/权重/提交ID/数据与测量协议
as_of: 事实核验日期（与原事件日期分开）
source: code / runtime / git / official-local / user-report / inference
evidence: 路径+函数/提交/运行记录，含反证
state: draft | active | blocked | resolved | superseded | archived
owner: 维护者；decided_by: 仅DECISION需要，保留用户确认依据
supersedes / superseded_by: 关联旧/新记录，不抹去旧内容
confidence: high | medium | low（仅对断言把握）
```

- HISTORICAL表达时间适用性，可写“历史FACT/历史DECISION/历史ASSUMPTION”，不把错误推测洗成曾经的事实。
- CONFLICT保留各分支来源，只有证据足够或用户裁决后才能resolved；解决不代表原材料应删除。
- 测试使用独立 `PASS/FAIL/SKIP/NOT_RUN`，必须附版本、输入、命令/日志、退出码和覆盖范围；SKIP/NOT_RUN不计PASS。
- 提交状态独立记录 `prepared/submitted/scored/technical_failed/fraud_flagged/under_review/invalidated`；“未知复核状态”不能填“通过”。这是文档建议，不迁移现有JSON schema。
- 决策与实验证据分离：“停止投入”是DECISION；“当前样本未改善”是FACT；“所有配置都不行”通常仍是ASSUMPTION。
- 合规不使用单一✅：至少分技术格式检查、图/权重来源、审核状态三项。风险接受保留决策日期，不改写事实。
- emoji只作装饰，不承担状态意义；旧Phase/链/波/版本保留历史命名空间，不重新编号历史。
- G中已有四类体系；是否将此建议纳入长期协作规范须用户确认，本轮未修改G。

## E. 重复与冗余信息

| 信息 | 当前重复位置 | 建议单一维护位置（尚未执行） |
|---|---|---|
| 成绩/提交状态 | R §2/10、P §2、SCORE、LED、ASSET、RULE案例 | 提交事件台账；P只汇总并指向它，R最多日期化摘要 |
| 评分经验模型/Q表 | R §4、A §7、D4/5/13、T、SCORE、LED、SP/CF/QF | reference中的版本化实验说明+原始读数；D只记是否采用及范围 |
| L1–L6/官方解释 | R、D12、RULE、ANN、RUNBOOK | 官方证据与团队政策分开；D保留团队决策，入口引用 |
| 环境与绝对路径 | R、M、A、WR、各交接 | M管资产路径；A管运行契约/环境要求，实际观测有日期 |
| 当前任务/下一步 | T、P、R、LED derived、RUNBOOK、交接提示 | T唯一现行任务；P只报状态/产物，历史计划不自动执行 |
| 风险/审计问题 | A §8/10、V、P §6/8、D11 | V统一问题ID；A仅保留影响架构理解的已知行为并引用 |
| 历史基线数值 | R、WR、旧规则快照、SCORE | 原始results JSON+协议；其他只引用，不再抄独立表 |
| 文档地图/分类规则 | R §11、M §5/8、三个INDEX、G §9 | G管治理；M管现存清单，INDEX仅导航，不另立真实性规则 |

不能机械删重：同一个读数的原始记录、后续解释、用户决策是三种不同证据。可合并的是“当前摘要的重复维护”，不是不可回溯地丢弃历史。

## F. 文档缺失的重要事实

1. **基线身份清单**：基座/LoRA/merged权重哈希、scale、seed、代码版本、包ID与输出图哈希之间的映射。首轮保存了部分runner/hash，但尚未覆盖生成来源链。
2. **离线与runner差异表**：VAE mode/sample、float32/fp16、DINO图块、对齐、PNG/JPEG、随机数生命周期；仅写“同主体”掩盖这些差异。
3. **完整配置来源优先级**：args.json与CLI默认值如何覆盖；INF会由CLI非None默认值覆盖JSON；某项命令省略不代表沿用训练配置。
4. **候选可用性**：v1–v3只有历史结果、v5不同命名；当前扫描器不能从目录名推断可执行候选。
5. **测量协议manifest**：样本ID、自然/字典序、原图/resize/crop/tiled、图像格式、指标版本、有效n、失败项、设备/空闲态与计时边界。
6. **未完成扫描状态**：Q10计划6配置，现只有2成功/1失败汇总，D已有部分图；应区分计划、部分产物和完成。
7. **已有工具缺口**：GATE忽略子退出码、EV吞缺项/覆盖重复case、QM的GT通配符未展开、QF配对身份错误；本轮只记录，不修改。
8. **来源与许可证清单**：外部模型/数据/代码的版本和使用条件；“全自研/开源”不能替代该清单。
9. **恢复与安全边界**：哪些关键资产未入Git、哪些已删除无Git记录、凭据暴露范围与撤销状态。未发现记录不等于不存在外部备份。
10. **审查覆盖范围**：静态/运行/归档包/现工作目录必须分别标记。当前包内容能验证，不意味着历史原包一直相同；未验证项目不能由PASS自动扩展。

## G. ASSUMPTION / UNKNOWN

**ASSUMPTION（不能升级为事实）**：官方固定开销与全局加速比上限；Q与速度完全可分离；MANIQA回归足以选优；有参权重零/负；对手高分来自某配方或作弊；未标fraud必然只是尚未复核；wavelet精确还原原配方；测试退化的估计等于官方生成器；既有失败证明全部同类路线不可能。

**UNKNOWN（现有材料不能闭合）**：最新官方规则/排名/复核/额度；JIT形式是否有书面例外；官方计时与IQA预处理；现场设备/时限；旧云资源真正关机与凭据有效性；完整历史M24生成manifest；v1–v3是否有其他位置备份；训练数据独立样本数；部分已删除文档的外部备份；DeepSeek此次独立onboarding的完整报告是否另存于未纳入本轮文件清单的位置。最后一项不阻塞本报告，也未把“独立完成onboarding”当成双方结论已相同。

其中环境、路径、代码行为已经能核验的部分均已自行核验，不列入用户裁决。需要官方回复的是外部事实，不应由用户凭偏好选择“事实版本”。

## H. 真正需要用户裁决的事项

| 事项 | 已知事实与问题 | 可选边界 / Architect建议 | 代价、风险与未知 |
|---|---|---|---|
| H1 未来是否继续D3解耦策略 | 历史D3风险接受与现行D12/T同主体要求并存，D14转硬实力 | 建议明确“D3只记录历史；未来新提交按同主体”；另一选择是保持既有风险接受，但不得称官方许可或自动执行 | 决定后续提交范围，不自动撤回既有提交；规则执行粒度仍UNKNOWN |
| H2 L4适用边界 | 五种行为被同一标签混写：普通评估、全局选模型、IQA损失训练、逐图择优、像素对抗；v7已有IQA训练历史 | 请分别确定允许/禁止，尤其现有v7是否可继续使用；可禁止新增IQA优化而允许既有权重评估，或采取更严格范围 | 严格追溯会影响主力可用性；允许使用不构成官方合规认证；未自行废弃或重训 |
| H3 已提交风险件与规则澄清的外部处置 | 两高分件不同主体且沿用同主体注释；M24非JIT | 建议先获取官方对同主体/JIT的明确口径，再决定是否补充说明、替换或撤回；也可暂不行动但保留风险 | 这些操作影响外部提交/资格，需用户授权；只改本地注释不能改变历史提交事实或合规性质 |
| H4 统一知识基线方案 | 六类标签、文档职责、台账唯一维护位置尚未正式接受 | 确认D/I建议后再下发文档整合；可先只统一现行六文档，历史只加索引/勘误 | 不需要修改业务接口/JSON schema；涉及AGENTS结构和历史移动/删除时仍须另行明确授权 |

本轮不请求用户裁定：Python版本、门禁项数、v5文件名、v1–v3本目录有无权重、VAE mode/sample、SP参数、QF锚点错配、旧TODO是否已有记录——这些已有代码/文件证据。安全凭据若需轮换和历史清理，是后续独立授权工作，不能混在文档美化里执行。

## I. 推荐的统一文档基线（Recommendation，尚未执行）

| 载体 | 应保留 | 应移出/合并/限定 |
|---|---|---|
| README | 项目目标、交付接口概览、当前主力一句话、可验证启动入口、规则来源与关键风险链接 | 不再维护完整成绩表/公式证明/长任务日志；动态摘要加日期；不把合规/GT天花板写成已证明 |
| architecture | 当前实际模块关系、数据流、接口、配置/精度/随机性/离线行为；认可设计与实装差异分栏 | 评分推断移reference；审查清单转review引用；历史A7等关闭并指向历史，不与新正文并存 |
| project_map | 当前仓库/外部资产路径、文件存在性、版本/哈希入口、权重可用性 | 不抄成绩、实验结论、任务计划；不存在路径标missing并保留恢复线索 |
| decisions | 已获确认的决定、决定人/日期、作用范围、理由证据、被何决定取代 | 实验结果不自动成为DECISION；D3/D12/D14等待H1/H2明确后建立效力关系 |
| progress | 当前里程碑状态、产物链接、已验证范围、失败/阻塞；成绩从指定台账引用 | 删除或降级重复的“下一步战略”，旧阶段任务转HISTORICAL；不把日志存在等同运行中 |
| current_task | 唯一当前获授权任务、候选资产清单、输入输出、预登记协议、成功/负结果/暂停标准、责任与审批 | 本轮当前任务就是事实对账；研究任务保留待恢复计划，不据旧“进行中”自动开工 |
| review | Architect发现、证据、冲突ID、风险、用户裁决请求、解决状态与验证范围；本报告归此 | 9/15自审保留历史署名；不要把自审pending与正式审计并列为同一当前状态 |
| reference | 可靠原始规则/样例来源指针、日期化转述、版本化实验说明、维护中的提交事件台账 | ANN战略分析、旧RUNBOOK判别树、调研旧推荐应分离或降级，不以“长期有效”总标签覆盖整篇 |
| handoff | 不可改写的阶段记录、历史指令和当时证据 | 继续保留原文；索引注明“不作为当前执行指令”、被取代关系；不要定义为“内容当时必真” |
| archive | 已停止使用的操作手册/旧方案/旧公式解释，保留原出处与替代入口 | 进入archive不等于所有科学判断已被证伪，也不自动授权删除 |

建议待确认后进行的材料处置：

- ANN保留公告转述作为证据，其旧分数/云训练/行动部分降级历史；不要把转述重新包装成官方原文。
- RUNBOOK先抽取仍与代码一致的提交流程，旧公式判读/旧波次操作部分移archive或保留显式历史区。
- `research_diffusion_candidates_2026.md`保留带日期的调研，旧推荐降级；使用外部模型前另查官方来源，不在本轮伪造最新结论。
- WR保留为旧自训模块说明，并指向当前项目入口；不要删除仍能解释LM/训练脚本的内容。
- SCORE与LED需要明确一个提交事件主台账；建议LED承担结构化事件、SCORE只作生成/引用视图，但这是维护方案建议，未修改数据模型。ASSET保留资产索引，不再竞争成绩真源。
- 首轮A/P审计段落与本报告以后通过问题ID互链；核准基线时应重写当前摘要并把旧断言降级，不能继续层层追加相反结论。
- **本轮不建议直接删除任何历史文件。** 删除重复正文、移动材料或调整G需在用户确认整合方案后执行；凭据需要撤销/轮换而非仅删文本。

### 本轮核验补记

QF只读复算输出：9个有效拟合点，probeA n=0被跳过、TinyV2缺失；MANIQA单指标系数118.2940、截距−15.26、LOOCV1.78。该结果复现的是现有输入表，**包含TR36的身份错配**，不能作为新可靠基线。GT_val未进入输出。

本轮源码SHA-256（固定本报告讨论的外部版本；未修改）：

```text
vosr_lora_infer2.py 11b664d403f3b662067f7e221badb2c8724d8d3c3b0b825398985be0d66fb94b
train_vosr_lora.py  61409f0ecd785a0a5070d234d7d156e97ffe483e4ff1b650511421800cddee46
preupload_gate.py   acff56c784e703a2cb8513519098098cfc08ead8c0047c15840e84b1014c8478
score_pred2.py      38e27eeda883279b1b674a99071d3035524de6ab4a4c15f6c62385937b949a15
```

**结论：已建立可追溯的对账清单，但尚未宣称所有冲突已解决或项目已获开发批准。** 60项中纯事实部分可作为后续整合依据；H1–H4保持待用户确认。完成本报告后停止，不进入实现阶段。
