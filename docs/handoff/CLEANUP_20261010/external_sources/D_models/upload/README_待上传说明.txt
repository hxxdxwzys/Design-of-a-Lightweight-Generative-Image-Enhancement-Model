待上传（2026-09-20 02:2x 更新）—— 合规线五代 + 解耦两件

★ 最终版（建议明早提交）：Track2_TinyV21.zip
   clipiqa 0.6306 / niqe 5.4843；同进程 512 时延 4.88 ms（r=41.5）⇒ 预测 50.9（保守映射）~ 58.9（乐观映射）
   合规：图与 runner 同一模型主体（runner↔ckpt fp16 同路 max u8 = 1）；PREUPLOAD_GATE_PASS

合规线其余各代（同族，可作对照/备份）：
   Track2_TinyV11b.zip   clipiqa 0.5481（预测 45.5~48.0）
   Track2_TinyV18.zip    clipiqa 0.5715（预测 47.3~51.5）
   Track2_TinyV19.zip    clipiqa 0.5876（预测 48.5~53.8）
   Track2_TinyV20b.zip   clipiqa 0.6128（预测 49.0~55.7）

解耦件（已提交，未出分；规则风险见 docs/handoff/P10003）：
   Track2_OPT10img_V4grun.zip    预测 103~109（S2+S3）
   Track2_OPT10img_V7ganrun.zip  预测  94~99（S2）

读数与日志：D:\models\opt11\{test30_*.json, ab*.log, consistency_final.log, gate_*.txt, build_*.log, prereg_*.txt, cleanup_log.json}
细化说明：docs/handoff/P10006_20260920_今晚最终版交付说明.md
