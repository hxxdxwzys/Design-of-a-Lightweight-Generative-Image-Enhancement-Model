# 研究候选，不是正式复核交付

V3保真微调，2026-09-27，3000步。仅改变V3既有细化器权重，没有新增模块，UNet与原计算图保持。

完整3张4K验证：PSNR均值23.70575→24.03399 dB；LPIPS均值0.397719→0.395456。保真度有小幅收益，但LPIPS改善约0.57%，未达预登记2%交付门槛。一张图LPIPS轻微退化。

状态：**RESEARCH_ONLY / NOT_APPROVED_FOR_DELIVERY**。没有完成test30退化警戒、时延门槛、100图生产或新版复核材料。不得称官方分数已超过V3，也不应作为已验收包直接提交。

最终模型文件跨独立进程的固定3输入8位结果完全一致；UNet/非ref状态精确保持，ref权重精确对应checkpoint转half。内存注入参考与保存文件的额外对照仍不完全相等，根因UNKNOWN，不能声称eager等价。

模型SHA256：6166649cdf34669810c7e666e65c9ec27ab140c08187f5c5f3055198bfc6e351。

完整证据位于本目录上两级，主要为full_val.json、holdout_3000.json、reload_gate.json、final_status.json。原V3正式复核包保持不变。
