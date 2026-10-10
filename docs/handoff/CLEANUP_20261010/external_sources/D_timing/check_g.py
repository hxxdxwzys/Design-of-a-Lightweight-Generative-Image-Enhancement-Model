# -*- coding: utf-8 -*-
"""核对 graph runner 与 eager 是否真的等价 (std 0.364 可疑)"""
import sys, torch
import torch.nn.functional as F
MD = r'D:/models/pkg_TinyV2g/model_dir'
sys.path.insert(0, MD)
import runner as R
r = R.Runner(MD)
print('graph =', r.graph is not None, 'native =', R._NATIVE, 'down =', R._DOWN)
x = (torch.rand(1, 3, 512, 512) * 2 - 1)
with torch.no_grad():
    y_graph = r.infer(x)
    # 手动 eager 路径
    xs = x.to('cuda').half()
    xs = F.interpolate(xs, size=(256, 256), mode='bicubic', align_corners=False)
    y_eager_model = r.m(xs)                       # 直接 traced 模块
    y_eager_full = F.interpolate(y_eager_model, size=(512, 512), mode='bicubic', align_corners=False).float().clamp(-1,1)
    # graph 路径
    r.static_in.copy_(xs); r.graph.replay(); torch.cuda.synchronize()
    y_graph_raw = r.static_out.clone()
    y_graph_full = F.interpolate(y_graph_raw, size=(512, 512), mode='bicubic', align_corners=False).float().clamp(-1,1)
print('runner.infer      std =', float(y_graph.std()))
print('eager full        std =', float(y_eager_full.std()))
print('graph raw vs eager raw maxdiff =', float((y_graph_raw - y_eager_model).abs().max()))
print('runner.infer vs eager full maxdiff =', float((y_graph - y_eager_full).abs().max()))
print('static_in std =', float(r.static_in.std()), ' (输入 std =', float(xs.std()), ')')
print('CHECK_G_DONE')
