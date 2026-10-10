# -*- coding: utf-8 -*-
"""DELIV-02 定位：4K 迁移偏差的空间分布与分块/后处理相关性（CPU，只读已有数组）。"""
import json
import sys
from pathlib import Path

import numpy as np

NPY = Path("C:/models/exports/DELIV02_20260918_073810/npy")
BASE = Path("D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01/attempt_01_072821_A_case1_full_r2.npy")


def main():
    out = Path(sys.argv[1])
    ours = np.load(NPY / "case1_full_A.npy", allow_pickle=False)[0].transpose(1, 2, 0)
    ref = np.load(BASE, allow_pickle=False)[0].transpose(1, 2, 0)
    d = np.abs(ours - ref)
    res = {"schema": "DELIV02-localize/1", "shape": list(d.shape),
           "max_abs": float(d.max()), "mean_abs": float(d.mean()),
           "n_gt_0.05": int((d > 0.05).sum()), "n_gt_0.2": int((d > 0.2).sum()),
           "n_gt_0.01": int((d > 0.01).sum())}
    md = d.max(axis=2)
    ys, xs = np.where(md > 0.05)
    res["bbox_gt_0.05"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    # 与分块边界（DiT stride 448 / VAE stride 896）的邻近关系
    def near_boundary(coords, stride, tol=4):
        n = 0
        for c in coords:
            r = c % stride
            if r <= tol or r >= stride - tol:
                n += 1
        return n
    res["near_dit_stride448"] = {"x": near_boundary(xs, 448), "y": near_boundary(ys, 448),
                                 "of": int(len(xs))}
    res["near_vae_stride896"] = {"x": near_boundary(xs, 896), "y": near_boundary(ys, 896),
                                 "of": int(len(xs))}
    # 聚类：连通域数量与大小
    try:
        import cv2
        mask = (md > 0.05).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = sorted([int(s) for s in stats[1:, 4]], reverse=True)
        res["clusters"] = {"n": int(n - 1), "top_sizes": sizes[:10]}
        cents = [tuple(map(int, c)) for c in cv2.connectedComponentsWithStats(mask)[3][1:]]
        res["cluster_centroids_top10"] = [list(c) for c in cents[:10]]
    except Exception as e:
        res["clusters"] = {"error": str(e)[:120]}
    # 偏差与图像高频/边缘的相关性（判断是否出现在纹理细节处）
    gray = ours.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gy ** 2 + gx ** 2)
    if len(xs):
        res["edge_at_diff"] = {"mean": float(edge[ys, xs].mean()),
                               "image_mean": float(edge.mean())}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("max_abs", "mean_abs", "n_gt_0.01", "n_gt_0.05",
                                          "n_gt_0.2", "bbox_gt_0.05", "near_dit_stride448",
                                          "near_vae_stride896", "clusters", "edge_at_diff")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
