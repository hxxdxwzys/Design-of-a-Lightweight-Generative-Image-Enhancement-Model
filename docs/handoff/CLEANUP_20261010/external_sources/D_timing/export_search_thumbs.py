# -*- coding: utf-8 -*-
"""为溯源导出测试集代表性缩略图 (给用户做以图搜图找回原图)
选 6 张内容辨识度高的测试 LQ 图, 缩到最长边 1024, 存 D:/tmp_timing/for_search/
"""
from pathlib import Path

from PIL import Image

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(r'D:/tmp_timing/for_search')
PICKS = ['case1', 'case5', 'case9', 'case14', 'case23', 'case31']  # 内容多样性


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in PICKS:
        p = TEST_DIR / f'{name}.jpg'
        if not p.exists():
            continue
        im = Image.open(p).convert('RGB')
        im.thumbnail((1024, 1024), Image.LANCZOS)
        im.save(OUT / f'{name}_thumb.jpg', quality=95)
        print(f'[ok] {name}_thumb.jpg {im.size}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
