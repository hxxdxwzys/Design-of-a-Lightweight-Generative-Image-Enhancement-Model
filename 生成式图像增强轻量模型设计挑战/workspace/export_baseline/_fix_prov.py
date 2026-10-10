# -*- coding: utf-8 -*-
"""把 phase1_provenance.json 规范化为无 BOM 的 UTF-8 JSON（PS 5.1 的 Out-File 会写 BOM）。"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
txt = raw.decode('utf-8-sig')
obj = json.loads(txt)
p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'fixed': str(p), 'n_files': len(obj.get('files', {})),
                  'bytes': p.stat().st_size}, ensure_ascii=False))
