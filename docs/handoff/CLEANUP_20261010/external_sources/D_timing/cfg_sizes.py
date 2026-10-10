import json
from pathlib import Path
for s in ('s','b','l','xl'):
    p = Path(rf'D:/models/newsrc/PixRestore/pixrestore-{s}/config.json')
    c = json.loads(p.read_text(encoding='utf-8'))
    print(f"{s:3s} hidden={c['hidden_size']:5d} depth={c['depth']:3d} heads={c['num_heads']:3d} "
          f"encoder={c['encoder_type']:9s} enc_dim={c['encoder_dim']:5d} layers={c['encoder_layers']}")
