import socket, os, re, json, sys

HOST = "10.96.138.171"
PORTS = [22, 2222, 3389, 2200, 20022, 22222, 222, 8022, 80, 443]

def probe(port, timeout=2.5):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((HOST, port))
        try:
            banner = s.recv(64)
        except Exception:
            banner = b""
        s.close()
        return True, banner[:60]
    except Exception as e:
        return False, type(e).__name__
    finally:
        try: s.close()
        except Exception: pass

print("== port probe on", HOST, "==")
for p in PORTS:
    results = [probe(p) for _ in range(2)]
    ok = any(r[0] for r in results)
    detail = [r[1] for r in results if r[0]]
    print(f"port {p:6d}: {'OPEN ' if ok else 'closed'} {detail if detail else results[0][1]}")

print()
print("== .secrets ==")
root = r"C:\生成式图像增强轻量模型设计挑战"
cands = [os.path.join(root, ".secrets"), os.path.join(root, "workspace", ".secrets")]
for c in cands:
    print(c, "exists" if os.path.exists(c) else "MISSING")
    if os.path.exists(c):
        with open(c, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip(); v = v.strip()
                    kind = "other"
                    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", v): kind = "IP"
                    elif re.fullmatch(r"\d{2,5}", v): kind = "PORTNUM"
                    elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,30}", v): kind = "WORD"
                    shown = v if kind in ("IP", "PORTNUM") else (v[:3] + "***" + f"(len={len(v)})")
                    print(f"  [{i}] {k} = {shown}   <{kind}>")
                else:
                    print(f"  [{i}] (non-kv, len={len(line)})")

print()
print("== tooling ==")
for p in [r"D:\tmp_timing\ssh_helper.py", r"D:\tmp_timing\ssh_cat_upload.py", r"D:\tmp_timing"]:
    print(p, os.path.exists(p))
try:
    import paramiko
    print("paramiko", paramiko.__version__)
except Exception as e:
    print("paramiko MISSING:", e)
