import os, re, sys, socket
import paramiko

SEC = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218" + "\\.secrets"
raw = open(SEC, "rb").read()
for enc in ("utf-8", "gbk", "utf-8-sig"):
    try:
        text = raw.decode(enc); used = enc; break
    except Exception:
        continue
print("secrets decoded as", used, "len", len(text))

entries = []
cur = {}
for line in text.splitlines():
    if "=" in line:
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if "\u7528\u6237" in k: cur["user"] = v
        elif "\u5bc6\u7801" in k:
            if cur.get("user"):
                entries.append((cur.get("user"), v, cur.get("port")))
            cur = {}
        elif "\u7aef\u53e3" in k: cur["port"] = v
print("credential blocks found:", [(u, "pw" if p else "-", pt) for u, p, pt in entries])

HOST = "10.96.138.171"
cands = [(8022, "tempuser"), (9077, "tempuser"), (8022, None), (2222, "tempuser")]
pw_by_user = {}
for u, p, _ in entries:
    pw_by_user.setdefault((u or "").strip(), p)

def try_ssh(port, user, pw, cmds, timeout=12):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=port, username=user, password=pw, timeout=timeout,
              banner_timeout=timeout, auth_timeout=timeout, allow_agent=False, look_for_keys=False)
    out = []
    for cmd in cmds:
        si, so, se = c.exec_command(cmd, timeout=20)
        out.append((cmd, so.read().decode("utf-8", "replace").strip(), se.read().decode("utf-8", "replace").strip()[:200]))
    c.close()
    return out

for port, user in cands:
    user = user or "tempuser"
    pw = pw_by_user.get(user)
    if pw is None:
        print(f"[{port}/{user}] no stored password -> skip"); continue
    try:
        res = try_ssh(port, user, pw, ["hostname", "whoami"])
        print(f"[{port}/{user}] AUTH OK")
        for cmd, o, e in res:
            print(f"   $ {cmd} -> {o}" + (f" | err: {e}" if e else ""))
        break
    except paramiko.AuthenticationException:
        print(f"[{port}/{user}] auth FAILED (password rejected)")
    except Exception as e:
        print(f"[{port}/{user}] {type(e).__name__}: {str(e)[:120]}")
