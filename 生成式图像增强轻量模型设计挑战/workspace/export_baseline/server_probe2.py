import socket, subprocess, concurrent.futures as cf

def port_open(host, port, t=2.5):
    s = socket.socket(); s.settimeout(t)
    try:
        s.connect((host, port))
        try: b = s.recv(64)
        except Exception: b = b""
        return True, b[:40]
    except Exception as e:
        return False, type(e).__name__
    finally:
        try: s.close()
        except Exception: pass

print("== DNS: bigdata.dynu.net ==")
try:
    ips = sorted({ai[4][0] for ai in socket.getaddrinfo("bigdata.dynu.net", None)})
    print("resolved:", ips)
except Exception as e:
    ips = []
    print("resolve FAILED:", type(e).__name__, e)
for ip in ips:
    for p in (9077, 22, 2222):
        ok, d = port_open(ip, p)
        print(f"  {ip}:{p} -> {'OPEN ' + str(d) if ok else 'closed ' + str(d)}")

print()
print("== extra SSH-ish ports on 10.96.138.171 ==")
cands = [2223, 2221, 2022, 6022, 10022, 9022, 7022, 8022, 22022, 50022]
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(port_open, "10.96.138.171", p, 2.0): p for p in cands}
    for f in cf.as_completed(futs):
        p = futs[f]; ok, d = f.result()
        if ok: print(f"  port {p}: OPEN  {d}")

print()
print("== TTL fingerprint (ping) ==")
r = subprocess.run(["ping", "-n", "3", "10.96.138.171"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
for line in (r.stdout or "").splitlines():
    if "TTL" in line.upper() or "ttl" in line:
        print("  ", line.strip())
