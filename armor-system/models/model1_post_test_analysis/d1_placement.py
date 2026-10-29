"""Derive pod->VM placement from D1 sample graphs. Read-only, non-executing.
Pickles are read with the repo's pickle_symbolic (opcode interpreter, no imports/calls);
edge tensors parsed as raw bytes by torch_blob. pickle.load is never used.
Usage: python -I d1_placement.py TOOLS_DIR D1_DIR
"""
import datetime, hashlib, json, os, sys
from collections import Counter, defaultdict
TOOLS, D1 = sys.argv[1], sys.argv[2]
sys.path.insert(0, TOOLS)
import numpy as np
import pickle_symbolic as ps
import torch_blob as tb

nh = ps.read_symbolic(os.path.join(D1, "hash_info", "node_hash.pkl"))
assert not ps.global_refs(nh), "node_hash references globals"
names = [n for n, _ in sorted(nh.items(), key=lambda kv: kv[1])]
assert sorted(nh.values()) == list(range(len(names)))
is_vm = np.array([n.startswith("node-") for n in names])
print("nodes:", len(names), "VMs:", [n for n in names if n.startswith("node-")])

def edges(g):
    st = g.state["_graph"].state.state
    s, d = (tb.tensor_to_array(t, np) for t in st[2])
    return s.astype(int), d.astype(int)

def iso(t): return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

sets = {}  # key -> dict
for split in ("train", "test"):
    raw = ps.read_symbolic(os.path.join(D1, "samples", f"{split}_samples.pkl"), keep_bytes=True)
    for t, g, _x in raw:
        s, d = edges(g)
        k = hashlib.md5(s.tobytes() + d.tobytes()).hexdigest()[:10]
        e = sets.setdefault(k, {"src": s, "dst": d, "n": Counter(), "ts": []})
        e["n"][split] += 1; e["ts"].append(int(t))
    del raw

placements = {}
for k, e in sorted(sets.items(), key=lambda kv: min(kv[1]["ts"])):
    s, d = e["src"], e["dst"]
    pairs = set(zip(s.tolist(), d.tolist()))
    cat = Counter(("vm" if is_vm[a] else "svc") + "->" + ("vm" if is_vm[b] else "svc") for a, b in zip(s, d))
    vmnb = defaultdict(set)
    for a, b in pairs:
        if is_vm[a] != is_vm[b]:
            svc, vm = (a, b) if not is_vm[a] else (b, a)
            vmnb[names[svc]].add(names[vm])
    svcs = [n for i, n in enumerate(names) if not is_vm[i]]
    counts = Counter(len(vmnb.get(n, ())) for n in svcs)
    sym = all((b, a) in pairs for a, b in pairs)
    pmap = {n: sorted(vmnb[n])[0] for n in svcs if len(vmnb.get(n, ())) == 1}
    placements[k] = {n: tuple(sorted(vmnb.get(n, ()))) for n in svcs}
    print(f"\nedge set {k}: edges={len(s)} unique_pairs={len(pairs)} selfloops={sum(a==b for a,b in pairs)} "
          f"symmetric={sym} samples={dict(e['n'])} span={iso(min(e['ts']))}..{iso(max(e['ts']))} UTC")
    print("  edge categories:", dict(cat))
    print("  #VM neighbours per service instance -> #services:", dict(sorted(counts.items())))
    if any(c != 1 for c in counts.elements() if False): pass
    odd = {n: sorted(vmnb.get(n, ())) for n in svcs if len(vmnb.get(n, ())) != 1}
    if odd: print("  services without exactly one VM neighbour:", odd)
    isolated = [names[i] for i in range(len(names)) if i not in set(s.tolist()) | set(d.tolist())]
    if isolated: print("  isolated nodes:", isolated)

keys = list(placements)
ref = placements[keys[0]]
print("\nplacement map (first edge set):")
by_vm = defaultdict(list)
for n, v in ref.items(): by_vm[v].append(n)
for v in sorted(by_vm): print(" ", v, "<-", sorted(by_vm[v]))
print("\nper edge set: VM -> hosted service instances (UTC+8 day counts):")
for k in keys:
    days = Counter(datetime.datetime.fromtimestamp(t + 8*3600, datetime.timezone.utc).strftime("%m-%d") for t in sets[k]["ts"])
    g = defaultdict(list)
    for n, v in placements[k].items(): g[v[0] if v else "NONE"].append(n)
    print(f"  {k} days(UTC+8)={dict(sorted(days.items()))}")
    for v in sorted(g): print(f"     {v} ({len(g[v])}): {sorted(g[v])}")
distinct = {tuple(sorted(m.items())) for m in placements.values() if all(len(x)==1 for x in m.values())}
print("\ndistinct complete (all 40 have one VM) placement maps:", len(distinct))

ss = {k: frozenset((a, b) for a, b in zip(e["src"].tolist(), e["dst"].tolist()) if not is_vm[a] and not is_vm[b] and a != b) for k, e in sets.items()}
print("\nsvc->svc non-self call edges per set:", {k: len(v) for k, v in ss.items()}, "identical across all sets:", len(set(ss.values())) == 1)
vv = {k: sorted((names[a], names[b]) for a, b in zip(e["src"].tolist(), e["dst"].tolist()) if is_vm[a] and is_vm[b]) for k, e in sets.items()}
print("vm->vm edges are self-loops only:", all(a == b for v in vv.values() for a, b in v))
