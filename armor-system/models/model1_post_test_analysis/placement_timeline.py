"""Per-sample pod->VM placement timeline from the D1 sample graphs. Read-only, non-executing.

Pickles are read with the repository's pickle_symbolic (opcode interpreter, no imports or calls) and edge tensors are parsed as
raw bytes by torch_blob; pickle.load is never used. Labels are never read here.

    .venv/bin/python -I placement_timeline.py TOOLS_DIR D1_DIR OUT_JSON
"""
import hashlib
import json
import os
import sys

TOOLS, D1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, TOOLS)
import numpy as np  # noqa: E402
import pickle_symbolic as ps  # noqa: E402
import torch_blob as tb  # noqa: E402

nh = ps.read_symbolic(os.path.join(D1, "hash_info", "node_hash.pkl"))
assert not ps.global_refs(nh), "node_hash references globals"
names = [n for n, _ in sorted(nh.items(), key=lambda kv: kv[1])]
is_vm = np.array([n.startswith("node-") for n in names])
pods = [n for i, n in enumerate(names) if not is_vm[i]]


def edges(g):
    st = g.state["_graph"].state.state
    s, d = (tb.tensor_to_array(t, np) for t in st[2])
    return s.astype(int), d.astype(int)


rows, maps = [], {}
for split in ("train", "test"):
    raw = ps.read_symbolic(os.path.join(D1, "samples", f"{split}_samples.pkl"), keep_bytes=True)
    for t, g, _x in raw:
        s, d = edges(g)
        nb = {}
        for a, b in set(zip(s.tolist(), d.tolist())):
            if is_vm[a] != is_vm[b]:
                p, v = (a, b) if not is_vm[a] else (b, a)
                nb.setdefault(names[p], set()).add(names[v])
        if any(len(v) > 1 for v in nb.values()):
            raise RuntimeError("a pod with more than one VM neighbour")
        placement = {p: sorted(nb[p])[0] for p in pods if p in nb}
        key = hashlib.md5(json.dumps(sorted(placement.items())).encode()).hexdigest()[:10]
        maps.setdefault(key, placement)
        rows.append({"ts": int(t), "split": split, "edge_set": hashlib.md5(s.tobytes() + d.tobytes()).hexdigest()[:10],
                     "placement": key, "n_pods_placed": len(placement), "complete": len(placement) == len(pods)})
    del raw
rows.sort(key=lambda r: r["ts"])
with open(OUT, "w") as fh:
    json.dump({"pods": pods, "vms": [n for i, n in enumerate(names) if is_vm[i]], "maps": maps, "samples": rows}, fh)
print(len(rows), "samples;", len(maps), "distinct placements (incl. partial);", sum(r["complete"] for r in rows), "complete samples")
