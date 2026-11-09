import json, os, sys
sys.path[:0] = [os.path.expanduser('~/298A_project/team-repo/armor-system/data')]
import numpy as np
from prep import loader
T = json.load(open('timeline.json')); S = T['samples']; maps = T['maps']
L = loader.read_labels()
cs = L['cases']; cts = np.array([c['timestamp'] for c in cs])
for i, r in enumerate(S):
    if r['complete']:
        continue
    prev = next(S[j] for j in range(i - 1, -1, -1) if S[j]['complete'])
    lost_vm = sorted({maps[prev['placement']][p] for p in set(T['pods']) - set(maps[r['placement']])})
    j = int(np.abs(cts - r['ts']).argmin()); c = cs[j]
    print(r['ts'], 'lost', lost_vm, '| nearest case', r['ts'] - c['timestamp'], 's after injection;', c['level'], c['cmdb_id'], c['failure_type'],
          '| VM-level fault on the same VM:', c['cmdb_id'] in lost_vm)
# baseline: fraction of all sample minutes within 203 s after/around an injection
ts = np.array([r['ts'] for r in S])
d = np.array([np.abs(cts - t).min() for t in ts])
print('fraction of all 7,200 minutes within 203 s of a case injection:', round(float((d <= 203).mean()), 4))
