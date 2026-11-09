import json, os, sys
from collections import Counter
sys.path[:0] = [os.path.expanduser('~/298A_project/team-repo/armor-system/data')]
from prep import loader
T = json.load(open('timeline.json')); S = T['samples']
part = [r['ts'] for r in S if not r['complete']]
cs = loader.read_labels()['cases']
ft = Counter(c['failure_type'] for c in cs)
disk = [c['timestamp'] for c in cs if c['failure_type'] == 'Node Disk Space Consumption']
tsset = {r['ts'] for r in S}
covered = [t for t in disk if any(t <= u <= t + 300 for u in tsset)]
with_partial = [t for t in disk if any(0 <= p - t <= 300 for p in part)]
print('fine failure types:', len(ft), '; Node Disk Space Consumption cases:', len(disk), '; with sample minutes in the next 300 s:', len(covered), '; followed by a partial-edge sample within 300 s:', len(with_partial))
