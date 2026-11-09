import json, os, sys
from collections import defaultdict
sys.path[:0] = [os.path.expanduser('~/298A_project/team-repo/armor-system/data')]
from prep import loader
a = json.load(open('analysis.json')); m = json.load(open('out/meta.json'))
L = loader.read_labels()['cases']
rec = [json.load(open(f'out/recon_seed_{s}.json')) for s in range(5)]
wrong = defaultdict(int)
for k in ('across', 'within'):
    for c, v in a['fault']['v1'][k]['cases'].items():
        wrong[int(c)] += v['n_seeds']
agg = defaultdict(lambda: [0, 0, 0, 0])
for c in m['test_cases']:
    r = L[c]
    key = r['cmdb_id'] if r['level'] == 'node' else r['level']
    t = str(m['cases']['timestamp'][c])
    agg[key][0] += 1
    agg[key][1] += sum(not x['detail']['ours/primary']['labels'][t]['hit'] for x in rec)
    agg[key][2] += wrong[c]
    agg[key][3] += sum(not x['detail']['ours/d3_m2_p90']['labels'][t]['hit'] for x in rec)
for k, v in sorted(agg.items()):
    print(f"{k}: cases {v[0]}, primary miss obs {v[1]}/{5*v[0]}, D3p90 miss obs {v[3]}/{5*v[0]}, v1 fault-error obs {v[2]}/{5*v[0]}")
