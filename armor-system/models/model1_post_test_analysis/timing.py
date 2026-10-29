"""Placement-change timing vs case injections, gaps, partial samples (reads the timeline and the label timestamps only)."""
import json, os, sys
from datetime import datetime, timezone
sys.path[:0] = [os.path.expanduser('~/298A_project/team-repo/armor-system/data')]
import numpy as np
from prep import loader
T = json.load(open('timeline.json'))
S = T['samples']; maps = T['maps']
L = loader.read_labels()
cases = np.array(sorted(c['timestamp'] for c in L['cases'])); ad = np.array(sorted(L['ad']))
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
ts = np.array([r['ts'] for r in S])
gaps = [(int(ts[i]), int(ts[i + 1])) for i in range(len(ts) - 1) if ts[i + 1] - ts[i] != 60]
out = {'n_samples': len(S), 'first': iso(ts[0]), 'last': iso(ts[-1]), 'step_60s_except_gaps': [(iso(a), iso(b), b - a) for a, b in gaps]}
comp = [r for r in S if r['complete']]
changes = []
for a, b in zip(comp, comp[1:]):
    if a['placement'] != b['placement']:
        lo, hi = a['ts'], b['ts']
        ent = {'last_before': iso(lo), 'first_after': iso(hi), 'bracket_s': hi - lo, 'from': a['placement'], 'to': b['placement'],
               'in_data_gap': any(g[0] == lo and g[1] == hi for g in gaps), 'moved_pods': sorted(p for p in T['pods'] if maps[a['placement']][p] != maps[b['placement']][p])}
        for nm, arr in (('cases_csv', cases), ('ad_cases', ad)):
            # distance from an injection to the change bracket (0 if inside); injections BEFORE the change
            before = arr[arr <= hi]
            d_before = np.maximum(lo - before, 0)
            d_all = np.where(arr < lo, lo - arr, np.where(arr > hi, arr - hi, 0))
            ent[nm] = {'min_s': int(d_all.min()), 'median_s': float(np.median(d_all)),
                       'nearest_before_s': int(d_before.min()) if len(before) else None,
                       'injections_within_30min_before_change': int((d_before <= 1800).sum()) if len(before) else 0,
                       'injections_inside_bracket': int(((arr >= lo) & (arr <= hi)).sum())}
        changes.append(ent)
out['placement_changes'] = changes
part = []
for i, r in enumerate(S):
    if r['complete']:
        continue
    prev = next((S[j] for j in range(i - 1, -1, -1) if S[j]['complete']), None)
    nxt = next((S[j] for j in range(i + 1, len(S)) if S[j]['complete']), None)
    pm = maps[prev['placement']] if prev else None
    m = maps[r['placement']]
    missing = sorted(set(T['pods']) - set(m))
    part.append({'ts': iso(r['ts']), 'split': r['split'], 'pods_without_vm': len(missing),
                 'vms_of_missing_pods_in_prior_map': sorted({pm[p] for p in missing}) if pm else None,
                 'prior_complete': iso(prev['ts']) if prev else None, 'prior_gap_s': r['ts'] - prev['ts'] if prev else None,
                 'next_complete': iso(nxt['ts']) if nxt else None,
                 'present_pods_agree_with_prior': all(pm[p] == v for p, v in m.items()) if pm else None,
                 'prior_equals_next_placement': (prev['placement'] == nxt['placement']) if prev and nxt else None,
                 'nearest_case_s': int(np.abs(cases - r['ts']).min()), 'nearest_ad_s': int(np.abs(ad - r['ts']).min())})
out['partial_samples'] = part
hosts = {}
for k, m in maps.items():
    for p, v in m.items():
        hosts.setdefault(v, set()).add(p)
out['pods_per_vm_any_placement'] = {v: len(hosts.get(v, ())) for v in T['vms']}
out['complete_maps'] = sorted({r['placement'] for r in comp})
# per-day placement (UTC+8 day)
day = {}
for r in comp:
    d = datetime.fromtimestamp(r['ts'] + 8 * 3600, timezone.utc).strftime('%m-%d')
    day.setdefault(d, set()).add(r['placement'])
out['complete_placements_per_day_utc8'] = {d: sorted(v) for d, v in sorted(day.items())}
out['vm_pod_counts_per_map'] = {k: {v: sum(x == v for x in maps[k].values()) for v in T['vms']} for k in out['complete_maps']}
json.dump(out, open('placement_timing.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
