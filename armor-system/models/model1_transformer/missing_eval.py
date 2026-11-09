"""Missing-data validation masks (V1_PROTOCOL.md section 7): ARMOR's six scenarios and the random-rate masks. Validation experiments only.

A mask is an `observed_entry` array over the whole sample timeline (samples, nodes, channels); `masked_view` returns a shallow copy of the
verified `Model1Data` whose windows then carry it (inputs AND the target minute), so effective = applicable & observed everywhere.
  * ARMOR scenarios: every channel of the dropped modalities hidden for every node and minute (ARMOR's `get_channel_indices_to_mask`,
    on our schema's `modality_index_ranges`). Scenarios containing `metric` hide all six VMs entirely.
  * Random masks: Bernoulli draw per (sample minute, node, applicable modality) from numpy Philox keyed by (rate index, mask index) only,
    so the same mask serves every model and seed (paired). A hidden sample is hidden in every window that uses it.
"""
import copy
import itertools

import numpy as np

from .sim_missing import MissingBlocks

MISS_SEED = 20261028
RATES = (0.0, 0.1, 0.2, 0.3, 0.4)
N_MASKS = 10
ARMOR_MODALITIES = ("log", "metric", "trace")                       # sorted keys of ARMOR's feat_span, as in build_missing_cases


def armor_scenarios():
    """[(name, dropped modalities)]: the full baseline plus every non-empty proper subset (six drop scenarios)."""
    out = [("full", ())]
    for k in range(1, len(ARMOR_MODALITIES)):
        for combo in itertools.combinations(ARMOR_MODALITIES, k):
            out.append(("drop_" + "+".join(sorted(combo)), combo))
    return out


def scenario_observed(D, drop):
    obs = np.ones(D.Xt.shape, bool)
    idx = sorted({i for m in drop for i in D.schema["modality_index_ranges"][m]})
    if idx:
        obs[:, :, idx] = False
    return obs


def random_observed(D, blocks, rate_index, mask_index, rate):
    """(samples, nodes, channels) observed array for one deterministic random mask."""
    S, N = D.Xt.shape[0], D.Xt.shape[1]
    ss = np.random.SeedSequence([MISS_SEED, int(rate_index), int(mask_index)])
    u = np.random.Generator(np.random.Philox(ss)).random((S, N, blocks.M))
    hide = (u < rate) & blocks.block_applicable(D.applicable)[None]
    hidden = np.zeros((S, N, blocks.C), bool)
    for m in range(blocks.M):
        hidden |= hide[:, :, m, None] & blocks.chan_block[None, :, m, :]
    return ~hidden


def masked_view(D, observed):
    if observed.shape != D.Xt.shape or observed.dtype != np.bool_:
        raise ValueError("observed must be a bool array shaped like the data")
    view = copy.copy(D)
    view.observed = observed
    return view


def hides_all_vms(D, spec, observed):
    """True when no VM node keeps an effective entry anywhere on the timeline (VM = the adapter spec's 'vm' node type)."""
    vm = spec.node_type == spec.type_names.index("vm")
    return not (observed[:, vm] & D.applicable[vm][None]).any()
