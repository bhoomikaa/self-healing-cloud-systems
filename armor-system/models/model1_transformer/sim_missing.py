"""Simulated-missing inputs for v1 training (V1_PROTOCOL.md section 4; amendment A-4). Training only, both stages.

Each applicable (node, modality) block is hidden with probability RATE (0.20), independently per window, input minute, node and
modality: all channels of that modality of that node at that minute become unavailable. Hidden entries are removed from the effective
mask (the existing adapter path zeroes their values with `torch.where` and receives the mask). Rules enforced here and tested:
  * natural absence (not applicable) is never droppable and never reported as artificial;
  * a true zero (applicable, observed, value 0) is data and stays visible unless a draw hides its block;
  * targets are never hidden: target values and `target_mask` are not read or changed;
  * the draws come from numpy Philox keyed by (seed, stage, epoch, window start index): deterministic, independent of batch order and
    size, and a stream of their own (torch's shuffle/dropout RNG is not consumed);
  * `artificial_missing` (hidden AND previously effective) is returned for logging and tests; the model never receives it.
"""
import numpy as np
import torch

RATE = 0.20
STAGE_IDS = {"s1": 1, "s2": 2}


class MissingBlocks:
    """Block structure of a dataset: which channels form the (node, modality) block of every node, from the adapter spec (never from positions)."""

    def __init__(self, spec):
        self.spec = spec
        N, M, C = len(spec.node_type), len(spec.modalities), spec.n_channels
        self.N, self.M, self.C = N, M, C
        self.chan_block = np.zeros((N, M, C), bool)
        for n in range(N):
            for j, m in enumerate(spec.modalities):
                self.chan_block[n, j, spec.channels(int(spec.node_type[n]), m)] = True

    def block_applicable(self, applicable):
        """(N, M): the block has at least one applicable channel (so it can be hidden)."""
        return (self.chan_block & np.asarray(applicable, bool)[:, None, :]).any(-1)

    def expand(self, hide_nm):
        """Block mask (..., N, M) -> channel mask (..., N, C)."""
        return (hide_nm[..., None] & self.chan_block).any(-2)

    def hidden_for_window(self, applicable, start_index, seed, stage, epoch, n_steps=6, rate=RATE):
        """(T, N, C) bool of entries hidden in one window. Always draws the full (T, N, M) block uniforms so the draw never depends on the structure."""
        ss = np.random.SeedSequence([int(seed), STAGE_IDS[stage], int(epoch), int(start_index)])
        u = np.random.Generator(np.random.Philox(ss)).random((n_steps, self.N, self.M))
        return self.expand((u < rate) & self.block_applicable(applicable)[None])


def simulate_missing(batch, blocks, seed, stage, epoch, rate=RATE):
    """Returns {"inputs", "input_mask", "artificial_missing"} for a batch; `batch` is not modified and targets are not touched."""
    ims = batch["input_mask"]
    T = ims.shape[1]
    app = batch["applicable_entry"].detach().cpu().numpy()
    if app.ndim == 2:                                                      # one (N, C) mask shared by the batch
        app = np.broadcast_to(app, (ims.shape[0],) + app.shape)
    start = batch["start_index"].detach().cpu().tolist()
    hidden = np.stack([blocks.hidden_for_window(app[b], start[b], seed, stage, epoch, T, rate) for b in range(ims.shape[0])])
    hidden = torch.from_numpy(hidden).to(ims.device)
    return {"inputs": torch.where(hidden, torch.zeros_like(batch["inputs"]), batch["inputs"]), "input_mask": ims & ~hidden, "artificial_missing": hidden & ims}
