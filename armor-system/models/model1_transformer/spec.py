"""Model configuration and the schema-derived adapter specification (design note 3.3, ARCHITECTURE.md).

`AdapterSpec` is the only place that knows which of the 130 channels belong to which (node type, modality). It is
derived from `schema.json` through the declared mask rule (names only, never from data values), so adding a node type
or a dataset changes configuration, not the core. Node positions are never hard-coded: the node-type vector comes from
the node names via the rule.
"""
import hashlib
import json
from dataclasses import dataclass

import numpy as np

MODALITIES = ("metric", "trace", "log")                 # token order inside the core (M = 3)
TYPE_NAMES = ("pod", "vm")                              # design note 2.2: node_type 0 = pod, 1 = VM
# Approved design note 3.3: channels per (type, modality). Any difference is a failure, not a warning.
D1_EXPECTED_CHANNELS = {("pod", "metric"): 53, ("pod", "trace"): 9, ("pod", "log"): 11, ("vm", "metric"): 57}


class SpecError(ValueError):
    """The schema or masks do not give the approved adapter layout."""


@dataclass(frozen=True)
class ModelConfig:
    """Approved v0 sizes (design note 4.2) plus the implementation decisions recorded in ARCHITECTURE.md."""
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ffn: int = 256
    dropout: float = 0.1
    n_classes: int = 5
    n_input_steps: int = 6
    n_node_types: int = 2
    n_modalities: int = 3
    fusion_eps: float = 1e-8

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise SpecError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")


@dataclass(frozen=True)
class AdapterSpec:
    node_names: tuple
    node_type: np.ndarray                               # (N,) int64, index into type_names
    n_channels: int
    type_channels: dict                                 # (type_index, modality) -> sorted int64 channel indices (absent key = unused)
    type_names: tuple = TYPE_NAMES
    modalities: tuple = MODALITIES

    @property
    def n_nodes(self):
        return len(self.node_names)

    def channels(self, type_index, modality):
        return self.type_channels.get((type_index, modality), np.zeros(0, np.int64))

    def output_channels(self, type_index):
        """Channels a node type predicts, concatenated in modality order (the output adapter's d -> len(...))."""
        parts = [self.channels(type_index, m) for m in self.modalities]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def fingerprint(self):
        body = {"nodes": list(self.node_names), "node_type": self.node_type.tolist(), "n_channels": self.n_channels,
                "type_names": list(self.type_names), "modalities": list(self.modalities),
                "channels": {f"{self.type_names[k]}/{m}": c.tolist() for (k, m), c in sorted(self.type_channels.items())}}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_d1(cls, schema, applicable=None, expected=D1_EXPECTED_CHANNELS):
        """Build from the D1 schema. `applicable` (46,130) bool, if given, must lie inside the declared channel lists."""
        from prep import m1_masks                       # data layer lives in armor-system/data (the caller puts it on sys.path)
        declared, m1_type = m1_masks.declared_entry_mask(schema)
        nodes, names, modality = m1_masks.read_layout(schema)
        rule_types = [t["name"] for t in m1_masks.D1_MASK_RULE["node_types"]]
        if sorted(rule_types) != sorted(TYPE_NAMES):
            raise SpecError(f"mask rule node types {rule_types} != {list(TYPE_NAMES)}")
        node_type = np.empty(len(nodes), np.int64)
        for i, k in enumerate(m1_type):
            node_type[i] = TYPE_NAMES.index(rule_types[k])
        mod = np.array(modality)
        channels = {}
        for k, tname in enumerate(TYPE_NAMES):
            rows = np.nonzero(node_type == k)[0]
            if rows.size == 0:
                continue
            if not (declared[rows] == declared[rows[0]]).all():
                raise SpecError(f"declared mask is not uniform across the nodes of type {tname!r}")
            for m in MODALITIES:
                cols = np.nonzero(declared[rows[0]] & (mod == m))[0].astype(np.int64)
                if cols.size:
                    channels[(k, m)] = cols
        got = {(TYPE_NAMES[k], m): int(c.size) for (k, m), c in channels.items()}
        if got != dict(expected):
            raise SpecError(f"channel counts per (type, modality) {got} differ from the approved {dict(expected)}")
        if applicable is not None:
            applicable = m1_masks.validate_bool_mask(applicable, declared.shape, "applicable_entry")
            if (applicable & ~declared).any():
                raise SpecError("applicable_entry has entries outside the declared channel lists")
        return cls(tuple(nodes), node_type, len(names), channels)
