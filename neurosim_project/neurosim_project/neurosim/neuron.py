"""
Neuron: a single unit living in 3D space.

Each neuron carries:
- a fixed or mobile 3D position (used for growth/pruning/movement rules)
- incoming synaptic weights (sparse dict: source_id -> weight)
- a running trace of its own activity and co-activity with its inputs,
  used purely for the *structural* plasticity rules (Hebbian-style),
  independent of whatever is used to learn the weights themselves.
"""
import numpy as np
import itertools

_id_counter = itertools.count()


class Neuron:
    __slots__ = (
        "id", "pos", "bias", "kind", "incoming", "act_ema",
        "last_output", "last_input_sum", "grad_bias", "birth_step",
        "coactivity", "pos_new",
    )

    def __init__(self, pos, kind="hidden", bias=0.0):
        self.id = next(_id_counter)
        self.pos = np.array(pos, dtype=np.float64)
        self.bias = bias
        self.kind = kind  # 'input' | 'hidden' | 'output'
        self.incoming = {}          # source_id -> weight
        self.coactivity = {}        # source_id -> running co-activation EMA
        self.act_ema = 0.0          # running mean activity (for pruning/growth)
        self.last_output = 0.0
        self.last_input_sum = 0.0
        self.grad_bias = 0.0
        self.birth_step = 0

    def is_fixed(self):
        # Only input/output neurons are geometrically & functionally fixed.
        return self.kind in ("input", "output")
