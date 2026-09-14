"""
NeuroSpace: a dynamic, spatially-embedded neural network.

Design decisions (read this before you extend it):

1. TOPOLOGY IS A DAG. New connections are only ever allowed to run from a
   neuron created "earlier" (lower topo index) to one created "later" or
   between input->hidden/output and hidden->output. We maintain an explicit
   topological order and rebuild it lazily whenever structure changes. This
   lets us do exact backprop through an irregular, sparse, changing graph
   with plain numpy - no autograd library needed. Real cortex has cycles and
   uses time; we approximate the "gets a real error signal" part with a
   clean feed-forward DAG, which is the pragmatic engineering trade-off that
   makes this actually trainable on a laptop.

2. WEIGHT LEARNING vs STRUCTURAL PLASTICITY ARE DECOUPLED.
   - Weight learning: exact backprop (reverse-mode, hand-rolled) OR a
     reward/error-modulated Hebbian rule (see learning.py) - selectable.
   - Structural plasticity (grow branches, prune synapses, move neurons):
     always driven by activity-correlation statistics, never by gradients.
     This part is the "brain-like" part and runs independently of which
     weight-learning rule you picked.

3. LIGHTWEIGHT ON PURPOSE. Pure numpy, sparse dict-of-weights per neuron
   (not dense matrices), so a network with a few hundred hidden neurons and
   a few thousand synapses trains in seconds on a laptop CPU.
"""
import numpy as np
from .neuron import Neuron


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def sigmoid_grad_from_out(y):
    return y * (1.0 - y)


def relu(x):
    return np.maximum(0.0, x)


def relu_grad_from_out(y):
    return (y > 0).astype(np.float64)


ACTS = {
    "sigmoid": (sigmoid, sigmoid_grad_from_out),
    "relu": (relu, relu_grad_from_out),
}


class NeuroSpace:
    def __init__(self, n_inputs, n_outputs, input_extent=1.0, output_z=10.0,
                 hidden_activation="relu", output_activation="sigmoid", seed=0,
                 n_seed_hidden=0, seed_fanin=8):
        self.rng = np.random.default_rng(seed)
        self.neurons = {}
        self.step_count = 0
        self.h_act, self.h_act_grad = ACTS[hidden_activation]
        self.o_act, self.o_act_grad = ACTS[output_activation]

        # Fixed input plane at z=0, fixed output plane at z=output_z.
        self.input_ids = []
        for i in range(n_inputs):
            pos = self._grid_pos(i, n_inputs, z=0.0, extent=input_extent)
            n = Neuron(pos, kind="input")
            self.neurons[n.id] = n
            self.input_ids.append(n.id)

        self.output_ids = []
        for i in range(n_outputs):
            pos = self._grid_pos(i, n_outputs, z=output_z, extent=input_extent)
            n = Neuron(pos, kind="output", bias=0.0)
            self.neurons[n.id] = n
            self.output_ids.append(n.id)

        self._topo = list(self.input_ids) + list(self.output_ids)
        self._topo_dirty = False

        # Start with every output wired to every input (fully connected
        # seed graph) so the network can learn something before any growth
        # happens at all - growth then adds capacity where it's needed.
        out_init_std = 1.0 / np.sqrt(max(1, n_inputs))
        for oid in self.output_ids:
            for iid in self.input_ids:
                self.neurons[oid].incoming[iid] = self.rng.normal(0, out_init_std)

        # Optionally seed a starting cortex of hidden neurons scattered in
        # the space between the input/output planes, each wired to a random
        # subset of inputs and feeding every output. This gives the network
        # real depth from step 0 instead of waiting for growth to bootstrap
        # it purely from input activity statistics.
        self.hidden_ids = []
        for _ in range(n_seed_hidden):
            pos = (self.rng.uniform(-input_extent, input_extent),
                   self.rng.uniform(-input_extent, input_extent),
                   self.rng.uniform(2.0, output_z - 2.0))
            hn = Neuron(pos, kind="hidden", bias=0.0)
            fanin = self.rng.choice(self.input_ids, size=min(seed_fanin, n_inputs), replace=False)
            h_std = 1.0 / np.sqrt(max(1, len(fanin)))
            for iid in fanin:
                hn.incoming[iid] = self.rng.normal(0, h_std)
            self.neurons[hn.id] = hn
            self.hidden_ids.append(hn.id)
            for oid in self.output_ids:
                on = self.neurons[oid]
                o_std = 1.0 / np.sqrt(max(1, len(on.incoming) + 1))
                on.incoming[hn.id] = self.rng.normal(0, o_std)
        if n_seed_hidden:
            self._topo_dirty = True

    # ---------------------------------------------------------------- utils
    def _grid_pos(self, i, n, z, extent):
        side = max(1, int(np.ceil(np.sqrt(n))))
        row, col = divmod(i, side)
        x = (col - side / 2) * extent
        y = (row - side / 2) * extent
        return (x, y, z)

    def n_hidden(self):
        return sum(1 for n in self.neurons.values() if n.kind == "hidden")

    def n_synapses(self):
        return sum(len(n.incoming) for n in self.neurons.values())

    def _rebuild_topo(self):
        # Kahn's algorithm over the "incoming" dependency graph.
        indeg = {nid: 0 for nid in self.neurons}
        for n in self.neurons.values():
            for src in n.incoming:
                indeg[n.id] += 1
        # inputs have indegree 0 by construction
        frontier = [nid for nid, d in indeg.items() if d == 0]
        order = []
        indeg = dict(indeg)
        # build reverse adjacency: src -> [dependents]
        dependents = {nid: [] for nid in self.neurons}
        for n in self.neurons.values():
            for src in n.incoming:
                dependents[src].append(n.id)
        frontier = list(frontier)
        while frontier:
            nid = frontier.pop()
            order.append(nid)
            for dep in dependents[nid]:
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    frontier.append(dep)
        self._topo = order
        self._topo_dirty = False

    def topo_order(self):
        if self._topo_dirty:
            self._rebuild_topo()
        return self._topo

    # ------------------------------------------------------------ forward
    def forward(self, x, hebbian_update=False, hebbian_rate=0.01):
        """x: array of shape (n_inputs,). Returns output activations."""
        for i, iid in enumerate(self.input_ids):
            n = self.neurons[iid]
            n.last_output = float(x[i])

        for nid in self.topo_order():
            n = self.neurons[nid]
            if n.kind == "input":
                n.act_ema = 0.98 * n.act_ema + 0.02 * n.last_output
                continue
            s = n.bias
            for src_id, w in n.incoming.items():
                s += w * self.neurons[src_id].last_output
            n.last_input_sum = s
            act, _ = (self.o_act, self.o_act_grad) if n.kind == "output" else (self.h_act, self.h_act_grad)
            n.last_output = act(s)

            # running activity trace + co-activity trace (for plasticity),
            # tracked regardless of which weight-learning rule is active.
            n.act_ema = 0.98 * n.act_ema + 0.02 * n.last_output
            for src_id in n.incoming:
                src_out = self.neurons[src_id].last_output
                prev = n.coactivity.get(src_id, 0.0)
                n.coactivity[src_id] = 0.98 * prev + 0.02 * (src_out * n.last_output)
                if hebbian_update:
                    # local reward-free Hebbian nudge (see learning.py for
                    # the modulated version used during actual training)
                    n.incoming[src_id] += hebbian_rate * src_out * n.last_output

        return np.array([self.neurons[oid].last_output for oid in self.output_ids])

    # ----------------------------------------------------------- backward
    def backward(self, target, lr=0.05, l2=1e-5):
        """Exact backprop through the current DAG for one sample.
        target: array shape (n_outputs,) for MSE loss on the outputs.
        Returns scalar loss.
        """
        grad = {nid: 0.0 for nid in self.neurons}
        out = np.array([self.neurons[oid].last_output for oid in self.output_ids])
        loss = float(np.mean((out - target) ** 2))
        for i, oid in enumerate(self.output_ids):
            grad[oid] = 2.0 * (out[i] - target[i]) / len(self.output_ids)

        for nid in reversed(self.topo_order()):
            n = self.neurons[nid]
            if n.kind == "input":
                continue
            act_grad = self.o_act_grad if n.kind == "output" else self.h_act_grad
            delta = grad[nid] * act_grad(n.last_output)  # dLoss/d(pre-activation)
            n.grad_bias = delta
            n.bias -= lr * delta
            for src_id, w in list(n.incoming.items()):
                src = self.neurons[src_id]
                grad[src_id] += delta * w
                dw = delta * src.last_output + l2 * w
                n.incoming[src_id] = w - lr * dw
        return loss

    # ------------------------------------------------------- plasticity
    def grow(self, activity_threshold=0.2, max_new=1, branch_std=0.6, max_hidden=48):
        """Spawn new hidden neurons ('branches') near existing neurons that
        are highly active but under-connected (few outgoing synapses) -
        i.e. neurons that seem 'useful' but have little downstream reach
        yet. This is the structural growth rule; it never looks at
        gradients, only at activity."""
        if self.n_hidden() >= max_hidden:
            return 0
        candidates = [n for n in self.neurons.values() if n.kind != "output" and n.act_ema > activity_threshold]
        candidates.sort(key=lambda n: -n.act_ema)
        added = 0
        for src in candidates:
            if added >= max_new or self.n_hidden() >= max_hidden:
                break
            new_pos = src.pos + self.rng.normal(0, branch_std, size=3)
            new_pos[2] = np.clip(new_pos[2], 0.5, 9.5)  # stay between input/output planes
            new_n = Neuron(new_pos, kind="hidden", bias=0.0)
            new_n.birth_step = self.step_count
            new_n.incoming[src.id] = self.rng.normal(0, 0.3)
            # also wire a couple of spatially-nearby existing neurons in
            nearby = self._nearest(new_pos, exclude={src.id, new_n.id}, k=2, before_z=new_pos[2])
            for nb in nearby:
                new_n.incoming[nb.id] = self.rng.normal(0, 0.2)
            self.neurons[new_n.id] = new_n
            # connect it forward into a random nearby-in-z output/hidden with
            # a near-zero weight: like real synaptic potentiation, a brand
            # new branch starts functionally silent and only grows in
            # influence if gradient descent (or Hebbian reinforcement) finds
            # it useful. Starting it loud would just inject noise into an
            # already-partially-trained network.
            downstream = self._nearest(new_pos, exclude={new_n.id}, k=1, after_z=new_pos[2])
            for dn in downstream:
                # scale by existing fan-in so we don't keep inflating the
                # pre-activation variance of a target that already has many
                # incoming synapses (that's what was saturating the output
                # sigmoids and collapsing predictions during testing).
                fan = max(1, len(dn.incoming))
                dn.incoming[new_n.id] = self.rng.normal(0, 0.05 / np.sqrt(fan))
            added += 1
        if added:
            self._topo_dirty = True
        return added

    def grow_synapses(self, max_new=40, radius=3.0, min_coactivity=0.002):
        """The literal 'new dendritic branch' rule: existing neurons reach
        out and form a brand new synapse onto another existing, spatially
        nearby, currently-unconnected neuron that tends to be co-active
        with them. This is how the network densifies its wiring over time
        without needing to keep minting new neurons."""
        order = self.topo_order()
        pos_idx = {nid: i for i, nid in enumerate(order)}
        added = 0
        candidates = list(self.neurons.values())
        self.rng.shuffle(candidates)
        for n in candidates:
            if added >= max_new:
                break
            if n.kind == "input":
                continue
            # look for nearby neurons that come earlier in topo order
            # (so the new synapse keeps the graph a DAG) and aren't
            # already connected.
            for other in self.neurons.values():
                if other.id == n.id or other.id in n.incoming or other.kind == "output":
                    continue
                if pos_idx.get(other.id, 1e9) >= pos_idx.get(n.id, -1):
                    continue  # must be strictly earlier to preserve DAG
                d = np.linalg.norm(other.pos - n.pos)
                if d > radius:
                    continue
                co = n.coactivity.get(other.id, None)
                # estimate a plausible correlation even for a never-yet-connected
                # pair using current activity levels as a proxy.
                proxy = other.act_ema * n.act_ema
                if (co if co is not None else proxy) < min_coactivity:
                    continue
                fan = max(1, len(n.incoming))
                n.incoming[other.id] = self.rng.normal(0, 0.05 / np.sqrt(fan))
                added += 1
                break
        if added:
            self._topo_dirty = True
        return added

    def prune(self, weight_thresh=0.02, min_age=5, dead_thresh=0.01):
        """Remove near-zero synapses, and remove hidden neurons that have
        gone quiet (low activity) and are old enough to judge."""
        removed_syn = 0
        for n in self.neurons.values():
            if n.kind == "input":
                continue
            for src_id in list(n.incoming.keys()):
                if abs(n.incoming[src_id]) < weight_thresh:
                    del n.incoming[src_id]
                    n.coactivity.pop(src_id, None)
                    removed_syn += 1

        removed_neurons = 0
        for nid in list(self.neurons.keys()):
            n = self.neurons[nid]
            if n.kind != "hidden":
                continue
            age = self.step_count - n.birth_step
            if age > min_age and n.act_ema < dead_thresh:
                # cut it out: remove it as a source everywhere too
                for other in self.neurons.values():
                    other.incoming.pop(nid, None)
                    other.coactivity.pop(nid, None)
                del self.neurons[nid]
                removed_neurons += 1
        if removed_syn or removed_neurons:
            self._topo_dirty = True
        return removed_syn, removed_neurons

    def move(self, lr=0.02, repel=0.01):
        """Hebbian-style wiring optimization: neurons drift toward inputs
        they're strongly co-active with (shortens 'axons' that matter) and
        drift slightly away from everything else (keeps the space from
        collapsing to a point). Only hidden neurons move."""
        deltas = {}
        for n in self.neurons.values():
            if n.kind != "hidden":
                continue
            pull = np.zeros(3)
            for src_id, co in n.coactivity.items():
                src = self.neurons[src_id]
                direction = src.pos - n.pos
                pull += co * direction
            push = self.rng.normal(0, repel, size=3)
            d = lr * pull + push
            n.pos_new = n.pos + d  # stash, apply after loop (sync update)
            deltas[n.id] = d
        for nid in deltas:
            n = self.neurons[nid]
            n.pos = np.clip(n.pos_new, [-5, -5, 0.3], [5, 5, 9.7])
            del n.pos_new

    def _nearest(self, pos, exclude, k, before_z=None, after_z=None):
        cands = []
        for n in self.neurons.values():
            if n.id in exclude:
                continue
            if before_z is not None and n.pos[2] >= before_z:
                continue
            if after_z is not None and n.pos[2] <= after_z:
                continue
            d = np.linalg.norm(n.pos - pos)
            cands.append((d, n))
        cands.sort(key=lambda t: t[0])
        return [n for _, n in cands[:k]]

    def plasticity_step(self, do_prune=True, do_move=True, do_grow_neurons=True,
                         do_grow_synapses=True, **kw):
        self.step_count += 1
        added_n = 0
        if do_grow_neurons:
            gkw = {k: v for k, v in kw.items() if k in ("activity_threshold", "branch_std")}
            if "max_new_neurons" in kw:
                gkw["max_new"] = kw["max_new_neurons"]
            added_n = self.grow(**gkw)
        added_s = 0
        if do_grow_synapses:
            skw = {k: v for k, v in kw.items() if k in ("radius", "min_coactivity")}
            if "max_new_synapses" in kw:
                skw["max_new"] = kw["max_new_synapses"]
            added_s = self.grow_synapses(**skw)
        rs, rn = (0, 0)
        if do_prune:
            rs, rn = self.prune()
        if do_move:
            self.move()
        return {"new_neurons": added_n, "new_synapses": added_s,
                "pruned_synapses": rs, "pruned_neurons": rn}
