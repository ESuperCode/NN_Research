# NeuroSpace — a 3D structurally-plastic neural network

A neural network where hidden neurons live at 3D coordinates, can **spawn
new branches** (new neurons and new dendrite-like synapses), **prune**
unused connections, and **drift through space** toward neurons they're
correlated with — while input/output neurons stay fixed. Pure Python +
NumPy, no GPU, no heavy dependencies.

## Honest framing, read this first

You asked whether this is "the next evolution of thinking computers."
Straight answer: **no, and I want to be upfront about why**, because it
changes what this project is actually good for.

- **Structural plasticity (growth/pruning/movement) is real and works.**
  It's driven purely by activity correlation, no gradients involved — the
  most genuinely "brain-like" part of this.
- **Weight learning is still backpropagation.** I implemented an exact
  hand-rolled backprop through the dynamic sparse DAG. This is the part
  that actually makes the network learn to classify/reconstruct anything.
- **I also implemented pure reward-modulated Hebbian learning** (a
  three-factor local rule, no backprop at all) so you could see the gap
  yourself instead of taking my word for it. The numbers below show it:
  Hebbian-only learning is barely better than chance on anything but the
  most trivial task. This isn't a bug — it's the actual state of
  biologically-plausible learning in ML research. Getting local rules to
  match backprop's credit assignment on deep/sparse networks is an open
  problem researchers are still working on (see: feedback alignment,
  predictive coding, equilibrium propagation, target propagation).
- So: this is a **legitimate, working research toy** that demonstrates
  dynamic structural plasticity alongside real gradient-based learning. It
  is not a new paradigm that outperforms standard deep learning — on every
  benchmark below, a plain fixed MLP with the same optimizer matches or
  beats it on accuracy, and trains 10-30x faster (vectorized numpy matmuls
  vs. this project's per-synapse Python dict loops).

Where NeuroSpace actually pulled ahead: the **autoencoder/reconstruction**
task, where it beat the MLP (0.026 vs 0.042 MSE) at the same bottleneck
size, likely because structural growth found extra useful pathways a fixed
16-unit bottleneck couldn't. That's a genuinely interesting result, not a
guaranteed one — rerun it and it may not always replicate exactly.

## Results (all reproducible, see "how to run")

### Digit classification (sklearn `digits`, 10 classes, 300 train / 200 test, 40 epochs)
| System | Test Acc | Train time | Params | Hidden units |
|---|---|---|---|---|
| NeuroSpace (backprop + plasticity) | 83.5% | 12.2s | 905 | 32 |
| NeuroSpace (pure Hebbian, no backprop) | 10.0% (chance) | 6.0s | 574 | 6 |
| Standard MLP (same optimizer) | 89.5% | 0.4s | 2410 | 32 |

### Text topic classification (synthetic 4-topic corpus, bag-of-words)
| System | Test Acc | Train time | Params |
|---|---|---|---|
| NeuroSpace (backprop + plasticity) | 94.8% | 2.5s | 530 |
| NeuroSpace (pure Hebbian) | 29.2% (~chance) | 1.8s | 203 |
| Standard MLP | 99.0% | 0.2s | 1304 |

### Image reconstruction / generation (autoencoder, 16-unit bottleneck)
| System | Reconstruction MSE (lower=better) |
|---|---|
| NeuroSpace (backprop + plasticity) | **0.0256** |
| Standard MLP | 0.0423 |

Takeaways:
- **Backprop-through-the-dynamic-graph works and is competitive**, using
  fewer parameters than the fixed MLP by letting structure adapt.
- **Pure Hebbian learning does not work well** on these tasks — expected,
  and included so you'd have real numbers instead of a hand-wave.
- **NeuroSpace is much slower to train** (Python-level sparse dict ops vs.
  NumPy's vectorized matmuls). This is an engineering ceiling, not a
  fundamental one — it would close a lot if rewritten with batched sparse
  tensor ops (e.g. in PyTorch with sparse adjacency matrices).

## Project layout
```
neurosim/
  neuron.py            - single neuron: 3D position, sparse incoming weights, activity trace
  network.py            - NeuroSpace: forward/backward pass (DAG backprop), grow/prune/move
  learning.py            - train_backprop() and train_hebbian() training loops
  baseline_mlp.py        - plain fixed MLP for fair comparison
tasks/
  bench_digits.py         - classification benchmark + comparison table
  bench_text.py           - text topic classification benchmark
  bench_reconstruction.py  - autoencoder/generation benchmark, saves before/after image grid
  visualize_network.py     - renders the trained network's 3D structure to a PNG
```

## How to run
```bash
pip install numpy scikit-learn matplotlib
python3 tasks/bench_digits.py
python3 tasks/bench_text.py
python3 tasks/bench_reconstruction.py     # saves reconstruction.png
python3 tasks/visualize_network.py        # saves network_3d.png
```
Each digit-classification run takes ~15-20 seconds total on a laptop CPU;
the whole suite runs in well under a minute. That's the "lightweight"
constraint honored — nothing here needs a GPU.

## Where to run heavier experiments for free
This code is pure NumPy (no GPU acceleration path as written), so a free
GPU notebook won't speed *this specific code* up much — the bottleneck is
Python-level looping over sparse dicts, not matrix-multiply throughput.
If you want to scale this up (bigger networks, more epochs, more data),
your options, roughly in order of convenience:
- **Google Colab** (free tier: decent CPU + a T4 GPU if you rewrite the
  hot loop with a tensor library) — easiest to get started, generous free
  compute, some session-length limits.
- **Kaggle Notebooks** — free GPU/TPU quota (~30 hrs/week), good for
  longer unattended runs.
- **Lightning AI Studios** — free monthly compute credits, persistent
  environment.
- If you keep it pure NumPy/CPU-only, any of the above's free CPU tier is
  fine — the real lever is rewriting the sparse per-synapse loops as
  batched operations (e.g. a PyTorch sparse tensor for the adjacency
  matrix), which would also make GPU accel actually pay off.

## Extending this
Things that would meaningfully improve it, roughly by effort:
1. **Vectorize the forward/backward pass** with batched sparse matrix ops
   instead of per-neuron Python loops — biggest speed win by far.
2. **Better structural plasticity criteria** — right now growth/pruning
   are activity-threshold heuristics; something like a running estimate of
   "would this connection reduce loss" (a cheap gradient proxy) would grow
   structure more purposefully.
3. **True recurrent/temporal dynamics** (allow cycles, unroll over time,
   add something like spiking neurons + STDP) if you want to chase the
   "more biologically real" direction further — this is a much bigger
   project and would need a different backward pass entirely (surrogate
   gradients or an actual local learning rule that works, which remains
   an open research question).
