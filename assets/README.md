# README figure sources

These assets preserve the original manuscript figure and archived experiment
measurements. The decomposition figure was rendered again with updated display
titles; its curves and aggregation are unchanged.

## Figure 1

- Files: [figure1.png](figure1.png), [figure1.pdf](figure1.pdf).
- Source: Figure 1 on PDF page 2 of the author's `main-neurips.pdf`,
  *Rethinking LoRA Initialization for Robust Asymmetric Learning Rates*.
- The vector PDF is cropped to the figure, excluding the manuscript caption.
  The crop uses `[105,565,507,722]` points in bottom-left PDF coordinates on
  the original 612×792-point page. The PNG renders this crop at 300 DPI.
- Width 4096; initialization scales 1, 5, and 25; first 100 training updates.
  The paper compares Init[A], Init[AB], and Init[A²].

These are the original paper trajectories. The reconstructed toy implementation
in this release cannot certify those exact values; see
[the experiment guide](../docs/EXPERIMENTS.md#limits-on-exact-numerical-reproduction).

## Feature decomposition

- Files: [feature-decomposition.png](feature-decomposition.png),
  [feature-decomposition.pdf](feature-decomposition.pdf).
- Source: archived measurements and the author's original
  `plot_llama_feature_decomposition_rms.py` plotting code.
- Archived model identities: Llama-3.2-1B and Llama-3.1-8B, trained on Math-10K.
- Display titles: Llama 3.2-1B and Llama 3-8B. The panel titles were updated
  for presentation; the archived model identities and measurements are retained.
- Initializations: Init[A] and Init[A²], both with B initially zero.
- Rank 8; alpha 96; multiplier 12; beta 0.5.
- Learning rates: A `1e-5`, B `2e-3`; ratio 200; three epochs.
- Seeds: 42, 43, and 44.
- Probe: one fixed example, probe seed 1234, at most 32 tokens, evaluated on
  eight adapted modules sampled across depth.
- Aggregation: average the module-level RMS values within each seed, then
  report the mean and ±1 sample standard deviation across the three seeds.
- Logging: 27 checkpoints through optimizer step 1860, including the first
  update, scheduled epoch fractions, 100-step intervals, and epoch boundaries.

The curves are the RMS norms of the initialization-noise term, learned-signal
term, and total scaled adapter output. The decomposition is exact, but its RMS
norms are not additive: total energy also includes a signed cross term. The
smaller noise curves under Init[A²] illustrate the noise suppression mechanism
for these runs; they do not establish that noise vanishes in every finite model.

To generate new measurements with these settings, use the commands in the
[release README](../README.md#feature-decomposition-suppressing-initialization-noise).
The plotting script requires both models, both initializations, and three seeds.
