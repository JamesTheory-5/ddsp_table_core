# ddsp_table_core
Below is the filled-out spec, followed by the full standalone `ddsp_table_core.py` module.

---

MODULE NAME:
**ddsp_table_core**

DESCRIPTION:
Phase-driven wavetable core that maps normalized phase inputs into wavetable samples with interpolation and optional multi-band crossfading. The module is pure functional JAX, GDSP-style, and provides `init`, `update_state`, `tick`, and `process` for use by oscillators and any other phase → table workflows.

INPUTS:

* **x (phase)** : normalized phase in `[0,1)` per sample (scalar in `tick`, vector in `process`)
* **params.band_target** : desired band index (can be fractional for crossfading or time-varying across a buffer)
* **params.band_smooth_coef** : smoothing coefficient in `[0,1]` for band index
* **params.interp_type** : integer interpolation mode selector (`0=nearest, 1=linear, 2=cubic, 3=hermite, 4=lagrange4 placeholder`)

OUTPUTS:

* **y** : interpolated wavetable sample corresponding to the given phase and smoothed band index

STATE VARIABLES:
`state = (table_stack, band_index_smooth)`

* **table_stack** : array of shape `(num_bands, table_len)` containing all wavetable bands (bandlimited variants or different waveforms)
* **band_index_smooth** : scalar tracking the smoothed band index for crossfading between adjacent tables

EQUATIONS / MATH:

* Phase to index mapping (per sample):

  * `N = table_len`
  * `index_float = phase[n] * (N - 1)`
  * `i0 = floor(index_float)`
  * `frac = index_float - i0`

* Interpolation (per band):

  * For **nearest**:

    * `y_band = table[band, i0]`
  * For **linear**:

    * `i1 = (i0 + 1) mod N`
    * `y_band = table[band, i0] + frac * (table[band, i1] - table[band, i0])`
  * For **cubic / Hermite / Lagrange**:

    * Use fixed 4-tap neighborhood `table[band, i0-1], table[band, i0], table[band, i0+1], table[band, i0+2]`, all modulo `N`
    * Apply the chosen polynomial kernel (Catmull-Rom or Hermite formulas)

* Band index smoothing:

  * `b_s[n+1] = b_s[n] + band_smooth_coef * (band_target[n] - b_s[n])`
  * `b_s[n+1]` is clamped into `[0, num_bands-1]`

* Band crossfade:

  * `b0 = floor(b_s[n+1])`
  * `b1 = min(b0 + 1, num_bands - 1)`
  * `w = b_s[n+1] - b0`
  * `y[n] = (1 - w) * y_band0 + w * y_band1`

state[n+1] = `(table_stack, b_s[n+1])`

* **through-zero rules**: not applicable directly (tables are static); phase is assumed pre-wrapped by the upstream phasor.
* **phase wrapping rules**: indices wrap via modulo `N` when fetching table samples.
* **nonlinearities**: only polynomial interpolation kernels; everything is smooth and differentiable.
* **interpolation rules**: cubic, Hermite, etc., use standard DSP formulas; mode chosen via integer `interp_type`.
* **time-varying coefficient rules**: band index smoothing uses a one-pole low-pass on band_target.

NOTES:

* `phase` must be in `[0,1)`. If an upstream module produces out-of-range values, it should wrap them before calling `ddsp_table_core_tick` / `_process`.
* `band_target` is not required to be integer; fractional values smoothly crossfade between adjacent tables.
* Interpolation mode is clipped to valid range `[0,4]`.
* Tables must be pre-allocated and passed in at init; no dynamic allocation is performed inside jit.
* The number of bands and table length are inferred from the static shape of `table_stack` and remain fixed for the lifetime of the module.
* All control flow inside jit is via `lax.switch` or arithmetic masks; there is no Python branching.

---

## `ddsp_table_core.py`

```python
"""
ddsp_table_core.py

GammaJAX DDSP – GDSP-Style Wavetable Core
-----------------------------------------

This module implements a fully differentiable, phase-driven wavetable lookup
engine in GDSP style:

    ddsp_table_core_init(...)
    ddsp_table_core_update_state(state, params)
    ddsp_table_core_tick(x, state, params)
    ddsp_table_core_process(x, state, params)

The core responsibilities:

    - Map normalized phase ∈ [0,1) to wavetable indices.
    - Apply interpolation (nearest, linear, cubic, Hermite, Lagrange-4 placeholder).
    - Support multi-band wavetable stacks with band index smoothing and
      crossfading between bands.

Design constraints:

    - Pure functional JAX.
    - No classes, no dicts, no dataclasses.
    - State = tuple only (arrays, scalars).
    - tick() returns (y, new_state).
    - process() is a lax.scan wrapper around tick().
    - No side effects.
    - No Python objects inside jit.
    - No Python branching inside @jit (use jnp.where / lax.switch).
    - No dynamic allocation or jnp.arange/jnp.zeros inside jit.
    - All shapes computed outside jit (table shapes are static).
    - Use modulo via jnp.mod for wrapping.
    - Fully differentiable.

State:
    state = (
        table_stack,        # (num_bands, table_len)
        band_index_smooth,  # scalar float
    )

Params:
    params = (
        band_target,        # desired band index (scalar or per-sample array)
        band_smooth_coef,   # smoothing coefficient in [0,1]
        interp_type,        # 0=nearest,1=linear,2=cubic,3=hermite,4=lagrange4
    )

Per-sample math:
    - Band smoothing:
        b_s[n+1] = b_s[n] + a * (band_target[n] - b_s[n])
        b_s[n+1] is clamped into [0, num_bands-1].

    - Band crossfade:
        b0 = floor(b_s[n+1])
        b1 = min(b0 + 1, num_bands-1)
        w  = b_s[n+1] - b0
        y  = (1 - w) * y_band0 + w * y_band1

    - Phase → index:
        N            = table_len
        index_float  = phase[n] * (N - 1)
        i0           = floor(index_float)
        frac         = index_float - i0

        Interpolation uses modulo indexing to handle wrap-around.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# 1. Interpolation kernels
# =============================================================================

def _linear_interp(y0, y1, frac):
    """Linear interpolation: y = y0 + frac * (y1 - y0)."""
    return y0 + frac * (y1 - y0)


def _cubic_catmull_rom(p_1, p0, p1, p2, t):
    """
    Catmull–Rom cubic interpolation.

    p(t) = 0.5 * (2p1 + (p2 - p_1)t +
                  (2p_1 - 5p1 + 4p2 - p2)t^2 +
                  (3p1 - p_1 - 3p2 + p2)t^3)
    """
    t2 = t * t
    t3 = t2 * t

    a0 = 2.0 * p0
    a1 = p1 - p_1
    a2 = 2.0 * p_1 - 5.0 * p0 + 4.0 * p1 - p2
    a3 = 3.0 * p0 - p_1 - 3.0 * p1 + p2

    return 0.5 * (a0 + a1 * t + a2 * t2 + a3 * t3)


def _hermite_interp(p_1, p0, p1, p2, t):
    """
    Hermite interpolation with tension=0, bias=0.

    Tangents:
        m0 = 0.5 * (p1 - p_1)
        m1 = 0.5 * (p2 - p0)

    Basis:
        h00 =  2t^3 - 3t^2 + 1
        h10 =      t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 =      t^3 -   t^2
    """
    t2 = t * t
    t3 = t2 * t

    m0 = 0.5 * (p1 - p_1)
    m1 = 0.5 * (p2 - p0)

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _lagrange4_interp(p_1, p0, p1, p2, t):
    """
    Placeholder 4-tap Lagrange interpolation.

    Currently implemented as Catmull–Rom; can be replaced with a true
    4th-order Lagrange kernel later.
    """
    return _cubic_catmull_rom(p_1, p0, p1, p2, t)


# =============================================================================
# 2. Core per-band interpolation helper
# =============================================================================

def _interp_sample_band(table_1d: jnp.ndarray,
                        phase: jnp.ndarray,
                        interp_type: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate a single sample from a single band's 1D table using the
    specified interpolation type.

    Args:
        table_1d    : (N,) wavetable for one band
        phase       : normalized phase in [0,1)
        interp_type : 0=nearest,1=linear,2=cubic,3=hermite,4=lagrange4

    Returns:
        y           : interpolated sample
    """
    N = table_1d.shape[0]
    N_minus_1 = float(N - 1)

    phase = jnp.asarray(phase, dtype=table_1d.dtype)
    i_float = phase * N_minus_1
    i0 = jnp.floor(i_float).astype(jnp.int32)
    frac = i_float - i0

    # Neighbor indices with wrap
    i_m1 = jnp.mod(i0 - 1, N)
    i_p0 = jnp.mod(i0, N)
    i_p1 = jnp.mod(i0 + 1, N)
    i_p2 = jnp.mod(i0 + 2, N)

    # Sample values
    p_m1 = table_1d[i_m1]
    p0 = table_1d[i_p0]
    p1 = table_1d[i_p1]
    p2 = table_1d[i_p2]

    interp_idx = jnp.clip(jnp.asarray(interp_type, dtype=jnp.int32), 0, 4)

    def mode_nearest(args):
        p_m1_, p0_, p1_, p2_, frac_ = args
        del p_m1_, p1_, p2_, frac_
        return p0_

    def mode_linear(args):
        p_m1_, p0_, p1_, p2_, frac_ = args
        del p_m1_, p2_
        return _linear_interp(p0_, p1_, frac_)

    def mode_cubic(args):
        p_m1_, p0_, p1_, p2_, frac_ = args
        return _cubic_catmull_rom(p_m1_, p0_, p1_, p2_, frac_)

    def mode_hermite(args):
        p_m1_, p0_, p1_, p2_, frac_ = args
        return _hermite_interp(p_m1_, p0_, p1_, p2_, frac_)

    def mode_lagrange4(args):
        p_m1_, p0_, p1_, p2_, frac_ = args
        return _lagrange4_interp(p_m1_, p0_, p1_, p2_, frac_)

    args = (p_m1, p0, p1, p2, frac)

    return lax.switch(
        interp_idx,
        (mode_nearest, mode_linear, mode_cubic, mode_hermite, mode_lagrange4),
        args,
    )


# =============================================================================
# 3. GDSP-style public API
# =============================================================================

def ddsp_table_core_init(
    table: jnp.ndarray,
    *,
    interp_type: int = 1,
    initial_band_index: float = 0.0,
    band_smooth_coef: float = 0.0,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Initialize ddsp_table_core.

    Args:
        table             : 1D (N,) or 2D (num_bands, N) wavetable.
                            If 1D, it is promoted to shape (1, N).
        interp_type       : interpolation mode (0..4)
        initial_band_index: starting band index (float; can be fractional)
        band_smooth_coef  : smoothing coefficient in [0,1]
        dtype             : JAX dtype

    Returns:
        state  : (table_stack, band_index_smooth)
        params : (band_target, band_smooth_coef, interp_type)
    """
    table = jnp.asarray(table, dtype=dtype)
    if table.ndim == 1:
        table_stack = table[None, :]
    elif table.ndim == 2:
        table_stack = table
    else:
        raise ValueError("table must be 1D or 2D (num_bands, N)")

    band_index_smooth = jnp.asarray(initial_band_index, dtype=dtype)

    state = (table_stack, band_index_smooth)

    band_target = jnp.asarray(initial_band_index, dtype=dtype)
    band_smooth_coef_arr = jnp.asarray(band_smooth_coef, dtype=dtype)
    interp_type_arr = jnp.asarray(interp_type, dtype=jnp.int32)

    params = (band_target, band_smooth_coef_arr, interp_type_arr)
    return state, params


def ddsp_table_core_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optional non-IO state update (band index smoothing) without producing audio.

    Args:
        state  : (table_stack, band_index_smooth)
        params : (band_target, band_smooth_coef, interp_type)

    Returns:
        new_state: (table_stack, band_index_smooth_next)
    """
    table_stack, band_index_smooth = state
    band_target, band_smooth_coef, interp_type = params

    del interp_type  # not needed here

    band_target = jnp.asarray(band_target, dtype=band_index_smooth.dtype)
    band_smooth_coef = jnp.asarray(band_smooth_coef, dtype=band_index_smooth.dtype)

    num_bands = table_stack.shape[0]
    max_band = float(num_bands - 1)

    band_next = band_index_smooth + band_smooth_coef * (band_target - band_index_smooth)
    band_next = jnp.clip(band_next, 0.0, max_band)

    return table_stack, band_next


@jax.jit
def ddsp_table_core_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Single-sample tick.

    Inputs:
        x      : phase sample in [0,1)
        state  : (table_stack, band_index_smooth)
        params : (band_target, band_smooth_coef, interp_type)

    Returns:
        y         : interpolated table sample
        new_state : (table_stack, band_index_smooth_next)
    """
    phase = jnp.asarray(x, dtype=jnp.float32)

    table_stack, band_index_smooth = state
    band_target, band_smooth_coef, interp_type = params

    band_target = jnp.asarray(band_target, dtype=band_index_smooth.dtype)
    band_smooth_coef = jnp.asarray(band_smooth_coef, dtype=band_index_smooth.dtype)
    interp_type = jnp.asarray(interp_type, dtype=jnp.int32)

    num_bands = table_stack.shape[0]
    max_band = float(num_bands - 1)

    # Band smoothing
    band_next = band_index_smooth + band_smooth_coef * (band_target - band_index_smooth)
    band_next = jnp.clip(band_next, 0.0, max_band)

    # Band crossfade
    b0 = jnp.floor(band_next).astype(jnp.int32)
    b1 = jnp.minimum(b0 + 1, num_bands - 1)
    w = band_next - b0.astype(band_next.dtype)

    table_0 = table_stack[b0]
    table_1 = table_stack[b1]

    y0 = _interp_sample_band(table_0, phase, interp_type)
    y1 = _interp_sample_band(table_1, phase, interp_type)

    y = (1.0 - w) * y0 + w * y1

    new_state = (table_stack, band_next)
    return y, new_state


@jax.jit
def ddsp_table_core_process(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer of phase samples using lax.scan.

    Args:
        x      : phase buffer, shape (T,) with values in [0,1)
        state  : (table_stack, band_index_smooth)
        params : (band_target, band_smooth_coef, interp_type)
                 - band_target can be scalar or shape (T,)

    Returns:
        y_buf      : buffer of table samples, shape (T,)
        final_state: (table_stack, band_index_smooth_final)
    """
    table_stack, band_index_smooth = state
    band_target, band_smooth_coef, interp_type = params

    x = jnp.asarray(x, dtype=table_stack.dtype)
    T = x.shape[0]

    band_target = jnp.asarray(band_target, dtype=band_index_smooth.dtype)
    band_smooth_coef = jnp.asarray(band_smooth_coef, dtype=band_index_smooth.dtype)
    interp_type = jnp.asarray(interp_type, dtype=jnp.int32)

    # Broadcast time-varying band_target if needed
    band_target = jnp.broadcast_to(band_target, (T,))
    band_smooth_coef = jnp.broadcast_to(band_smooth_coef, (T,))

    init_state = (table_stack, band_index_smooth)

    def body(carry, xs):
        st = carry
        phase_t, b_target_t, b_coef_t = xs
        params_t = (b_target_t, b_coef_t, interp_type)
        y_t, st_next = ddsp_table_core_tick(phase_t, st, params_t)
        return st_next, y_t

    final_state, y_buf = lax.scan(
        body,
        init_state,
        (x, band_target, band_smooth_coef),
    )
    return y_buf, final_state


# =============================================================================
# 4. Smoke test, plot, listen example
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    print("=== ddsp_table_core: smoke test ===")

    # Create a simple 2-band table stack outside JAX:
    # band 0: sine
    # band 1: saw-like
    sample_rate = 48000
    table_len = 2048
    num_bands = 2

    n = onp.arange(table_len, dtype=onp.float64)
    phase = 2.0 * onp.pi * n / float(table_len)

    sine_table = onp.sin(phase)

    # crude saw via partials
    max_h = table_len // 8
    saw = onp.zeros_like(phase)
    for h in range(1, max_h + 1):
        saw += (1.0 / h) * onp.sin(h * phase)
    saw /= onp.max(onp.abs(saw) + 1e-12)

    table_stack_np = onp.stack([sine_table, saw], axis=0)
    table_stack = jnp.asarray(table_stack_np, dtype=jnp.float32)

    # Init table core
    state, params = ddsp_table_core_init(
        table_stack,
        interp_type=2,            # cubic
        initial_band_index=0.0,   # start on band 0 (sine)
        band_smooth_coef=0.01,
        dtype=jnp.float32,
    )

    # Generate phase buffer for a 440 Hz tone
    duration = 1.0
    N = int(sample_rate * duration)
    n_idx = jnp.arange(N, dtype=jnp.float32)
    freq_hz = 440.0
    freq_cyc = freq_hz / sample_rate
    phase_buf = (n_idx * freq_cyc) - jnp.floor(n_idx * freq_cyc)

    # Band_target ramps from 0→1 over the duration (crossfade sine→saw)
    band_target_buf = jnp.linspace(0.0, 1.0, N, dtype=jnp.float32)
    band_smooth_coef_buf = jnp.full_like(band_target_buf, 0.02)

    params_tv = (
        band_target_buf,
        band_smooth_coef_buf,
        params[2],  # interp_type (scalar)
    )

    y_buf, state_out = ddsp_table_core_process(phase_buf, state, params_tv)
    y_np = onp.asarray(y_buf)

    # Plot first few cycles
    plt.figure(figsize=(10, 4))
    plt.plot(y_np[:2000])
    plt.title("ddsp_table_core: sine → saw crossfade (cubic interp)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    if HAVE_SD:
        print("Playing output...")
        sd.play(y_np * 0.2, samplerate=sample_rate, blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping playback.")
```

---

If you want, the next natural step is to:

* Generate a **`ddsp_wavetable_osc_from_phasor_core.py`** module that uses:

  * `phasor_core` → phase
  * `ddsp_table_core` → wavetable lookup
  * optional BLEP/BLAMP hooks at table endpoints.
