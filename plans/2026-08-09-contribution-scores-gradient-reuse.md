# Reuse Gradients Across Classes in `contribution_scores` — Implementation Plan

## Overview

`crested.tl.contribution_scores` computes contribution scores for one or more
target classes of the same input region(s). For the gradient-based methods
(`saliency_map`, `integrated_grad`, `expected_integrated_grad`), the current
implementation loops over classes in Python and, for each class, independently
reruns the *entire* gradient computation from scratch — including regenerating
the interpolated baseline sequences and rerunning the model's forward pass —
even though none of that depends on which class is being explained (only the
final backward pass, which selects a different output column, differs per
class). This plan reworks the gradient-based backends so a single call
computes gradients for all requested classes at once, reusing one forward
pass (and, for the integrated-gradient variants, one set of interpolated
baselines) across classes instead of recomputing them per class.

Mutagenesis and window-shuffle methods do not use gradients (they call
`model.predict`) and are out of scope — see "What We're Not Doing".

## Current State Analysis

- `crested.tl.contribution_scores` (`src/crested/tl/_tools.py:386-610`) loops
  `for i, class_index in enumerate(target_idx):` (`_tools.py:518`) and, for
  gradient methods, calls `integrated_grad(...)` or `saliency_map(...)` once
  per class index (`_tools.py:519-553`).
- `integrated_grad` (`src/crested/tl/_explainer.py:84-210`) takes a single
  `class_index: int | None`. For each input sequence it builds baselines
  (`make_baselines`, `_explainer.py:179`) and interpolates from baseline to
  sequence (`_explainer.py:186-191`) — this interpolation is identical
  regardless of `class_index`, but because `integrated_grad` is invoked once
  per class from `_tools.py`, it is rebuilt from scratch for every class.
  The actual gradient computation happens in `function_batch(..., _saliency_map, ...)`
  (`_explainer.py:194-201`), which is the expensive part (one model forward +
  backward pass per chunk of interpolated sequences).
- `saliency_map` (`_explainer.py:40-81`) similarly takes a single
  `class_index: int | None` and delegates straight to `function_batch`.
- `_saliency_map` is backend-specific:
  - TF (`src/crested/tl/_explainer_tf.py:16-47`): opens a
    `tf.GradientTape()`, calls the model once, selects `outputs[:, class_index]`,
    and returns `tape.gradient(outputs, X)`. A fresh (non-persistent) tape and
    forward pass is created on every call.
  - Torch (`src/crested/tl/_explainer_torch.py:16-49`): sets
    `requires_grad_(True)` on `X`, calls the model once, and uses
    `outputs.backward(...)` to populate `X.grad`. A fresh forward pass and
    backward graph is built on every call.
- `function_batch` (`_explainer.py:433-487`) batches over `X`'s first axis
  (`batch_size`) and concatenates each batch's output along axis 0. It works
  generically with whatever shape `fun` returns per batch, as long as axis 0
  of that output corresponds to the batch axis it sliced.
- Tests directly import and call these functions with a scalar `class_index`
  (`tests/test_tl.py:252-306`, `tests/test_tl.py:121-179`), so scalar behavior
  and return shapes must not change.
- `contribution_scores_specific` (`_tools.py:613-727`) calls
  `contribution_scores` once per class with *different* regions per class (the
  regions specific to that class) — there is no shared region set across
  classes there, so gradient reuse does not apply and this function is
  unaffected.

### Key Discoveries

- `target_idx` is always normalized to a list before the per-model loop in
  `contribution_scores` (`_tools.py:493-500`), including the `[None]`
  "combined class" case. This means every call site we need to change already
  has a list available — no new call convention needs to be invented.
- Because `make_baselines` is seeded (`seed` param, default 42) and otherwise
  deterministic, computing it once for all classes instead of once per class
  produces numerically identical baselines — this refactor is a pure
  performance change with no expected difference in output values.
- `function_batch`'s "concatenate along axis 0" logic keeps working unchanged
  as long as the per-class dimension is added as a *non-zero* axis in
  `_saliency_map`'s output (e.g. axis 1), since axis 0 stays the batch axis it
  slices on.
- TF's `GradientTape` must have the target tensor computed *while the tape is
  recording* for `tape.gradient()` to work later; the fix is to do the model
  call and per-class indexing inside one `with tape:` block, then call
  `tape.gradient()` multiple times afterwards on a `persistent=True` tape.
- Torch's `torch.autograd.grad(target, X, grad_outputs=..., retain_graph=...)`
  can be called repeatedly against the same forward pass by passing
  `retain_graph=True` on all but the last call — this is a cleaner mechanism
  than accumulating into `X.grad` for the multi-class case and can replace the
  existing `.backward()`-based single-class implementation too (same result,
  since `outputs.backward(torch.ones_like(outputs))` and
  `torch.autograd.grad(outputs, X, grad_outputs=torch.ones_like(outputs))[0]`
  compute the same value).
- The `crested_py313` mamba env has both `tensorflow` and `torch` installed
  (torch added during planning). `tests/__init__.py:6-19` always forces
  `KERAS_BACKEND=tensorflow` when tensorflow is importable, so `pytest` in
  this env can only exercise the TF backend — this is pre-existing repo
  behavior (the real per-backend CI runs each backend in its own env via the
  `hatch-test` matrix in `pyproject.toml:99-106`), not something this plan
  changes. The torch backend will be verified with a standalone script that
  imports `crested` directly with `KERAS_BACKEND=torch` set before import,
  bypassing `tests/__init__.py`.

## Desired End State

`contribution_scores` calls each gradient-based backend function **once per
model** with the full `target_idx` list, instead of once per class. The
backend functions compute one forward pass per batch of (interpolated) input
sequences and reuse it to produce gradients for every requested class via
either a persistent `GradientTape` (TF) or `retain_graph=True` (torch). Passing
a single `int` or `None` for `class_index` continues to work exactly as
before (existing direct callers/tests are unaffected).

Verification: for a fixed seed, `contribution_scores(..., target_idx=[0,1,2], method=...)`
produces bit-identical results to today's per-class loop, for all three
gradient methods, on both backends — plus a wall-clock speedup when
`n_classes > 1` (fewer forward passes through the model).

## What We're NOT Doing

- Not touching `mutagenesis` or `window_shuffle` — they use `model.predict`,
  not gradients, so "reuse gradients" doesn't apply. They also redundantly
  rerun the forward pass per class today, but that's a separate optimization
  outside this plan's stated scope.
- Not changing `contribution_scores_specific` — each class there uses a
  different region set, so there's nothing to share across classes.
- Not vectorizing the backward pass itself (e.g. via a single Jacobian /
  `vmap` call) — we reuse the *forward pass* and computational graph across
  classes and do one backward call per class, which is the straightforward,
  numerically-transparent option and doesn't require batching class outputs
  into the model's compute graph in a fragile way.
- Not adding a new public parameter to `crested.tl.contribution_scores`
  itself — its existing signature and return shapes are unchanged; this is an
  internal efficiency change.
- Not fixing the pre-existing `tests/__init__.py` backend-selection quirk
  that prevents testing torch when tensorflow is also installed.

## Implementation Approach

Work bottom-up: fix the backend-specific `_saliency_map` first (the shared
primitive both gradient methods rely on), then `integrated_grad`/`saliency_map`
in `_explainer.py`, then update `contribution_scores` in `_tools.py` to call
them with the full class list. Add equivalence tests at each layer that would
catch a regression before moving to the next phase.

---

## Phase 1: Backend `_saliency_map` supports a list of class indices

### Overview
Make `_saliency_map` in both backends accept `class_index: int | list[int] | None`.
When given a list, compute one forward pass and reuse it for one backward
pass per class, returning gradients stacked on a new axis 1 (shape
`(batch, n_classes, seq_len, nuc)`). Scalar/`None` behavior and return shape
(`(batch, seq_len, nuc)`) are unchanged.

### Changes Required

#### 1. TF backend
**File**: `src/crested/tl/_explainer_tf.py`
**Changes**: Rewrite `_saliency_map` to compute the model's raw outputs and
all per-class targets inside a single `with tf.GradientTape(persistent=...) as tape:`
block, then call `tape.gradient()` once per class after the block exits.

```python
def _saliency_map(
    X: tf.Tensor,
    model: keras.Model,
    class_index: int | list[int] | None = None,
    func: Callable[[tf.Tensor], tf.Tensor] = tf.math.reduce_mean,
) -> tf.Tensor:
    """Fast function to generate saliency maps.

    Parameters
    ----------
    X
        tf.Tensor of sequences/model inputs, of shape (n_sequences, seq_len, nuc).
    model
        Your Keras model, or any object that supports __call__ with gradients, so it can also be a non-Keras TensorFlow model.
    class_index
        Index (or list of indices) of model output(s) to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
        If a list, gradients for all requested classes are computed from a single forward pass, which is reused across classes.
    func
        Function to reduce model outputs to one value with, for any class_index entries that are None.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc), if class_index is a single int or None.
    If class_index is a list, gradients of shape (batch, n_classes, seq_len, nuc).
    """
    if func is None:
        func = tf.math.reduce_mean
    is_multi_class = isinstance(class_index, (list, tuple))
    class_indices = class_index if is_multi_class else [class_index]
    with tf.GradientTape(persistent=len(class_indices) > 1) as tape:
        tape.watch(X)
        raw_outputs = model(X, training=False)
        targets = [
            raw_outputs[:, idx] if idx is not None else func(raw_outputs)
            for idx in class_indices
        ]
    grads = [tape.gradient(target, X) for target in targets]
    if len(class_indices) > 1:
        del tape  # release the persistent tape's resources
    return tf.stack(grads, axis=1) if is_multi_class else grads[0]
```

#### 2. Torch backend
**File**: `src/crested/tl/_explainer_torch.py`
**Changes**: Rewrite `_saliency_map` to do one forward pass, then one
`torch.autograd.grad` call per class with `retain_graph=True` on all but the
last, replacing the `.backward()`/`X.grad` pattern (numerically equivalent).

```python
def _saliency_map(
    X: torch.Tensor,
    model: keras.Model,
    class_index: int | list[int] | None = None,
    func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    """Fast function to generate saliency maps.

    Parameters
    ----------
    X
        torch.Tensor of sequences/model inputs, of shape (n_sequences, seq_len, nuc).
    model
        Your Keras model, or any object that supports __call__ with gradients, so it can also be a non-Keras PyTorch model.
    class_index
        Index (or list of indices) of model output(s) to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
        If a list, gradients for all requested classes are computed from a single forward pass, which is reused across classes.
    func
        Function to reduce model outputs to one value with, for any class_index entries that are None.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc), if class_index is a single int or None.
    If class_index is a list, gradients of shape (batch, n_classes, seq_len, nuc).
    """
    if func is None:
        func = torch.mean
    is_multi_class = isinstance(class_index, (list, tuple))
    class_indices = class_index if is_multi_class else [class_index]
    X = X.clone().detach().requires_grad_(True)
    outputs = model(X)
    n = len(class_indices)
    grads = []
    for i, idx in enumerate(class_indices):
        target = outputs[:, idx] if idx is not None else func(outputs)
        grad = torch.autograd.grad(
            target, X, grad_outputs=torch.ones_like(target), retain_graph=(i < n - 1)
        )[0]
        grads.append(grad)
    return torch.stack(grads, dim=1) if is_multi_class else grads[0]
```

### Success Criteria

#### Automated Verification:
- [x] TF backend unit check (standalone script, `crested_py313` env, default `KERAS_BACKEND=tensorflow`): `_saliency_map(X, model, class_index=[0,1])[:, 0]` matches `_saliency_map(X, model, class_index=0)` and `[:, 1]` matches `class_index=1`, via `np.testing.assert_allclose`.
- [x] Torch backend unit check (standalone script setting `KERAS_BACKEND=torch` before importing `crested`): same equivalence check as above.
- [x] `python -m pytest tests/test_tl.py -k explainer_dtype_handling -q` passes unchanged (scalar `class_index` path untouched).

#### Manual Verification:
- [ ] None beyond the above — this phase has no user-facing entry point yet.

---

## Phase 2: `integrated_grad` and `saliency_map` support a list of class indices

### Overview
Thread `class_index: int | list[int] | None` through `saliency_map()` and
`integrated_grad()` in `_explainer.py`. `saliency_map()` needs no logic
change (it already delegates to `function_batch(..., _saliency_map, ...)`,
which now handles lists). `integrated_grad()` needs its baseline/output
allocation to account for an optional class axis, but the interpolation and
integration math is unchanged.

### Changes Required

#### 1. `saliency_map`
**File**: `src/crested/tl/_explainer.py`
**Changes**: Widen the `class_index` type hint and docstring only — no
functional change, since `function_batch`'s axis-0 concatenation already
supports whatever shape `_saliency_map` returns per batch.

#### 2. `integrated_grad`
**File**: `src/crested/tl/_explainer.py`
**Changes**: Widen `class_index` type hint; preallocate `outputs` with an
extra class axis when `class_index` is a list; pass `class_index` straight
through to `function_batch`/`_saliency_map` (which now returns a
class-stacked result); reshape generically instead of hardcoding a 4D shape.

```python
    is_multi_class = isinstance(class_index, (list, tuple))
    if is_multi_class:
        outputs = np.zeros((X.shape[0], len(class_index), X.shape[1], X.shape[2]), dtype=X.dtype)
    else:
        outputs = np.zeros_like(X)

    for i, x in enumerate(X):
        x = np.expand_dims(x, axis=0)

        x_full = []
        for baseline in baselines[i, ...]:
            steps = np.linspace(start=0.0, stop=1.0, num=num_steps + 1)
            x_interp = interpolate_data(baseline, x, steps)
            x_full.append(x_interp)
        x_full = np.concatenate(x_full, axis=0)

        grad = function_batch(
            x_full,
            _saliency_map,
            model=model,
            class_index=class_index,
            func=func,
            batch_size=batch_size,
        )
        # grad shape: (n_baselines * (n_steps + 1), [n_classes,] seq_len, nuc)
        grad = grad.reshape((num_baselines, num_steps + 1) + grad.shape[1:])

        avg_grad = integral_approximation(grad)
        outputs[i, ...] = np.mean(avg_grad, axis=0)
    return outputs
```

(`interpolate_data` and `integral_approximation` are unchanged — both already
operate generically over trailing dimensions via `...`/broadcasting.)

Update both functions' docstrings: `class_index` accepts "an int, a list of
ints, or None. If a list, gradients for all classes are computed together,
reusing the same forward pass (and, for `integrated_grad`, the same
interpolated baselines) across classes instead of recomputing them per
class." Also add a note to the `batch_size` docstring that memory scales with
the number of requested classes when reusing gradients this way (the
computation graph / interpolated batch is now shared across classes rather
than freed after each one).

#### 3. `function_batch` docstring
**File**: `src/crested/tl/_explainer.py`
**Changes**: Correct the `Returns` docstring line ("Numpy array of the same
shape as X") to note it matches X's leading (batch) dimension, with any
additional trailing dimensions `fun` adds (e.g. a class axis) preserved.

### Success Criteria

#### Automated Verification:
- [x] `python -m pytest tests/test_tl.py -k "explainer_dtype_handling or contribution_scores" -q` passes (TF backend, `crested_py313`).
- [x] New equivalence test (see Phase 4) passes: `integrated_grad(X, model, class_index=[0,1])` equals stacking two scalar calls, for both `baseline_type="zeros"` and `"random"` with a fixed `seed`.
- [x] Standalone torch script: same equivalence check passes with `KERAS_BACKEND=torch`.

#### Manual Verification:
- [ ] None beyond the above.

---

## Phase 3: `contribution_scores` calls gradient methods once per model

### Overview
Replace the per-class loop for `integrated_grad`, `expected_integrated_grad`,
and `saliency_map` with a single call using the full `target_idx` list, using
the existing `_gradient_methods` set already defined in this function. Leave
`mutagenesis`/`window_shuffle`/`window_shuffle_uniform` on the existing
per-class loop.

### Changes Required

#### 1. `contribution_scores`
**File**: `src/crested/tl/_tools.py`
**Changes**: Restructure the per-model score computation (`_tools.py:514-577`):

```python
    scores_per_model = []
    for m in tqdm(model, desc="Model", disable=not verbose):
        scores = np.zeros((N, n_classes, L, D))  # Shape: (N, C, L, 4)

        if method in _gradient_methods:
            # Request gradients for all classes in one call so the forward
            # pass (and, for the integrated-gradient variants, the
            # interpolated baselines) is reused across classes instead of
            # being recomputed once per class.
            if method == "integrated_grad":
                scores[:, :, :, :] = integrated_grad(
                    input_sequences,
                    model=m,
                    class_index=target_idx,
                    baseline_type="zeros",
                    num_baselines=1,
                    num_steps=25,
                    batch_size=batch_size,
                )
            elif method == "expected_integrated_grad":
                scores[:, :, :, :] = integrated_grad(
                    input_sequences,
                    model=m,
                    class_index=target_idx,
                    baseline_type="random",
                    num_baselines=25,
                    num_steps=25,
                    batch_size=batch_size,
                    seed=seed,
                )
            elif method == "saliency_map":
                scores[:, :, :, :] = saliency_map(
                    input_sequences,
                    model=m,
                    class_index=target_idx,
                    batch_size=batch_size,
                )
        else:
            for i, class_index in enumerate(target_idx):
                if method == "mutagenesis":
                    scores[:, i, :, :] = mutagenesis(
                        input_sequences,
                        model=m,
                        class_index=class_index,
                        batch_size=batch_size,
                    )
                elif method == "window_shuffle":
                    scores[:, i, :, :] = window_shuffle(
                        input_sequences,
                        model=m,
                        class_index=class_index,
                        window_size=window_size,
                        n_shuffles=n_shuffles,
                        uniform=False,
                        batch_size=batch_size,
                    )
                elif method == "window_shuffle_uniform":
                    scores[:, i, :, :] = window_shuffle(
                        input_sequences,
                        model=m,
                        class_index=class_index,
                        window_size=window_size,
                        n_shuffles=n_shuffles,
                        uniform=True,
                        batch_size=batch_size,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

        scores_per_model.append(scores)
```

Note `scores` stays preallocated as a `float64` `np.zeros((N, n_classes, L, D))`
array with a full-slice assignment (`scores[:, :, :, :] = ...`), matching
today's per-class assignment dtype behavior exactly (no dtype change to the
returned scores from `contribution_scores`).

### Success Criteria

#### Automated Verification:
- [x] `python -m pytest tests/test_tl.py -k "contribution_scores" -q` passes (covers shape assertions for single/multiple targets, multiple models, batching, and mutagenesis) in `crested_py313` (TF backend).
- [x] `python -m pytest tests/test_pipeline.py -q` passes (exercises `contribution_scores` with `target_idx=[1, 2]` end-to-end, including `expected_integrated_grad` with region inputs).
- [x] New equivalence test (Phase 4): `contribution_scores(..., target_idx=[0,1,2], method=X)` output equals the concatenation of three separate `contribution_scores(..., target_idx=k, method=X)` calls, for `method` in `{"integrated_grad", "expected_integrated_grad", "saliency_map"}`, same `seed`.
- [x] Manual timing check (see below) shows `target_idx=[0,1,2,3,4]` takes noticeably less than ~5x the time of `target_idx=0` for `method="integrated_grad"`.

#### Manual Verification:
- [ ] Torch backend: rerun the standalone script from Phase 1/2 against `crested.tl.contribution_scores` directly (with `KERAS_BACKEND=torch` set before import) to confirm the full multi-class path works end-to-end, since pytest in this env can't select the torch backend.

---

## Phase 4: Tests and changelog

### Overview
Add regression tests that pin the "same result, reused computation" contract
established above, and document the change.

### Changes Required

#### 1. Equivalence tests
**File**: `tests/test_tl.py`
**Changes**: Add a test (e.g. `test_contribution_scores_multiclass_equivalence`)
that runs `crested.tl.contribution_scores` with `target_idx=[0, 1]` and
separately with `target_idx=0` / `target_idx=1`, for `method` in
`{"integrated_grad", "expected_integrated_grad", "saliency_map"}` (fixed
`seed=42`), and asserts the combined result's per-class slices are
`np.testing.assert_allclose` to the separate calls' outputs.

Optionally (if useful for isolating regressions to the right layer), add a
lower-level test importing `saliency_map`/`integrated_grad` from
`crested.tl._explainer` directly with `class_index=[0, 1]` vs. two scalar
calls.

#### 2. Changelog
**File**: `docs/changelog.md`
**Changes**: Add an entry under `## Unreleased`, in a new `### Performance`
subsection (matching the precedent at `docs/changelog.md:25` from the 1.9.0
release):

```markdown
### Performance
- {func}`crested.tl.contribution_scores` now computes gradients for all requested classes of a region in a single pass for the gradient-based methods (`saliency_map`, `integrated_grad`, `expected_integrated_grad`), reusing the model's forward pass (and, for the integrated-gradient variants, the interpolated baseline sequences) across classes instead of recomputing them once per class.
```

### Success Criteria

#### Automated Verification:
- [x] `python -m pytest tests/test_tl.py tests/test_pipeline.py -q` passes in `crested_py313` (TF backend).
- [x] `pre-commit run --files src/crested/tl/_tools.py src/crested/tl/_explainer.py src/crested/tl/_explainer_tf.py src/crested/tl/_explainer_torch.py tests/test_tl.py docs/changelog.md` passes (ruff/formatting) — `pre-commit` itself wasn't installed in the environment, substituted with `ruff check` + `ruff format --check` directly (same checks pre-commit's hooks run); only my own added/changed lines were required to be clean, per surgical-changes guidance pre-existing formatting deviations elsewhere in `_tools.py`/`test_tl.py`/`_explainer_tf.py` were left untouched.

#### Manual Verification:
- [ ] Full test suite passes under the torch backend via the standalone-script approach (or, if available, `hatch test -b pytorch`), given `tests/__init__.py` can't select torch in a mixed-backend env.

---

## Testing Strategy

### Unit Tests
- Backend-level (`_saliency_map`) equivalence between scalar and list
  `class_index`, both backends.
- `_explainer.py`-level (`integrated_grad`, `saliency_map`) equivalence,
  covering both `baseline_type="zeros"` and `"random"`.
- `contribution_scores`-level equivalence across all three gradient methods,
  plus the existing shape/mutagenesis/multi-model tests continuing to pass
  unchanged.

### Integration Tests
- `tests/test_pipeline.py` already exercises `contribution_scores` with
  `target_idx=[1, 2]` end-to-end (sequence and region inputs, both
  `integrated_grad` and `expected_integrated_grad`) — rerun as part of
  verification.

### Manual Testing Steps
1. In `crested_py313`, time `crested.tl.contribution_scores` with
   `target_idx=list(range(5))` vs. `target_idx=0` called 5 times in a loop,
   for `method="integrated_grad"`, on the test model/fixtures — confirm the
   multi-class call is meaningfully faster than 5x the single-class call.
2. With `KERAS_BACKEND=torch` set before importing `crested` in a standalone
   script (not pytest), repeat the equivalence and timing checks against the
   torch backend.

## Performance Considerations

- Memory: retaining the forward-pass computation graph (TF persistent tape /
  torch `retain_graph=True`) across all requested classes means peak memory
  no longer shrinks between classes the way independent per-class calls did.
  For very large `n_classes` combined with large `batch_size`, this could
  increase peak memory versus today; this is called out in the updated
  `batch_size` docstring so users can lower `batch_size` if needed.
- Expected speedup scales with how much of the total cost was the forward
  pass vs. backward pass, and is largest for `integrated_grad`/
  `expected_integrated_grad` (where the interpolated-baseline construction
  and forward pass through `num_baselines * (num_steps + 1)` sequences was
  previously repeated verbatim for every class).

## Migration Notes

None — this is an internal efficiency change. `contribution_scores`'s public
signature, return shapes, and output values are unchanged. `saliency_map`/
`integrated_grad`/`_saliency_map`'s scalar `class_index` behavior is
unchanged; the `list[int]` option is new and additive.

## References

- `crested.tl.contribution_scores`: `src/crested/tl/_tools.py:386-610`
- `crested.tl.contribution_scores_specific` (unaffected): `src/crested/tl/_tools.py:613-727`
- `saliency_map` / `integrated_grad` / `function_batch`: `src/crested/tl/_explainer.py`
- TF backend `_saliency_map`: `src/crested/tl/_explainer_tf.py:16-47`
- Torch backend `_saliency_map`: `src/crested/tl/_explainer_torch.py:16-49`
- Existing tests: `tests/test_tl.py:121-306`, `tests/test_pipeline.py:68-88`
- Backend-selection quirk in tests: `tests/__init__.py:6-19`
- `hatch-test` per-backend matrix: `pyproject.toml:99-106`
