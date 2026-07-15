# Model Abstraction

The model abstraction separates three concerns:

- the parameter vector being evaluated;
- the coordinate space that vector belongs to;
- the execution context for that specific evaluation.

## Callable Models

Models are callable:

```{code-block} python
value = model(x, context)
```

The model instance has one input space. It does not switch between constrained
and unconstrained interpretation at call time. If a different space is needed,
wrap the model in another model.

This keeps calls explicit and avoids hidden coordinate changes through mutable
state.

## Evaluation Context

Models do not own randomness.

Any randomness, cache, run identifier, or backend-specific context is passed
into the operation:

```{code-block} python
context = EvaluationContext(rng=np.random.default_rng(123))
value = model(x, context)
```

This keeps model objects reusable across runs and makes distributed evaluation
easier to inspect.

## Parameter Spaces

`ParameterSpace.CONSTRAINED` means the model accepts parameters in their native
domain. `ParameterSpace.UNCONSTRAINED` means the model accepts vectors in
Euclidean coordinates.

Inference engines usually want one of these spaces explicitly. PyVBMC is
adapted through an unconstrained callable plus bounds.

## Transformed Models

Use `TransformedModel` to expose a constrained model in unconstrained
coordinates. The transformed model evaluates:

```{code-block} python
base_model(to_constrained(z)) + log_abs_det_jacobian(z)
```

Automatic bound transformation is not part of the first implementation.
Transformed models require explicit bounds in their own input space when an
engine needs bounds.

## Optional Gradients

Gradient support is a capability, not a requirement. A `CallableModel` without a
gradient function only supports scalar log-density evaluation. A model with a
gradient function supports `log_density_and_gradient`.

This keeps black-box targets usable while leaving a path for engines that can
exploit derivative information.

## PyVBMC Adapter

PyVBMC receives a plain callable:

```{code-block} python
log_density = as_pyvbmc_log_density(model, context)
bounds = pyvbmc_bounds(model)
```

The adapter requires the model input space to be unconstrained.
