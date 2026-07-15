# Parameter Transforms

Transforms are separate from the base model.

They are useful when an engine expects unconstrained Euclidean vectors but a
model is naturally defined in constrained coordinates.

`TransformedModel` wraps a constrained model and a `ParameterTransform`. It
evaluates:

```{code-block} python
base_model(transform.to_constrained(z), context) + transform.log_abs_det_jacobian(z)
```

Automatic bounds transformation is not part of the first implementation. If an
engine needs bounds for a transformed model, attach explicit bounds in the
transformed input space with `WithBounds`.
