# Parameter Transforms

Parameter transforms relate two coordinate systems for the same model.

A model may be naturally defined on constrained parameters
{math}`x \in \mathcal{X}`, while an inference engine may operate more cleanly on
unconstrained vectors {math}`z \in \mathbb{R}^k`. A transform gives an explicit
map between those spaces.

Let:

```{math}
T : \mathbb{R}^k \rightarrow \mathcal{X},
\qquad
x = T(z).
```

If the constrained-space target has density {math}`\pi_X(x)`, then the
corresponding unconstrained-space density is:

```{math}
\pi_Z(z) = \pi_X(T(z))\,\left|\det J_T(z)\right|,
```

where {math}`J_T(z)` is the Jacobian matrix of the transform. At log-density
level:

```{math}
\log \pi_Z(z) =
\log \pi_X(T(z)) + \log \left|\det J_T(z)\right|.
```

This is why `ParameterTransform` includes `to_constrained` and
`log_abs_det_jacobian`.

## Transforms And Gradients

The Jacobian correction is not the same capability as model differentiability.

`TransformedModel` can wrap any `Model`, including a black-box `CallableModel`.
It only needs to evaluate:

```{math}
\log \pi_X(T(z)) + \log \left|\det J_T(z)\right|.
```

This requires the transform to provide the scalar log absolute determinant. It
does not require the base model to provide
{math}`\nabla_x \log \pi_X(x)`.

A gradient-aware transformed model is a stronger capability. For an engine that
needs gradients in the unconstrained coordinates, the transformed gradient is:

```{math}
\nabla_z \log \pi_Z(z)
=
J_T(z)^{\mathsf{T}}\nabla_x \log \pi_X(T(z))
+
\nabla_z \log \left|\det J_T(z)\right|.
```

That requires both a `DifferentiableModel` base model and a transform capable of
providing derivative information beyond the scalar Jacobian correction. The
current `ParameterTransform` abstraction deliberately does not require that
extra structure.

## Example: Positive Scalar

For a positive parameter {math}`x > 0`, a common unconstraining transform is:

```{math}
x = T(z) = \exp(z),
\qquad z \in \mathbb{R}.
```

The Jacobian term is:

```{math}
\log \left|\frac{d}{dz}\exp(z)\right| = z.
```

So the transformed log density is:

```{math}
\log \pi_Z(z) = \log \pi_X(\exp(z)) + z.
```

## Implementation Boundary

`TransformedModel` wraps a constrained model and a `ParameterTransform`. It
evaluates the constrained model at `transform.to_constrained(z)` and adds
`transform.log_abs_det_jacobian(z)`.

Transforms are separate from [bounds](bounds.md). A transform changes coordinate
systems. Bounds describe a rectangular domain in a particular coordinate system.
If an engine needs bounds for a transformed model, attach explicit bounds in the
transformed input space with `WithBounds`.

## References

The change-of-variables formula and common constrained-parameter transforms are
documented in the Stan reference manual
([Stan Reference Manual: Constraint Transforms](https://mc-stan.org/docs/reference-manual/transforms.html)).
Betancourt discusses why unconstrained Euclidean parameterizations are useful
for Hamiltonian Monte Carlo and related inference algorithms
([Betancourt, 2017](https://arxiv.org/abs/1701.02434)). Normalizing-flow
literature uses the same Jacobian-corrected density transformation as a general
modeling tool ([Papamakarios et al., 2019](https://arxiv.org/abs/1912.02762)).
