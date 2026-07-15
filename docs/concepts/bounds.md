# Bounds

Bounds describe a rectangular parameter domain.

For a parameter vector {math}`x \in \mathbb{R}^d`, box bounds define the set:

```{math}
B = \left\{x \in \mathbb{R}^d \mid l_i \le x_i \le u_i,\ i = 1,\ldots,d\right\}.
```

Bounds are useful whenever an inference or optimization procedure needs an
explicit domain of valid evaluation. They can represent physical constraints,
numerical validity limits, finite design regions, or a region where an expensive
model is intended to be queried.

Bounds are optional because not every model has finite box constraints and not
every engine needs them. A model on all of {math}`\mathbb{R}^d` should not be
forced to invent artificial bounds just to satisfy the base model protocol.

## Hard Bounds And Support

If bounds are part of the mathematical support of a target density, they can be
understood as a support restriction:

```{math}
\tilde{\pi}(x) \propto \pi(x)\,\mathbf{1}\{x \in B\}.
```

Equivalently, at log-density level:

```{math}
\log \tilde{\pi}(x) =
\begin{cases}
\log \pi(x) + C, & x \in B, \\
-\infty, & x \notin B,
\end{cases}
```

where {math}`C` is the normalizing constant induced by truncation. In software,
however, bounds do not always mean "truncate the target." They can also be
metadata used by an engine to choose initial points, define a finite search
region, avoid invalid numerical evaluations, or communicate a legal domain.

The important design point is that bounds are a capability attached to a model,
not part of the core callable model semantics.

## Plausible Bounds

Some engines distinguish hard bounds from a smaller region where useful mass or
reasonable initial evaluations are expected. Distributed Inference represents
that with optional plausible bounds:

```{math}
l_i \le l_i^{\mathrm{plaus}} \le u_i^{\mathrm{plaus}} \le u_i,
\qquad i = 1,\ldots,d.
```

Hard bounds define validity. Plausible bounds define a useful region inside the
valid domain. They should not be used as a substitute for the target density.

Attach bounds by composition:

```{code-block} python
import numpy as np

from distributed_inference import Bounds, WithBounds

bounds = Bounds(
    lower=np.array([-5.0, -5.0]),
    upper=np.array([5.0, 5.0]),
    plausible_lower=np.array([-2.0, -2.0]),
    plausible_upper=np.array([2.0, 2.0]),
)
bounded_model = WithBounds(model, bounds)
```

This keeps the general model semantics small while allowing engines to request
the `BoundedModel` capability explicitly.

## References

Stan's reference manual discusses support, bounds, and constrained parameters in
the context of probability functions and transformations
([Stan Reference Manual: Constraint Transforms](https://mc-stan.org/docs/reference-manual/transforms.html)).
For optimization algorithms, bound-constrained problems are commonly written
with simple box constraints and motivated algorithmically by methods such as
L-BFGS-B ([Byrd, Lu, Nocedal, and Zhu, 1995](https://doi.org/10.1137/0916069);
[Zhu, Byrd, Lu, and Nocedal, 1997](https://doi.org/10.1145/279232.279236)).
