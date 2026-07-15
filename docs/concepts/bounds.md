# Bounds

Bounds are optional.

The base model protocol does not require bounds because not every inference
engine needs them. PyVBMC does need bounds, so bounds are represented as a
separate capability.

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
