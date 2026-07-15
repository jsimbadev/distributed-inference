# Set Up A PyVBMC Model

PyVBMC needs a log-density callable and bounds. Distributed Inference keeps
those as separate pieces: a model provides the log density, and `WithBounds`
adds the bounds capability.

## Define The Model

```{code-block} python
import numpy as np

from distributed_inference import CallableModel

def log_density(x, context):
    return -0.5 * float(np.dot(x, x))

model = CallableModel(name="gaussian", dimension=2, fn=log_density)
```

## Add Bounds For PyVBMC

```{code-block} python
from distributed_inference import Bounds, WithBounds

bounds = Bounds(
    lower=np.array([-5.0, -5.0]),
    upper=np.array([5.0, 5.0]),
    plausible_lower=np.array([-2.0, -2.0]),
    plausible_upper=np.array([2.0, 2.0]),
)
bounded_model = WithBounds(model, bounds)
```

## Build PyVBMC Inputs

```{code-block} python
from distributed_inference.engines.pyvbmc import as_pyvbmc_log_density, pyvbmc_bounds

log_density_for_pyvbmc = as_pyvbmc_log_density(bounded_model)
pyvbmc_box = pyvbmc_bounds(bounded_model)
```

`log_density_for_pyvbmc` is a plain callable. `pyvbmc_box` contains the lower,
upper, plausible lower, and plausible upper bounds PyVBMC needs.

## Pass Values To PyVBMC

```{code-block} python
from pyvbmc import VBMC

vbmc = VBMC(
    log_density_for_pyvbmc,
    x0=np.array([0.0, 0.0]),
    lower_bounds=pyvbmc_box.lower,
    upper_bounds=pyvbmc_box.upper,
    plausible_lower_bounds=pyvbmc_box.plausible_lower,
    plausible_upper_bounds=pyvbmc_box.plausible_upper,
)
```

The model abstraction stays independent of PyVBMC, while the adapter provides
the PyVBMC-specific shape.
