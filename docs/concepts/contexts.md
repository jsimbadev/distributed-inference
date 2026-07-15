# Evaluation Contexts

Models do not own randomness.

Any randomness, cache, run identifier, or backend-specific context is passed
into the model evaluation:

```{code-block} python
import numpy as np

from distributed_inference import EvaluationContext

context = EvaluationContext(rng=np.random.default_rng(123), run_id="run-001")
value = model(x, context)
```

This keeps model objects reusable across local runs, worker processes, and
cluster jobs.

The context is for evaluation-time information. It is not sampler state. Sampler
state should live in the inference engine or runner that owns the algorithm.
