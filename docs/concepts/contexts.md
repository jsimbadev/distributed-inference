# Evaluation Contexts

Evaluation contexts make model evaluation explicit.

A distributed inference system evaluates the same mathematical model many times
under different operational conditions: different runs, worker processes,
random-number streams, caches, diagnostic settings, or backend handles. Those
conditions should not be hidden inside the model object. They should be supplied
at the point of evaluation.

In Distributed Inference, a model evaluation has the shape:

```{math}
\ell : \mathbb{R}^d \times C \rightarrow \mathbb{R},
\qquad
(x, c) \mapsto \ell(x; c),
```

where `x` is the parameter vector and `c` is an `EvaluationContext`. If no
context is needed, `c` can be `None`.

This keeps the model reusable and makes the operational inputs visible:

```{code-block} python
import numpy as np

from distributed_inference import EvaluationContext

context = EvaluationContext(rng=np.random.default_rng(123), run_id="run-001")
value = model(x, context)
```

## Why Not Store This On The Model?

Models should represent the log-density semantics. Runtime resources belong to
the evaluation.

This distinction matters once work is distributed:

- A local run, a cluster job, and a retry should be able to evaluate the same
  model object with different run metadata.
- Randomness should be passed explicitly so independent workers can receive
  independent streams without mutating shared model state.
- Caches and backend resources should be scoped to the evaluation or runner that
  owns their lifecycle.

Parallel Monte Carlo work has a long history of treating random-number streams
as an explicit distributed-systems concern. Bauke and Mertens describe random
number generation as a specific issue for simulations on clusters and grids
([Bauke and Mertens, 2006](https://arxiv.org/abs/cond-mat/0609584)). Salmon et
al. introduce counter-based random-number generation as a practical way to make
parallel random-number generation reproducible and decomposable across many
workers ([Salmon et al., 2011](https://doi.org/10.1145/2063384.2063405)).

`EvaluationContext` does not prescribe a particular RNG scheme. It creates the
place where a runner can pass one deliberately.

## Context Is Not Sampler State

The context is for evaluation-time information. It is not sampler state. Sampler
state should live in the inference engine or runner that owns the algorithm.

For example, a Markov chain position, adaptation window, acceptance statistic,
or variational approximation parameter belongs to the engine. A run identifier,
worker-local RNG, memoization cache, or backend session handle can belong to the
evaluation context.
