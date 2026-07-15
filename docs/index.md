# Distributed Inference

Distributed Inference provides tools for running, collecting, and combining
repeated inference runs on local and distributed infrastructure.

The first model interface is callable-first: a model is an object that accepts a
parameter vector and an optional evaluation context, then returns a scalar log
density.

```{toctree}
:maxdepth: 2

concepts/index
how-to/index
api
development
```
