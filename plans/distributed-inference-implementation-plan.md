# Distributed Inference Implementation Plan

## Problem

PyVBMC is useful for fitting expensive black-box Bayesian models on one machine. In practice, many users have access to internal clusters where the required workflow is to run several independent inference runs in parallel, compare the results, combine the useful runs, and keep enough records to understand what happened.

The problem to solve is operational:

- run many PyVBMC jobs without hand-written job scripts for every run;
- keep results organized across interrupted or partially failed cluster runs;
- combine completed runs into a usable posterior summary;
- generate diagnostics that show whether the combined result is sensible;
- make the workflow portable from local development to internal HPC systems.

## Objective

Build a small open-source tool for running, collecting, and combining repeated inference runs on local and distributed infrastructure.

The tool should let a user define an inference problem once, execute multiple runs, collect completed outputs, combine posterior approximations, and produce a report.

Cloud support is deferred. The first target is local development plus internal HPC-style execution because many users run models tied to local filesystems, protected data, licensed simulators, or institution-managed compute.

## Target User Workflow

Command line:

```bash
di init inference.yaml
di submit inference.yaml --backend mini-cluster --runs 64
di status runs/my-inference
di collect runs/my-inference
di combine runs/my-inference
di report runs/my-inference
```

Python:

```python
from distributed_inference import InferenceRunSet, MiniClusterBackend

runset = InferenceRunSet.from_yaml("inference.yaml")
backend = MiniClusterBackend.from_config("cluster.yaml")

runset.submit(backend, runs=64)
runset.collect()
posterior = runset.combine()
runset.report(posterior)
```

## What The Tool Provides

- A config format for inference problems.
- A runner for one PyVBMC fit.
- An `InferenceRunSet` abstraction for repeated runs.
- A local backend for testing.
- A mini-cluster backend for local cluster simulation.
- A small backend interface for later SLURM support.
- A run manifest format for provenance and recovery.
- A posterior-combination layer.
- Diagnostics for run quality and disagreement.
- Markdown or HTML reports.

## Scope

### In Scope

- Running independent PyVBMC jobs in parallel.
- Generating per-run seeds and initialization settings.
- Saving run outputs in a structured directory.
- Resuming collection after interruption.
- Combining completed posterior approximations.
- Reporting failures, warnings, and comparison metrics.
- Developing against `CCMI-CDT/mini-cluster`.
- Adding SLURM support after the mini-cluster backend works.

### Out Of Scope Initially

- Modifying PyVBMC internals.
- Implementing asynchronous active learning within a single PyVBMC run.
- Supporting every scheduler.
- Building a cloud service.
- Building a general distributed inference framework.

## Assumptions

- `mini-cluster` can act as a local stand-in for cluster-style execution.
- PyVBMC remains the inference engine.
- Independent PyVBMC runs are the first unit of parallelism.
- The tool should wrap PyVBMC rather than fork it.
- Each run should be reproducible from saved config, seed, package versions, and output files.

## Repository Access Note

The GitHub connector could not access `CCMI-CDT/mini-cluster` by the exact repository name at planning time. Stage 0 includes validating access and documenting the actual interface exposed by `mini-cluster`.

## Design Questions

1. What is the smallest backend interface needed for local, mini-cluster, and SLURM execution?
2. Which PyVBMC outputs must be saved to reload, sample, and compare completed runs?
3. What should the default posterior-combination method be?
4. How should failed, low-quality, duplicate, or near-duplicate runs be handled?
5. Which warnings should appear before producing a combined posterior report?
6. What output directory layout will remain stable across local, mini-cluster, SLURM, and later cloud execution?

## Package Shape

Package name:

```text
distributed-inference
```

Core concepts:

- `InferenceProblem`: target function, bounds, PyVBMC options, and run settings.
- `InferenceRun`: one PyVBMC execution with one seed and initialization.
- `InferenceRunSet`: the collection of runs for one inference problem.
- `CombinedPosterior`: the posterior object produced from completed runs.
- `Backend`: where runs execute.
- `Report`: diagnostics and summaries.

Core modules:

- `run`: execute one PyVBMC run from config.
- `runset`: manage repeated inference runs.
- `manifest`: record inference and run metadata.
- `combine`: combine completed VBMC approximations.
- `diagnostics`: compute checks and warnings.
- `report`: generate human-readable summaries.
- `backends`: submit, monitor, and collect jobs.

## Staged Plan

### Stage 0: Mini-Cluster Check And Project Scaffold

Goal: establish the local development substrate.

Tasks:

- Clone or vendor `CCMI-CDT/mini-cluster`.
- Document how jobs are submitted, monitored, logged, and failed.
- Create the Python project scaffold.
- Pin PyVBMC and core dependencies.
- Add a trivial mini-cluster smoke test.

Deliverables:

- `README.md`.
- `pyproject.toml`.
- `docs/mini-cluster-interface.md`.
- Smoke test that submits a trivial function and collects its output.

Acceptance criteria:

- A new checkout can run one local mini-cluster job.
- The mini-cluster execution model is documented enough to implement a backend.

### Stage 1: Single-Run PyVBMC Wrapper

Goal: make one PyVBMC run reproducible from a config file.

Tasks:

- Implement a single-run PyVBMC wrapper.
- Support YAML configuration for target, bounds, plausible bounds, seed, and PyVBMC options.
- Save run metadata, posterior object, ELBO, runtime, evaluation count, stdout, stderr, and logs.
- Add simple validation targets for smoke testing.

Deliverables:

- `src/distributed_inference/run.py`.
- `examples/targets.py`.
- `configs/baseline/*.yaml`.
- `docs/run-output-schema.md`.

Acceptance criteria:

- Running the same config and seed produces the same output structure.
- Simple validation targets complete successfully.

### Stage 2: InferenceRunSet Launcher

Goal: run many independent PyVBMC fits as separate jobs.

Tasks:

- Generate per-run configs with distinct seeds and initialization regions.
- Submit each run as an independent mini-cluster job.
- Track job status: queued, running, complete, failed, timed out.
- Support retries and partial result collection.
- Keep PyVBMC internal code unchanged.

Deliverables:

- `src/distributed_inference/runset.py`.
- `src/distributed_inference/backends/base.py`.
- `src/distributed_inference/backends/mini_cluster.py`.
- CLI commands: `submit`, `status`, `collect`.

Acceptance criteria:

- An `InferenceRunSet` can launch multiple independent PyVBMC runs.
- Completed runs can be collected after interruption.
- Failed runs are recorded without corrupting the runset directory.

### Stage 3: Manifests And Recovery

Goal: make inference output inspectable and recoverable.

Tasks:

- Define `inference.json` and per-run `run.json` schemas.
- Store package versions, git commit, config hash, seed, host, backend, and scheduler metadata.
- Store stdout, stderr, exception tracebacks, and PyVBMC termination reason.
- Add validation for incomplete or inconsistent runsets.

Deliverables:

- `src/distributed_inference/manifest.py`.
- JSON schema or Pydantic models for inference outputs.
- CLI command: `validate`.

Acceptance criteria:

- A partial or failed runset can be inspected without rerunning jobs.
- Invalid or incomplete runset directories produce actionable validation errors.

### Stage 4: Posterior Combination

Goal: combine completed PyVBMC runs into a single result object.

Tasks:

- Load variational posteriors from completed runs.
- Implement a first posterior-combination method.
- Support simpler fallback combinations, such as evidence-weighted mixtures, for comparison.
- Keep individual approximations available for inspection.
- Export a `CombinedPosterior` sampling interface.

Deliverables:

- `src/distributed_inference/combine.py`.
- CLI command: `combine`.
- `CombinedPosterior` sampling API.
- Comparison against individual best-run selection.

Acceptance criteria:

- The combined posterior can be sampled and summarized.
- The combined posterior can be compared with each individual run.

### Stage 5: Diagnostics And Reports

Goal: show users what happened across the inference runs.

Tasks:

- Report run-level ELBOs, runtimes, evaluation counts, and termination states.
- Detect duplicate or near-duplicate posterior components.
- Compare individual runs with the combined posterior.
- Flag runsets with one-mode collapse, high ELBO disagreement, many failures, or unstable combination weights.
- Generate a human-readable report.

Deliverables:

- `src/distributed_inference/diagnostics.py`.
- `src/distributed_inference/report.py`.
- CLI command: `report`.
- Markdown and HTML report outputs.

Acceptance criteria:

- Reports show whether combination changed the result relative to individual runs.
- Reports surface failed or suspicious runs clearly.

### Stage 6: Validation Scenarios

Goal: test the workflow against controlled examples.

Tasks:

- Add validation targets:
  - unimodal Gaussian;
  - banana/Rosenbrock;
  - Gaussian mixtures with separated modes;
  - narrow ring or funnel-style targets;
  - noisy likelihood variants.
- Compare:
  - single PyVBMC run;
  - best-of-N PyVBMC;
  - evidence-weighted multi-start mixture;
  - stacked posterior combination;
  - longer sequential PyVBMC under matched wall-clock budget.
- Run fixed likelihood-budget and fixed wall-clock-budget comparisons.
- Repeat across seeds and worker counts.

Deliverables:

- `validation/run_matrix.py`.
- `validation/metrics.py`.
- `reports/validation-summary.md`.
- Plotting scripts.

Acceptance criteria:

- Validation runs produce repeatable metrics.
- Results separate posterior quality, evidence quality, and wall-clock behavior.

### Stage 7: HPC Backend

Goal: move from mini-cluster to real internal HPC usage.

Tasks:

- Extract a minimal backend interface:

```python
class Backend:
    def submit(self, command: list[str], resources: dict) -> str: ...
    def status(self, job_id: str) -> str: ...
    def collect(self, job_id: str) -> JobResult: ...
    def cancel(self, job_id: str) -> None: ...
```

- Add a local process backend for testing.
- Add templates for SLURM job scripts.
- Consider HTCondor or PBS only after SLURM works.
- Avoid hard-coding institution-specific assumptions.

Deliverables:

- `src/distributed_inference/backends/local.py`.
- `src/distributed_inference/backends/slurm.py`.
- `docs/hpc-usage.md`.
- Example SLURM scripts.

Acceptance criteria:

- The same inference config can run on local, mini-cluster, and SLURM-style backends with backend-specific resource settings.

### Stage 8: Noisy Likelihood Handling

Goal: make warnings explicit when noisy likelihoods affect combined results.

Tasks:

- Support noisy likelihood outputs where PyVBMC supports them.
- Track noise estimates per run.
- Measure ELBO and evidence instability under noisy targets.
- Add warnings when combination weights appear dominated by noisy evidence estimates.
- Compare combination based on evidence, posterior diversity, and held-out checks where available.

Deliverables:

- Noisy validation configs.
- Evidence reliability diagnostics.
- Report section for evidence and weight stability.

Acceptance criteria:

- Noisy inference runs report evidence instability explicitly.
- The package warns before presenting a combined posterior when diagnostics indicate unstable evidence or weights.

### Stage 9: Packaging And Documentation

Goal: make the package usable outside the initial development environment.

Tasks:

- Add API documentation.
- Add tutorials:
  - local single-run baseline;
  - mini-cluster repeated inference runs;
  - combined posterior report;
  - SLURM deployment.
- Add tests for manifests, backends, combination, diagnostics, and CLI commands.
- Package the project cleanly.
- Open upstream discussion with PyVBMC maintainers once wrapper requirements are clear.

Deliverables:

- Documentation site or structured `docs/`.
- CI test suite.
- Example notebooks.
- Release checklist.

Acceptance criteria:

- A user can run repeated VBMC inference from documented setup steps.
- Core tests pass in a clean environment.

### Stage 10: Cloud Backend

Goal: reuse the backend abstraction for cloud execution after the local and HPC workflow is stable.

Tasks:

- Identify which backend methods map cleanly to cloud batch systems.
- Add object-storage-compatible run collection if needed.
- Add one prototype cloud backend.
- Preserve the same inference config and report format.

Deliverables:

- Cloud backend design note.
- Prototype backend for one cloud execution system.
- Cost and reproducibility notes.

Acceptance criteria:

- Cloud support behaves as another backend rather than a separate workflow.

## Suggested Directory Layout

```text
distributed-inference/
  README.md
  pyproject.toml
  plans/
    distributed-inference-implementation-plan.md
  docs/
    mini-cluster-interface.md
    hpc-usage.md
    posterior-combination.md
    run-output-schema.md
  src/
    distributed_inference/
      __init__.py
      cli.py
      run.py
      runset.py
      manifest.py
      combine.py
      diagnostics.py
      report.py
      backends/
        __init__.py
        base.py
        local.py
        mini_cluster.py
        slurm.py
  examples/
    targets.py
  validation/
    metrics.py
    run_matrix.py
  configs/
    baseline/
    multirun/
    noisy/
  tests/
  reports/
```

## Metrics

Inference metrics:

- Mode coverage.
- Posterior moment error where ground truth is available.
- Symmetrized KL or Gaussianized symmetrized KL.
- Error to known log marginal likelihood where available.
- Stability of combination weights across seeds.
- Agreement between individual posteriors and combined posterior.

Systems metrics:

- Wall-clock time to target quality.
- Number of completed, failed, and retried jobs.
- Scheduler overhead.
- Worker utilization.
- Queue wait time.
- Reproducibility under resume.

Usability metrics:

- Time from config to report.
- Number of manual steps required.
- Clarity of failure diagnostics.
- Portability across local, mini-cluster, and SLURM-like execution.

## Risks

- Posterior combination may improve mode coverage while degrading evidence reliability.
- PyVBMC posterior objects may not serialize cleanly across versions.
- Independent runs may be too similar unless initialization is deliberately diversified.
- Real HPC systems vary enough that one backend abstraction may be too narrow.
- The project may accumulate scheduler complexity that does not improve inference outputs.

## Mitigations

- Treat posterior quality and evidence quality as separate outputs.
- Persist raw run outputs and version metadata.
- Add explicit seed and initialization strategies.
- Start with mini-cluster plus SLURM only.
- Keep the core backend abstraction small: submit, status, collect, cancel.
- Avoid modifying PyVBMC until wrapper requirements are clear.

## Near-Term Milestones

### Week 1

- Validate access to `mini-cluster`.
- Create project scaffold.
- Run one trivial mini-cluster job.
- Run one sequential PyVBMC baseline.

### Week 2

- Implement baseline run wrapper.
- Define inference and run manifest schemas.
- Add simple validation targets.

### Week 3

- Implement independent `InferenceRunSet` launcher on mini-cluster.
- Add `submit`, `status`, and `collect` commands.
- Demonstrate interruption and recovery.

### Week 4

- Implement first posterior-combination method.
- Compare combined posterior to best individual run on Gaussian mixtures.
- Generate first diagnostic report.

### Week 5-6

- Expand validation scenarios.
- Add mode coverage and evidence diagnostics.
- Decide whether the next implementation focus is posterior combination, noisy likelihoods, or SLURM portability.

## Decision Point After Week 6

Use validation and usability results to choose the next stage:

1. If posterior combination improves multimodal recovery, expand validation scenarios and reporting.
2. If evidence estimates are unstable, prioritize evidence diagnostics and noisy-likelihood handling.
3. If execution friction dominates, prioritize SLURM support and documentation.
4. If independent runs are insufficient, evaluate whether asynchronous or batched active sampling should be added later.

## Later Extensions

- Asynchronous active sampling inside a single VBMC run.
- Batch acquisition functions for PyVBMC.
- Cost-aware noisy likelihood allocation.
- Cloud-native backend.
- Upstream integration with PyVBMC if the wrapper proves stable.

## References

- Acerbi, L. 2018. Variational Bayesian Monte Carlo. https://arxiv.org/abs/1810.05558
- Acerbi, L. 2020. Variational Bayesian Monte Carlo with Noisy Likelihoods. https://arxiv.org/abs/2006.08655
- Huggins et al. 2023. PyVBMC: Efficient Bayesian inference in Python. https://doi.org/10.21105/joss.05428
- PyVBMC repository. https://github.com/acerbilab/pyvbmc
- Stacking VBMC. https://arxiv.org/abs/2504.05004
- Planned local substrate: https://github.com/CCMI-CDT/mini-cluster.git
