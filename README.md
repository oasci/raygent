<h1 align="center">raygent</h1>

<h4 align="center">Parallelism, Delegated</h4>

<p align="center">
    <a href="https://github.com/oasci/raygent/actions/workflows/tests.yml">
        <img src="https://github.com/oasci/raygent/actions/workflows/tests.yml/badge.svg" alt="Build Status ">
    </a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/raygent">
    <a href="https://codecov.io/gh/oasci/raygent">
        <img src="https://codecov.io/gh/oasci/raygent/branch/main/graph/badge.svg" alt="codecov">
    </a>
    <a href="https://github.com/oasci/raygent/releases">
        <img src="https://img.shields.io/github/v/release/oasci/raygent" alt="GitHub release (latest by date)">
    </a>
    <a href="https://github.com/oasci/raygent/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/oasci/raygent" alt="License">
    </a>
    <a href="https://github.com/oasci/raygent/" target="_blank">
        <img src="https://img.shields.io/github/repo-size/oasci/raygent" alt="GitHub repo size">
    </a>
</p>

Raygent simplifies parallel execution in Python by providing an intuitive interface to Ray's distributed computing framework.
It removes boilerplate code and offers a modular approach to managing parallel tasks with directed acyclic graphs (DAG), making it easier to scale your workflow across multiple computational cores and nodes.

> [!CAUTION]
> Raygent is under active development.
> Breaking changes are could happen without warning.

## Features

- **Simple Task-Based API**: Define your computational logic once and automatically scale to any size.
- **Flexible Execution Modes**: Run tasks either in parallel using Ray or sequentially with a single parameter.
- **Resource Optimization**: Automatically detect available CPU cores and manage resource allocation.
- **Batched Processing**: Efficiently process data in optimal batch sizes.
- **DAG workflows**: Easy-to-use workflows using parallelized computational tasks (nodes) and data transformations (edges).

## Quick Start

### Defining Tasks

All workflows are built from raygent `Task`s that specify an independent calculation.
At minimum, `Task`s must have a `do` method that performs the desired computation.
Tasks are not limited to native Python; they can use NumPy arrays, Polars DataFrames, anything.

```python
from typing import override

from raygent import Task

class SquareTask(Task):
    @override
    def do(self, batch: list[int]) ->  list[int]:
        return [item ** 2 for item in batch]

class PrefactorTask(Task):
    @override
    def do(self, batch: list[int], factor: int = 1) -> list[int]:
        return [factor * x for x in batch]

class CombineTask(Task):
    @override
    def do(self, a: list[int], b: list[int]) -> dict[str, list[int]]:
        return {"a": a, "b": b}

class SumTask(Task):
    @override
    def do(self, batch: dict[str, list[int]]) ->  list[int]:
        return [a + b for a, b in zip(batch["a"], batch["b"])]
```

`Task`s must follow a few rules:

- Positional arguments take batches of data, not a single element.
- Only data can be passed through positional arguments; all other parameters must be specified with keyword arguments.
- Each `Task` should represent an isolated computation (i.e., no `Task` can call another `Task`).
- Internal multithreading in `do` methods should be explicitly annotated in the docstring.

Abiding by these rules will make creating workflows with a `DAG` a breeze.

> [!IMPORTANT]
> DAG workflows do not restrict `Tasks.do()` to a single thread; `Task`s could perform their own multithreading.
> This is a limitation of ray, so take this into account when specifying resources for DAG nodes.

### Creating workflows

Here is an example directed acyclic graph (DAG) workflow.
It creates two source queues (`source_1` and `source_2`) that inject data using our `Task`s defined in the previous section.

```mermaid
flowchart LR
    A[source_1] --> B((PrefactorTask))
    B --> C((SquareTask))
    D[source_2] --> E((SquareTask))

    C --> F((CombineTask))
    E --> F

    F --> G((SumTask))

    G --> I([sink_1])
    E --> H([sink_2])
```

This DAG also provides two sinks to receive messages from: `sink_1` is our final processed data, and `sink_2` provides messages from intermediate nodes.
Do note that queues consume computational resources; adding an excessive number of sinks will reduce your workflow's parallelization.

Below is how we build this DAG in raygent.

```python
from raygent.workflow import DAG

dag = DAG()

# Add source nodes we can send data into the DAG
# Returns a source node and a queue
source_n1, source_1 = dag.add_source()
source_n2, source_2 = dag.add_source()

# Add fully-connected nodes to process our workflow
#   Top
prefactor_n = dag.add(PrefactorTask(), inputs=source_n1, task_kwargs={"factor": 2})
square_n1 = dag.add(SquareTask(), inputs=prefactor_n)
#   Bottom
square_n2 = dag.add(SquareTask(), inputs=source_n2)
#   Merged
comb = dag.add(CombineTask(), inputs=(square_n1, square_n2))
summed = dag.add(SumTask(), inputs=comb)

# Attach sinks to get data out of our DAG
sink_1 = dag.add_sink(summed)
sink_2 = dag.add_sink(square_n2)
```

> [!NOTE]
> All inputs to a node must be defined before adding the task to the DAG.
> Defining nodes from source to sink helps prevent cycles.

### Running workflows

Injecting data into sources must be done with `DAG.stream()` to process data in batches.
It will handle batching into source queues and returning messages from all sinks.

```python
data1 = [1, 2, 3, 4]  # For source_1
data2 = [5, 6, 7, 8]  # For source_2

# Workflow starts waiting for source messages
dag.start()

# Streams data into sources and collects any sink messages
# Order of data specifies which source queue it goes into
for q_idx, msg in dag.stream(
    data1,
    data2,
    source_queues=(source_1, source_2),
    sink_queues=(sink_1, sink_2),
    batch_size=2,
    max_inflight=100,
    sink_wait=0.01,
):
    print(f"from sink #{q_idx} -> batch_idx={msg.index}, payload={msg.payload}")

# Releases all resources by terminating all DAG nodes
dag.stop()

# sink_1 payloads: [29, 52], [85, 128]
# sink_2 payloads: [25, 36], [49, 64]
```

> [!TIP]
> Sinks use queues (i.e., first-in, first-out); messages will not be in the order they are sent.

## Installation

You can install `raygent` directly from the [GitHub repository](https://github.com/oasci/raygent).
First, clone the [repository](https://github.com/oasci/raygent).

```bash
git clone git@github.com:oasci/raygent.git
```

Install `raygent` using `pip` after moving into the directory.

```sh
pip install .
```

This will install all dependencies and `raygent` into your current Python environment.
To enable `raygent`'s full parallelization capabilities with DAG workflows, install it with the `workflow` extra.

```python
pip install .[workflow]
```

## Development

We use [pixi](https://pixi.sh/latest/) to manage Python environments and simplify the developer workflow.
Once you have [pixi](https://pixi.sh/latest/) installed, move into `raygent` directory (e.g., `cd raygent`) and install the  environment using the command

```bash
pixi install
```

Now you can activate the new virtual environment using

```sh
pixi shell
```

## License

This project is released under the Apache-2.0 License as specified in `LICENSE.md`.
