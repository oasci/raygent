<h1 align="center">raygent</h1>

<h4 align="center">Parallelism, Delegated</h4>

<p align="center">
    <a href="https://github.com/oasci/raygent/actions/workflows/tests.yml">
        <img src="https://github.com/oasci/raygent/actions/workflows/tests.yml/badge.svg" alt="Build Status ">
    </a>
    <!-- <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/raygent"> -->
    <a href="https://codecov.io/gh/oasci/raygent">
        <img src="https://codecov.io/gh/oasci/raygent/branch/main/graph/badge.svg" alt="codecov">
    </a>
    <!-- <a href="https://github.com/oasci/raygent/releases">
        <img src="https://img.shields.io/github/v/release/oasci/raygent" alt="GitHub release (latest by date)">
    </a> -->
    <a href="https://github.com/oasci/raygent/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/oasci/raygent" alt="License">
    </a>
    <a href="https://github.com/oasci/raygent/" target="_blank">
        <img src="https://img.shields.io/github/repo-size/oasci/raygent" alt="GitHub repo size">
    </a>
</p>

Raygent simplifies parallel execution in Python by providing an intuitive interface to Ray's distributed computing framework.
It removes boilerplate code and offers a modular approach to managing parallel tasks, making it easier to scale your computations across multiple cores or nodes.

## Features

-   **Simple Task-Based API**: Define your computational logic once and automatically scale across cores.
-   **Flexible Execution Modes**: Run tasks either in parallel using Ray or sequentially with a single parameter.
-   **Resource Optimization**: Automatically detect available CPU cores and manage resource allocation.
-   **Batched Processing**: Efficiently process data in optimal batch sizes.
-   **Intermediate Result Saving**: Save results at customizable intervals with pluggable savers.
-   **Error Handling**: Built-in error capture and logging.

## Quick Start

```python
from raygent import Task, TaskManager
from raygent.results.handlers import ResultsCollector

# Define your task
class SquareTask(
    Task[list[float], list[float]]
):
    def do(self, batch):
        return [item ** 2 for item in batch]

# Create a task manager
manager = TaskManager[
    list[float], ResultsCollector[list[float]]
](SquareTask, ResultsCollector, use_ray=True)

# Process items in parallel
handler = manager.submit_tasks(batch=[1., 2., 3., 4., 5.])
results = handler.get()  # [1., 4., 9., 16., 25.]
```

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
To enable `raygent`'s full parallelization capabilities using Ray, install it with the `ray` extra.

```python
pip install .[ray]
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
