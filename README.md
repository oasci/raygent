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
-   **Flexible Execution Modes**: Run tasks either in parallel (using Ray) or sequentially with a single parameter.
-   **Resource Optimization**: Automatically detect available CPU cores and manage resource allocation.
-   **Chunked Processing**: Efficiently process data in optimal batch sizes.
-   **Intermediate Result Saving**: Save results at customizable intervals with pluggable savers.
-   **Error Handling**: Built-in error capture and logging.

## Why Raygent?

Parallelizing code with Ray requires significant boilerplate and infrastructure management.
Raygent abstracts away this complexity with a clean, task-focused API that lets you concentrate on your actual computation logic rather than parallelization mechanics.

## Quick Start

```python
from raygent import Task, TaskManager

# Define your task
class SquareTask(Task):
    def process_item(self, item):
        return item ** 2

# Create a task manager
manager = TaskManager(SquareTask, use_ray=True)

# Process items in parallel
manager.submit_tasks(items=[1, 2, 3, 4, 5])
results = manager.get_results()
print(results)  # [1, 4, 9, 16, 25]
```

## Installation

Clone the [repository](https://github.com/oasci/raygent):

```bash
git clone git@github.com:oasci/raygent.git
```

Install `raygent` using `pip` after moving into the directory.

```sh
pip install .
```

This will install all dependencies and `raygent` into your current Python environment.

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
