# Development

This comprehensive guide provides detailed instructions to help maintainers effectively develop, test, document, build, and release new versions of `raygent`.

## Setting up the Development Environment

`raygent` utilizes [`pixi`](https://pixi.sh/latest/) for managing environments and dependencies, streamlining the setup process.
Follow these precise steps to configure your development environment:

1.  **Clone the repository:**
    Begin by obtaining a local copy of the `raygent` codebase:

    ```bash
    git clone git@github.com:oasci/raygent.git
    cd raygent
    ```
2.  **Install dependencies:**
    Install all necessary dependencies by running:

    ```bash
    pixi install
    ```
3.  **Activate the development environment:**
    To enter the isolated virtual environment configured specifically for `raygent` development, execute:

    ```bash
    pixi shell
    ```

You are now fully prepared and equipped to develop `raygent`.

## Code Formatting and Style Guide

Maintaining consistent style and formatting across the codebase is crucial for readability and maintainability.
`raygent` employs automated formatting tools configured to enforce standardized style guidelines.
Execute the following command to apply formatting automatically:

```bash
pixi run format
```

This command sequentially runs `ruff` for Python formatting, `isort` for managing imports, and `markdownlint-cli2` to enforce markdown formatting standards, ensuring your contributions align with project conventions.

## Documentation

`raygent`'s documentation is built using MkDocs, allowing easy creation and maintenance of high-quality documentation.
To locally preview documentation changes, serve the documentation by running:

```bash
pixi run -e docs serve-docs
```

After execution, open your web browser and visit [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) to review changes in real-time.

## Testing

Writing and maintaining tests is essential for ensuring code correctness, reliability, and stability.
Execute `raygent`'s tests with:

```bash
pixi run -e dev tests
```

Additionally, you can evaluate test coverage to identify untested areas and improve overall reliability by running:

```bash
pixi run -e dev coverage
```

Review the generated coverage reports to address any gaps in testing.

## Bumping Version

`raygent` uses [setuptools-scm](https://pypi.org/project/setuptools-scm/) to automatically derive version numbers from Git tags.
To bump the version for a new release, you simply need to add a new Git tag.
Use [semantic versioning](https://semver.org/) (e.g., `v1.2.0`) to reflect the nature of the changes:

-   **MAJOR**: Incompatible API changes
-   **MINOR**: Backward-compatible new features
-   **PATCH**: Bug fixes and minor improvements

Run the following command to create a new tag:

```bash
git tag vX.Y.Z
```

Then push the tag to the remote repository:

```bash
git push origin vX.Y.Z
```

> [!note]
> Do not manually set the version in `pyproject.toml`.
> It is managed automatically through Git tags.

After tagging, you're ready to [Build the Package](#building-the-package) and [Publish to PyPI](#publishing-to-pypi).

## Building the Package

Prepare `raygent` for publishing or distribution by building the package.
Execute:

```bash
pixi run build
```

Upon completion, inspect the `dist` directory for the generated distribution files, which are ready for publication.

## Publishing to PyPI

Once the version number is updated and the package is built, it can be published to PyPI.

```bash
pixi run publish
```

For preliminary testing or release candidates, it is highly recommended to publish to TestPyPI first.

```bash
pixi run publish-test
```

Publishing to TestPyPI allows you to validate packaging correctness and installation processes without affecting production users.

## Maintenance Best Practices

To maintain high quality and reliability of `raygent`, adhere to the following best practices:

-   Regularly synchronize your local repository with the main branch to incorporate the latest updates:

    ```bash
    git pull origin main
    ```
-   Frequently review and address open issues and pull requests on GitHub.
-   Clearly document changes in commit messages, issue descriptions, and pull requests.
-   Routinely verify dependencies and update them as necessary to maintain compatibility and security.
