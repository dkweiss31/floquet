# Contributing to floquet

We welcome your contribution, whether it be additional cost functions, new functionality, better documentation, etc.

## Requirements

The project was written using Python 3.10+, you must have a compatible version of Python (i.e. >= 3.10) installed on your computer. Building the documentation (see below) requires Python 3.11+, since that's the minimum supported by [Zensical](https://zensical.org/).

## Setup

Clone the repository:

```shell
git clone git@github.com:dkweiss31/floquet.git
cd floquet
```

It is good practice to use a virtual environment to install the dependencies, such as conda. Once this environment has been activated, you can run 

```shell
pip install -e .
```

to install the package and its dependencies. As a developer you also need to install the developer dependencies:

```shell
pip install -e ".[dev]"
```

## Code style

This project follows PEP8 and uses automatic formatting and linting tools to ensure that the code is compliant.

## Workflow

### Before submitting a pull request (run all tasks)

Run all tasks before each commit:

```shell
task all
```

### Build the documentation

The documentation is built using [Zensical](https://zensical.org/) with the Material for MkDocs `classic` theme. Zensical reads the same `mkdocs.yml` and generates a static website based on the markdown files in the `docs/` directory.

To preview the changes to the documentation as you edit the docstrings or the markdown files in `docs/`, we recommend starting a live preview server, which will automatically rebuild the website upon modifications:

```shell
task docserve
```

Open <http://localhost:8000/floquet/> in your web browser to preview the documentation website (the `/floquet/` path mirrors `site_url` in `mkdocs.yml`).

You can build the static documentation website locally with:

```shell
task docbuild
```

This will create a `site/` directory with the contents of the documentation website. You can then simply open `site/index.html` in your web browser to view the documentation website.

### Example notebooks

`docs/examples/*.ipynb` are the source of truth for the tutorial notebooks, and are shipped as-is so readers can download and run them directly (linked from the top of each rendered page). Since Zensical can't render notebooks itself, each `.ipynb` is also exported to a `.md` page that Zensical does render. If you edit one of the notebooks, regenerate its `.md` export with:

```shell
task docs-convert-notebooks
```

### Run specific tasks

You can also execute tasks individually:

```shell
> task --list
lint         lint the code (ruff)
format       auto-format the code (ruff)
codespell    check for misspellings (codespell)
clean        clean the code (ruff + codespell)
test         run the unit tests suite (pytest)
docbuild     build the documentation website
docserve     preview documentation website with hot-reloading
all          run all tasks before a commit (ruff + codespell + pytest)
ci           run all the CI checks
```
