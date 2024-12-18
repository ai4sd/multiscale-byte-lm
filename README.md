# Multiscale Byte Language Model

Multiscale Byte Language Model is a hierarchical byte-level sequence-to-sequence model for multimodal tasks.

<p align="center">
    <img src="assets/mblm.png" alt="mblm-architecture" width="600"/>
</p>

## Quickstart

TBD.

## Development setup

We use `uv` for packaging and dependency management. Before proceeding, install a recent version (>= `0.4.17`) via the instructions on [the homepage](https://docs.astral.sh/uv/getting-started/installation/).

### Install dependencies

- With CUDA: `make install_cuda`
- CPU only (e.g., MacOS): `make install_cpu`

### Environment and Datasets

If you want to run the analysis notebooks and work with the datasets locally, you will need to create an `.env` file in the project root directory with the following entries:

```
DATASET_PG19_DIR= # Path to the PG19 dataset on you system
DATASET_CLEVR_DIR= # Path to the Clevr dataset on you system
```

Note that all paths to datasets must be _absolute_. The datasets can be downloaded here:

- [PG19](https://github.com/google-deepmind/pg19)
- [Clevr](https://cs.stanford.edu/people/jcjohns/clevr/) v1.0

## Running scripts

- Project-related tasks (e.g., installing dependencies, running tests) are defined in the [Makefile](Makefile)

## Dependency management

Most of the dependencies are installed via `uv`. However, some dependencies that require no build isolation, special index urls or a specific build environment are installed via `uv pip` directly, see the Makefile.

- Whenever **Mamba/SSM dependencies** are upgraded (currently, `mamba-ssm` and `causal-conv1d`), set the versions in the Makefile

### Installing Mamba dependencies

If you've noticed, there are two SSM/Mamba dependencies:

- `mambapy`, defined in `pyproject.toml`
- `mamba-ssm` (with `causal-conv1d`), defined in `Makefile`

Because the official Mamba implementation `mamba-ssm` requires a Linux machine and a GPU available during installation, we shim the dependencies. `mambapy` is used as a fallback for all unsupported platforms or when `mamba-ssm` is not installed. Because `mamba-ssm` is so delicate, it needs to be installed manually:

```sh
make install_mamba
```

For any experiments, we wish to use the new Mamba 2 block from `mamba-ssm`. If the import of this module fails, we fall back to a Mamba 1 block from `mambapy`, which is written in pure PyTorch.

## Pre-Commit Hooks

Before every commit, we lint the _staged_ Python and Jupyter Notebook files and check if they are formatted correctly. Doing this locally speeds up development because one does not have to wait for the CI to catch issues. Errors of these checks are not fixed automatically, instead, you will have to fix the files yourself before committing. You may bypass hooks with `git commit -m <message> --no-verify`. However, the CI will likely fail in this case.

All Pre-commit hooks can be run manually as well:

- `pre-commit run lint`
- `pre-commit run check-format`

Note that:

- The `lint` command is similar to the `make lint` command, but the `make` command operates on _all_ files in the project and not just the staged files
- While `check-format` simply _checks_ the format, `make format` will _actually_ format the files

## Guidelines

Don't forget to follow the guidelines decided here [here](https://github.ibm.com/AI4SD/ai4sd-misc/blob/main/markdown/coding_guidelines.md).

## Citation

TBD.
