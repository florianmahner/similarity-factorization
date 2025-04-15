# srf

Similarity-based Representation Factorization (SRF).

### Installation


This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with your repository URL
    cd srf
    ```

2.  **Add the `tools` subdirectory as a subtree:**
    ```bash
    git subtree add --prefix srf git@github.com:florianmahner/ml-toolkit.git main --squash
    ```

3.  **Install Poetry:**
    Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation) if you don't have it installed.

4.  **Install dependencies:**
    This command installs the main project dependencies listed in `pyproject.toml`.
    ```bash
    poetry install
    ```

### TODOs and Open Questions

This is kind of a project page where we can keep track of pending tasks and future development ideas.

- [ ] Scaling ambiguity in $S = W A H^T$. Normalizing columns?
- [ ] Currently the project uses tri factor optimization based on block coordinate descent. ADMM might be more stable, not yet implemented
- [ ] Rank selection not working yet.
