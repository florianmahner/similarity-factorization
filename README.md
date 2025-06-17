# srf

Similarity-based Representation Factorization (SRF).

### Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and includes the `ml-toolkit` as a git subtree in the `srf/ml-toolkit` directory.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/florianmahner/srf.git
    cd srf
    ```

2.  **[Optional] If ml-toolkit directory is missing or needs updating:**
    The ml-toolkit directory should be included when you clone the repository. However, if you need to add or update it manually:
    ```bash
    # To add the tools subtree if missing:
    git subtree add --prefix=src/tools https://github.com/florianmahner/tools.git main --squash

    # To update the tools subtree later:
    git subtree pull --prefix=src/tools https://github.com/florianmahner/tools.git main --squash
    ```

3.  **Install Poetry:**
    Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation) if you don't have it installed.

4.  **Install dependencies:**
    This command installs the main project dependencies and the ml-toolkit package:
    ```bash
    poetry install
    ```

### Project Structure

- `srf/`: Main project code
- `srf/ml-toolkit/`: Machine Learning toolkit (included as a git subtree) providing utilities for data handling, feature extraction, and analysis

### TODOs and Open Questions

This is kind of a project page where we can keep track of pending tasks and future development ideas.

- [ ] Scaling ambiguity in $S = W A H^T$. Normalizing columns?
- [ ] Currently the project uses tri factor optimization based on block coordinate descent. ADMM might be more stable, not yet implemented
- [ ] Rank selection not working yet.
