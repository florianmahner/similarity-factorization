"""Download the preprocessed data (similarity matrices, consensus embeddings) from OSF.

Set ``OSF_PROJECT_ID`` to the project's id, then run::

    make data                                  # or:
    poetry run python scripts/download_data.py

The raw source datasets (THINGS images, NSD, macaque recordings) are obtained from
their original providers; see the data-availability statement in the paper.
"""

from __future__ import annotations

from pathlib import Path

OSF_PROJECT_ID = ""  # TODO: set to the OSF project id, e.g. "ab3cd"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    if not OSF_PROJECT_ID:
        raise SystemExit(
            "Set OSF_PROJECT_ID in scripts/download_data.py to the OSF project id."
        )
    try:
        from osfclient.api import OSF
    except ImportError:
        raise SystemExit("osfclient is required: pip install osfclient")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    project = OSF().project(OSF_PROJECT_ID)
    for store in project.storages:
        for remote in store.files:
            target = DATA_DIR / remote.path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {remote.path}")
            with open(target, "wb") as fh:
                remote.write_to(fh)


if __name__ == "__main__":
    main()
