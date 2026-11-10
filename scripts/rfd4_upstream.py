"""On-demand fetch of native RFD4-Proteina benchmark/test data.

Downloads the GitHub tarball of RosettaCommons/RFD4-Proteina at a pinned
commit (once) and extracts the `data/benchmarks` and `tests/data` subtrees
relative to the current working directory (run from the repository root), so
the native-parity scripts do not depend on a local upstream checkout. The
downloaded locations are gitignored.
"""

from __future__ import annotations

import io
import pathlib
import tarfile
import urllib.request

UPSTREAM_REPO = "RosettaCommons/RFD4-Proteina"
UPSTREAM_COMMIT = "6fc44c060ef96d4c759bafeaa1eae1a66a5be67c"


def _extract(prefix: str, dest: pathlib.Path) -> None:
    url = f"https://github.com/{UPSTREAM_REPO}/archive/{UPSTREAM_COMMIT}.tar.gz"
    print(f"fetching {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:8]} ...", flush=True)
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as tar:
            members = [
                m for m in tar.getmembers()
                if m.name.startswith(prefix) and m.isfile()
            ]
            for member in members:
                member.name = str(pathlib.PurePosixPath(member.name).relative_to(prefix))
            tar.extractall(dest, members=members, filter="data")


def ensure_benchmarks() -> pathlib.Path:
    """Return the benchmarks directory, downloading it if absent (run from the repo root)."""
    dest = pathlib.Path("data") / "benchmarks"
    if not dest.exists():
        _extract(f"RFD4-Proteina-{UPSTREAM_COMMIT}/data/benchmarks", dest)
    return dest


def ensure_upstream_tests_data() -> pathlib.Path:
    """Return the upstream tests/data directory, downloading it if absent (run from the repo root)."""
    dest = pathlib.Path("tests") / "data"
    if not dest.exists():
        _extract(f"RFD4-Proteina-{UPSTREAM_COMMIT}/tests/data", dest)
    return dest


if __name__ == "__main__":
    ensure_benchmarks()
    ensure_upstream_tests_data()
