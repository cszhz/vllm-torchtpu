# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: record the `actual` operand of every numpy assertion a test makes.

    RECORD_ACTUALS=out.npz pytest -p record_actuals ...

Running the same tests under two settings and comparing the two files byte for
byte (tests/glue/compare_actuals.py) checks the kernels' outputs are identical,
not merely within the tests' tolerances.
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

_store = {}
_current = {"id": None, "n": 0}


def _wrap(name):
    orig = getattr(npt, name)

    def wrapper(actual, *args, **kwargs):
        if _current["id"] is not None:
            a = np.asarray(actual)
            if a.dtype.kind in "fiub" or a.dtype.name.startswith(("bfloat", "float8")):
                key = f"{_current['id']}#{name}#{_current['n']}"
                _current["n"] += 1
                _store[key] = np.ascontiguousarray(a).reshape(-1).view(np.uint8).copy()
        return orig(actual, *args, **kwargs)

    setattr(npt, name, wrapper)
    setattr(np.testing, name, wrapper)


for _n in (
    "assert_allclose",
    "assert_array_equal",
    "assert_array_almost_equal",
    "assert_equal",
):
    _wrap(_n)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _current["id"], _current["n"] = item.nodeid, 0
    yield
    _current["id"] = None


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("RECORD_ACTUALS")
    if out:
        np.savez(out, **{k.replace("/", "|"): v for k, v in _store.items()})
        print(f"recorded {len(_store)} actuals -> {out}")
