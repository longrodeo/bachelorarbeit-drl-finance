# tests/integration/test_data_sanity.py
import pytest
from validation.sanity import check_data

@pytest.mark.integration
def test_data_sanity():
    ok, combined, _ = check_data()
    if not ok:
        msg = "\n".join(["[WARN] "+w for w in combined.warns] + ["[FAIL] "+e for e in combined.errors])
        pytest.fail(msg)
