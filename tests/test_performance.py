import pytest
from pnm.performance import distribution


def test_percentiles_and_empty_measurement_are_explicit():
    result=distribution([1,2,3,4])
    assert result['p50']==2.5
    assert result['p95']==pytest.approx(3.85)
    assert result['p99']==pytest.approx(3.97)
    assert distribution([])['p50'] is None
