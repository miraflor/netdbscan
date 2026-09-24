import math
import pytest
from netdbscan import NetDBSCANConfig

@pytest.mark.parametrize("value", [0,-1,float("nan"),float("inf")])
def test_eps_must_be_positive_finite(value):
    with pytest.raises(ValueError):
        NetDBSCANConfig(eps=value)


def test_inf_snap_limit_means_no_limit():
    NetDBSCANConfig(eps=1, max_snap_distance=math.inf)
