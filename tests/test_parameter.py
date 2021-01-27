"""
Tests for DDSP Parameters
"""

import numpy as np
import torch.nn as nn
import torch.tensor as T

from ddspdrum.parameter import ParameterRange, TorchParameter


class TestParameterRange:

    def test_empty_construction(self):
        param_range = ParameterRange()
        assert param_range.minimum == 0.0
        assert param_range.maximum == 1.0
        assert param_range.curve_type == "linear"

    def test_to_0to1(self):
        # Test linear scaling
        param_range = ParameterRange(0.0, 10.0)
        assert param_range.to_0to1(5.0) == 0.5

        param_range = ParameterRange(0.0, 10.0)


class TestTorchParameter:

    def test_empty_construction(self):
        param = TorchParameter()
        assert issubclass(TorchParameter, nn.Parameter)
        assert isinstance(param, nn.Parameter)
        assert isinstance(param, TorchParameter)

    def test_construction(self):
        param = TorchParameter(T(1.0))
        assert param == 1.0
        assert param.item() == 1.0

    def test_get_float(self):
        param = TorchParameter(T(0.33))
        np.testing.assert_almost_equal(param.get_float(), 0.33)

    def test_get_in_range(self):
        # Test with no range first -- should get original value back
        param = TorchParameter(T(0.45))
        np.testing.assert_almost_equal(param.get_float(), 0.45)

        # Test with range now
        param_range = ParameterRange(0.0, 10.0)
        param = TorchParameter(T(0.5), parameter_range=param_range)
        np.testing.assert_almost_equal(param.get_in_range(), 5.0)

    def test_set_with_range(self):
        param = TorchParameter(T(0.5))