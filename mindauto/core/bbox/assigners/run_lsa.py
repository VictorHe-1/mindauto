from mindspore.scipy.ops import LinearSumAssignment
from mindspore.scipy.utils import _mstype_check, _dtype_check
from mindspore.common import dtype as mstype
from mindspore import Tensor, nn


class NetLsap(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = LinearSumAssignment().set_device(device_target='CPU')

    def construct(self, cost_matrix, maximize, dimension_limit):
        func_name = 'linear_sum_assignment'
        _mstype_check(func_name, cost_matrix, mstype.TensorType, 'cost_matrix')
        _mstype_check(func_name, dimension_limit,
                      mstype.TensorType, 'dimension_limit')
        _mstype_check(func_name, maximize, mstype.TensorType, 'maximize')
        _dtype_check(func_name, cost_matrix, [mstype.float32, mstype.float64])
        _dtype_check(func_name, dimension_limit, [mstype.int64])
        _dtype_check(func_name, maximize, [mstype.bool_])
        return self.op(cost_matrix, dimension_limit, maximize)
