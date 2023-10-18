import sys
from mindspore.scipy.ops import LinearSumAssignment
from mindspore.scipy.utils import _mstype_check, _dtype_check
from mindspore.common import dtype as mstype
from mindspore import Tensor


def linear_sum_assignment(cost_matrix, maximize, dimension_limit=Tensor(sys.maxsize)):
    func_name = 'linear_sum_assignment'
    _mstype_check(func_name, cost_matrix, mstype.TensorType, 'cost_matrix')
    _mstype_check(func_name, dimension_limit,
                  mstype.TensorType, 'dimension_limit')
    _mstype_check(func_name, maximize, mstype.TensorType, 'maximize')
    _dtype_check(func_name, cost_matrix, [mstype.float32, mstype.float64])
    _dtype_check(func_name, dimension_limit, [mstype.int64])
    _dtype_check(func_name, maximize, [mstype.bool_])

    solve = LinearSumAssignment()
    solve.set_device(device_target="CPU")
    return solve(cost_matrix, dimension_limit, maximize)