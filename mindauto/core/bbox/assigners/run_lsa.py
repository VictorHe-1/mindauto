from mindspore.scipy.ops import LinearSumAssignment
from mindspore import Tensor, nn


class NetLsap(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = LinearSumAssignment()

    def construct(self, cost_matrix, maximize, dimension_limit):
        return self.op(cost_matrix, dimension_limit, maximize)
