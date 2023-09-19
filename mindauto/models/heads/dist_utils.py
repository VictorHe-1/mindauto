from mindspore.ops import ReduceOp
from mindspore import ops
import mindspore as ms


# Warning: this api mmdet.core.reduce_mean and not tested yet
def reduce_mean(tensor):
    tensor = tensor.copy()
    all_reduce_op = ops.AllReduce(ReduceOp.SUM)
    tensor = all_reduce_op(tensor)
    tensor = tensor / ms.context.get_auto_parallel_context('device_num')
    return tensor
