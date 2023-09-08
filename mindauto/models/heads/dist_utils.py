from mindspore.ops import operations as P
import mindspore as ms


# Warning: this api mmdet.core.reduce_mean and not tested yet
def reduce_mean(tensor):
    tensor = tensor.copy()
    all_reduce_op = P.AllReduce(P.ReduceOp.SUM, ms.context.get_context("device_num"))
    tensor = all_reduce_op(tensor)
    tensor = tensor / ms.context.get_context("device_num")
    return tensor
