# convert pytorch model to mindspore
import json

from torchvision.models import resnet50

import mindspore as ms

# load param_map json
with open("mapping.json", "r") as json_file:
    param_name_map = json.load(json_file)

net = resnet50(pretrained=True)

params_dict = net.state_dict()
# conversion
ms_params = []
for name, value in params_dict.items():
    each_param = dict()
    if name not in param_name_map:
        continue
    each_param["name"] = param_name_map[name]
    each_param["data"] = ms.Tensor(value.numpy())
    ms_params.append(each_param)

ms.save_checkpoint(ms_params, "resnet50_ms.ckpt")
