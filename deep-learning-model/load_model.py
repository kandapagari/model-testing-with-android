import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
script_model_path = r'data/deeplabv3_resnet50.pt'

script_model = torch.jit.script(model)
script_model.save(script_model_path)

torch_mobile_model = optimize_for_mobile(script_model)
torch_mobile_model._save_for_lite_interpreter(script_model_path + 'l')
