import torch
from unet import UNet

model = UNet()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
script_model = torch.jit.script(model)
script_model.save("saved_models/converted_jit_model.pt")
print("Done")
