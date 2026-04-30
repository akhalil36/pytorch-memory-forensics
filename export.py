import torch
from model import CNN

model = CNN()
model.load_state_dict(torch.load("cnn_model.pt", map_location='cpu'))

model.eval()
model.to('cpu')

torchscript_model = torch.jit.script(model)
torchscript_model.save('cnn_model_torchscript.pt')
print("TorchScript model saved as cnn_model_torchscript.pt")