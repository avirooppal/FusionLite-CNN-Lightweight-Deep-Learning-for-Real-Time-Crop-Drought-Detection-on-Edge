import torch

model = torch.load('fusion_model.pth')
print(model)
model.eval()